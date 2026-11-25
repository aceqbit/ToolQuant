"""ONNX Runtime tuning sweep (prompt 2.4): threads x graph optimisation level.

    uv run python scripts/tune_ort.py int4          # model dir name under data/cache/onnx

Each setting runs the Phase 1 benchmark (128-token prompt, 128 new tokens, 5 warm-up, 20 runs)
in its own process. IO binding for the KV cache is not exposed by onnxruntime-genai (it manages
past/present buffers itself), so it is not swept; recorded in the result file.
Writes results/tuning_<utc>.json with the change vs the Phase 1 defaults.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

THREADS = [2, 4, 6, 8, 10, 12]
LEVELS = ["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"]


def run(name: str, args: list[str], tag: str) -> dict:
    cmd = [
        sys.executable, "-m", "toolquant.bench.speed",
        "--onnx-dir", f"data/cache/onnx/{name}", "--name", name, "--tag", tag, *args,
    ]  # fmt: skip
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    path = out.stdout.strip().split()[0]
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row(res: dict) -> dict:
    s = res["summary"]
    return {
        "threads": res["ort_threads"],
        "opt_level": res["ort_opt_level"],
        "decode_tps": s["decode_tokens_per_s_mean"],
        "ttft_ms": s["ttft_ms_mean"],
        "peak_rss_mb": res["peak_rss_mb"],
    }


def main(name: str) -> None:
    base = run(name, [], "_tune_default")
    rows = [row(base)]
    for t in THREADS:
        rows.append(row(run(name, ["--threads", str(t)], f"_tune_t{t}")))
    best_t = max(rows[1:], key=lambda r: r["decode_tps"])["threads"]
    for lvl in LEVELS:
        rows.append(row(run(name, ["--threads", str(best_t), "--opt-level", lvl], f"_tune_{lvl}")))
    best = max(rows, key=lambda r: r["decode_tps"])
    d = rows[0]
    out = {
        "utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "model": name,
        "hardware": base["hardware"],
        "rows": rows,
        "phase1_default": d,
        "best": best,
        "change_pct": {
            "decode_tps": (best["decode_tps"] / d["decode_tps"] - 1) * 100,
            "ttft_ms": (best["ttft_ms"] / d["ttft_ms"] - 1) * 100,
            "peak_rss_mb": (best["peak_rss_mb"] / d["peak_rss_mb"] - 1) * 100,
        },
        "io_binding": "not swept: onnxruntime-genai manages KV buffers internally",
        "cpu_count": os.cpu_count(),
    }
    p = Path("results") / f"tuning_{out['utc']}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main(sys.argv[1])
