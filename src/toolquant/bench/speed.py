"""Speed benchmark: 128-token prompt, 128 new tokens, 5 warm-up + 20 timed runs.

Run one model per process so peak resident memory belongs to that model:
    uv run python -m toolquant.bench.speed --onnx-dir data/cache/onnx/int4 --name int4
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROMPT_TOKENS = 128
OUTPUT_TOKENS = 128
WARMUP = 5
RUNS = 20
RESULTS = Path("results")
_TEXT = (
    "The quick brown fox jumps over the lazy dog while the engineer checks the weather report "
    "for Paris, books a table for two at seven, and asks for the stock price of a company. "
)


def fixed_prompt_ids(tok: Any, n: int = PROMPT_TOKENS) -> list[int]:
    ids = tok(_TEXT * 20).input_ids
    return ids[:n]


def summarize_runs(runs: list[dict[str, float]]) -> dict[str, float]:
    ttft = np.array([r["ttft_s"] for r in runs])
    total = np.array([r["total_s"] for r in runs])
    decode = np.array([r["decode_tps"] for r in runs])
    return {
        "ttft_ms_mean": float(ttft.mean() * 1000),
        "ttft_ms_p95": float(np.percentile(ttft, 95) * 1000),
        "decode_tokens_per_s_mean": float(decode.mean()),
        "decode_tokens_per_s_std": float(decode.std()),
        "latency_s_mean": float(total.mean()),
        "latency_s_p95": float(np.percentile(total, 95)),
        "n_runs": len(runs),
    }


def one_run(model: Any, ids: list[int], out_tokens: int) -> dict[str, float]:
    import onnxruntime_genai as og

    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False, max_length=len(ids) + out_tokens, min_length=len(ids) + out_tokens
    )
    t0 = time.perf_counter()
    gen = og.Generator(model, params)
    gen.append_tokens(ids)
    gen.generate_next_token()
    t1 = time.perf_counter()
    n = 1
    while not gen.is_done() and n < out_tokens:
        gen.generate_next_token()
        n += 1
    t2 = time.perf_counter()
    return {
        "ttft_s": t1 - t0,
        "total_s": t2 - t0,
        "decode_tps": (n - 1) / (t2 - t1) if n > 1 else 0.0,
        "tokens": n,
    }


def main() -> None:
    import onnxruntime_genai as og
    from transformers import AutoTokenizer

    from toolquant.bench.memory import peak_rss_bytes
    from toolquant.evalrun.run_eval import hardware
    from toolquant.tracking import git_sha, package_versions

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hf-path", default="data/cache/qwen3-0.6b-hf")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--runs", type=int, default=RUNS)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.hf_path)
    ids = fixed_prompt_ids(tok)
    model = og.Model(a.onnx_dir)
    for _ in range(a.warmup):
        one_run(model, ids, OUTPUT_TOKENS)
    runs = [one_run(model, ids, OUTPUT_TOKENS) for _ in range(a.runs)]
    d = Path(a.onnx_dir)
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    res = {
        "name": a.name,
        "utc": utc,
        "prompt_tokens": len(ids),
        "output_tokens": OUTPUT_TOKENS,
        "warmup": a.warmup,
        "summary": summarize_runs(runs),
        "peak_rss_mb": peak_rss_bytes() / 1e6,
        "model_size_mb": sum(p.stat().st_size for p in d.iterdir() if "onnx" in p.name) / 1e6,
        "hardware": hardware(),
        "ort_threads": "onnxruntime-genai default (no override)",
        "git_sha": git_sha(),
        "packages": package_versions(),
        "runs": runs,
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"bench_{a.name}_{utc}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(path, json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
