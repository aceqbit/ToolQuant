"""Single-group sensitivity probe: quantize one group to 4-bit, all others stay full precision.

Writes results/sensitivity_partial_<tag>.json after every group (resumable) and, when finished,
results/sensitivity_<model>_<utc>.json plus a CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from toolquant.quant.fakequant import GROUP_SIZE, quantized
from toolquant.quant.groups import derive_groups

RESULTS = Path("results")


def _score(model: Any, tok: Any, tasks: list[Any], batch_size: int) -> dict[str, Any]:
    from toolquant.evalrun.generate import generate_hf
    from toolquant.scoring.metrics import aggregate, score_output

    texts = generate_hf(model, tok, tasks, batch_size)
    recs = [score_output(t, texts[t.id]) for t in tasks]
    agg = aggregate(recs)
    return {
        "success": agg["tool_call_success"],
        "missed_call": agg["missed_call"],
        "silent_failure": agg["silent_failure"],
        "per_prompt": {r["id"]: r["outcomes"]["tool_call_success"] for r in recs},
    }


def run_probe(
    model: Any,
    tok: Any,
    tasks: list[Any],
    ppl_ids: Any,
    granularity: str = "block",
    bits: int = 4,
    group_size: int = GROUP_SIZE,
    batch_size: int = 16,
    partial_path: Path | None = None,
    limit_groups: int | None = None,
) -> dict[str, Any]:
    from toolquant.data.wikitext import perplexity

    groups = derive_groups(model, granularity)
    names = list(groups)[:limit_groups]
    state: dict[str, Any] = {"baseline": None, "groups": {}}
    if partial_path and partial_path.exists():
        state = json.loads(partial_path.read_text(encoding="utf-8"))

    def save() -> None:
        if partial_path:
            partial_path.parent.mkdir(exist_ok=True)
            partial_path.write_text(json.dumps(state), encoding="utf-8")

    if state["baseline"] is None:
        base = _score(model, tok, tasks, batch_size)
        base["ppl"] = perplexity(model, ppl_ids)
        state["baseline"] = base
        save()
        print("baseline", base["success"], base["ppl"], flush=True)
    base = state["baseline"]
    for g in names:
        if g in state["groups"]:
            continue
        t0 = time.time()
        with quantized(model, groups[g], bits, group_size):
            r = _score(model, tok, tasks, batch_size)
            r["ppl"] = perplexity(model, ppl_ids)
        r["sensitivity"] = base["success"] - r["success"]
        r["ppl_change"] = r["ppl"] - base["ppl"]
        state["groups"][g] = r
        save()
        print(
            g, r["sensitivity"], round(r["ppl_change"], 4), f"{time.time() - t0:.0f}s", flush=True
        )
    return state


def finalize(state: dict[str, Any], meta: dict[str, Any], results_dir: Path = RESULTS) -> Path:
    rows = [
        {
            "group": g,
            "tool_call_drop": v["sensitivity"],
            "ppl_change": v["ppl_change"],
            "success": v["success"],
            "ppl": v["ppl"],
        }
        for g, v in state["groups"].items()
    ]
    rho = spearmanr([r["tool_call_drop"] for r in rows], [r["ppl_change"] for r in rows])
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = meta["model"].split("/")[-1].lower()
    out = {
        **meta,
        "utc": utc,
        "baseline": state["baseline"],
        "groups": state["groups"],
        "spearman": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
    }
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"sensitivity_{tag}_{utc}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    from toolquant.data.wikitext import load_slice
    from toolquant.evalrun.generate import load_hf
    from toolquant.evalrun.run_eval import (
        DEFAULT_HF_PATH,
        MODEL_ID,
        get_tasks,
        hardware,
        model_revision,
    )
    from toolquant.tracking import git_sha, package_versions

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_HF_PATH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--granularity", default="block", choices=["block", "attn_mlp"])
    ap.add_argument("--limit-groups", type=int)
    ap.add_argument("--probe-limit", type=int)
    a = ap.parse_args()

    model, tok = load_hf(a.model_path, a.dtype, a.device)
    tasks = get_tasks("probe", a.probe_limit)
    ids = load_slice(tok)
    tag = f"{MODEL_ID.split('/')[-1].lower()}_{a.granularity}_{a.dtype}"
    state = run_probe(
        model,
        tok,
        tasks,
        ids,
        a.granularity,
        batch_size=a.batch_size,
        partial_path=RESULTS / f"sensitivity_partial_{tag}.json",
        limit_groups=a.limit_groups,
    )
    meta = {
        "model": MODEL_ID,
        "revision": model_revision(a.model_path),
        "seed": 0,
        "granularity": a.granularity,
        "bits": 4,
        "group_size": GROUP_SIZE,
        "dtype": a.dtype,
        "git_sha": git_sha(),
        "packages": package_versions(),
        "hardware": hardware(),
        "n_probe": len(tasks),
        "ppl_slice": {"n_seq": 64, "seq_len": 512, "dataset": "wikitext-2-raw-v1 test"},
    }
    print("wrote", finalize(state, meta))


if __name__ == "__main__":
    main()
