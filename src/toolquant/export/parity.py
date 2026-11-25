"""Simulation versus real: fake-quantized PyTorch vs the ONNX model built from the same map.

(a) Greedy next-token agreement on 200 prompts: both models decode N tokens; we record whether
    the first token agrees, the position of first divergence, and whether all N tokens agree.
(b) Tool-call success on 100 validation prompts for both (full 256-token generation).
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from toolquant.quant.fakequant import apply_map

RESULTS = Path("results")
N_TOKENS = 32


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first differing token; min(len) if one is a prefix of the other."""
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def agreement_stats(pt: dict[str, list[int]], ox: dict[str, list[int]]) -> dict[str, Any]:
    ids = sorted(pt)
    div = {i: first_divergence(pt[i], ox[i]) for i in ids}
    same_len = {i: len(pt[i]) == len(ox[i]) for i in ids}
    full = [div[i] == len(pt[i]) and same_len[i] for i in ids]
    first = [bool(pt[i]) and bool(ox[i]) and pt[i][0] == ox[i][0] for i in ids]
    return {
        "n": len(ids),
        "next_token_agreement": sum(first) / len(ids),
        "full_prefix_agreement": sum(full) / len(ids),
        "mean_first_divergence": sum(div.values()) / len(ids),
        "per_prompt_first_divergence": div,
    }


def pt_tokens(model: Any, tok: Any, tasks: list[Any], n_tokens: int) -> dict[str, list[int]]:
    import torch

    from toolquant.evalrun.generate import build_prompt

    out = {}
    for t in tasks:
        enc = tok(build_prompt(tok, t), return_tensors="pt")
        with torch.no_grad():
            g = model.generate(
                **enc,
                max_new_tokens=n_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        out[t.id] = g[0, enc.input_ids.shape[1] :].tolist()
    return out


def run_parity(
    name: str,
    bits_per_group: dict[str, int],
    onnx_dir: str,
    hf_path: str,
    granularity: str = "block",
    n_prompts: int = 200,
    n_success_prompts: int = 100,
) -> Path:
    from toolquant.data.splits import load_splits
    from toolquant.evalrun.generate import generate_hf, generate_onnx, load_hf
    from toolquant.evalrun.run_eval import get_tasks, score_texts, summarize
    from toolquant.quant.groups import derive_groups

    model, tok = load_hf(hf_path, "float32")
    groups = derive_groups(model, granularity)
    val = get_tasks("validation")
    rng = random.Random(0)
    chosen = rng.sample(val, n_prompts)
    sp = load_splits()
    assert not {t.id for t in chosen} & set(sp["probe"])
    with apply_map(model, groups, bits_per_group):
        pt = pt_tokens(model, tok, chosen, N_TOKENS)
        succ_tasks = chosen[:n_success_prompts]
        pt_texts = generate_hf(model, tok, succ_tasks, batch_size=8)
    ox = generate_onnx(onnx_dir, tok, chosen, N_TOKENS, return_ids=True)
    ox_texts = generate_onnx(onnx_dir, tok, succ_tasks)
    stats = agreement_stats(pt, ox)
    sp_pt = score_texts(succ_tasks, pt_texts)
    sp_ox = score_texts(succ_tasks, ox_texts)
    both = [
        a["outcomes"]["tool_call_success"] == b["outcomes"]["tool_call_success"]
        for a, b in zip(sp_pt, sp_ox, strict=True)
    ]
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    res = {
        "name": name,
        "utc": utc,
        "n_tokens": N_TOKENS,
        "agreement": stats,
        "success_pytorch_fakequant": summarize(sp_pt)["tool_call_success"],
        "success_onnx": summarize(sp_ox)["tool_call_success"],
        "success_outcome_agreement": sum(both) / len(both),
        "n_success_prompts": len(succ_tasks),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"parity_{name}_{utc}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="map JSON, or int4 / int8")
    ap.add_argument("--onnx-dir")
    ap.add_argument("--hf-path", default="data/cache/qwen3-0.6b-hf")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--n-success-prompts", type=int, default=100)
    a = ap.parse_args()
    if a.map in ("int4", "int8"):
        from toolquant.evalrun.generate import load_hf
        from toolquant.quant.groups import derive_groups

        m, _ = load_hf(a.hf_path)
        bits = {g: int(a.map[3:]) for g in derive_groups(m)}
        name = a.map
    else:
        mp = json.loads(Path(a.map).read_text(encoding="utf-8"))
        bits, name = mp["bits_per_group"], Path(a.map).stem
    print(
        run_parity(
            name,
            bits,
            a.onnx_dir or f"data/cache/onnx/{name}",
            a.hf_path,
            n_prompts=a.n_prompts,
            n_success_prompts=a.n_success_prompts,
        )
    )


if __name__ == "__main__":
    main()
