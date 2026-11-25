"""Greedy budgeted allocation: lowest-sensitivity groups get 4-bit, the rest 8-bit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toolquant.quant.sizing import GROUP_SIZE, budget, total_bits

MAPS = Path("maps")


def allocate(
    sensitivity: dict[str, float],
    group_params: dict[str, int],
    other_params: int,
    f: float,
    group_size: int = GROUP_SIZE,
) -> dict[str, int]:
    """Start from all-INT8; flip groups to INT4 in ascending sensitivity until size <= budget.

    Ties in sensitivity are broken by group name (neutral and deterministic).
    """
    limit = budget(f, group_params, other_params, group_size)
    bits = {g: 8 for g in group_params}
    for g in sorted(group_params, key=lambda g: (sensitivity[g], g)):
        if total_bits(group_params, bits, other_params, group_size) <= limit:
            break
        bits[g] = 4
    return bits


def make_map(
    kind: str,
    f: float,
    bits: dict[str, int],
    group_params: dict[str, int],
    other_params: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": meta["model"],
        "revision": meta["revision"],
        "group_size": meta.get("group_size", GROUP_SIZE),
        "granularity": meta["granularity"],
        "bits_per_group": bits,
        "size_bits": total_bits(
            group_params, bits, other_params, meta.get("group_size", GROUP_SIZE)
        ),
        "seed": meta.get("seed", 0),
        "git_sha": meta["git_sha"],
        "budget_f": f,
        "kind": kind,
        "sensitivity_file": meta.get("sensitivity_file"),
    }


def map_path(model: str, f: float, kind: str, maps_dir: Path = MAPS) -> Path:
    return maps_dir / f"{model.split('/')[-1].lower()}_{f}_{kind}.json"


def n4(m: dict[str, Any]) -> list[str]:
    return sorted(g for g, b in m["bits_per_group"].items() if b == 4)


def compare_maps(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Number of groups at 4-bit in each map, the overlap, and the groups that differ."""
    sa, sb = set(n4(a)), set(n4(b))
    return {
        "n4_a": len(sa),
        "n4_b": len(sb),
        "overlap": len(sa & sb),
        "differ": len(sa ^ sb),
        "identical": sa == sb,
        "only_a": sorted(sa - sb),
        "only_b": sorted(sb - sa),
    }


def main() -> None:
    from toolquant.quant.groups import group_info
    from toolquant.tracking import git_sha

    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5])
    a = ap.parse_args()
    s = json.loads(Path(a.sensitivity).read_text(encoding="utf-8"))

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("data/cache/qwen3-0.6b-hf")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    info = group_info(model, s["granularity"])
    names = list(info["params"])
    missing = [g for g in names if g not in s["groups"]]
    if missing:
        raise SystemExit(f"sensitivity file lacks groups: {missing}")
    tool = {g: s["groups"][g]["sensitivity"] for g in names}
    ppl = {g: s["groups"][g]["ppl_change"] for g in names}
    meta = {**s, "git_sha": git_sha(), "sensitivity_file": Path(a.sensitivity).name}
    MAPS.mkdir(exist_ok=True)
    for f in a.budgets:
        built = {}
        for kind, sens in (("toolcall", tool), ("ppl", ppl)):
            bits = allocate(sens, info["params"], info["other_params"], f, s.get("group_size", 128))
            m = make_map(kind, f, bits, info["params"], info["other_params"], meta)
            map_path(s["model"], f, kind).write_text(json.dumps(m, indent=1), encoding="utf-8")
            built[kind] = m
        print(f, json.dumps(compare_maps(built["toolcall"], built["ppl"])))


if __name__ == "__main__":
    main()
