"""Layer groups derived from module names (never hard-coded indices).

A group is a set of nn.Linear modules inside decoder blocks. The embedding table and `lm_head`
are not in any group.
"""

from __future__ import annotations

import re
from typing import Any

from torch import nn

_BLOCK = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


def _part(rest: str) -> str:
    head = rest.split(".")[0]
    return "attn" if head in {"self_attn", "attn", "attention"} else "mlp"


def derive_groups(model: nn.Module, granularity: str = "block") -> dict[str, list[str]]:
    """Map group name -> linear module names. granularity: 'block' or 'attn_mlp'."""
    if granularity not in {"block", "attn_mlp"}:
        raise ValueError(f"unknown granularity {granularity!r}")
    groups: dict[str, list[str]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        m = _BLOCK.search(name)
        if not m:
            continue  # embedding / lm_head / anything outside decoder blocks
        idx = int(m.group(1))
        key = f"layer{idx:02d}" if granularity == "block" else f"layer{idx:02d}.{_part(m.group(2))}"
        groups.setdefault(key, []).append(name)
    return dict(sorted(groups.items()))


def group_param_counts(model: nn.Module, groups: dict[str, list[str]]) -> dict[str, int]:
    mods = dict(model.named_modules())
    return {g: sum(mods[n].weight.numel() for n in names) for g, names in groups.items()}


def other_param_count(model: nn.Module, groups: dict[str, list[str]]) -> int:
    """Parameters outside the groups' linear weights (unique tensors; tied weights counted once)."""
    mods = dict(model.named_modules())
    in_groups = {id(mods[n].weight) for names in groups.values() for n in names}
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in in_groups or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def group_info(model: nn.Module, granularity: str = "block") -> dict[str, Any]:
    groups = derive_groups(model, granularity)
    return {
        "groups": groups,
        "params": group_param_counts(model, groups),
        "other_params": other_param_count(model, groups),
    }
