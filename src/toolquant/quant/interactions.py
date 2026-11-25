"""Interaction test and capped swap local search (CLAUDE.md section 7 step 4).

Selection uses the probe set only. `score(bits_per_group) -> float` is injected (probe success
rate of the fake-quantized model), so the logic is testable without a model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Score = Callable[[dict[str, int]], float]


def interaction_test(
    score: Score,
    groups: list[str],
    sensitivity: dict[str, float],
    k: int,
    base_bits: int = 8,
    low_bits: int = 4,
) -> dict[str, Any]:
    """Joint drop of the k least sensitive groups at 4-bit vs the sum of their single drops.

    Drops are measured against the all-`base_bits` configuration with the same scorer.
    """
    least = sorted(groups, key=lambda g: (sensitivity[g], g))[:k]
    base = score({g: base_bits for g in groups})
    singles = {}
    for g in least:
        singles[g] = base - score({**{x: base_bits for x in groups}, g: low_bits})
    joint = base - score({g: (low_bits if g in least else base_bits) for g in groups})
    total = sum(singles.values())
    return {
        "k": k,
        "groups": least,
        "single_drops": singles,
        "sum_single_drops": total,
        "joint_drop": joint,
        "superadditive": joint > total,
    }


def swap_search(
    score: Score,
    bits: dict[str, int],
    max_iters: int = 20,
    low_bits: int = 4,
    high_bits: int = 8,
) -> dict[str, Any]:
    """Swap one 4-bit group with one 8-bit group (size-neutral for equal-size groups); accept if
    the probe score improves. First-improvement, fixed iteration cap on evaluated swaps."""
    cur = dict(bits)
    cur_score = score(cur)
    start = cur_score
    history = []
    evals = 0
    improved = True
    while improved and evals < max_iters:
        improved = False
        lows = sorted(g for g, b in cur.items() if b == low_bits)
        highs = sorted(g for g, b in cur.items() if b == high_bits)
        for a in lows:
            for b in highs:
                if evals >= max_iters:
                    break
                cand = {**cur, a: high_bits, b: low_bits}
                s = score(cand)
                evals += 1
                if s > cur_score:
                    history.append({"to_8bit": a, "to_4bit": b, "score": s})
                    cur, cur_score, improved = cand, s, True
                    break
            if improved or evals >= max_iters:
                break
    return {
        "bits_per_group": cur,
        "start_score": start,
        "final_score": cur_score,
        "swaps": history,
        "evaluations": evals,
    }
