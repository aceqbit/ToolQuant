"""Multi-turn scoring (per-call, per-trajectory, silent failure as separate fields).

BFCL multi-turn ground truth is a list of turns; each turn is a list of Python call strings such
as ``cd(folder='document')``. This scorer compares the model's calls per turn to those strings
with the same normalisation as the single-turn matcher. It does not execute the calls against
BFCL's simulated backends (state-based checking); that is a documented limitation.
"""

from __future__ import annotations

import ast
import itertools
from typing import Any

from toolquant.scoring.match import normalize


def parse_gt_call(s: str) -> dict[str, Any]:
    node = ast.parse(s.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise TypeError(f"not a call: {s}")
    name = ast.unparse(node.func)
    args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
    return {"name": name, "arguments": args}


def _eq_call(pred: dict[str, Any], gt: dict[str, Any]) -> bool:
    if pred["name"].split(".")[-1] != gt["name"].split(".")[-1]:
        return False
    return normalize(pred["arguments"]) == normalize(gt["arguments"])


def score_turn(pred: list[dict[str, Any]], gt_strings: list[str]) -> dict[str, Any]:
    gt = [parse_gt_call(s) for s in gt_strings]
    n_correct = 0
    ok = False
    if len(pred) == len(gt):
        ok = any(
            all(_eq_call(p, g) for p, g in zip(perm, gt, strict=True))
            for perm in itertools.permutations(pred)
        )
    if ok:
        n_correct = len(gt)
    else:  # partial credit per call: greedy one-to-one matching
        left = list(gt)
        for p in pred:
            hit = next((g for g in left if _eq_call(p, g)), None)
            if hit is not None:
                left.remove(hit)
                n_correct += 1
    return {
        "n_gt": len(gt),
        "n_pred": len(pred),
        "n_correct": n_correct,
        "turn_ok": ok,
        "silent_failure": bool(pred) and not ok,  # well-formed calls made, but wrong
        "missed": not pred and bool(gt),
    }


def score_trajectory(turns: list[dict[str, Any]]) -> dict[str, Any]:
    n_gt = sum(t["n_gt"] for t in turns)
    n_ok = sum(t["n_correct"] for t in turns)
    return {
        "per_call": n_ok / n_gt if n_gt else None,
        "trajectory_ok": all(t["turn_ok"] for t in turns),
        "silent_failure": any(t["silent_failure"] for t in turns),
        "n_turns": len(turns),
    }


def aggregate(trajs: list[dict[str, Any]]) -> dict[str, float | None]:
    n = len(trajs)
    pc = [t["per_call"] for t in trajs if t["per_call"] is not None]
    return {
        "n": n,
        "per_call": sum(pc) / len(pc) if pc else None,
        "per_trajectory": sum(t["trajectory_ok"] for t in trajs) / n if n else None,
        "silent_failure_rate": sum(t["silent_failure"] for t in trajs) / n if n else None,
    }
