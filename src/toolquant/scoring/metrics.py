"""Per-prompt record and aggregate metrics. Definitions follow CLAUDE.md section 6."""

from __future__ import annotations

from collections import Counter
from typing import Any

from toolquant.data.bfcl import Task
from toolquant.scoring.match import ast_match, schema_valid
from toolquant.scoring.parse import parse_tool_calls
from toolquant.scoring.taxonomy import label


def score_output(task: Task, text: str) -> dict[str, Any]:
    p = parse_tool_calls(text)
    calls_made = max(p.attempted, len(p.calls))  # a malformed attempt still counts as a call
    valid = bool(p.calls) and not p.malformed and all(schema_valid(c, task.tools) for c in p.calls)
    matched = (
        valid and task.reference is not None and ast_match(p.calls, task.reference, task.tools)
    )
    missed = task.needs_call and calls_made == 0
    silent = valid and calls_made > 0 and not matched
    abstain = (not task.needs_call) and calls_made == 0
    success = matched if task.needs_call else abstain
    rec: dict[str, Any] = {
        "id": task.id,
        "category": task.category,
        "needs_call": task.needs_call,
        "calls_made": calls_made,
        "predicted": p.calls,
        "reference": task.reference,
        "outcomes": {
            "schema_valid": valid,
            "ast_match": matched,
            "missed_call": missed,
            "silent_failure": silent,
            "abstain_ok": abstain,
            "tool_call_success": bool(success),
        },
    }
    rec["label"] = label(rec)
    return rec


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Rates. Denominators: success, missed_call, silent_failure over all prompts;
    ast_match over call-needed prompts; abstain_ok over no-call prompts;
    schema_valid over prompts where a call was made."""

    def col(rs: list[dict[str, Any]], k: str) -> list[bool]:
        return [r["outcomes"][k] for r in rs]

    need = [r for r in records if r["needs_call"]]
    nocall = [r for r in records if not r["needs_call"]]
    made = [r for r in records if r["calls_made"] > 0]
    return {
        "n": len(records),
        "tool_call_success": _rate(col(records, "tool_call_success")),
        "missed_call": _rate(col(records, "missed_call")),
        "silent_failure": _rate(col(records, "silent_failure")),
        "ast_match": _rate(col(need, "ast_match")),
        "abstain_ok": _rate(col(nocall, "abstain_ok")),
        "schema_valid": _rate(col(made, "schema_valid")),
        "labels": dict(Counter(r["label"] for r in records)),
    }
