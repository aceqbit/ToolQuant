"""Failure taxonomy on top of the six metrics (one label per prompt)."""

from __future__ import annotations

from typing import Any


def label(rec: dict[str, Any]) -> str:
    o = rec["outcomes"]
    if o["tool_call_success"]:
        return "success"
    if rec["needs_call"]:
        if o["missed_call"]:
            return "missed_call"
        if o["silent_failure"]:
            return "silent_failure"
        return "malformed_or_invalid"
    return "false_call"  # a call was made when none was needed
