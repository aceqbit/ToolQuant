"""Schema validation and the minimal AST matcher (CLAUDE.md section 6).

The BFCL checker is not importable without heavy dependencies (docs/SPIKE.md), so this is the
minimal matcher. Differences from BFCL are recorded in DECISIONS.md.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

_STRIP = re.compile(r"[ ,./\-_*^]")


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _type_ok(value: Any, typ: str | None) -> bool:
    if typ is None:
        return True
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return _is_num(value)
    if typ == "string":
        return isinstance(value, str)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "array":
        return isinstance(value, list)
    if typ == "object":
        return isinstance(value, dict)
    return True


def _value_ok(value: Any, schema: dict[str, Any]) -> bool:
    if not _type_ok(value, schema.get("type")):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        return all(_value_ok(v, schema["items"]) for v in value)
    return True


def _fn(name: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((t["function"] for t in tools if t["function"]["name"] == name), None)


def schema_valid(call: dict[str, Any], tools: list[dict[str, Any]]) -> bool:
    """Name is an offered tool; required args present; given args have the declared type."""
    fn = _fn(call["name"], tools)
    if fn is None:
        return False
    params = fn.get("parameters", {})
    props = params.get("properties", {})
    args = call["arguments"]
    if any(r not in args for r in params.get("required", [])):
        return False
    return all(_value_ok(v, props[k]) for k, v in args.items() if k in props)


def normalize(v: Any) -> Any:
    """Type normalisation: ints equal floats; strings compared case/space/punctuation-blind."""
    if isinstance(v, bool):
        return v
    if _is_num(v):
        return float(v)
    if isinstance(v, str):
        return _STRIP.sub("", v.lower()).replace("'", '"')
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {str(k): normalize(x) for k, x in v.items()}
    return v


def _eq(a: Any, b: Any) -> bool:
    """Equality that never treats a bool as a number."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    return a == b


def _arg_ok(value: Any, accepted: list[Any]) -> bool:
    nv = normalize(value)
    return any(_eq(nv, normalize(a)) for a in accepted)


def _call_matches(
    call: dict[str, Any], ref: dict[str, dict[str, list[Any]]], tools: list[dict[str, Any]]
) -> bool:
    ((name, accepted_args),) = ref.items()
    if call["name"] != name:
        return False
    fn = _fn(name, tools) or {}
    required = fn.get("parameters", {}).get("required", [])
    args = call["arguments"]
    for r in required:
        if r not in args or not _arg_ok(args[r], accepted_args.get(r, [])):
            return False
    for k, v in args.items():
        if k in required or k not in accepted_args:
            continue  # extra optional arguments are allowed
        if not _arg_ok(v, accepted_args[k]):
            return False
    return True


def ast_match(
    calls: list[dict[str, Any]],
    reference: list[dict[str, dict[str, list[Any]]]],
    tools: list[dict[str, Any]],
) -> bool:
    """All reference calls matched one-to-one (order-insensitive) by the predicted calls."""
    if len(calls) != len(reference):
        return False
    return any(
        all(_call_matches(c, r, tools) for c, r in zip(perm, reference, strict=True))
        for perm in itertools.permutations(calls)
    )
