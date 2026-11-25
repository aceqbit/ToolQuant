"""Load BFCL v4 single-turn tasks from a local directory (see README, "Reproduce")."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Categories used by the project: one function, several functions to choose from, and
# irrelevance (no function applies, the correct behaviour is to not call).
CATEGORIES = ("simple_python", "multiple", "irrelevance")
_FILES = {
    "simple_python": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
    "live_simple": "BFCL_v4_live_simple.json",
}
_JSON_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": None}


@dataclass
class Task:
    id: str
    category: str
    messages: list[dict[str, str]]
    tools: list[dict[str, Any]]  # BFCL function docs converted to OpenAI/JSON-schema tools
    reference: list[dict[str, dict[str, list[Any]]]] | None  # None: no call is correct
    raw_functions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def needs_call(self) -> bool:
        return self.reference is not None


def data_dir() -> Path:
    return Path(os.environ.get("BFCL_DATA_DIR", "data/cache/bfcl"))


def to_json_schema(node: Any) -> Any:
    """Convert BFCL type names (dict, float, tuple, any) to JSON-schema types, recursively."""
    if isinstance(node, list):
        return [to_json_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            mapped = _JSON_TYPES.get(v, v)
            if mapped is not None:
                out["type"] = mapped
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: to_json_schema(pv) for pk, pv in v.items()}
        else:
            out[k] = to_json_schema(v)
    return out


def _tool(fn: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": to_json_schema(fn.get("parameters", {"type": "dict", "properties": {}})),
        },
    }


def load_category(category: str, root: Path | None = None) -> list[Task]:
    root = root or data_dir()
    path = root / _FILES[category]
    answers: dict[str, Any] = {}
    ans_path = root / "possible_answer" / _FILES[category]
    if ans_path.exists():
        for line in ans_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                answers[row["id"]] = row["ground_truth"]
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref = answers.get(row["id"])
        if category != "irrelevance" and ref is None:
            continue  # no ground truth available: cannot score
        tasks.append(
            Task(
                id=row["id"],
                category=category,
                messages=row["question"][0],
                tools=[_tool(f) for f in row["function"]],
                reference=ref if category != "irrelevance" else None,
                raw_functions=row["function"],
            )
        )
    return tasks


def load_all(categories: tuple[str, ...] = CATEGORIES, root: Path | None = None) -> list[Task]:
    tasks: list[Task] = []
    for c in categories:
        tasks.extend(load_category(c, root))
    return tasks
