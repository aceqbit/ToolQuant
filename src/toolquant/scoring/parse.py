"""Parse Qwen3 tool calls: JSON objects inside <tool_call>...</tool_call> tags.

Format verified from the tokenizer chat template of Qwen/Qwen3-0.6B:
``<tool_call>\\n{"name": <function-name>, "arguments": <args-json-object>}\\n</tool_call>``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ParsedOutput:
    calls: list[dict[str, Any]] = field(default_factory=list)  # well-formed {name, arguments}
    attempted: int = 0  # number of <tool_call> openings seen
    malformed: bool = False


def parse_tool_calls(text: str) -> ParsedOutput:
    out = ParsedOutput()
    out.attempted = text.count("<tool_call>")
    closed = _BLOCK.findall(text)
    if out.attempted > len(closed):
        out.malformed = True  # truncated or unterminated block
    for body in closed:
        try:
            obj = json.loads(body.strip())
        except json.JSONDecodeError:
            out.malformed = True
            continue
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("name"), str)
            and isinstance(obj.get("arguments"), dict)
        ):
            out.calls.append({"name": obj["name"], "arguments": obj["arguments"]})
        else:
            out.malformed = True
    return out
