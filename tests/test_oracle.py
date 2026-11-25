"""Feeding each task's own reference back as model output must score as success."""

import json

import pytest

from toolquant.data.bfcl import data_dir, load_all
from toolquant.scoring.metrics import score_output

pytestmark = pytest.mark.skipif(not data_dir().exists(), reason="BFCL data not fetched")


def oracle_text(task) -> str:
    if task.reference is None:
        return "None of the tools apply."
    blocks = []
    for ref in task.reference:
        ((name, accepted),) = ref.items()
        args = {k: v[0] for k, v in accepted.items() if v and v[0] != ""}
        blocks.append(
            "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"
        )
    return "\n".join(blocks)


def test_oracle_success_rate_is_high():
    tasks = load_all()
    ok = [score_output(t, oracle_text(t))["outcomes"]["tool_call_success"] for t in tasks]
    # A handful of BFCL items have schema/type quirks; the scorer must accept nearly all.
    assert sum(ok) / len(ok) > 0.95, sum(ok) / len(ok)
