import pytest

from toolquant.data.bfcl import Task, to_json_schema
from toolquant.scoring.match import ast_match, normalize, schema_valid
from toolquant.scoring.metrics import aggregate, score_output
from toolquant.scoring.parse import parse_tool_calls

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "math.area",
            "description": "area",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "integer"},
                    "height": {"type": "number"},
                    "unit": {"type": "string"},
                    "mode": {"type": "string", "enum": ["a", "b"]},
                },
                "required": ["base", "height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "other",
            "description": "x",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
REF = [{"math.area": {"base": [10], "height": [5.0], "unit": ["units", ""]}}]


def task(ref=REF) -> Task:
    return Task("t", "simple_python", [{"role": "user", "content": "q"}], TOOLS, ref)


def call(args: str, name: str = "math.area") -> str:
    return f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'


def outcomes(text: str, t: Task | None = None) -> dict:
    return score_output(t or task(), text)["outcomes"]


# ---- parser
def test_parse_single():
    p = parse_tool_calls(call('{"a": 1}'))
    assert p.calls == [{"name": "math.area", "arguments": {"a": 1}}] and not p.malformed


def test_parse_two_calls():
    assert len(parse_tool_calls(call("{}") + "\n" + call("{}")).calls) == 2


def test_parse_plain_text_no_call():
    p = parse_tool_calls("Sorry, I cannot help.")
    assert p.calls == [] and p.attempted == 0 and not p.malformed


def test_parse_bad_json_is_malformed():
    p = parse_tool_calls('<tool_call>\n{"name": "x", "arguments": {\n</tool_call>')
    assert p.calls == [] and p.malformed and p.attempted == 1


def test_parse_truncated_block():
    p = parse_tool_calls('<tool_call>\n{"name": "x", "arguments": {}}')
    assert p.malformed and p.attempted == 1


def test_parse_arguments_not_object():
    assert parse_tool_calls(call('"str"')).malformed


# ---- normalisation and matcher
def test_normalize_numbers():
    assert normalize(1) == normalize(1.0)
    assert normalize(True) is True


def test_normalize_strings():
    assert normalize("New  York") == normalize("new_york")


def test_ast_match_int_float_equal():
    assert ast_match([{"name": "math.area", "arguments": {"base": 10, "height": 5}}], REF, TOOLS)


def test_ast_match_extra_optional_allowed_if_unreferenced():
    args = {"base": 10, "height": 5, "mode": "a"}
    assert ast_match([{"name": "math.area", "arguments": args}], REF, TOOLS)


def test_ast_match_optional_must_match_when_referenced():
    args = {"base": 10, "height": 5, "unit": "cm"}
    assert not ast_match([{"name": "math.area", "arguments": args}], REF, TOOLS)


def test_ast_match_bool_is_not_number():
    ref = [{"math.area": {"base": [1], "height": [1]}}]
    assert not ast_match(
        [{"name": "math.area", "arguments": {"base": True, "height": 1}}], ref, TOOLS
    )


def test_ast_match_parallel_order_insensitive():
    ref = [{"math.area": {"base": [1], "height": [1]}}, {"other": {}}]
    calls = [
        {"name": "other", "arguments": {}},
        {"name": "math.area", "arguments": {"base": 1, "height": 1}},
    ]
    assert ast_match(calls, ref, TOOLS)


def test_ast_match_wrong_count():
    assert not ast_match([], REF, TOOLS)


def test_schema_valid_cases():
    ok = {"name": "math.area", "arguments": {"base": 1, "height": 2}}
    assert schema_valid(ok, TOOLS)
    assert not schema_valid({"name": "nope", "arguments": {}}, TOOLS)
    assert not schema_valid({"name": "math.area", "arguments": {"base": 1}}, TOOLS)
    assert not schema_valid({"name": "math.area", "arguments": {"base": "1", "height": 2}}, TOOLS)
    assert not schema_valid(
        {"name": "math.area", "arguments": {"base": 1, "height": 2, "mode": "z"}}, TOOLS
    )


def test_to_json_schema_types():
    s = to_json_schema(
        {"type": "dict", "properties": {"a": {"type": "float"}, "b": {"type": "any"}}}
    )
    assert s["type"] == "object" and s["properties"]["a"]["type"] == "number"
    assert "type" not in s["properties"]["b"]


# ---- six metrics, one per scenario
def test_correct_call():
    o = outcomes(call('{"base": 10, "height": 5}'))
    assert o["schema_valid"] and o["ast_match"] and o["tool_call_success"]
    assert not o["missed_call"] and not o["silent_failure"]


def test_wrong_name_is_silent_failure():
    o = outcomes(call("{}", name="other"))
    assert o["schema_valid"] and not o["ast_match"] and o["silent_failure"]
    assert not o["tool_call_success"]


def test_wrong_argument_is_silent_failure():
    o = outcomes(call('{"base": 11, "height": 5}'))
    assert o["schema_valid"] and o["silent_failure"] and not o["tool_call_success"]


def test_malformed_json_not_silent():
    o = outcomes('<tool_call>\n{"name": broken\n</tool_call>')
    assert not o["schema_valid"] and not o["silent_failure"] and not o["missed_call"]
    assert not o["tool_call_success"]


def test_schema_invalid_wrong_type_not_silent():
    o = outcomes(call('{"base": "ten", "height": 5}'))
    assert not o["schema_valid"] and not o["silent_failure"] and not o["tool_call_success"]


def test_missing_call_when_needed():
    r = score_output(task(), "I can't do that.")
    assert r["outcomes"]["missed_call"] and r["label"] == "missed_call"
    assert not r["outcomes"]["tool_call_success"]


def test_no_call_when_not_needed():
    o = outcomes("No suitable tool.", task(ref=None))
    assert o["abstain_ok"] and o["tool_call_success"] and not o["missed_call"]


def test_call_when_not_needed():
    r = score_output(task(ref=None), call('{"base": 1, "height": 2}'))
    assert not r["outcomes"]["abstain_ok"] and not r["outcomes"]["tool_call_success"]
    assert r["label"] == "false_call"


def test_empty_output():
    assert outcomes("")["missed_call"]


def test_aggregate_rates():
    t = task()
    recs = [
        score_output(t, call('{"base": 10, "height": 5}')),
        score_output(t, call('{"base": 1, "height": 5}')),
        score_output(t, "nothing"),
        score_output(task(ref=None), "nothing"),
    ]
    a = aggregate(recs)
    assert a["n"] == 4
    assert a["tool_call_success"] == pytest.approx(0.5)
    assert a["missed_call"] == pytest.approx(0.25)
    assert a["silent_failure"] == pytest.approx(0.25)
    assert a["abstain_ok"] == 1.0
    assert a["labels"]["success"] == 2
