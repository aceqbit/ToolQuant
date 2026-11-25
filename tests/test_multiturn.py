import pytest

from toolquant.scoring.multiturn import (
    aggregate,
    parse_gt_call,
    score_trajectory,
    score_turn,
)


def call(name, **kw):
    return {"name": name, "arguments": kw}


def test_parse_gt_call():
    assert parse_gt_call("cd(folder='document')") == call("cd", folder="document")
    assert parse_gt_call("a.b(x=1, y=[1, 2])") == call("a.b", x=1, y=[1, 2])
    with pytest.raises(TypeError):
        parse_gt_call("x")


def test_turn_correct():
    r = score_turn([call("cd", folder="Document")], ["cd(folder='document')"])
    assert r["turn_ok"] and r["n_correct"] == 1 and not r["silent_failure"]


def test_turn_wrong_args_is_silent():
    r = score_turn([call("cd", folder="x")], ["cd(folder='document')"])
    assert not r["turn_ok"] and r["silent_failure"] and r["n_correct"] == 0


def test_turn_missed():
    r = score_turn([], ["cd(folder='document')"])
    assert r["missed"] and not r["silent_failure"]


def test_turn_partial_credit_and_order():
    gt = ["a(x=1)", "b(y=2)"]
    assert score_turn([call("b", y=2), call("a", x=1)], gt)["turn_ok"]
    r = score_turn([call("a", x=1), call("b", y=3)], gt)
    assert r["n_correct"] == 1 and not r["turn_ok"] and r["silent_failure"]


def test_trajectory_and_aggregate():
    good = score_trajectory([score_turn([call("a", x=1)], ["a(x=1)"])])
    bad = score_trajectory(
        [score_turn([call("a", x=1)], ["a(x=1)"]), score_turn([call("a", x=2)], ["a(x=1)"])]
    )
    assert good["trajectory_ok"] and not bad["trajectory_ok"]
    assert bad["per_call"] == 0.5 and bad["silent_failure"]
    agg = aggregate([good, bad])
    assert agg["per_trajectory"] == 0.5 and agg["silent_failure_rate"] == 0.5
    assert agg["per_call"] == 0.75
