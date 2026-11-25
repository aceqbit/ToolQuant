import json

from toolquant.report import build_tables, config_order, latest


def fake_eval(path, cfg, outcomes, size):
    recs = [
        {
            "id": f"p{i}",
            "outcomes": {
                "tool_call_success": bool(o),
                "missed_call": False,
                "silent_failure": not o,
            },
        }
        for i, o in enumerate(outcomes)
    ]
    iv = {
        k: {"mean": 0.5, "lo": 0.4, "hi": 0.6}
        for k in ("tool_call_success", "missed_call", "silent_failure")
    }
    path.write_text(
        json.dumps(
            {"summary": {"intervals": iv}, "records": recs, "extra_metrics": {"size_mb": size}}
        )
    )


def test_latest_picks_newest_per_config(tmp_path):
    for utc in ("20260101T000000Z", "20260102T000000Z"):
        (tmp_path / f"eval_int4_validation_{utc}.json").write_text("{}")
    got = latest("eval_*_validation_*.json", tmp_path)
    assert got["int4"].name.endswith("20260102T000000Z.json")


def test_config_order():
    names = ["a_0.5_ppl", "int4", "fp16", "int8", "a_0.25_toolcall"]
    assert sorted(names, key=config_order) == [
        "fp16",
        "int8",
        "int4",
        "a_0.25_toolcall",
        "a_0.5_ppl",
    ]


def test_build_tables_pairs(tmp_path):
    ok = [1, 1, 0, 1, 0, 1, 1, 0]
    fake_eval(tmp_path / "eval_m_0.25_toolcall_validation_20260101T000000Z.json", "x", ok, 700)
    fake_eval(tmp_path / "eval_m_0.25_ppl_validation_20260101T000000Z.json", "x", ok[::-1], 700)
    fake_eval(tmp_path / "eval_int4_validation_20260101T000000Z.json", "x", [0] * 8, 500)
    table, rows = build_tables(tmp_path)
    assert "m_0.25_toolcall" in table and "McNemar" in table
    assert len(rows) == 3
