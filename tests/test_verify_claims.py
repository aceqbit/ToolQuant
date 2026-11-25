import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_claims as vc


def test_matches_percent_and_raw(tmp_path):
    (tmp_path / "eval_x.json").write_text(json.dumps({"summary": {"m": 0.853}, "size_mb": 1098.0}))
    doc = tmp_path / "doc.md"
    doc.write_text("Success was 85.3% and size 1098 MB.\n")
    assert vc.check([doc], tmp_path) == []


def test_unmatched_number_reported(tmp_path):
    (tmp_path / "eval_x.json").write_text(json.dumps({"summary": {"m": 0.853}}))
    doc = tmp_path / "doc.md"
    doc.write_text("Success was 91.7%.\n")
    assert vc.check([doc], tmp_path) == [("doc.md", "91.7")]


def test_code_and_headings_ignored(tmp_path):
    (tmp_path / "eval_x.json").write_text(json.dumps({"summary": {}}))
    doc = tmp_path / "doc.md"
    doc.write_text("# Heading 99\n```\n123.4\n```\nUse `n=777`.\n")
    assert vc.check([doc], tmp_path) == []
