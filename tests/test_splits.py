import json
from pathlib import Path

import pytest

from toolquant.data.bfcl import CATEGORIES, data_dir, load_all
from toolquant.data.splits import SPLITS_PATH, make_splits

needs_data = pytest.mark.skipif(not data_dir().exists(), reason="BFCL data not fetched")


def synthetic():
    from toolquant.data.bfcl import Task

    return [
        Task(f"{c}_{i}", c, [], [], [] if c != "irrelevance" else None)
        for c in CATEGORIES
        for i in range(50)
    ]


def test_no_overlap_and_balance_synthetic():
    sp = make_splits(synthetic())
    assert not set(sp["probe"]) & set(sp["validation"])
    assert len(sp["probe"]) == 100
    assert len(sp["probe"]) + len(sp["validation"]) == 150


def test_deterministic():
    assert make_splits(synthetic()) == make_splits(synthetic())


def test_committed_splits_no_overlap():
    if not Path(SPLITS_PATH).exists():
        pytest.skip("splits.json not generated")
    sp = json.loads(Path(SPLITS_PATH).read_text(encoding="utf-8"))
    assert len(sp["probe"]) == 100
    assert not set(sp["probe"]) & set(sp["validation"])


@needs_data
def test_committed_splits_match_data():
    sp = json.loads(Path(SPLITS_PATH).read_text(encoding="utf-8"))
    tasks = load_all()
    assert make_splits(tasks) == sp
