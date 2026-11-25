import numpy as np
import pytest

from toolquant.evalrun.stats import bootstrap_ci, compare, mcnemar_exact, paired_bootstrap


def test_identical_inputs_zero_diff_interval_contains_zero():
    x = np.random.default_rng(1).integers(0, 2, 200)
    r = paired_bootstrap(x, x)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_seeded_reproducible():
    rng = np.random.default_rng(2)
    a, b = rng.integers(0, 2, 100), rng.integers(0, 2, 100)
    assert paired_bootstrap(a, b) == paired_bootstrap(a, b)


def test_clear_difference_excludes_zero():
    a = np.ones(200)
    b = np.zeros(200)
    r = paired_bootstrap(a, b)
    assert r["diff"] == 1 and r["lo"] > 0.9


def test_interval_contains_point_estimate():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    r = paired_bootstrap(a, b)
    assert r["lo"] <= r["diff"] <= r["hi"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])


def test_bootstrap_ci_known():
    m, lo, hi = bootstrap_ci([1] * 50 + [0] * 50)
    assert m == 0.5 and 0.39 < lo < 0.42 and 0.58 < hi < 0.61


def test_mcnemar_no_discordant():
    r = mcnemar_exact([1, 0, 1], [1, 0, 1])
    assert r["p"] == 1.0 and r["n10"] == 0 and r["n01"] == 0


def test_mcnemar_known_value():
    # 8 vs 2 discordant pairs: exact two-sided p = 0.109375
    a = [1] * 8 + [0] * 2 + [1] * 5
    b = [0] * 8 + [1] * 2 + [1] * 5
    r = mcnemar_exact(a, b)
    assert (r["n10"], r["n01"]) == (8, 2)
    assert r["p"] == pytest.approx(0.109375)


def test_mcnemar_symmetric():
    a, b = [1, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0]
    assert mcnemar_exact(a, b)["p"] == pytest.approx(mcnemar_exact(b, a)["p"])


def test_compare_has_both():
    r = compare([1, 0, 1, 1], [1, 1, 0, 0])
    assert {"diff", "lo", "hi", "p"} <= r.keys()
