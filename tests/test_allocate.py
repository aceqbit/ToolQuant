import random

import pytest

from toolquant.quant.allocate import allocate, compare_maps, make_map
from toolquant.quant.sizing import budget, total_bits

GP = {f"layer{i:02d}": 1000 * 128 for i in range(10)}
OTHER = 50_000


@pytest.mark.parametrize("f", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_size_at_or_under_budget(f):
    sens = {g: random.Random(i).random() for i, g in enumerate(GP)}
    bits = allocate(sens, GP, OTHER, f)
    assert total_bits(GP, bits, OTHER) <= budget(f, GP, OTHER) + 1e-6


def test_4bit_goes_to_lowest_sensitivity():
    sens = {g: float(i) for i, g in enumerate(GP)}
    sens["layer07"] = -1.0
    bits = allocate(sens, GP, OTHER, 0.5)
    fours = {g for g, b in bits.items() if b == 4}
    assert fours
    assert "layer07" in fours
    assert max(sens[g] for g in fours) <= min(sens[g] for g in GP if g not in fours)


def test_budget_extremes():
    sens = {g: float(i) for i, g in enumerate(GP)}
    assert set(allocate(sens, GP, OTHER, 0.0).values()) == {4}
    assert set(allocate(sens, GP, OTHER, 1.0).values()) == {8}


def test_monotone_in_budget():
    sens = {g: float(i) for i, g in enumerate(GP)}
    counts = [
        sum(b == 4 for b in allocate(sens, GP, OTHER, f).values()) for f in (0.1, 0.25, 0.5, 0.75)
    ]
    assert counts == sorted(counts, reverse=True)


def test_two_maps_equal_size_within_one_group():
    rng = random.Random(1)
    a = allocate({g: rng.random() for g in GP}, GP, OTHER, 0.25)
    b = allocate({g: rng.random() for g in GP}, GP, OTHER, 0.25)
    sa, sb = total_bits(GP, a, OTHER), total_bits(GP, b, OTHER)
    one_group = GP["layer00"] * 4
    assert abs(sa - sb) <= one_group


def test_ties_broken_deterministically():
    sens = {g: 0.0 for g in GP}
    assert allocate(sens, GP, OTHER, 0.5) == allocate(sens, GP, OTHER, 0.5)


def test_compare_maps():
    meta = {"model": "m", "revision": "r", "granularity": "block", "git_sha": "x"}
    sens1 = {g: float(i) for i, g in enumerate(GP)}
    sens2 = {g: float(-i) for i, g in enumerate(GP)}
    m1 = make_map("toolcall", 0.5, allocate(sens1, GP, OTHER, 0.5), GP, OTHER, meta)
    m2 = make_map("ppl", 0.5, allocate(sens2, GP, OTHER, 0.5), GP, OTHER, meta)
    c = compare_maps(m1, m2)
    assert c["n4_a"] == c["n4_b"] and c["differ"] == 2 * (c["n4_a"] - c["overlap"])
    assert compare_maps(m1, m1)["identical"]
    assert {
        "model",
        "revision",
        "group_size",
        "granularity",
        "bits_per_group",
        "size_bits",
        "seed",
        "git_sha",
    } <= m1.keys()
