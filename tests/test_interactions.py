from toolquant.quant.interactions import interaction_test, swap_search

G = [f"g{i}" for i in range(6)]


def additive_score(bits):
    drops = {g: 0.01 * i for i, g in enumerate(G)}
    return 1.0 - sum(d for g, d in drops.items() if bits[g] == 4)


def synergy_score(bits):
    n4 = sum(1 for b in bits.values() if b == 4)
    return 1.0 - 0.01 * n4 - (0.05 if n4 >= 3 else 0)


def test_additive_not_superadditive():
    sens = {g: 0.01 * i for i, g in enumerate(G)}
    r = interaction_test(additive_score, G, sens, k=3)
    assert not r["superadditive"] and abs(r["joint_drop"] - r["sum_single_drops"]) < 1e-9


def test_superadditive_detected():
    sens = {g: 0.01 * i for i, g in enumerate(G)}
    assert interaction_test(synergy_score, G, sens, k=4)["superadditive"]


def test_swap_search_improves_and_keeps_size():
    bits = {g: (4 if g in ("g5", "g4", "g3") else 8) for g in G}  # worst groups at 4-bit
    r = swap_search(additive_score, bits, max_iters=50)
    assert r["final_score"] > r["start_score"]
    assert sum(b == 4 for b in r["bits_per_group"].values()) == 3


def test_swap_search_respects_cap():
    bits = {g: (4 if g in ("g5", "g4", "g3") else 8) for g in G}
    assert swap_search(additive_score, bits, max_iters=2)["evaluations"] <= 2


def test_swap_search_no_gain_returns_same():
    bits = {g: (4 if g in ("g0", "g1", "g2") else 8) for g in G}
    r = swap_search(additive_score, bits, max_iters=50)
    assert r["bits_per_group"] == bits and not r["swaps"]
