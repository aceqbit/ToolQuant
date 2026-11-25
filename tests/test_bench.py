from toolquant.bench.memory import peak_rss_bytes
from toolquant.bench.speed import summarize_runs
from toolquant.export.parity import agreement_stats, first_divergence


def test_peak_rss_positive():
    assert peak_rss_bytes() > 10_000_000


def test_summarize_runs():
    runs = [{"ttft_s": 0.1, "total_s": 2.0, "decode_tps": 50.0} for _ in range(20)]
    s = summarize_runs(runs)
    assert abs(s["ttft_ms_mean"] - 100.0) < 1e-6 and s["latency_s_p95"] == 2.0
    assert s["decode_tokens_per_s_mean"] == 50.0 and s["n_runs"] == 20


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) == 3
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_agreement_stats():
    pt = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7]}
    ox = {"a": [1, 2, 3], "b": [4, 0, 6], "c": [8]}
    s = agreement_stats(pt, ox)
    assert s["next_token_agreement"] == 2 / 3
    assert s["full_prefix_agreement"] == 1 / 3
    assert s["per_prompt_first_divergence"] == {"a": 3, "b": 1, "c": 0}
