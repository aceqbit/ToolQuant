"""Paired bootstrap and exact McNemar over per-prompt binary outcomes (CLAUDE.md section 7.8)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest

N_RESAMPLES = 10_000
SEED = 0


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def bootstrap_ci(
    x: Any, n_resamples: int = N_RESAMPLES, seed: int = SEED, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval of one binary outcome vector."""
    a = _arr(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(a.mean()), float(lo), float(hi)


def paired_bootstrap(
    a: Any, b: Any, n_resamples: int = N_RESAMPLES, seed: int = SEED, alpha: float = 0.05
) -> dict[str, float]:
    """Difference mean(a) - mean(b) over the same prompts, resampling prompts jointly."""
    x, y = _arr(a), _arr(b)
    if x.shape != y.shape:
        raise ValueError("paired outcomes must have the same length")
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    diffs = d[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(d)}


def mcnemar_exact(a: Any, b: Any) -> dict[str, float]:
    """Exact McNemar test (two-sided binomial on the discordant pairs)."""
    x, y = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n10 = int(np.sum(x & ~y))  # a yes, b no
    n01 = int(np.sum(~x & y))
    n = n10 + n01
    p = 1.0 if n == 0 else float(binomtest(n10, n, 0.5, alternative="two-sided").pvalue)
    return {"n10": n10, "n01": n01, "p": p}


def compare(a: Any, b: Any, **kw: Any) -> dict[str, float]:
    """Both reported together: interval of mean(a)-mean(b) and McNemar p-value."""
    out = paired_bootstrap(a, b, **kw)
    out.update(mcnemar_exact(a, b))
    return out
