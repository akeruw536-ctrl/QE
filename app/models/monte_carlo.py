"""
Monte Carlo probability engine.

Ensemble of two resampling methods (spec section 18):
  1. Empirical bootstrap -- i.i.d. draws from the recent return distribution.
  2. Block bootstrap -- draws contiguous blocks of returns to preserve
     short-horizon autocorrelation/momentum structure that i.i.d. resampling
     destroys.

Both are volatility-normalized: returns are rescaled so the simulation's
realized volatility matches the *current* regime's volatility rather than
the full-history average, per section 17. Path count is adaptive -- more
simulations only when the candidate is close to the trade threshold
(section 19).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MCResult:
    probability: float           # ensemble mean probability of finishing in-range
    probability_uncertainty: float  # spread across ensemble methods + bootstrap resampling noise
    model_disagreement: float    # |method_1 - method_2|
    mean_final_price: float
    median_final_price: float
    path_count: int


def _simulate_paths(current_price: float, returns_pool: np.ndarray, steps: int,
                     n_paths: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Block bootstrap path simulation, fully vectorized. block_size=1
    reduces to i.i.d. bootstrap. Returns final prices only.

    Previously this looped in Python once per path (and again per block),
    which cost ~0.2s per 10k paths and ~2.6s per 100k -- multiplied across
    the 81-243 candidates a single scan scores, that blocked the asyncio
    event loop for tens of seconds per symbol, long enough to starve the
    WebSocket pump and trip the 60s ping timeout. Same sampling scheme,
    now one numpy gather + sum.
    """
    returns_pool = np.asarray(returns_pool, dtype=float)
    n = len(returns_pool)
    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(steps / block_size))

    starts = rng.integers(0, n - block_size + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_size)).reshape(n_paths, n_blocks * block_size)[:, :steps]
    final_log_moves = returns_pool[idx].sum(axis=1)
    return current_price * np.exp(final_log_moves)


def estimate_probability_in_range(
    current_price: float,
    returns_pool: np.ndarray,
    steps: int,
    lower_barrier: float,
    upper_barrier: float,
    current_volatility: float,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """
    steps: number of return-increments to simulate forward (e.g. minutes-to-
    expiry mapped onto the granularity of returns_pool).
    """
    rng = np.random.default_rng(seed)

    if len(returns_pool) < 10 or steps <= 0:
        # not enough history to simulate -- return a maximally uncertain result
        return MCResult(0.5, 0.5, 1.0, current_price, current_price, 0)

    pool_vol = float(np.std(returns_pool, ddof=1)) if len(returns_pool) > 1 else current_volatility
    scale = (current_volatility / pool_vol) if pool_vol > 0 else 1.0
    normalized_pool = returns_pool * scale

    half = max(1, n_paths // 2)

    empirical_final = _simulate_paths(current_price, normalized_pool, steps, half, block_size=1, rng=rng)
    block_size = max(2, min(10, steps // 3 or 2))
    block_final = _simulate_paths(current_price, normalized_pool, steps, half, block_size=block_size, rng=rng)

    def in_range_prob(finals: np.ndarray) -> float:
        return float(np.mean((finals >= lower_barrier) & (finals <= upper_barrier)))

    p_empirical = in_range_prob(empirical_final)
    p_block = in_range_prob(block_final)

    all_finals = np.concatenate([empirical_final, block_final])
    p_ensemble = in_range_prob(all_finals)

    # Monte Carlo standard error of the pooled estimate. This is the exact
    # quantity the old 30x "bootstrap-of-the-bootstrap" resampling loop was
    # approximating (the std of a resampled Bernoulli mean), without the
    # extra 30 passes over every path.
    n_total = len(all_finals)
    uncertainty = float(np.sqrt(max(p_ensemble * (1.0 - p_ensemble), 0.0) / n_total)) if n_total else 0.5

    disagreement = abs(p_empirical - p_block)

    return MCResult(
        probability=p_ensemble,
        probability_uncertainty=uncertainty,
        model_disagreement=disagreement,
        mean_final_price=float(np.mean(all_finals)),
        median_final_price=float(np.median(all_finals)),
        path_count=len(all_finals),
    )


def adaptive_path_count(preliminary_probability: float, threshold: float,
                         minimum: int, default: int, maximum: int, band: float) -> int:
    """More simulations only when the candidate sits close to the decision
    boundary -- section 19: don't waste 100k paths on obviously-dead candidates."""
    distance = abs(preliminary_probability - threshold)
    if distance > band * 3:
        return minimum
    if distance > band:
        return default
    return maximum
