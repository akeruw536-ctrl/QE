import numpy as np

from app.models.monte_carlo import estimate_probability_in_range, adaptive_path_count


def test_probability_bounded_zero_one():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.002, size=500)
    result = estimate_probability_in_range(
        current_price=100.0, returns_pool=returns, steps=5,
        lower_barrier=95.0, upper_barrier=105.0, current_volatility=0.002,
        n_paths=2000, seed=42,
    )
    assert 0.0 <= result.probability <= 1.0
    assert result.path_count > 0


def test_wider_barriers_increase_probability():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.002, size=500)
    narrow = estimate_probability_in_range(
        100.0, returns, 5, 99.5, 100.5, 0.002, n_paths=4000, seed=1,
    )
    wide = estimate_probability_in_range(
        100.0, returns, 5, 90.0, 110.0, 0.002, n_paths=4000, seed=1,
    )
    assert wide.probability >= narrow.probability


def test_insufficient_history_returns_uncertain_default():
    result = estimate_probability_in_range(100.0, np.array([]), 5, 95, 105, 0.01, n_paths=1000)
    assert result.probability == 0.5
    assert result.probability_uncertainty == 0.5


def test_adaptive_path_count_scales_with_distance_to_threshold():
    near = adaptive_path_count(0.705, 0.71, minimum=1000, default=5000, maximum=20000, band=0.03)
    far = adaptive_path_count(0.20, 0.71, minimum=1000, default=5000, maximum=20000, band=0.03)
    assert near >= far
    assert far == 1000


def test_matches_analytic_normal_random_walk():
    # For i.i.d. normal log-returns the in-range probability has a closed form;
    # the vectorized simulator must reproduce it.
    from math import erf, log, sqrt
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.001, size=5000)
    sd = float(np.std(returns, ddof=1)) * sqrt(4)
    res = estimate_probability_in_range(100.0, returns, 4, 99.8, 100.2, float(np.std(returns, ddof=1)),
                                        n_paths=40000, seed=7)
    expected = 0.5 * (erf(log(100.2 / 100) / sd / sqrt(2)) - erf(log(99.8 / 100) / sd / sqrt(2)))
    assert abs(res.probability - expected) < 0.01
    assert res.probability_uncertainty < 0.005


def test_block_bootstrap_uses_contiguous_blocks():
    from app.models.monte_carlo import _simulate_paths
    # a strictly increasing pool: contiguous-block sampling of 3 steps from
    # block_size 3 must produce sums of 3 consecutive pool entries only
    pool = np.arange(10, dtype=float) * 1e-3
    finals = _simulate_paths(1.0, pool, steps=3, n_paths=500, block_size=3, rng=np.random.default_rng(0))
    sums = np.round(np.log(finals) * 1e3).astype(int)
    valid = {3 * i + 3 for i in range(8)}  # i + (i+1) + (i+2)
    assert set(sums) <= valid


def test_fast_enough_not_to_block_event_loop():
    import time
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.001, size=300)
    t = time.perf_counter()
    estimate_probability_in_range(100.0, returns, 10, 99.7, 100.3, 0.001, n_paths=100_000, seed=1)
    assert time.perf_counter() - t < 1.0
