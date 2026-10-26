from app.strategy.filters import evaluate, expected_value, implied_probability, payout_multiplier


def test_payout_multiplier():
    assert payout_multiplier(0.49, 0.35) - 1.4 < 1e-9


def test_implied_probability_matches_break_even_math():
    p = implied_probability(1.40)
    assert abs(p - (1 / 1.40)) < 1e-9


def test_ev_positive_when_model_beats_implied():
    ev = expected_value(calibrated_probability=0.80, payout_multiplier_value=1.40)
    assert ev > 0


def test_rejects_low_payout():
    d = evaluate(
        calibrated_probability=0.9, probability_uncertainty=0.01, payout=0.40, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
    )
    assert not d.accept
    assert "PAYOUT" in d.reason


def test_rejects_insufficient_edge():
    # payout implies ~71.4% breakeven; model only slightly above -> reject
    d = evaluate(
        calibrated_probability=0.72, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
    )
    assert not d.accept
    assert "EDGE" in d.reason


def test_accepts_strong_edge_and_payout():
    d = evaluate(
        calibrated_probability=0.85, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
    )
    assert d.accept


def test_rejects_high_model_disagreement():
    d = evaluate(
        calibrated_probability=0.85, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
        model_disagreement=0.10, max_model_disagreement=0.05,
    )
    assert not d.accept
    assert "DISAGREEMENT" in d.reason


def test_rejects_low_regime_confidence():
    d = evaluate(
        calibrated_probability=0.85, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
        regime_confidence=0.3, min_regime_confidence=0.55,
    )
    assert not d.accept
    assert "REGIME CONFIDENCE" in d.reason


def test_edge_requirement_scales_with_duration():
    kwargs = dict(
        calibrated_probability=0.78, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
        edge_duration_scaling=0.02,
    )
    short = evaluate(**kwargs, duration_minutes=0)
    long = evaluate(**kwargs, duration_minutes=10)
    assert short.accept
    assert not long.accept  # same edge, but a 10-minute trade needs more margin than a baseline one


def test_caution_penalty_raises_required_edge():
    kwargs = dict(
        calibrated_probability=0.78, probability_uncertainty=0.01, payout=0.49, stake=0.35,
        min_payout_multiplier=1.40, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06,
    )
    normal = evaluate(**kwargs, extra_edge_requirement=0.0)
    cautious = evaluate(**kwargs, extra_edge_requirement=0.10)
    assert normal.accept
    assert not cautious.accept


def test_random_walk_is_not_mostly_trending_and_trend_confidence_scales():
    import numpy as np
    from app.features.stats import Regime, detect_regime
    rng = np.random.default_rng(0)
    labels = []
    for _ in range(400):
        prices = 1000 * np.exp(np.cumsum(rng.normal(0, 0.001, 120)))
        labels.append(detect_regime(prices).regime)
    trending = sum(r in (Regime.TRENDING_UP, Regime.TRENDING_DOWN) for r in labels) / len(labels)
    assert trending < 0.25          # was ~0.74 before the scale fix

    rets = rng.normal(0, 0.001, 119)
    rets[-20:] += 0.0015            # strong 20-bar drift
    strong = detect_regime(1000 * np.exp(np.concatenate([[0], np.cumsum(rets)])))
    assert strong.regime == Regime.TRENDING_UP and strong.confidence > 0.6


def test_no_payout_floor_low_payout_judged_by_edge_alone():
    import os
    if "MIN_PAYOUT_MULTIPLIER" not in os.environ:
        from app.config import Config
        assert Config().min_payout_multiplier == 1.0
    # payout 1.20x -> implied 83.3%; 90% calibrated clears a 5% edge -> accepted
    d = evaluate(calibrated_probability=0.90, probability_uncertainty=0.01, payout=0.42, stake=0.35,
                 min_payout_multiplier=1.0, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06)
    assert d.accept, d.reason
    # same payout, 86% calibrated -> edge too small -> rejected on edge, not payout
    d = evaluate(calibrated_probability=0.86, probability_uncertainty=0.01, payout=0.42, stake=0.35,
                 min_payout_multiplier=1.0, min_edge=0.05, min_ev=0.0, max_probability_uncertainty=0.06)
    assert not d.accept and d.reason == "INSUFFICIENT EDGE"
    # payout not above stake is never accepted
    d = evaluate(calibrated_probability=0.999, probability_uncertainty=0.0, payout=0.35, stake=0.35,
                 min_payout_multiplier=1.0, min_edge=0.0, min_ev=-1.0, max_probability_uncertainty=0.06)
    assert not d.accept and d.reason == "PAYOUT DOES NOT EXCEED STAKE"
