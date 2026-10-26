"""
Feature calculations used by both the Monte Carlo engine and the regime
detector. Pure numpy -- no lookahead, operates only on data available up
to the last supplied index.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


def log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    prices = prices[prices > 0]
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(prices))


def realized_volatility(returns: np.ndarray, annualize_periods: Optional[int] = None) -> float:
    if len(returns) < 2:
        return 0.0
    vol = float(np.std(returns, ddof=1))
    if annualize_periods:
        vol *= np.sqrt(annualize_periods)
    return vol


def rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    if len(returns) < window:
        return np.array([])
    out = np.empty(len(returns) - window + 1)
    for i in range(len(out)):
        out[i] = np.std(returns[i:i + window], ddof=1)
    return out


def skew(returns: np.ndarray) -> float:
    if len(returns) < 3:
        return 0.0
    m = np.mean(returns)
    s = np.std(returns, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((returns - m) / s) ** 3))


def kurtosis_excess(returns: np.ndarray) -> float:
    if len(returns) < 4:
        return 0.0
    m = np.mean(returns)
    s = np.std(returns, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((returns - m) / s) ** 4) - 3.0)


def lag1_autocorrelation(returns: np.ndarray) -> float:
    if len(returns) < 3:
        return 0.0
    r = returns - np.mean(returns)
    num = np.sum(r[:-1] * r[1:])
    den = np.sum(r ** 2)
    return float(num / den) if den != 0 else 0.0


def momentum(prices: np.ndarray, lookback: int) -> float:
    prices = np.asarray(prices, dtype=float)
    if len(prices) < lookback + 1:
        return 0.0
    return float(prices[-1] / prices[-lookback - 1] - 1.0)


class Regime(str, Enum):
    LOW_VOLATILITY_RANGE = "LOW_VOLATILITY_RANGE"
    NORMAL_RANGE = "NORMAL_RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    VOLATILITY_CONTRACTION = "VOLATILITY_CONTRACTION"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    MEAN_REVERTING = "MEAN_REVERTING"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class RegimeAssessment:
    regime: Regime
    confidence: float
    volatility: float
    momentum_val: float
    autocorr: float


def detect_regime(prices: np.ndarray, short_window: int = 20, long_window: int = 60) -> RegimeAssessment:
    """
    Heuristic, explainable regime classifier. Not a learned model -- it's
    deliberately simple so the Monte Carlo sample-selection logic that
    consumes it stays auditable. Swap in a learned classifier later without
    changing the interface.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < long_window + 5:
        return RegimeAssessment(Regime.UNCERTAIN, 0.2, 0.0, 0.0, 0.0)

    rets = log_returns(prices)
    short_vol = realized_volatility(rets[-short_window:])
    long_vol = realized_volatility(rets[-long_window:])
    mom = momentum(prices, short_window)
    ac = lag1_autocorrelation(rets[-long_window:])

    vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0

    # trend check first -- strong directional momentum dominates.
    #
    # SCALE FIX: `mom` is a short_window-bar move, `long_vol` a 1-bar
    # volatility. Comparing them directly (the old `mom > 1.5 * long_vol`)
    # meant a pure random walk -- whose 20-bar move is typically
    # long_vol * sqrt(20) ~ 4.5x a 1-bar move -- read as "trending" ~74% of
    # the time. And confidence was `0.5 + abs(mom)`, adding a raw fractional
    # return (~0.001) to 0.5, which pinned every trend at 0.50-0.52 and
    # below the 0.55 trading gate regardless of how strong it was. Both now
    # use the move measured in its own standard deviations.
    trend_z = mom / (long_vol * np.sqrt(short_window)) if long_vol > 0 else 0.0
    trend_conf = min(0.9, 0.5 + 0.15 * (abs(trend_z) - 1.5))
    if trend_z > 1.5:
        return RegimeAssessment(Regime.TRENDING_UP, trend_conf, short_vol, mom, ac)
    if trend_z < -1.5:
        return RegimeAssessment(Regime.TRENDING_DOWN, trend_conf, short_vol, mom, ac)

    if ac < -0.15:
        return RegimeAssessment(Regime.MEAN_REVERTING, min(0.85, 0.5 + abs(ac)), short_vol, mom, ac)

    if vol_ratio > 1.4:
        return RegimeAssessment(Regime.VOLATILITY_EXPANSION, min(0.85, 0.4 + (vol_ratio - 1)), short_vol, mom, ac)
    if vol_ratio < 0.7:
        return RegimeAssessment(Regime.VOLATILITY_CONTRACTION, min(0.85, 0.4 + (1 - vol_ratio)), short_vol, mom, ac)

    if long_vol == 0:
        return RegimeAssessment(Regime.UNCERTAIN, 0.2, short_vol, mom, ac)

    percentile_proxy = short_vol / long_vol
    if percentile_proxy < 0.85:
        return RegimeAssessment(Regime.LOW_VOLATILITY_RANGE, 0.6, short_vol, mom, ac)
    if percentile_proxy > 1.15:
        return RegimeAssessment(Regime.HIGH_VOLATILITY, 0.6, short_vol, mom, ac)

    return RegimeAssessment(Regime.NORMAL_RANGE, 0.55, short_vol, mom, ac)
