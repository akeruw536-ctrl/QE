"""
Online probability calibration (spec section 24).

Tracks predicted-probability vs actual-outcome by bucket (e.g. "0.70-0.75")
and nudges raw Monte Carlo probabilities toward the empirically observed
win rate for that bucket. Deliberately conservative: with few observations
in a bucket, the calibrated probability stays close to the raw estimate
(shrinkage toward the model, not toward an arbitrary prior) and only pulls
harder toward the observed rate as evidence accumulates.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def bucket_for(probability: float, width: float = 0.05) -> str:
    probability = min(max(probability, 0.0), 0.999999)
    lo = int(probability / width) * width
    hi = lo + width
    return f"{lo:.2f}-{hi:.2f}"


class CalibrationTracker:
    def __init__(self, storage, min_observations_for_full_trust: int = 40):
        self.storage = storage
        self.min_observations_for_full_trust = min_observations_for_full_trust

    def snapshot(self) -> Dict[str, Tuple[int, int]]:
        """Current bucket counts; pass to calibrate() to avoid a DB read per call."""
        return self.storage.get_calibration()

    def calibrate(self, raw_probability: float, buckets: Optional[Dict[str, Tuple[int, int]]] = None,
                  regime: Optional[str] = None) -> float:
        """Maps a RAW model probability to a calibrated one. Buckets are keyed
        by raw probability, so record_outcome() must also be given the raw
        probability -- see its docstring."""
        if buckets is None:
            buckets = self.storage.get_calibration()
        key = bucket_for(raw_probability)
        n, wins = buckets.get(key, (0, 0))
        if n == 0:
            return raw_probability
        observed_rate = wins / n
        weight = min(1.0, n / self.min_observations_for_full_trust)
        return (1 - weight) * raw_probability + weight * observed_rate

    def record_outcome(self, raw_probability: float, won: bool) -> None:
        """MUST be called with the RAW model probability, never the calibrated
        one. calibrate() looks buckets up by raw probability; recording by the
        calibrated value (as run.py previously did) files each outcome under
        a different bucket than the one later consulted, so the calibration
        layer learned a relationship that was never the one it applied."""
        self.storage.update_calibration(bucket_for(raw_probability), won)


# ---------------------------------------------------------------------------
# Faster-learning calibration
# ---------------------------------------------------------------------------
def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    z = max(min(z, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class CalibrationParams:
    """calibrated = sigmoid(a + b * logit(raw) + offset[regime]).
    The identity (a=0, b=1, no offsets) is the starting point and the prior."""
    a: float = 0.0
    b: float = 1.0
    offsets: Dict[str, float] = field(default_factory=dict)


class AdaptiveCalibrator(CalibrationTracker):
    """
    Logistic (Platt-style) recalibration that learns from EVERY labelled
    outcome at once, instead of 0.05-wide buckets that each need ~40 trades.

      * 2 shared parameters (intercept, slope) are fitted on all data, so a
        systematic optimism such as "the model says 80% but it is really 72%"
        is learned from a few dozen outcomes, not a few hundred.
      * Per-regime intercept offsets let a regime that is worse (or better)
        than the model thinks be corrected -- with a stronger prior, so a
        regime has to earn its own correction.
      * A ridge prior toward "no correction" keeps it stable on little data:
        with no data it returns the raw probability unchanged.
      * Recent outcomes count more (exponential half-life), so it follows the
        market instead of averaging over all history.
      * Data = settled trades + resolved probes (priced-but-not-traded
        candidates resolved at expiry, weighted down because the candidates
        of one signal share one terminal price).
    """

    def __init__(self, storage, probe_weight: float = 0.4, halflife_hours: float = 96.0,
                 prior_strength: float = 3.0, regime_prior_strength: float = 10.0,
                 refit_after: int = 5, max_rows: int = 20000, min_observations_for_full_trust: int = 40):
        super().__init__(storage, min_observations_for_full_trust)
        self.probe_weight = probe_weight
        self.halflife_hours = halflife_hours
        self.prior_strength = prior_strength
        self.regime_prior_strength = regime_prior_strength
        self.refit_after = refit_after
        self.max_rows = max_rows
        self.params = CalibrationParams()
        self._dirty = 0
        self.last_fit_at: Optional[float] = None
        self._summary: Dict = {"n_trades": 0, "n_probes": 0, "a": 0.0, "b": 1.0, "regime_offsets": {}}

    # ---------------------------------------------------------------- use
    def calibrate(self, raw_probability: float, buckets=None, regime: Optional[str] = None) -> float:
        p = self.params
        z = p.a + p.b * _logit(raw_probability) + p.offsets.get(regime or "", 0.0)
        return min(max(_sigmoid(z), 0.001), 0.999)

    def record_outcome(self, raw_probability: float, won: bool) -> None:
        super().record_outcome(raw_probability, won)      # legacy bucket table, kept for diagnostics
        self._dirty += 1

    def mark_dirty(self, n: int = 1) -> None:
        self._dirty += n

    def needs_refit(self) -> bool:
        return self._dirty >= self.refit_after

    # ---------------------------------------------------------------- fit
    def fit(self, rows: List[dict], now: Optional[float] = None) -> Tuple[CalibrationParams, Dict]:
        now = time.time() if now is None else now
        n = len(rows)
        n_trades = sum(1 for r in rows if r["source"] == "trade")
        summary = {"n_trades": n_trades, "n_probes": n - n_trades, "a": 0.0, "b": 1.0, "regime_offsets": {}}
        if n < 3:
            return CalibrationParams(), summary

        regimes = sorted({r["regime"] for r in rows if r.get("regime")})
        ridx = {name: i for i, name in enumerate(regimes)}
        k = 2 + len(regimes)
        X = np.zeros((n, k)); X[:, 0] = 1.0
        y = np.zeros(n); w = np.zeros(n)
        for i, r in enumerate(rows):
            X[i, 1] = _logit(r["raw"])
            if r.get("regime") in ridx:
                X[i, 2 + ridx[r["regime"]]] = 1.0
            y[i] = r["won"]
            weight = 1.0 if r["source"] == "trade" else self.probe_weight
            if self.halflife_hours and r.get("opened_at"):
                age_h = max(now - float(r["opened_at"]), 0.0) / 3600.0
                weight *= 0.5 ** (age_h / self.halflife_hours)
            w[i] = weight

        theta0 = np.zeros(k); theta0[1] = 1.0                       # prior mean: no correction
        lam = np.array([self.prior_strength, self.prior_strength] + [self.regime_prior_strength] * len(regimes))
        theta = theta0.copy()
        for _ in range(40):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ theta, -30, 30)))
            grad = X.T @ (w * (y - p)) - lam * (theta - theta0)
            hess = (X.T * (w * p * (1 - p))) @ X + np.diag(lam)
            step = np.linalg.solve(hess, grad)
            theta = theta + step
            if np.max(np.abs(step)) < 1e-8:
                break
        a = float(np.clip(theta[0], -3.0, 3.0))
        b = float(np.clip(theta[1], 0.2, 2.0))                     # keep it monotonic and sane
        offsets = {name: float(np.clip(theta[2 + i], -2.0, 2.0)) for name, i in ridx.items()}

        raw_p = 1.0 / (1.0 + np.exp(-X[:, 1]))
        cal_z = a + b * X[:, 1] + np.array([offsets.get(r.get("regime") or "", 0.0) for r in rows])
        cal_p = 1.0 / (1.0 + np.exp(-cal_z))
        summary.update({
            "a": round(a, 3), "b": round(b, 3),
            "regime_offsets": {k_: round(v, 3) for k_, v in offsets.items()},
            "actual_win_rate": round(float(y.mean()), 4),
            "mean_raw": round(float(raw_p.mean()), 4), "mean_calibrated": round(float(cal_p.mean()), 4),
            "brier_raw": round(float(np.mean((raw_p - y) ** 2)), 4),
            "brier_calibrated": round(float(np.mean((cal_p - y) ** 2)), 4),
        })
        return CalibrationParams(a=a, b=b, offsets=offsets), summary

    def refit(self) -> Dict:
        rows = self.storage.calibration_dataset(self.max_rows)
        self.params, self._summary = self.fit(rows)
        self._dirty = 0
        self.last_fit_at = time.time()
        return self._summary

    def summary(self) -> Dict:
        out = dict(self._summary)
        out["last_fit_age_seconds"] = round(time.time() - self.last_fit_at, 1) if self.last_fit_at else None
        return out
