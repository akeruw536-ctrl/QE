import math
import os
import tempfile
import time

import numpy as np

from app.data.storage import Storage
from app.models.calibration import AdaptiveCalibrator, CalibrationTracker, _logit, _sigmoid


class _Log:
    def info(self, *a, **k): pass
    warning = error = debug = info


def _store():
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    s = Storage("", path, _Log()); s.init_schema()
    return s


def _rows(n, seed, a=-0.4, b=0.85, regime=None, source="trade", age_h=0.0, raw_lo=0.6, raw_hi=0.97, regime_shift=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        raw = float(rng.uniform(raw_lo, raw_hi))
        p = _sigmoid(a + b * _logit(raw) + regime_shift)
        out.append({"raw": raw, "won": int(rng.random() < p), "regime": regime,
                    "opened_at": time.time() - age_h * 3600, "source": source})
    return out


def test_identity_without_data():
    cal = AdaptiveCalibrator(_store())
    assert math.isclose(cal.calibrate(0.8), 0.8, abs_tol=1e-9)
    params, summ = cal.fit([])
    assert params.a == 0.0 and params.b == 1.0 and summ["n_trades"] == 0


def test_learns_a_systematic_optimism_from_few_outcomes_much_faster_than_buckets():
    # the true win probability is lower than the model says (like QE: says 80%, wins ~72%)
    grid = np.linspace(0.62, 0.96, 60)
    truth = np.array([_sigmoid(-0.4 + 0.85 * _logit(p)) for p in grid])
    err_log, err_bucket, err_raw = [], [], []
    for seed in range(6):
        rows = _rows(60, seed)                                  # only 60 labelled outcomes
        cal = AdaptiveCalibrator(_store())
        cal.params, _ = cal.fit(rows)
        bucket = CalibrationTracker(_store())
        for r in rows:
            bucket.record_outcome(r["raw"], bool(r["won"]))
        err_log.append(np.mean([abs(cal.calibrate(p) - t) for p, t in zip(grid, truth)]))
        err_bucket.append(np.mean([abs(bucket.calibrate(p) - t) for p, t in zip(grid, truth)]))
        err_raw.append(np.mean(np.abs(grid - truth)))
    assert np.mean(err_log) < 0.6 * np.mean(err_raw)            # closes most of the optimism gap
    assert np.mean(err_log) < np.mean(err_bucket)               # and beats the bucket tracker at this sample size


def test_calibration_is_monotonic_and_bounded():
    cal = AdaptiveCalibrator(_store())
    cal.params, _ = cal.fit(_rows(200, 1))
    vals = [cal.calibrate(p) for p in np.linspace(0.05, 0.999, 40)]
    assert all(b >= a for a, b in zip(vals, vals[1:])) and 0.001 <= min(vals) and max(vals) <= 0.999


def test_all_wins_does_not_blow_up():
    rows = [{"raw": 0.8, "won": 1, "regime": "NORMAL_RANGE", "opened_at": time.time(), "source": "trade"}] * 25
    cal = AdaptiveCalibrator(_store())
    cal.params, _ = cal.fit(rows)
    assert 0.8 < cal.calibrate(0.8) < 0.999 and abs(cal.params.a) <= 3 and 0.2 <= cal.params.b <= 2.0


def test_a_worse_regime_earns_its_own_offset():
    rows = _rows(250, 3, regime="NORMAL_RANGE") + _rows(250, 4, regime="MEAN_REVERTING", regime_shift=-0.9)
    cal = AdaptiveCalibrator(_store())
    cal.params, summ = cal.fit(rows)
    off = cal.params.offsets
    assert off["MEAN_REVERTING"] < off["NORMAL_RANGE"] - 0.3
    assert cal.calibrate(0.85, regime="MEAN_REVERTING") < cal.calibrate(0.85, regime="NORMAL_RANGE") - 0.04
    assert cal.calibrate(0.85, regime="NEVER_SEEN") == cal.calibrate(0.85)       # unknown regime -> shared curve


def test_recent_outcomes_outweigh_old_ones():
    old = _rows(150, 5, a=0.6, b=1.0, age_h=600)               # long ago the model was pessimistic
    new = _rows(150, 6, a=-0.8, b=1.0, age_h=1)                # now it is optimistic
    decayed = AdaptiveCalibrator(_store(), halflife_hours=48)
    flat = AdaptiveCalibrator(_store(), halflife_hours=0)
    decayed.params, _ = decayed.fit(old + new)
    flat.params, _ = flat.fit(old + new)
    assert decayed.calibrate(0.8) < flat.calibrate(0.8) - 0.03


def test_probes_are_downweighted_and_can_be_switched_off():
    trades = _rows(40, 7, a=-0.5, source="trade")
    probes = _rows(400, 8, a=0.5, source="probe")
    zero = AdaptiveCalibrator(_store(), probe_weight=0.0)
    some = AdaptiveCalibrator(_store(), probe_weight=0.4)
    zero.params, sz = zero.fit(trades + probes)
    some.params, ss = some.fit(trades + probes)
    only_trades = AdaptiveCalibrator(_store()); only_trades.params, _ = only_trades.fit(trades)
    assert math.isclose(zero.calibrate(0.8), only_trades.calibrate(0.8), abs_tol=1e-6)
    assert some.calibrate(0.8) > zero.calibrate(0.8) + 0.03
    assert ss["n_trades"] == 40 and ss["n_probes"] == 400


def test_refit_reads_trades_and_resolved_probes_and_reports_improvement():
    s = _store()
    now = time.time()
    for i, r in enumerate(_rows(60, 9)):
        s.record_trade({"trade_id": f"t{i}", "symbol": "1HZ90V", "stake": 1.0, "payout": 1.3,
                        "raw_probability": r["raw"], "regime": "NORMAL_RANGE", "opened_at": now - 100, "shadow": 0})
        s.settle_trade(f"t{i}", "WIN" if r["won"] else "LOSS", 0.3 if r["won"] else -1.0)
    probes = _rows(120, 10)
    s.record_probes([{"symbol": "1HZ90V", "regime": "NORMAL_RANGE", "raw_probability": r["raw"],
                      "lower_barrier": 1.0, "upper_barrier": 2.0, "opened_at": now - 100, "expires_at": now - 40}
                     for r in probes])
    for p, r in zip(s.open_probes(now), probes):
        s.resolve_probe(p["id"], bool(r["won"]), 1.5)
    cal = AdaptiveCalibrator(s)
    out = cal.refit()
    assert out["n_trades"] == 60 and out["n_probes"] == 120
    assert out["brier_calibrated"] < out["brier_raw"] and out["mean_calibrated"] < out["mean_raw"]
    assert out["b"] != 1.0 and cal.calibrate(0.9) < 0.9


def test_dirty_tracking_triggers_refit():
    cal = AdaptiveCalibrator(_store(), refit_after=3)
    assert not cal.needs_refit()
    cal.record_outcome(0.8, True); cal.record_outcome(0.8, False)
    assert not cal.needs_refit()
    cal.mark_dirty(1)
    assert cal.needs_refit()
    cal.refit()
    assert not cal.needs_refit() and cal.summary()["last_fit_age_seconds"] is not None


def test_reliability_table_reports_predicted_vs_actual_by_bucket():
    rows = []
    rng = np.random.default_rng(11)
    for raw, win_p, n, src in [(0.62, 0.55, 300, "probe"), (0.85, 0.70, 300, "probe"), (0.97, 0.80, 100, "probe"),
                               (0.85, 0.90, 20, "trade")]:
        for _ in range(n):
            rows.append({"raw": raw, "won": int(rng.random() < win_p), "regime": None,
                         "opened_at": time.time(), "source": src})
    cal = AdaptiveCalibrator(_store())
    _, summ = cal.fit(rows)
    rel = {r["bucket"]: r for r in summ["reliability"]}
    assert set(rel) == {"0.60-0.70", "0.80-0.90", ">=0.95"}
    assert rel["0.80-0.90"]["n"] == 320 and rel["0.80-0.90"]["n_trades"] == 20
    assert abs(rel["0.60-0.70"]["actual"] - 0.55) < 0.07 and abs(rel[">=0.95"]["actual"] - 0.80) < 0.12
    assert rel["0.80-0.90"]["actual_probes"] < rel["0.80-0.90"]["actual"]        # the 20 good trades lift the blend
    line = AdaptiveCalibrator.format_reliability(summ)
    assert "0.60-0.70:" in line and "n=300" in line
    assert "brier_constant" in summ and summ["raw_logit_sd"] > 0.3


def test_slope_prior_pins_the_slope_when_raw_probabilities_span_a_narrow_range():
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(1500):                                       # raw probabilities only in 0.70-0.80
        raw = float(rng.uniform(0.70, 0.80))
        rows.append({"raw": raw, "won": int(rng.random() < _sigmoid(0.2 + 0.2 * _logit(raw))), "regime": None,
                     "opened_at": time.time(), "source": "probe"})
    free = AdaptiveCalibrator(_store(), slope_prior_strength=3.0)
    pinned = AdaptiveCalibrator(_store(), slope_prior_strength=2000.0)
    free.params, _ = free.fit(rows)
    pinned.params, _ = pinned.fit(rows)
    assert free.params.b < 0.6                                  # the data pull the slope flat
    assert pinned.params.b > 0.85                               # the anchor keeps it near 1
    assert abs(pinned.calibrate(0.75) - free.calibrate(0.75)) < 0.03   # while the level is still corrected either way
