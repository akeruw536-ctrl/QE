"""Width (within a signal) and market level (between signals) are calibrated separately."""
import os
import sqlite3
import tempfile
import time

import numpy as np
import pytest

from app.data.storage import Storage
from app.models.calibration import AdaptiveCalibrator, CalibrationTracker, _logit, _sigmoid


class _Log:
    def info(self, *a, **k): pass
    warning = error = debug = info


def _store():
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    s = Storage("", path, _Log()); s.init_schema()
    return s, path


def _signals(n, seed, a=1.3, b_level=-0.4, b_width=1.0, offsets_d=(-0.6, -0.2, 0.2, 0.6), stored=True, symbol="1HZ10V"):
    """n signals x 4 widths. The model's raw logit = level + width offset; the truth uses the two slopes."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time() - 3600
    for i in range(n):
        level = float(rng.normal(1.1, 0.5))
        opened = t0 + i * 60
        xs = [level + d for d in offsets_d]
        m = float(np.mean(xs))
        for x in xs:
            p = _sigmoid(a + b_level * m + b_width * (x - m))
            rows.append({"raw": _sigmoid(x), "won": int(rng.random() < p), "regime": "NORMAL_RANGE", "opened_at": opened,
                         "source": "probe", "symbol": symbol, "signal_mean": m if stored else None})
    return rows


def test_recovers_a_flat_or_negative_level_slope_and_a_unit_width_slope():
    cal = AdaptiveCalibrator(_store()[0], halflife_hours=0)
    cal.params, summ = cal.fit(_signals(500, 1))
    assert cal.params.b_within == pytest.approx(1.0, abs=0.25)
    assert cal.params.b < 0.2                                   # level is NOT trustworthy (truth: -0.4)
    assert summ["b_within"] == pytest.approx(cal.params.b_within, abs=1e-3)


def test_older_probes_without_a_stored_mean_are_grouped_by_signal():
    stored, grouped = AdaptiveCalibrator(_store()[0], halflife_hours=0), AdaptiveCalibrator(_store()[0], halflife_hours=0)
    stored.params, _ = stored.fit(_signals(300, 2, stored=True))
    grouped.params, _ = grouped.fit(_signals(300, 2, stored=False))      # same data, mean recomputed from siblings
    assert grouped.params.b_within == pytest.approx(stored.params.b_within, abs=0.03)
    assert grouped.params.b == pytest.approx(stored.params.b, abs=0.03)


def test_wider_barriers_get_higher_probability_and_tight_ones_are_not_flattered():
    rows = _signals(400, 3)
    two = AdaptiveCalibrator(_store()[0], halflife_hours=0)
    two.params, _ = two.fit(rows)
    one = AdaptiveCalibrator(_store()[0], halflife_hours=0)
    one.params, _ = one.fit([{**r, "signal_mean": None, "symbol": None} for r in rows])   # the old, single-slope view

    level = 1.1
    xs = [level + d for d in (-0.6, -0.2, 0.2, 0.6)]
    m = float(np.mean(xs))
    truth = [_sigmoid(1.3 - 0.4 * m + 1.0 * (x - m)) for x in xs]
    p_two = [two.calibrate(_sigmoid(x), regime="NORMAL_RANGE", signal_mean=m) for x in xs]
    p_one = [one.calibrate(_sigmoid(x), regime="NORMAL_RANGE") for x in xs]

    assert p_two == sorted(p_two) and p_two[0] < p_two[-1] - 0.15          # monotone, strong width response
    assert abs(p_two[0] - truth[0]) < 0.06                                  # narrowest barrier is priced honestly
    assert p_one[0] - truth[0] > 0.05                                       # the single slope overstated it
    assert sum(abs(a - b) for a, b in zip(p_two, truth)) < sum(abs(a - b) for a, b in zip(p_one, truth))


def test_two_slopes_beat_one_slope_on_held_out_signals():
    train, test = _signals(400, 4), _signals(200, 5)
    two = AdaptiveCalibrator(_store()[0], halflife_hours=0); two.params, _ = two.fit(train)
    one = AdaptiveCalibrator(_store()[0], halflife_hours=0); one.params, _ = one.fit([{**r, "signal_mean": None, "symbol": None} for r in train])
    y = np.array([r["won"] for r in test])
    p2 = np.array([two.calibrate(r["raw"], regime=r["regime"], signal_mean=r["signal_mean"]) for r in test])
    p1 = np.array([one.calibrate(r["raw"], regime=r["regime"]) for r in test])
    base = np.full(len(y), np.mean([r["won"] for r in train]))
    brier = lambda p: float(np.mean((p - y) ** 2))
    assert brier(p2) < brier(base) < brier(p1)       # two slopes beat the base rate; one slope is worse than it


def test_without_siblings_it_behaves_like_the_single_feature_model():
    cal = AdaptiveCalibrator(_store()[0])
    assert cal.calibrate(0.8) == pytest.approx(0.8, abs=1e-6)               # identity prior, no signal mean needed
    assert CalibrationTracker(_store()[0]).calibrate(0.8, signal_mean=1.0) == pytest.approx(0.8, abs=0.02)


# ---------------------------------------------------------------- storage
def test_signal_mean_is_persisted_and_returned_by_the_dataset():
    st, _ = _store()
    now = time.time()
    st.record_trade({"trade_id": "t", "symbol": "1HZ10V", "stake": 1.0, "payout": 1.3, "raw_probability": 0.8,
                     "regime": "NORMAL_RANGE", "opened_at": now - 100, "shadow": 0, "signal_logit_mean": 1.234})
    st.settle_trade("t", "WIN", 0.3)
    st.record_probes([{"symbol": "1HZ10V", "regime": "NORMAL_RANGE", "raw_probability": 0.7, "lower_barrier": 1.0,
                       "upper_barrier": 2.0, "opened_at": now - 100, "expires_at": now - 40, "signal_logit_mean": 0.9}])
    for p in st.open_probes(now):
        st.resolve_probe(p["id"], True, 1.5)
    ds = {d["source"]: d for d in st.calibration_dataset()}
    assert ds["trade"]["signal_mean"] == pytest.approx(1.234) and ds["probe"]["signal_mean"] == pytest.approx(0.9)
    assert ds["probe"]["symbol"] == "1HZ10V"


def test_a_database_created_before_this_column_is_migrated_in_place():
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE calibration_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TIMESTAMP, symbol VARCHAR,"
                " regime VARCHAR, raw_probability FLOAT, lower_barrier FLOAT, upper_barrier FLOAT, opened_at FLOAT,"
                " expires_at FLOAT, resolved INTEGER, won INTEGER, exit_spot FLOAT)")
    con.execute("INSERT INTO calibration_samples (symbol, raw_probability, opened_at, expires_at, resolved, won) "
                "VALUES ('1HZ10V', 0.8, 1.0, 2.0, 1, 1)")
    con.commit(); con.close()
    st = Storage("", path, _Log()); st.init_schema()
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(calibration_samples)")}
    assert "signal_logit_mean" in cols
    assert st.calibration_dataset()[0]["signal_mean"] is None            # old row kept, mean recomputed by grouping
