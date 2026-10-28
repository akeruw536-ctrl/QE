"""Learning from priced-but-not-bought candidates ("probes")."""
import asyncio
import os
import tempfile
import time

import pytest

import run as run_mod
from app.models.calibration import AdaptiveCalibrator
from app.strategy.engine import evaluate_signal
from app.strategy.staking import build_staking_engine
from tests.test_engine_and_flow import FakeClient, WidthPricedClient, _Log, _candles, _cfg, _closes, _storage


def _run(cfg, client, closes, **kw):
    st = _storage()
    out = asyncio.run(evaluate_signal("1HZ90V", "DOWN", closes, float(closes[-1]), client, AdaptiveCalibrator(st),
                                      build_staking_engine(cfg.staking), cfg, _Log(), **kw))
    return out, st


# ---------------------------------------------------------------- storage
def test_probe_lifecycle_and_dataset():
    st = _storage()
    now = time.time()
    rows = [{"symbol": "1HZ90V", "regime": "NORMAL_RANGE", "raw_probability": p, "lower_barrier": 99.0,
             "upper_barrier": 101.0, "opened_at": now - 200, "expires_at": now - 80} for p in (0.7, 0.8, 0.9)]
    assert st.record_probes(rows + [{"raw_probability": None}]) == 3          # rows without a probability are skipped
    assert st.probe_counts() == {"open": 3, "resolved": 0, "abandoned": 0}
    assert st.open_probes(now - 500) == []                                    # not due yet
    due = st.open_probes(now)
    assert len(due) == 3
    assert st.resolve_probe(due[0]["id"], True, 100.0) is True
    assert st.resolve_probe(due[0]["id"], False, 100.0) is False              # idempotent: never counted twice
    st.abandon_probe(due[1]["id"])
    assert st.probe_counts() == {"open": 1, "resolved": 1, "abandoned": 1}
    ds = st.calibration_dataset()
    assert len(ds) == 1 and ds[0]["source"] == "probe" and ds[0]["won"] == 1
    assert st.prune_probes(now + 1) == 2 and st.probe_counts()["open"] == 1   # resolved/abandoned pruned, open kept


# ----------------------------------------------------------------- engine
def test_unbought_candidates_become_probes_and_the_bought_one_does_not():
    cfg = _cfg(use_edge_gate=False, use_regime_gate=False, barrier_scales=[1.0, 1.5, 2.0, 2.5])
    out, _ = _run(cfg, WidthPricedClient(), _closes())
    assert out.traded
    assert len(out.probes) == 3                                                # 4 priced - 1 bought
    bought_lo = out.trade_row["lower_barrier"]
    assert all(abs(p["lower_barrier"] - bought_lo) > 1.0 for p in out.probes)  # the traded width is not duplicated
    p = out.probes[0]
    assert p["symbol"] == "1HZ90V" and 0 < p["raw_probability"] < 1 and p["expires_at"] - p["opened_at"] == 120


def test_probe_only_prices_everything_but_never_asks_deriv_for_a_quote():
    cfg = _cfg(use_edge_gate=False, use_regime_gate=False, barrier_scales=[1.0, 1.5, 2.0])
    client = FakeClient()
    out, _ = _run(cfg, client, _closes(), probe_only=True)
    assert not out.traded and client.proposals == [] and len(out.probes) == 3


def test_a_regime_blocked_signal_still_teaches_the_model():
    cfg = _cfg(use_edge_gate=False, barrier_scales=[1.0, 1.5])                 # regime gate on
    client = FakeClient()
    out, _ = _run(cfg, client, _closes(drift=0.004, sigma=0.0005))            # trending -> blocked
    assert not out.traded and client.proposals == []
    assert "NOT RANGE-FRIENDLY" in out.rejections[0]["rejection_reason"]
    assert len(out.probes) == 2 and out.probes[0]["regime"].startswith("TRENDING")


def test_learning_can_be_switched_off():
    cfg = _cfg(use_edge_gate=False, use_regime_gate=False, barrier_scales=[1.0, 1.5], learn_from_rejected=False)
    out, _ = _run(cfg, FakeClient(), _closes())
    assert out.traded and out.probes == []


def test_calibration_regime_offset_is_used_by_the_engine():
    cfg = _cfg(use_edge_gate=False, use_regime_gate=False, barrier_scales=[1.0])
    st = _storage()
    cal = AdaptiveCalibrator(st)
    base = asyncio.run(evaluate_signal("1HZ90V", "DOWN", _closes(), 28000.0, FakeClient(), cal,
                                       build_staking_engine(cfg.staking), cfg, _Log()))
    from app.models.calibration import CalibrationParams
    cal.params = CalibrationParams(a=-0.5, b=1.0, offsets={})
    shifted = asyncio.run(evaluate_signal("1HZ90V", "DOWN", _closes(), 28000.0, FakeClient(), cal,
                                          build_staking_engine(cfg.staking), cfg, _Log()))
    assert shifted.trade_row["calibrated_probability"] < base.trade_row["calibrated_probability"] - 0.03
    assert shifted.trade_row["raw_probability"] == pytest.approx(base.trade_row["raw_probability"], abs=0.03)


# -------------------------------------------------------------------- bot
class SpotClient(FakeClient):
    def __init__(self, exit_price):
        super().__init__()
        self.exit_price = exit_price
        self.spot_calls = []

    async def get_spot_at(self, symbol, epoch):
        self.spot_calls.append((symbol, epoch))
        return (self.exit_price, epoch)


@pytest.fixture
def bot(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    monkeypatch.setenv("SQLITE_PATH", path)
    monkeypatch.setenv("USE_EDGE_GATE", "false")
    monkeypatch.setenv("USE_REGIME_GATE", "false")
    monkeypatch.setenv("BARRIER_SCALES", "1.0,1.5,2.0")
    monkeypatch.setenv("MC_PATHS", "3000")
    monkeypatch.setenv("MIN_WIN_PROBABILITY", "0")
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(run_mod, "SHADOW_RESOLVE_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(run_mod, "SHADOW_RESOLVE_GRACE_SECONDS", 0)
    b = run_mod.Bot()
    b.storage.init_schema()
    b.client = FakeClient()
    return b


def _signal(bot, black=True):
    async def go():
        bot.client.candles = _candles(black)
        await bot._prime_candles()
        await bot._prime_ticks()
        await bot._handle_signal("1HZ90V", "DOWN")
    asyncio.run(go())


def test_signal_while_a_position_is_open_is_learned_from_but_not_traded(bot):
    _signal(bot)                                       # first signal trades (shadow) and occupies the slot
    assert len(bot.storage.recent_trades()) == 1 and bot._open_count() == 1
    before = bot.storage.probe_counts()["open"]
    _signal(bot)                                       # second signal: slot busy
    assert len(bot.storage.recent_trades()) == 1       # no second trade
    assert bot.storage.probe_counts()["open"] == before + 3      # all 3 widths recorded as probes
    assert bot._open_count() == 1                      # the running trade is untouched


def test_session_stop_still_collects_learning_data(bot):
    bot.cfg.stop_loss = 0.5
    bot.storage.record_trade({"trade_id": "x", "symbol": "1HZ90V", "stake": 1.0, "payout": 1.3,
                              "opened_at": time.time() - 500, "expires_at": time.time() - 300, "shadow": 1})
    bot.storage.settle_trade("x", "LOSS", -1.0)
    bot._session_cache = None
    _signal(bot)
    assert len(bot.storage.recent_trades()) == 1 and bot.storage.probe_counts()["open"] == 3


def test_a_learning_only_error_does_not_pause_trading(bot, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("pricing exploded")
    monkeypatch.setattr(run_mod, "evaluate_signal", boom)
    bot._open_count = lambda symbol=None: 1            # busy slot -> probe-only path
    _signal(bot)
    assert not bot.sm.is_symbol_cautioned("1HZ90V")


def _insert_due_probes(bot, lo, hi):
    now = time.time()
    bot.storage.record_probes([
        {"symbol": "1HZ90V", "regime": "NORMAL_RANGE", "raw_probability": p, "lower_barrier": lo,
         "upper_barrier": hi, "opened_at": now - 300, "expires_at": now - 100} for p in (0.7, 0.8, 0.9)])


def _run_resolver(bot, seconds=0.4):
    async def go():
        task = asyncio.create_task(bot._probe_resolver_loop())
        await asyncio.sleep(seconds)
        bot._stop.set()
        await task
    asyncio.run(go())


def test_probe_resolver_settles_a_whole_signal_with_one_tick_lookup(bot):
    bot.client = SpotClient(exit_price=100.0)
    _insert_due_probes(bot, 99.0, 101.0)               # exit strictly inside -> all won
    _run_resolver(bot)
    c = bot.storage.probe_counts()
    assert c["resolved"] == 3 and c["open"] == 0
    assert len(bot.client.spot_calls) == 1             # one lookup for the shared expiry
    assert all(r["won"] == 1 for r in bot.storage.calibration_dataset())
    assert bot.calibration._dirty >= 3                  # outcomes arrived -> a refit is due


def test_probe_win_rule_matches_deriv_exit_strictly_inside(bot):
    bot.client = SpotClient(exit_price=101.0)          # exactly on the upper barrier -> loses
    _insert_due_probes(bot, 99.0, 101.0)
    _run_resolver(bot)
    assert [r["won"] for r in bot.storage.calibration_dataset()] == [0, 0, 0]


def test_resolving_probes_marks_the_calibrator_for_refit_and_it_learns(bot):
    bot.client = SpotClient(exit_price=150.0)          # everything lands outside
    for _ in range(8):
        _insert_due_probes(bot, 99.0, 101.0)
    _run_resolver(bot)
    assert bot.calibration.needs_refit()
    before = bot.calibration.calibrate(0.9)
    bot.calibration.refit()
    assert bot.calibration.calibrate(0.9) < before - 0.05      # 24 straight misses pull the 90% estimate down


def test_unreachable_old_probes_are_abandoned_not_retried_forever(bot):
    bot.client = SpotClient(exit_price=100.0)
    bot.storage.record_probes([{"symbol": "1HZ90V", "regime": "NORMAL_RANGE", "raw_probability": 0.8,
                                "lower_barrier": 99.0, "upper_barrier": 101.0,
                                "opened_at": time.time() - 4000, "expires_at": time.time() - 3900}])
    _run_resolver(bot)
    assert bot.storage.probe_counts() == {"open": 0, "resolved": 0, "abandoned": 1}
    assert bot.client.spot_calls == []


# ------------------------------------------------- signal mean through the engine
def test_every_candidate_of_a_signal_carries_the_same_signal_mean():
    from app.models.calibration import _logit
    cfg = _cfg(use_edge_gate=False, use_regime_gate=False, barrier_scales=[1.0, 1.5, 2.0, 2.5])
    out, _ = _run(cfg, WidthPricedClient(), _closes())
    means = {round(p["signal_logit_mean"], 9) for p in out.probes} | {round(out.trade_row["signal_logit_mean"], 9)}
    assert len(means) == 1
    logits = [_logit(p["raw_probability"]) for p in out.probes] + [_logit(out.trade_row["raw_probability"])]
    assert means.pop() == pytest.approx(sum(logits) / len(logits), abs=1e-6)
