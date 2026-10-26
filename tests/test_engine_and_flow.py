"""Decision pipeline + signal-driven Bot flow against a fake Deriv client (no network)."""
import asyncio
import os
import tempfile
import time

import numpy as np
import pytest

from app.config import Config
from app.models.calibration import CalibrationTracker
from app.strategy.engine import evaluate_signal
from app.strategy.staking import build_staking_engine


class _Log:
    def __init__(self): self.lines = []
    def info(self, m, *a, **k): self.lines.append(m)
    warning = error = debug = info


class FakeClient:
    def __init__(self, payout=1.30, price=28000.0):
        self.last_message_at = time.time()
        self.proposals = []
        self.payout_mult = payout
        self.price = price
        self.ticks_cb = {}
        self.candles = []

    async def request_proposal(self, symbol, duration_minutes, stake, lower_barrier, upper_barrier, currency="USD"):
        self.proposals.append((symbol, duration_minutes, stake, lower_barrier, upper_barrier))
        return {"id": f"P{len(self.proposals)}", "payout": round(stake * self.payout_mult, 2), "spot": self.price}

    async def get_candle_history(self, symbol, granularity_seconds, count):
        return self.candles

    async def get_tick_history(self, symbol, count):
        return [self.price + (i % 3) * 0.1 for i in range(count)]

    async def subscribe_ticks(self, symbol, callback):
        self.ticks_cb[symbol] = callback
        return f"ticks:{symbol}"

    async def get_spot_at(self, symbol, epoch):
        return None

    async def portfolio(self):
        return []


def _closes(n=300, price=28000.0, sigma=0.00104, seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    return price * np.exp(np.cumsum(rng.normal(drift, sigma, n)))


def _cfg(**kw):
    c = Config()
    c.symbols = ["1HZ90V"]
    c.monte_carlo.default_paths = 4000
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _storage():
    from app.data.storage import Storage
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    s = Storage("", path, _Log()); s.init_schema()
    return s


def _run(cfg, client, closes, losses=0, storage=None):
    st = storage or _storage()
    out = asyncio.run(evaluate_signal(
        "1HZ90V", "DOWN", closes, float(closes[-1]), client, CalibrationTracker(st),
        build_staking_engine(cfg.staking), cfg, _Log(), consecutive_losses=losses))
    return out


def test_edge_gate_off_buys_at_the_xml_barriers_and_follows_the_ladder():
    cfg = _cfg(use_edge_gate=False, use_regime_gate=False, barrier_scales=[1.0])
    cfg.staking.maximum_stake = 1000.0
    client = FakeClient()
    out = _run(cfg, client, _closes(), losses=2)
    assert out.traded and out.trade_row["stake"] == 6.25          # 1 -> 2.5 -> 6.25
    sym, dur, stake, lo, up = client.proposals[0]
    assert (dur, lo, up) == (2, -41.31, 41.46)                    # XML offsets, 2dp
    row = out.trade_row
    assert row["lower_barrier"] == pytest.approx(28000.0 - 41.31)
    assert row["model_version"] == "qe-v1"


def test_edge_gate_rejects_when_the_model_sees_no_edge():
    cfg = _cfg(use_regime_gate=False, barrier_scales=[1.0], min_edge=0.05)
    out = _run(cfg, FakeClient(payout=1.05), _closes())          # payout too thin for any edge
    assert not out.traded and out.rejections
    assert all(r["rejection_reason"] for r in out.rejections)


def test_regime_gate_blocks_a_trending_market_without_spending_a_proposal():
    cfg = _cfg(use_edge_gate=False, barrier_scales=[1.0])
    client = FakeClient()
    trending = _closes(drift=0.004, sigma=0.0005)                 # strong drift => TRENDING_UP
    out = _run(cfg, client, trending)
    assert not out.traded and client.proposals == []
    assert "NOT RANGE-FRIENDLY" in out.rejections[0]["rejection_reason"]


def test_warmup_and_scaled_widths_are_all_priced():
    cfg = _cfg(use_edge_gate=False, use_regime_gate=False)
    assert not _run(cfg, FakeClient(), _closes(n=40)).traded
    client = FakeClient()
    out = _run(cfg, client, _closes())
    assert len(client.proposals) == len(cfg.barrier_scales)
    assert out.traded


# ------------------------------------------------------------------ Bot flow
def _candles(last_closed_black=True):
    now = time.time()
    forming = int(now // 60) * 60
    closes = _closes(300)
    cs = []
    for i, c in enumerate(closes):
        cs.append({"epoch": forming - (len(closes) - 1 - i) * 60, "open": float(c), "high": float(c) + 1,
                   "low": float(c) - 1, "close": float(c)})
    closed = cs[-2]
    closed["open"], closed["close"] = (closed["close"] + 3, closed["close"]) if last_closed_black else (closed["close"], closed["close"] + 3)
    return cs


@pytest.fixture
def bot(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    monkeypatch.setenv("SQLITE_PATH", path)
    monkeypatch.setenv("USE_EDGE_GATE", "false")
    monkeypatch.setenv("USE_REGIME_GATE", "false")
    monkeypatch.setenv("BARRIER_SCALES", "1.0")
    monkeypatch.setenv("MC_PATHS", "3000")
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import run
    b = run.Bot()
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


def test_signal_with_black_candle_records_a_shadow_trade_and_occupies_the_slot(bot):
    _signal(bot, black=True)
    rows = bot.storage.recent_trades()
    assert len(rows) == 1 and rows[0]["shadow"] == 1 and rows[0]["stake"] == 1.0
    assert bot._open_count() == 1


def test_signal_with_white_candle_is_skipped(bot):
    _signal(bot, black=False)
    assert bot.storage.recent_trades() == []


def test_no_second_trade_while_one_is_open(bot):
    _signal(bot, black=True)
    asyncio.run(bot._handle_signal("1HZ90V", "DOWN"))
    assert len(bot.storage.recent_trades()) == 1


def test_stop_loss_pauses_trading_and_ladder_follows_journal(bot):
    for i in range(2):  # two settled shadow losses -> streak 2, day P&L -2 -> still trading
        bot.storage.record_trade({"trade_id": f"l{i}", "symbol": "1HZ90V", "stake": 1.0, "payout": 1.3,
                                  "opened_at": time.time() - 500 + i, "expires_at": time.time() - 300, "shadow": 1})
        bot.storage.settle_trade(f"l{i}", "LOSS", -1.0)
    bot.cfg.stop_loss = 2.0
    bot._session_cache = None
    assert bot._session()[0] == "STOP"
    _signal(bot, black=True)
    assert len(bot.storage.recent_trades()) == 2          # no new trade while halted
    bot.cfg.stop_loss = 200.0
    bot._session_cache = None
    assert bot._session()[0] == "OK"
    asyncio.run(bot._handle_signal("1HZ90V", "DOWN"))
    t = bot.storage.recent_trades()[0]
    assert t["stake"] == 6.25                              # streak of 2 -> 1 -> 2.5 -> 6.25


def test_tick_callback_only_spawns_on_a_confirmed_cross(bot):
    async def go():
        bot.client.candles = _candles(True)
        await bot._prime_candles(); await bot._prime_ticks()
        spawned = []
        bot._spawn = lambda coro, name: (spawned.append(name), coro.close())
        det = bot.detectors["1HZ90V"]
        det.ema_fast, det.ema_slow, det.last_price = 100.02, 100.0, 100.5   # fast just above slow
        bot._on_tick("1HZ90V", {"tick": {"quote": 90.0}})                    # big drop -> cross down
        bot._on_tick("1HZ90V", {"tick": {"quote": 89.0}})                    # no new cross
        return spawned
    assert asyncio.run(go()) == ["signal-1HZ90V"]
