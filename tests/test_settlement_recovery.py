"""A settlement the stream missed (socket dropped at expiry) must never leave
a position slot occupied forever."""
import asyncio
import os
import tempfile
import time

import pytest


class _Log:
    def info(self, *a, **k): pass
    warning = error = debug = info


def test_reconnect_delivers_an_already_sold_snapshot_to_the_contract_callback():
    from app.deriv.client import DerivClient, Subscription

    got = []
    c = DerivClient(endpoint="wss://x", app_id="1", api_token="t", logger=_Log())

    async def noop(*a, **k): return None
    async def fake_send(payload, timeout=None):
        return {"proposal_open_contract": {"contract_id": 9, "is_sold": 1, "profit": 0.13, "status": "won"}}

    c._open_socket = noop
    c._recv_pump = noop
    c.send = fake_send
    c._subscriptions["contract:9"] = Subscription(
        request={"proposal_open_contract": 1, "contract_id": 9, "subscribe": 1},
        callback=lambda msg: got.append(msg), subscription_id="old")
    c._subscriptions["ticks:R_10"] = Subscription(
        request={"ticks": "R_10", "subscribe": 1}, callback=lambda msg: got.append("TICK"), subscription_id="t")

    async def go():
        await c._reconnect()
        await asyncio.sleep(0)
    asyncio.run(go())
    assert len(got) == 1 and got[0]["proposal_open_contract"]["is_sold"] == 1   # contract only, not the tick sub


class FakeClient:
    def __init__(self, poc):
        self.poc = poc
        self.queried = []
        self.last_message_at = time.time()
        self.forgotten = []

    async def get_contract(self, cid):
        self.queried.append(cid)
        return self.poc

    async def forget_subscription(self, key):
        self.forgotten.append(key)


@pytest.fixture
def bot(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    monkeypatch.setenv("SQLITE_PATH", path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import run
    monkeypatch.setattr(run, "LIVE_WATCHDOG_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(run, "LIVE_WATCHDOG_GRACE_SECONDS", 0)
    b = run.Bot()
    b.storage.init_schema()
    return b


def _live_trade(bot, tid, expires_in):
    now = time.time()
    bot.storage.record_trade({"trade_id": tid, "symbol": "R_10", "stake": 0.35, "payout": 0.48,
                              "raw_probability": 0.79, "calibrated_probability": 0.78,
                              "implied_probability": 0.73, "lower_barrier": 1.0, "upper_barrier": 2.0,
                              "duration_minutes": 4, "opened_at": now - 300, "expires_at": now + expires_in,
                              "shadow": 0, "contract_id": "C9"})
    bot.open_positions[tid] = "R_10"


def _run_watchdog(bot, seconds=0.4):
    async def go():
        task = asyncio.create_task(bot._live_settlement_watchdog())
        await asyncio.sleep(seconds)
        bot._stop.set()
        await task
    asyncio.run(go())


def test_watchdog_settles_a_missed_win_and_frees_the_slot(bot):
    bot.client = FakeClient({"is_sold": 1, "profit": 0.13, "status": "won", "exit_tick": 4935.8})
    _live_trade(bot, "t1", expires_in=-60)
    _run_watchdog(bot)
    t = bot.storage.recent_trades()[0]
    assert t["result"] == "WIN" and t["profit_loss"] == pytest.approx(0.13)
    assert bot._open_count() == 0 and "contract:C9" in bot.client.forgotten
    assert bot.client.queried.count("C9") == 1            # settled -> not queried again


def test_watchdog_settles_a_missed_loss(bot):
    bot.client = FakeClient({"is_sold": 1, "profit": -0.35, "status": "lost"})
    _live_trade(bot, "t2", expires_in=-60)
    _run_watchdog(bot)
    assert bot.storage.recent_trades()[0]["result"] == "LOSS" and bot._open_count() == 0


def test_watchdog_leaves_unexpired_and_unsold_contracts_alone(bot):
    bot.client = FakeClient({"is_sold": 0})
    _live_trade(bot, "t3", expires_in=+600)               # not expired yet -> never queried
    _run_watchdog(bot, 0.2)
    assert bot.client.queried == [] and bot._open_count() == 1


def test_watchdog_keeps_retrying_an_expired_contract_deriv_reports_still_open(bot):
    bot.client = FakeClient({"is_sold": 0})
    _live_trade(bot, "t4", expires_in=-60)
    _run_watchdog(bot, 0.3)
    assert len(bot.client.queried) >= 2 and bot._open_count() == 1
    assert bot.storage.recent_trades()[0]["result"] == "OPEN"
