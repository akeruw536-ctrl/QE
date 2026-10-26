"""End-to-end checks of settlement/restart/shadow logic in run.py against a
fake Deriv client -- no network."""
import asyncio
import os
import tempfile
import time

import pytest


class FakeClient:
    def __init__(self):
        self.last_message_at = time.time()
        self.subscribed = {}
        self.forgotten = []
        self.spots = {}
        self.portfolio_contracts = []

    async def subscribe_contract(self, contract_id, callback):
        self.subscribed[contract_id] = callback
        return f"contract:{contract_id}"

    async def forget_subscription(self, key):
        self.forgotten.append(key)

    async def get_spot_at(self, symbol, epoch):
        return self.spots.get(symbol)

    async def portfolio(self):
        return self.portfolio_contracts


@pytest.fixture
def bot(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("SQLITE_PATH", path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import run
    b = run.Bot()
    b.storage.init_schema()
    b.client = FakeClient()
    return b


def _row(trade_id, **kw):
    now = time.time()
    r = {"trade_id": trade_id, "symbol": "RDBULL", "stake": 0.35, "payout": 0.55,
         "raw_probability": 0.62, "calibrated_probability": 0.80, "implied_probability": 0.64,
         "lower_barrier": 99.0, "upper_barrier": 101.0, "duration_minutes": 2,
         "opened_at": now - 200, "expires_at": now - 60, "shadow": 1}
    r.update(kw)
    return r


def test_live_settlement_counts_once_and_calibrates_by_raw_probability(bot):
    async def go():
        bot.storage.record_trade(_row("t1", shadow=0, contract_id="C1"))
        bot.open_positions["t1"] = "RDBULL"
        await bot._track_contract("t1", "RDBULL", 0.62, "C1")
        cb = bot.client.subscribed["C1"]
        msg = {"proposal_open_contract": {"is_sold": 1, "profit": 0.2, "status": "won", "exit_tick": 100.1}}
        cb(msg)
        cb(msg)  # duplicate final update
        await asyncio.sleep(0)
    asyncio.run(go())
    buckets = bot.storage.get_calibration()
    # filed under RAW 0.62's bucket (0.60-0.65), exactly once -- not under calibrated 0.80
    assert buckets == {"0.60-0.65": (1, 1)}
    assert bot.open_positions == {}
    assert "contract:C1" in bot.client.forgotten


def test_restart_restores_open_positions(bot):
    bot.storage.record_trade(_row("live", shadow=0, contract_id="C9", expires_at=time.time() + 60))
    bot.storage.record_trade(_row("sh", shadow=1, expires_at=time.time() + 60))
    asyncio.run(bot._restore_open_positions())
    assert set(bot.open_positions) == {"live", "sh"}
    assert "C9" in bot.client.subscribed


def test_shadow_resolver_settles_against_exit_spot(bot):
    bot.storage.record_trade(_row("win"))
    bot.storage.record_trade(_row("loss", symbol="RDBEAR"))
    bot.open_positions.update({"win": "RDBULL", "loss": "RDBEAR"})
    bot.client.spots = {"RDBULL": (100.5, 0), "RDBEAR": (101.5, 0)}

    async def go():
        task = asyncio.create_task(bot._shadow_resolver_loop())
        await asyncio.sleep(0.2)
        bot._stop.set()
        await task
    asyncio.run(go())
    by_id = {t["trade_id"]: t for t in bot.storage.recent_trades()}
    assert by_id["win"]["result"] == "WIN" and by_id["win"]["profit_loss"] == pytest.approx(0.20)
    assert by_id["loss"]["result"] == "LOSS" and by_id["loss"]["profit_loss"] == pytest.approx(-0.35)
    assert by_id["win"]["exit_spot"] == 100.5
    assert bot.open_positions == {}
    assert bot.stats()["shadow"]["settled"] == 2


def test_ambiguous_buy_finds_contract_in_portfolio(bot):
    since = time.time() - 5
    bot.client.portfolio_contracts = [
        {"contract_id": 7, "symbol": "RDBULL", "contract_type": "EXPIRYRANGE",
         "purchase_time": time.time(), "buy_price": 0.35},
        {"contract_id": 8, "symbol": "RDBEAR", "contract_type": "EXPIRYRANGE", "purchase_time": time.time()},
    ]
    found = asyncio.run(bot._reconcile_ambiguous_buy("RDBULL", since))
    assert found["contract_id"] == "7"
    bot.client.portfolio_contracts = []
    assert asyncio.run(bot._reconcile_ambiguous_buy("RDBULL", since)) is None


def test_ambiguous_buy_ignores_another_bots_contract(bot):
    """Two bots on one Deriv account: same symbol, same moment, different
    order -- the other bot's contract must not be adopted."""
    since = time.time() - 5
    now = time.time()
    bot.client.portfolio_contracts = [
        {"contract_id": 21, "symbol": "RDBULL", "contract_type": "EXPIRYRANGE",
         "purchase_time": now, "buy_price": 1.00, "payout": 1.30},           # the other bot's
        {"contract_id": 22, "symbol": "RDBULL", "contract_type": "EXPIRYRANGE",
         "purchase_time": now, "buy_price": 0.35, "payout": 0.49},           # ours
    ]
    found = asyncio.run(bot._reconcile_ambiguous_buy("RDBULL", since, stake=0.35, payout=0.49))
    assert found["contract_id"] == "22"
    bot.client.portfolio_contracts = bot.client.portfolio_contracts[:1]
    assert asyncio.run(bot._reconcile_ambiguous_buy("RDBULL", since, stake=0.35, payout=0.49)) is None


def test_health_goes_stale(bot):
    assert bot.is_healthy()[0]
    bot.client.last_message_at = time.time() - 600
    assert not bot.is_healthy()[0]
