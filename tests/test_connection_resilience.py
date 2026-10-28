"""A connection that opens but never answers must be detected and replaced, and a dead
data feed must never crash the bot."""
import asyncio
import json
import os
import tempfile
import time

import pytest

import app.deriv.client as dc
import run as run_mod
from app.deriv.client import DerivClient


class _Log:
    def __init__(self): self.lines = []
    def info(self, m, *a, **k): self.lines.append(("info", m))
    def warning(self, m, *a, **k): self.lines.append(("warning", m))
    def error(self, m, *a, **k): self.lines.append(("error", m))
    debug = info


class FakeWS:
    """Minimal websocket: replies to every request when `answers`, otherwise stays silent."""
    def __init__(self, answers):
        self.answers = answers
        self.state = dc.WsState.OPEN
        self.sent = []
        self._q = asyncio.Queue()

    async def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        if self.answers:
            await self._q.put(json.dumps({"req_id": msg["req_id"], "msg_type": "ping", "ping": "pong"}))

    async def close(self):
        self.state = dc.WsState.CLOSED
        await self._q.put(None)

    def __aiter__(self): return self

    async def __anext__(self):
        item = await self._q.get()
        if item is None:
            raise StopAsyncIteration
        return item


def _client(sockets, monkeypatch):
    monkeypatch.setattr(dc, "LIVENESS_TIMEOUT_SECONDS", 0.15)
    log = _Log()
    c = DerivClient(endpoint="wss://x", app_id="1", api_token="t", logger=log)
    opened = []

    async def fake_open():
        ws = sockets[len(opened)]
        opened.append(ws)
        c._ws = ws
        c.last_message_at = time.time()

    c._open_socket = fake_open
    return c, opened, log


def _connect_and_stop(c):
    async def go():
        await c.connect()
        await asyncio.sleep(0.05)
        await c.close()
    asyncio.run(go())


def test_a_silent_socket_is_replaced_with_a_fresh_handshake(monkeypatch):
    silent, good = FakeWS(False), FakeWS(True)
    c, opened, log = _client([silent, good], monkeypatch)
    _connect_and_stop(c)
    assert opened == [silent, good]                       # re-handshaked once
    assert silent.state is dc.WsState.CLOSED
    assert any(k == "warning" and "not answering" in m for k, m in log.lines)
    assert any(m.startswith("Deriv connected") for _, m in log.lines)


def test_a_healthy_socket_is_left_alone(monkeypatch):
    good = FakeWS(True)
    c, opened, _ = _client([good], monkeypatch)
    _connect_and_stop(c)
    assert opened == [good] and good.sent[0]["ping"] == 1


def test_gives_up_quietly_after_the_attempts_and_never_raises(monkeypatch):
    socks = [FakeWS(False) for _ in range(dc.LIVENESS_ATTEMPTS)]
    c, opened, log = _client(socks, monkeypatch)
    _connect_and_stop(c)                                  # must not raise
    assert len(opened) == dc.LIVENESS_ATTEMPTS
    assert any(k == "error" and "still unresponsive" in m for k, m in log.lines)


def test_force_reconnect_drops_the_socket(monkeypatch):
    good = FakeWS(True)
    c, _, log = _client([good], monkeypatch)
    async def go():
        await c.connect()
        await c.force_reconnect("no ticks for 95s")
        await asyncio.sleep(0.05)
        await c.close()
    asyncio.run(go())
    assert good.state is dc.WsState.CLOSED
    assert any("forcing a reconnect" in m for _, m in log.lines)


# ------------------------------------------------------------------ bot
class FlakyClient:
    def __init__(self, fail_subscribes=2):
        self.fail = fail_subscribes
        self.subscribed = []
        self.forced = []
        self.last_message_at = time.time()
        self.candles = []

    async def get_tick_history(self, symbol, count):
        return [100 + (i % 3) * 0.1 for i in range(count)]

    async def subscribe_ticks(self, symbol, callback):
        if self.fail > 0:
            self.fail -= 1
            raise RuntimeError("Timeout: no response within 15.0s")
        self.subscribed.append(symbol)

    async def get_candle_history(self, symbol, g, count):
        return self.candles

    async def force_reconnect(self, reason=""):
        self.forced.append(reason)


@pytest.fixture
def bot(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    monkeypatch.setenv("SQLITE_PATH", path)
    monkeypatch.setenv("SYMBOLS", "1HZ10V,RDBEAR")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(run_mod, "DATA_RECOVERY_INTERVAL_SECONDS", 0.05)
    b = run_mod.Bot()
    b.storage.init_schema()
    b.client = FlakyClient()
    return b


def _run_loop(bot, seconds=0.5):
    async def go():
        task = asyncio.create_task(bot._data_recovery_loop())
        await asyncio.sleep(seconds)
        bot._stop.set()
        await task
    asyncio.run(go())


def test_a_failed_tick_subscription_does_not_crash_startup_and_is_retried(bot):
    asyncio.run(bot._prime_ticks())                      # both subscriptions fail -> no exception
    assert bot._tick_subscribed == set()
    _run_loop(bot)                                       # the recovery loop gets them subscribed
    assert bot._tick_subscribed == {"1HZ10V", "RDBEAR"}
    assert bot.detectors["1HZ10V"].ready                 # EMA state seeded from history


def test_empty_candle_history_is_retried_until_it_arrives(bot):
    bot.client.fail = 0
    asyncio.run(bot._prime_ticks())
    now = time.time()
    bot.client.candles = [{"epoch": int(now // 60) * 60 - 60 * (300 - i), "open": 100.0, "high": 101.0,
                           "low": 99.0, "close": 100.5} for i in range(300)]
    _run_loop(bot, 0.3)
    assert len(bot.closes_by_symbol["1HZ10V"]) == 300 and len(bot.closes_by_symbol["RDBEAR"]) == 300


def test_silent_feed_forces_one_reconnect_then_respects_the_cooldown(bot):
    bot.client.fail = 0
    asyncio.run(bot._prime_ticks())
    bot.client.candles = [{"epoch": 1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}]
    for s in list(bot._last_tick_at):
        bot._last_tick_at[s] = time.time() - 200          # no ticks for 200s
    _run_loop(bot, 0.4)
    assert len(bot.client.forced) == 1 and "no ticks" in bot.client.forced[0]


def test_a_live_feed_is_never_reconnected(bot):
    bot.client.fail = 0
    asyncio.run(bot._prime_ticks())
    bot.client.candles = [{"epoch": 1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}]
    async def go():
        task = asyncio.create_task(bot._data_recovery_loop())
        for _ in range(8):
            bot._on_tick("1HZ10V", {"tick": {"quote": 100.1}})
            await asyncio.sleep(0.05)
        bot._stop.set()
        await task
    asyncio.run(go())
    assert bot.client.forced == []
