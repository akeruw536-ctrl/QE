"""The keepalive must keep the socket busy: Deriv closes a connection that
sends no API request for ~2 minutes, even while ticks are streaming in."""
import asyncio
from types import SimpleNamespace

import app.deriv.client as dc
from app.deriv.client import DerivClient


class _Log:
    def info(self, *a, **k): pass
    warning = error = debug = info


def _client():
    return DerivClient(endpoint="wss://x", app_id="1", api_token="t", logger=_Log())


def test_keepalive_sends_application_pings_and_counts_them(monkeypatch):
    monkeypatch.setattr(dc, "KEEPALIVE_INTERVAL_SECONDS", 0.01)
    c = _client()
    sent = []

    async def fake_send(payload, timeout=None):
        sent.append(payload)
        return {"ping": "pong"}

    c.send = fake_send
    c._ws = SimpleNamespace(state=dc.WsState.OPEN)

    async def go():
        task = asyncio.create_task(c._keepalive())
        await asyncio.sleep(0.2)
        c._closed = True
        await task
    asyncio.run(go())
    assert len(sent) >= 3 and all(p == {"ping": 1} for p in sent)
    assert c.ping_count == len(sent) and c.last_ping_ok > 0


def test_keepalive_survives_a_failed_ping_and_skips_a_closed_socket(monkeypatch):
    monkeypatch.setattr(dc, "KEEPALIVE_INTERVAL_SECONDS", 0.01)
    c = _client()
    calls = {"n": 0}

    async def flaky(payload, timeout=None):
        calls["n"] += 1
        raise RuntimeError("boom")

    c.send = flaky
    c._ws = SimpleNamespace(state=dc.WsState.OPEN)

    async def go():
        task = asyncio.create_task(c._keepalive())
        await asyncio.sleep(0.1)
        c._ws = SimpleNamespace(state=dc.WsState.CLOSED)   # closed: no further pings attempted
        n_before = calls["n"]
        await asyncio.sleep(0.1)
        c._closed = True
        await task
        return n_before
    n_before = asyncio.run(go())
    assert n_before >= 2 and calls["n"] == n_before and c.ping_count == 0
