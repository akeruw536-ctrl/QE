"""
Deriv WebSocket connection layer -- v2.

=====================================================================
WHY THIS CHANGED FROM v1, AND WHAT WAS AND WASN'T PORTED FROM THE
EXTERNAL REFERENCE FILE THIS WAS BASED ON (a "Reversal-System" repo's
deriv/client.py). Both points below were independently verified
against Deriv's own current documentation (developers.deriv.com) before
being adopted -- nothing here was ported on the reference file's word
alone.
=====================================================================

1. DERIV HAS A NEW API GENERATION. Confirmed live against
   developers.deriv.com/llms/*.md: the "New API" (product name
   "Options") uses a REST OTP exchange for WebSocket auth instead of
   the legacy post-connect `{"authorize": token}` message, connects to
   `wss://api.derivws.com/trading/v1/options/ws/{demo|real|public}`
   instead of `wss://ws.derivws.com/websockets/v3`, and renames the
   `proposal` request's `symbol` field to `underlying_symbol`.
   EXPIRYRANGE (our contract type) is confirmed present in the current
   `contract_type` enum. `ticks`/`ticks_history` are NOT affected --
   those calls take the symbol as the *value* of the `ticks`/
   `ticks_history` key itself, not a separately-named field, so no
   rename applies there.

   NOT ported: the reference file's claim of a special 360/minute
   budget shared across proposal/proposal_open_contract/buy/sell.
   Deriv's current docs (api-overview.md) state the limit is 100
   requests/second per connection, generically -- no narrower
   per-message-family budget is documented. `_RateLimiter` below paces
   against the documented number, not the unverified one.

2. A REAL DEADLOCK BUG IN OUR OWN v1 CLIENT, found by cross-checking
   the reference file's rules 1-3 against our code rather than by
   independent testing. v1's `_read_loop()` was the only coroutine
   that read the socket, but on `ConnectionClosed` it called
   `_handle_disconnect()` -- INLINE, in the same coroutine -- which
   then tried to resubscribe via `send()`, which awaits a future that
   only the read loop can resolve. Since the read loop was itself
   blocked inside `_handle_disconnect` at that point, every resubscribe
   after every reconnect would have hung for the full request timeout.
   This was never hit in your logs yet only because the bot never
   reached a successful connect to reconnect FROM (see the 520 issue).
   It would have surfaced the first time a real disconnect happened
   after a real connect.

   Fixed the way the reference file fixes it: the socket reader
   (`_recv_pump`) is its own task, never doing the reconnecting itself.
   A single long-lived `_supervise_pump` task watches it and reconnects
   whenever it exits, for any reason. `_reconnect()` starts the NEW
   pump task on the new socket BEFORE resubscribing anything -- so the
   resubscribe requests always have a live reader able to resolve their
   futures.

Everything else (idempotent single-attempt buy, subscribed rather than
polled settlement, `.state` instead of the removed `.closed` property)
carries the same public method signatures as v1
(`connect`, `close`, `request_proposal`, `buy_contract`,
`subscribe_candles`, `subscribe_contract`, `get_candle_history`,
`balance`) so `run.py` and `app/strategy/engine.py` did not need to
change.

REQUIRES: `httpx` for the REST OTP exchange (`pip install httpx`, or
add it to requirements.txt) when using the default OTP auth mode.
`auth_mode="legacy"` keeps the old connect-then-authorize flow against
`wss://ws.derivws.com/websockets/v3` and needs no extra dependency --
useful as a fallback if OTP auth is ever unavailable for your account.

NEW OPTIONAL ENV VARS (all have safe defaults, nothing else needs to
change to pick them up since DerivClient reads them itself when not
passed explicitly):
  DERIV_API_BASE_URL   default "https://api.derivws.com"
  DERIV_AUTH_MODE      default "otp"  (or "legacy")
  DERIV_ACCOUNT_ID     default unset -- auto-resolved (demo/real) via
                        GET /trading/v1/options/accounts on first
                        connect if not set
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State as WsState

AUTH_OTP = "otp"
AUTH_LEGACY = "legacy"


KEEPALIVE_INTERVAL_SECONDS = 30.0   # Deriv closes a socket with no API request for ~2 min
LIVENESS_ATTEMPTS = 3               # fresh handshakes tried when a new socket never answers
LIVENESS_TIMEOUT_SECONDS = 8.0


class DerivAPIError(Exception):
    def __init__(self, code: str, message: str, raw: dict | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.raw = raw or {}


class DerivAuthError(DerivAPIError):
    def __init__(self, message: str):
        super().__init__("AuthFailed", message)


class BuyAmbiguousError(DerivAPIError):
    """Raised when a buy's outcome could not be confirmed (timeout or
    disconnect mid-request). The buy may or may not have gone through.
    Callers MUST reconcile via `portfolio()` rather than retry --
    retrying a buy whose first attempt actually succeeded opens a
    second contract."""


@dataclass
class Subscription:
    request: dict
    callback: "callable"
    subscription_id: str | None = None


class _RateLimiter:
    """Sliding-window limiter paced to Deriv's documented 100 req/sec
    per connection (api-overview.md). Waits before sending rather than
    firing and handling a 429 after the fact -- a rejected request
    still cost a round trip and still leaves the caller with nothing."""

    def __init__(self, max_per_window: int = 80, window_seconds: float = 1.0):
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                self._timestamps = [t for t in self._timestamps if t > cutoff]
                if len(self._timestamps) < self.max_per_window:
                    self._timestamps.append(now)
                    return
                await asyncio.sleep(max(self._timestamps[0] - cutoff, 0.01))


class DerivClient:
    def __init__(self, endpoint: str, app_id: str, api_token: str, logger,
                 on_reconnect=None, *,
                 api_base_url: str | None = None,
                 auth_mode: str | None = None,
                 use_real_account: bool | None = None,
                 account_id: str | None = None,
                 request_timeout: float = 15.0,
                 max_requests_per_second: int = 80):
        self.legacy_endpoint = endpoint  # only used when auth_mode == "legacy"
        self.app_id = app_id
        self.api_token = api_token
        self.logger = logger
        self.on_reconnect = on_reconnect

        self.api_base_url = (api_base_url or os.getenv("DERIV_API_BASE_URL", "https://api.derivws.com")).rstrip("/")
        self.auth_mode = auth_mode or os.getenv("DERIV_AUTH_MODE", AUTH_OTP)
        if use_real_account is None:
            use_real_account = os.getenv("DERIV_ACCOUNT_MODE", "DEMO").upper() == "LIVE"
        self.use_real_account = use_real_account
        self.account_id = account_id or os.getenv("DERIV_ACCOUNT_ID") or None
        self.request_timeout = request_timeout

        self._ws = None
        self._req_id_counter = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._subscriptions: dict[str, Subscription] = {}
        self._contract_queues: dict[str, asyncio.Queue] = {}

        self._pump_task: asyncio.Task | None = None
        self._supervisor_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self.ping_count = 0
        self.last_ping_ok: float = 0.0
        self.reconnect_count = 0
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self.last_message_at: float = 0.0
        self.rate_limiter = _RateLimiter(max_per_window=max_requests_per_second)

    # ------------------------------------------------------------- lifecycle
    async def connect(self) -> None:
        self._closed = False
        async with self._connect_lock:
            await self._open_socket()
            # Rule 3: reader task exists before anything else can need it.
            self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-pump")
            await self._verify_alive()
        # Rule 2: exactly one long-lived supervisor for the client's whole life.
        self._supervisor_task = asyncio.create_task(self._supervise_pump(), name="deriv-pump-supervisor")
        self._keepalive_task = asyncio.create_task(self._keepalive(), name="deriv-keepalive")
        self.logger.info(f"Deriv connected (auth_mode={self.auth_mode})")

    async def _verify_alive(self) -> None:
        """A socket can open (OTP accepted, handshake done) and then never answer
        a single request -- seen in production: every candle and tick request
        timed out after 'Deriv connected', then the bot crashed. Prove the new
        socket responds with a ping before anyone relies on it; if it does not,
        drop it and handshake again with a fresh OTP. Called with the connect
        lock held and the pump running. Never raises: if every attempt fails it
        logs and carries on, and the callers' own retry logic takes over."""
        for attempt in range(1, LIVENESS_ATTEMPTS + 1):
            try:
                await self._send_raw({"ping": 1}, timeout=LIVENESS_TIMEOUT_SECONDS)
                return
            except DerivAPIError as exc:
                if exc.code != "Disconnected":
                    return                       # it answered (even with an error) -> alive
            except asyncio.TimeoutError:
                pass
            self.logger.warning(
                f"Deriv socket opened but is not answering (attempt {attempt}/{LIVENESS_ATTEMPTS})"
                + (" -- re-handshaking with a fresh OTP" if attempt < LIVENESS_ATTEMPTS else ""))
            if attempt == LIVENESS_ATTEMPTS:
                break
            await self._replace_socket()
        self.logger.error("Deriv socket still unresponsive after fresh handshakes; continuing -- "
                          "requests will retry and the bot's recovery loop keeps trying")

    async def _replace_socket(self) -> None:
        """Closes the current socket and opens a new one with its own pump.
        The old pump is left to end on its own (cancelling it would be read by
        the supervisor as a shutdown)."""
        old_ws, old_pump = self._ws, self._pump_task
        try:
            if old_ws is not None:
                await old_ws.close()
        except Exception:  # noqa: BLE001
            pass
        if old_pump is not None and not old_pump.done():
            await asyncio.wait({old_pump}, timeout=2.0)
        await self._open_socket()
        self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-pump")

    async def force_reconnect(self, reason: str = "") -> None:
        """Drops the socket; the supervisor sees the pump end and reconnects
        (fresh OTP, liveness check, resubscribe everything)."""
        self.logger.warning(f"Deriv: forcing a reconnect ({reason})")
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:  # noqa: BLE001
            pass

    async def _keepalive(self) -> None:
        """Deriv drops a socket that carries no API requests for ~2 minutes
        (WebSocket-level pings don't count). With nothing to scan or settle the
        bot went quiet and reconnected every ~2 minutes for hours. A tiny
        application ping every 30s keeps the connection up."""
        while not self._closed:
            await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
            if self._closed:
                return
            try:
                if self._ws is not None and self._ws.state is WsState.OPEN:
                    await self.send({"ping": 1}, timeout=10.0)
                    self.ping_count += 1
                    self.last_ping_ok = time.time()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.logger.debug(f"Deriv keepalive ping failed (non-fatal): {exc!r}")

    async def close(self) -> None:
        self._closed = True
        for t in (self._pump_task, self._supervisor_task, self._keepalive_task):
            if t:
                t.cancel()
        if self._ws is not None:
            await self._ws.close()
        self.logger.info("Deriv WebSocket closed cleanly")

    async def _open_socket(self) -> None:
        """Opens a new socket and assigns self._ws. Does not start the
        pump or resubscribe -- callers own that ordering (rule 3).

        EVERYTHING in this method -- account resolution, the OTP REST
        exchange, and the socket connect itself -- runs inside ONE
        retry-with-backoff loop. A prior version only wrapped the
        websockets.connect() call, so a failure in account resolution or
        the OTP exchange (missing httpx, a bad token, a transient network
        error -- anything) propagated straight out of connect() and killed
        the whole process instead of retrying. That is exactly the
        "never crash the process" principle the rest of this bot follows
        (see run.py's docstrings) -- this method now actually honors it.
        """
        attempt = 0
        backoff = 1.0
        while not self._closed:
            try:
                if self.auth_mode == AUTH_OTP:
                    if not self.account_id:
                        self.account_id = await self._resolve_account_id()
                        self.logger.info(
                            f"Resolved Deriv account {self.account_id} "
                            f"(wanted {'real' if self.use_real_account else 'demo'})"
                        )
                    # Single-use, valid 120s -- minted immediately before every
                    # socket open, including every reconnect. Caching it across
                    # reconnects would fail exactly when it matters, mid-outage.
                    url = await self._exchange_otp(self.account_id)
                else:
                    url = f"{self.legacy_endpoint}?app_id={self.app_id}"

                self.logger.info(f"Connecting to Deriv ({self.auth_mode})...")
                self._ws = await websockets.connect(
                    url, ping_interval=20, ping_timeout=60, close_timeout=5,
                    max_size=4 * 1024 * 1024,
                )
                self.last_message_at = time.time()
                if self.auth_mode == AUTH_LEGACY and self.api_token:
                    await self._legacy_authorize()
                return
            except Exception as exc:  # noqa: BLE001 - must never crash the process
                attempt += 1
                self.logger.error(f"Deriv connect failed (attempt {attempt}): {exc!r}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    # ---------------------------------------------------------- REST/OTP auth
    def _auth_headers(self) -> dict:
        return {"Deriv-App-ID": str(self.app_id), "Authorization": f"Bearer {self.api_token}"}

    @staticmethod
    def _http_client():
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise DerivAuthError(
                "auth_mode='otp' requires httpx: pip install httpx, or set "
                "DERIV_AUTH_MODE=legacy to use the old connect-then-authorize flow"
            ) from exc
        return httpx.AsyncClient(timeout=15.0)

    async def _resolve_account_id(self) -> str:
        if not self.api_token:
            raise DerivAuthError("DERIV_API_TOKEN is not set")
        url = f"{self.api_base_url}/trading/v1/options/accounts"
        async with self._http_client() as client:
            resp = await client.get(url, headers=self._auth_headers())
        if resp.status_code != 200:
            raise DerivAuthError(
                f"fetching accounts failed: HTTP {resp.status_code} {resp.text[:300]} "
                f"(token_len={len(self.api_token)}, "
                f"token_has_surrounding_whitespace={self.api_token != self.api_token.strip()}, "
                f"app_id={self.app_id!r})"
            )
        body = resp.json()
        accounts = body.get("data") or body.get("accounts") or (body if isinstance(body, list) else [])
        if not accounts:
            raise DerivAuthError(f"no Options accounts for this token: {body}")

        wanted = "real" if self.use_real_account else "demo"
        for acc in accounts:
            kind = str(acc.get("type") or acc.get("account_type") or "").lower()
            if kind == wanted:
                account_id = acc.get("account_id") or acc.get("id")
                if account_id:
                    return account_id

        first = accounts[0]
        account_id = first.get("account_id") or first.get("id")
        if not account_id:
            raise DerivAuthError(f"account entry had no id field: {first}")
        self.logger.warning(f"no {wanted!r} account found; falling back to {account_id}")
        return account_id

    async def _exchange_otp(self, account_id: str) -> str:
        if not self.api_token:
            raise DerivAuthError("DERIV_API_TOKEN is not set")
        url = f"{self.api_base_url}/trading/v1/options/accounts/{account_id}/otp"
        async with self._http_client() as client:
            resp = await client.post(url, headers=self._auth_headers())
        if resp.status_code != 200:
            raise DerivAuthError(f"OTP exchange failed: HTTP {resp.status_code} {resp.text[:300]}")
        body = resp.json()
        data = body.get("data", body)
        auth_url = data.get("url") or data.get("websocket_url") or data.get("ws_url")
        if not auth_url:
            raise DerivAuthError(f"OTP response had no usable URL: {body}")
        return auth_url

    async def _legacy_authorize(self) -> None:
        await self._send_raw({"authorize": self.api_token})

    # ------------------------------------------------------------- read pump
    async def _recv_pump(self) -> None:
        """THE ONLY coroutine that reads the socket (rule 1). Never awaits a
        handler -- everything it does is synchronous: put_nowait onto a
        queue, or set_result on a pending future."""
        try:
            async for raw in self._ws:
                self.last_message_at = time.time()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    self.logger.warning("Deriv: undecodable frame dropped")
                    continue
                self._route(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"Deriv pump exited: {exc!r}")
        finally:
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(DerivAPIError("Disconnected", "socket closed"))
            self._pending.clear()

    def _route(self, msg: dict) -> None:
        req_id = msg.get("req_id")
        if req_id is not None and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                if "error" in msg:
                    err = msg["error"]
                    fut.set_exception(DerivAPIError(err.get("code", "Unknown"), err.get("message", ""), msg))
                else:
                    fut.set_result(msg)
            # A subscribe call's own response also carries the first payload
            # and the subscription id -- route it below too so no opening
            # tick/candle/contract state is ever lost.

        sub_id = (msg.get("subscription") or {}).get("id")
        for sub in self._subscriptions.values():
            if sub.subscription_id and sub.subscription_id == sub_id:
                try:
                    sub.callback(msg)
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Deriv subscription callback error (non-fatal): {exc!r}")
                return

        if msg.get("msg_type") == "tick":
            # Fallback: route by symbol if the stream carried no usable
            # subscription id (only reached when no id matched above).
            t = msg.get("tick") or {}
            sub = self._subscriptions.get(f"ticks:{t.get('symbol') or t.get('underlying_symbol')}")
            if sub is not None:
                try:
                    sub.callback(msg)
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Deriv tick callback error (non-fatal): {exc!r}")
            return

        if msg.get("msg_type") == "proposal_open_contract":
            poc = msg.get("proposal_open_contract") or {}
            q = self._contract_queues.get(str(poc.get("contract_id")))
            if q is not None:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(poc)

    async def _supervise_pump(self) -> None:
        """Rule 2: reconnects whenever the pump exits, for any reason. One
        long-lived instance for the client's whole life -- never re-spawned
        per reconnect, or two supervisors race to read the same socket."""
        backoff = 1.0
        while not self._closed:
            try:
                if self._pump_task:
                    await self._pump_task
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            if self._closed:
                return
            self.logger.warning("Deriv pump exited -- reconnecting")
            while not self._closed:
                await asyncio.sleep(backoff)
                try:
                    await self._reconnect()
                    backoff = 1.0
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    backoff = min(backoff * 2, 30.0)
                    self.logger.error(f"Deriv reconnect failed ({exc!r}); next attempt in {backoff:.1f}s")

    async def ensure_connected(self) -> None:
        """Reconnects a dead socket from the request path. THE OPEN CHECK IS
        OUTSIDE THE LOCK ON PURPOSE: _reconnect() holds _connect_lock while
        resubscribing, and resubscribing goes through _send() ->
        ensure_connected(). asyncio.Lock is not reentrant, so acquiring the
        lock before checking would deadlock the first resubscribe of every
        reconnect against itself."""
        if self._ws is not None and self._ws.state is WsState.OPEN:
            return
        await self._reconnect()

    async def _reconnect(self) -> None:
        async with self._connect_lock:
            if self._ws is not None and self._ws.state is WsState.OPEN:
                return  # someone else reconnected while this caller waited
            self.reconnect_count += 1
            await self._open_socket()
            # RULE 3, the whole point: start the reader on the NEW socket
            # before any resubscribe, because resubscribes await responses
            # only the reader can deliver.
            self._pump_task = asyncio.create_task(self._recv_pump(), name="deriv-pump")
            await self._verify_alive()
            for key, sub in list(self._subscriptions.items()):
                try:
                    resp = await self.send(sub.request)
                    sub.subscription_id = (resp.get("subscription") or {}).get("id")
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Deriv: failed to resubscribe {key}: {exc!r}")
                    continue
                # A contract that expired while the socket was down comes back as an
                # already-sold snapshot in this very response -- with no subscription
                # id, so the router never delivers it. Hand it to the callback here,
                # otherwise the settlement is lost and the position slot stays
                # occupied forever (seen in production: one trade stuck for hours).
                poc = resp.get("proposal_open_contract")
                if key.startswith("contract:") and poc:
                    try:
                        sub.callback({"proposal_open_contract": poc,
                                      "subscription": {"id": sub.subscription_id}})
                    except Exception as exc:  # noqa: BLE001
                        self.logger.error(f"Deriv: contract snapshot callback failed after resubscribe: {exc!r}")
        if self.on_reconnect:
            await self.on_reconnect()

    # ------------------------------------------------------- request/response
    async def _send_raw(self, payload: dict, timeout: float | None = None) -> dict:
        """Sends without pacing or auto-reconnect -- only for use during the
        connect/auth sequence itself, before rate limiting or resubscribe
        logic would make sense."""
        self._req_id_counter += 1
        req_id = self._req_id_counter
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._ws.send(json.dumps(payload))
            return await asyncio.wait_for(fut, timeout=timeout or self.request_timeout)
        finally:
            self._pending.pop(req_id, None)

    async def send(self, payload: dict, timeout: float | None = None) -> dict:
        await self.rate_limiter.acquire()
        await self.ensure_connected()
        if self._ws is None or self._ws.state is not WsState.OPEN:
            raise DerivAPIError("Disconnected", "socket is not open")
        self._req_id_counter += 1
        req_id = self._req_id_counter
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._ws.send(json.dumps(payload))
            return await asyncio.wait_for(fut, timeout=timeout or self.request_timeout)
        except asyncio.TimeoutError:
            raise DerivAPIError("Timeout", f"no response within {timeout or self.request_timeout}s")
        finally:
            self._pending.pop(req_id, None)

    # ------------------------------------------------------------ high level
    async def get_candle_history(self, symbol: str, granularity_seconds: int, count: int) -> list[dict]:
        resp = await self.send({
            "ticks_history": symbol, "style": "candles",
            "granularity": granularity_seconds, "count": min(count, 5000),
            "end": "latest", "adjust_start_time": 1,
        })
        raw = resp.get("candles", [])
        out = []
        for i, c in enumerate(raw if isinstance(raw, list) else []):
            try:
                out.append({
                    "epoch": int(c["epoch"]), "open": float(c["open"]), "high": float(c["high"]),
                    "low": float(c["low"]), "close": float(c["close"]),
                })
            except (KeyError, TypeError, ValueError) as exc:
                self.logger.warning(f"{symbol}: skipping malformed candle at index {i}: {exc}")
        out.sort(key=lambda c: c["epoch"])
        return out

    async def get_tick_history(self, symbol: str, count: int) -> list[float]:
        """The last `count` tick quotes (oldest first) -- seeds the EMA state
        so the crossover logic is warm from the first live tick."""
        resp = await self.send({
            "ticks_history": symbol, "style": "ticks", "end": "latest",
            "count": min(int(count), 5000),
        })
        prices = (resp.get("history") or {}).get("prices") or []
        out = []
        for p in prices:
            try:
                out.append(float(p))
            except (TypeError, ValueError):
                continue
        return out

    async def subscribe_candles(self, symbol: str, granularity_seconds: int, count: int, callback) -> str:
        key = f"candles:{symbol}:{granularity_seconds}"
        request = {
            "ticks_history": symbol, "style": "candles", "granularity": granularity_seconds,
            "count": count, "subscribe": 1, "end": "latest",
        }
        resp = await self.send(request)
        sub_id = (resp.get("subscription") or {}).get("id")
        self._subscriptions[key] = Subscription(request=request, callback=callback, subscription_id=sub_id)
        return key

    async def subscribe_ticks(self, symbol: str, callback) -> str:
        key = f"ticks:{symbol}"
        request = {"ticks": symbol, "subscribe": 1}
        resp = await self.send(request)
        sub_id = (resp.get("subscription") or {}).get("id")
        self._subscriptions[key] = Subscription(request=request, callback=callback, subscription_id=sub_id)
        return key

    async def request_proposal(self, symbol: str, duration_minutes: int, stake: float,
                                lower_barrier: float, upper_barrier: float,
                                currency: str = "USD") -> dict:
        """EXPIRYRANGE == Deriv's "Ends Between" contract. `underlying_symbol`
        (not `symbol`) per the current API -- confirmed in
        developers.deriv.com/llms/contract-types.md.

        Field-to-value mapping confirmed against developers.deriv.com/docs/
        ends-between: `barrier` is the HIGH barrier offset, `barrier2` is the
        LOW barrier offset -- despite `barrier2`'s generic schema description
        elsewhere just saying "low barrier" without saying `barrier` is high.
        Getting this backwards produces a 100%-reproducible
        `InvalidHighBarrier: High barrier must be higher than low barrier`
        from Deriv on every single candidate, which is exactly what an
        earlier version of this method did.
        """
        resp = await self.send({
            "proposal": 1, "amount": round(float(stake), 2), "basis": "stake",
            "contract_type": "EXPIRYRANGE", "currency": currency,
            "duration": duration_minutes, "duration_unit": "m",
            "underlying_symbol": symbol,
            "barrier": f"+{upper_barrier}" if upper_barrier >= 0 else str(upper_barrier),
            "barrier2": f"+{lower_barrier}" if lower_barrier >= 0 else str(lower_barrier),
        })
        return resp.get("proposal", {})

    async def buy_contract(self, proposal_id: str, price: float) -> dict:
        """Exactly one attempt, ever. On an ambiguous failure (timeout or
        disconnect) the outcome is genuinely unknown -- the buy may have
        been accepted. Raises BuyAmbiguousError so the caller can reconcile
        via portfolio() instead of retrying (a retry that lands after a
        successful first attempt opens a second contract)."""
        try:
            resp = await self.send({"buy": proposal_id, "price": round(float(price), 2)})
        except DerivAPIError as exc:
            if exc.code in ("Timeout", "Disconnected"):
                raise BuyAmbiguousError(
                    "BuyAmbiguous",
                    f"buy outcome unknown for proposal {proposal_id} ({exc.code}) -- "
                    f"reconcile via portfolio(), do not retry",
                ) from exc
            raise
        return resp.get("buy", {})

    async def portfolio(self) -> list[dict]:
        resp = await self.send({"portfolio": 1})
        return resp.get("portfolio", {}).get("contracts", [])

    async def subscribe_contract(self, contract_id: str, callback) -> str:
        key = f"contract:{contract_id}"
        cid_key = str(contract_id)  # queues/routing are keyed by the string form
        if cid_key not in self._contract_queues:
            self._contract_queues[cid_key] = asyncio.Queue(maxsize=50)
        # Deriv's current API rejects a string contract_id ("InputValidationFailed:
        # contract_id"): it must go over the wire as an integer. Everything else in
        # the bot keeps the id as a string, so convert only here.
        try:
            wire_id = int(contract_id)
        except (TypeError, ValueError):
            wire_id = contract_id
        request = {"proposal_open_contract": 1, "contract_id": wire_id, "subscribe": 1}
        resp = await self.send(request)
        sub_id = (resp.get("subscription") or {}).get("id")
        self._subscriptions[key] = Subscription(request=request, callback=callback, subscription_id=sub_id)
        poc = resp.get("proposal_open_contract")
        if poc:
            try:
                callback({"proposal_open_contract": poc, "subscription": {"id": sub_id}})
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"Deriv: initial contract callback failed (non-fatal): {exc!r}")
        return key

    async def get_contract(self, contract_id) -> dict:
        """One-shot (non-subscribing) state of a contract -- the fallback for
        settlements the stream missed."""
        try:
            wire_id = int(contract_id)
        except (TypeError, ValueError):
            wire_id = contract_id
        resp = await self.send({"proposal_open_contract": 1, "contract_id": wire_id}, timeout=15.0)
        return resp.get("proposal_open_contract") or {}

    async def forget_subscription(self, key: str) -> None:
        """Stops tracking a subscription locally and asks Deriv to end the
        stream. Removing it locally is what matters: every entry left in
        _subscriptions is re-sent by _reconnect(), so settled contracts that
        were never removed piled up and got resubscribed after every
        reconnect for the life of the process."""
        sub = self._subscriptions.pop(key, None)
        if key.startswith("contract:"):
            self._contract_queues.pop(key.split(":", 1)[1], None)
        if sub and sub.subscription_id:
            try:
                await self.send({"forget": sub.subscription_id}, timeout=5.0)
            except Exception as exc:  # noqa: BLE001 - stream may already be closed server-side
                self.logger.debug(f"Deriv: forget {key} ignored ({exc!r})")

    async def get_spot_at(self, symbol: str, epoch: int) -> tuple[float, int] | None:
        """The last tick at or before `epoch` -> (price, tick_epoch), or None.
        Used to resolve shadow trades against the real market, the same way
        Deriv takes the exit spot at expiry."""
        resp = await self.send({
            "ticks_history": symbol, "style": "ticks", "end": int(epoch), "count": 1,
        })
        hist = resp.get("history") or {}
        prices, times = hist.get("prices") or [], hist.get("times") or []
        if not prices or not times:
            return None
        return float(prices[-1]), int(times[-1])

    async def balance(self) -> dict:
        resp = await self.send({"balance": 1})
        return resp.get("balance", {})
