"""
Production entrypoint. `python run.py`

Quantum Ends -- Python port of the "QUANTUM ENDS-IN-OUT" Deriv Bot, built on
the ER bot's infrastructure (Deriv client, journal, settlement, restart
safety). Unlike ER, which scans on a timer, this bot is SIGNAL-DRIVEN: every
tick feeds an EMA(7)/EMA(14) cross detector; a bearish cross confirmed by the
tick direction and the last 60s candle triggers the decision pipeline
(app/strategy/engine.py), and an accepted candidate is bought as an Ends
Between (EXPIRYRANGE) contract.

Startup: connect DB -> connect Deriv -> prime candles + tick EMA state ->
subscribe ticks -> reattach contracts still open from before a restart ->
run. Every step logs clearly; an ordinary Railway restart needs no manual
intervention.
"""
from __future__ import annotations

import asyncio
import json
import signal
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np

from app.config import load_config
from app.data.storage import Storage
from app.deriv.client import BuyAmbiguousError, DerivClient
from app.models.calibration import AdaptiveCalibrator
from app.monitoring.logging_utils import setup_logging
from app.state_machine import State, StateMachine
from app.strategy.engine import evaluate_signal
from app.strategy.risk import exposure_allows, session_status, utc_day_start
from app.strategy.signals import TickSignalDetector, candle_confirms, direction_allowed, reference_candle
from app.strategy.staking import build_staking_engine

CANDLE_HISTORY_COUNT = 300
SESSION_CACHE_SECONDS = 3.0
SHADOW_RESOLVE_INTERVAL_SECONDS = 10
SHADOW_RESOLVE_GRACE_SECONDS = 3      # let the expiry tick land before asking for it
STATS_LOG_INTERVAL_SECONDS = 15 * 60
LIVE_WATCHDOG_INTERVAL_SECONDS = 20
LIVE_WATCHDOG_GRACE_SECONDS = 20     # let the stream settle it first
HEALTH_STALE_AFTER_SECONDS = 120


class Bot:
    def __init__(self):
        self.cfg = load_config()
        self.logger = setup_logging(self.cfg.log_level)
        self.sm = StateMachine(self.logger)
        self.storage = Storage(self.cfg.database_url, self.cfg.sqlite_path, self.logger, schema=self.cfg.db_schema)
        self.calibration = AdaptiveCalibrator(
            self.storage, probe_weight=self.cfg.probe_weight,
            halflife_hours=self.cfg.calibration_halflife_hours,
            prior_strength=self.cfg.calibration_prior_strength,
            regime_prior_strength=self.cfg.calibration_regime_prior_strength)
        self.staking = build_staking_engine(self.cfg.staking)
        self.client: DerivClient | None = None
        self.closes_by_symbol: dict[str, deque] = {}
        self.candles_by_symbol: dict[str, list] = {}
        self.detectors: dict[str, TickSignalDetector] = {}
        self._evaluating: set[str] = set()
        self._probe_only: dict[str, bool] = {}
        self._session_cache: tuple | None = None   # (fetched_at, pnl)
        self._session_state_logged = "OK"
        # trade_id -> symbol for every position occupying a slot. Shadow
        # trades occupy slots too, until resolved, so shadow mode behaves like
        # live (one contract at a time) and its record predicts live results.
        self.open_positions: dict[str, str] = {}
        self._background: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self.started_at = time.time()

    # ------------------------------------------------------------- helpers
    def _spawn(self, coro, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def _open_count(self, symbol: str | None = None) -> int:
        if symbol is None:
            return len(self.open_positions)
        return sum(1 for s in self.open_positions.values() if s == symbol)

    # ------------------------------------------------------------- startup
    async def start(self):
        self.sm.transition(State.INITIALIZING)
        problems = self.cfg.validate()
        for p in problems:
            self.logger.error(f"CONFIG PROBLEM: {p}")
        if self.cfg.is_live() and problems:
            self.logger.error("Refusing to start in LIVE mode with unresolved config problems. Falling back to DEMO behaviour (shadow-only).")
            self.cfg.shadow_mode = True

        self.logger.info(f"ACCOUNT MODE: {self.cfg.account_mode}  |  SHADOW MODE: {self.cfg.shadow_mode}")
        self.logger.info(
            f"STRATEGY: {self.cfg.symbols} | EMA {self.cfg.ema_fast}/{self.cfg.ema_slow} cross {self.cfg.signal_directions.upper()} | "
            f"{self.cfg.duration_minutes}m Ends Between | barriers={self.cfg.barrier_mode} scales={self.cfg.barrier_scales} | "
            f"regime_gate={self.cfg.use_regime_gate} edge_gate={self.cfg.use_edge_gate}"
        )
        if hasattr(self.staking, "ladder"):
            self.logger.info(f"STAKE LADDER (after 0,1,2.. losses): {self.staking.ladder()} "
                             f"| target={self.cfg.target_profit} stop_loss={self.cfg.stop_loss} halt={self.cfg.session_halt}")

        self.storage.init_schema()
        retired = self.storage.retire_legacy_shadow_rows()
        if retired:
            self.logger.info(f"Marked {retired} pre-upgrade shadow rows as UNRESOLVED (no expiry recorded, cannot be settled)")
        cal = self.calibration.refit()
        self.logger.info(f"CALIBRATION loaded: {cal['n_trades']} trades + {cal['n_probes']} probes -> "
                         f"a={cal['a']} b={cal['b']} offsets={cal['regime_offsets']}")
        self.sm.transition(State.SYNCING_DATA)

        self.sm.transition(State.CONNECTING)
        self.client = DerivClient(
            endpoint=self.cfg.deriv_endpoint,
            app_id=self.cfg.deriv_app_id,
            api_token=self.cfg.deriv_api_token,
            logger=self.logger,
            on_reconnect=self._on_reconnect,
        )
        await self.client.connect()

        self.sm.transition(State.WARMING_UP)
        await self._prime_candles()
        await self._prime_ticks()
        await self._restore_open_positions()

        self._spawn(self._shadow_resolver_loop(), "shadow-resolver")
        self._spawn(self._live_settlement_watchdog(), "live-settlement-watchdog")
        self._spawn(self._probe_resolver_loop(), "probe-resolver")
        self._spawn(self._stats_loop(), "stats-logger")

        self.sm.transition(State.READY)
        self.logger.info("BOT READY -- waiting for EMA crossover signals")

    async def _prime_candles(self):
        for symbol in self.cfg.symbols:
            await self._refresh_candles(symbol)
            self.logger.info(f"{symbol}: primed with {len(self.closes_by_symbol.get(symbol, ()))} candles")

    async def _prime_ticks(self):
        """Seed each symbol's EMA state from tick history, then subscribe. A
        cross is detected on the exact tick it happens, like the XML."""
        for symbol in self.cfg.symbols:
            det = TickSignalDetector(self.cfg.ema_fast, self.cfg.ema_slow, self.cfg.require_tick_confirmation)
            try:
                history = await self.client.get_tick_history(symbol, self.cfg.tick_buffer)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"{symbol}: tick history failed ({exc!r}) -- will warm up from live ticks")
                history = []
            if history:
                det.seed(history)
            self.detectors[symbol] = det
            await self.client.subscribe_ticks(symbol, lambda msg, s=symbol: self._on_tick(s, msg))
            self.logger.info(f"{symbol}: tick stream subscribed (EMA state {'seeded from %d ticks' % len(history) if det.ready else 'warming up'})")

    async def _refresh_candles(self, symbol: str) -> None:
        """Fresh fetch of candle history (OHLC). Deliberately NOT built on the
        candle subscription: that stream pushes an update on every tick of the
        still-forming candle, which corrupts a close series."""
        try:
            candles = await self.client.get_candle_history(
                symbol, self.cfg.candle_granularity_seconds, CANDLE_HISTORY_COUNT)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"{symbol}: candle refresh failed (non-fatal, using previous history): {exc!r}")
            return
        if not candles:
            return
        self.candles_by_symbol[symbol] = candles
        buf = self.closes_by_symbol.setdefault(symbol, deque(maxlen=CANDLE_HISTORY_COUNT))
        buf.clear()
        for c in candles:
            buf.append(float(c["close"]))

    async def _restore_open_positions(self):
        """Previously, open-contract tracking lived only in memory: after a
        restart, contracts bought before it were never settled in the DB,
        never fed calibration, didn't count toward concurrency limits, and
        were invisible to the consecutive-loss logic. Reattach them."""
        for t in self.storage.open_trades(shadow=True):
            if t.get("expires_at"):
                self.open_positions[t["trade_id"]] = t["symbol"]
        live = [t for t in self.storage.open_trades(shadow=False) if t.get("contract_id")]
        for t in live:
            self.open_positions[t["trade_id"]] = t["symbol"]
            await self._track_contract(t["trade_id"], t["symbol"], t.get("raw_probability"), t["contract_id"])
        if self.open_positions:
            self.logger.info(f"Restored {len(self.open_positions)} open position(s) from the journal "
                             f"({len(live)} live contract(s) re-subscribed)")

    async def _on_reconnect(self):
        self.logger.info("Reconnect handler: subscriptions re-established, resuming normal operation")

    # --------------------------------------------------------------- loop
    async def run_forever(self):
        """Housekeeping only: signals are tick-driven (see _on_tick)."""
        while not self._stop.is_set():
            self.sm.reassess_cautions()
            if self.calibration.needs_refit():
                try:
                    await asyncio.to_thread(self.calibration.refit)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(f"calibration refit failed (non-fatal): {exc!r}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.cfg.scan_interval_seconds)
            except asyncio.TimeoutError:
                pass

    def _on_tick(self, symbol: str, msg: dict) -> None:
        """Called by the Deriv client for every tick (inside the event loop,
        so it must stay cheap -- all real work is spawned as a task)."""
        quote = (msg.get("tick") or {}).get("quote")
        det = self.detectors.get(symbol)
        if quote is None or det is None or self._stop.is_set():
            return
        try:
            direction = det.feed(float(quote))
        except (TypeError, ValueError):
            return
        if direction is None or not direction_allowed(direction, self.cfg.signal_directions):
            return
        if symbol in self._evaluating:
            return
        if (not self.cfg.learn_from_rejected
                and (self._open_count(symbol) >= self.cfg.max_concurrent_per_symbol
                     or self._open_count() >= self.cfg.max_concurrent_contracts)):
            return  # a contract is running; the XML also waits for its result
        self._spawn(self._handle_signal(symbol, direction), f"signal-{symbol}")

    def _session(self) -> tuple[str, float]:
        """(status, pnl): the XML's Target Profit / Stop Loss check, over the
        current UTC day, read from the journal (restart-safe, cached briefly)."""
        now = time.time()
        if self._session_cache is None or now - self._session_cache[0] > SESSION_CACHE_SECONDS:
            pnl = self.storage.pnl_since(utc_day_start(), shadow=self.cfg.shadow_mode)
            self._session_cache = (now, pnl)
        pnl = self._session_cache[1]
        status = session_status(pnl, self.cfg.target_profit, self.cfg.stop_loss) if self.cfg.session_halt else "OK"
        if status != self._session_state_logged:
            if status == "TARGET":
                self.logger.warning(f"SESSION TARGET REACHED: day P&L {pnl:+.2f} >= {self.cfg.target_profit} -- trading paused until 00:00 UTC")
            elif status == "STOP":
                self.logger.warning(f"SESSION STOP-LOSS HIT: day P&L {pnl:+.2f} <= -{abs(self.cfg.stop_loss)} -- trading paused until 00:00 UTC")
            else:
                self.logger.info(f"Session limits clear again (day P&L {pnl:+.2f}) -- trading resumed")
            self._session_state_logged = status
        return status, pnl

    async def _handle_signal(self, symbol: str, direction: str) -> None:
        if symbol in self._evaluating:
            return
        self._evaluating.add(symbol)
        try:
            await self._process_signal(symbol, direction)
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: signal handling failed (non-fatal): {exc!r}")
            if not self._probe_only.get(symbol):
                self.sm.enter_caution(symbol, f"signal error: {exc!r}", self.cfg.caution_cooldown_seconds)
        finally:
            self._evaluating.discard(symbol)
            if self._open_count() == 0 and self.sm.state not in (State.READY, State.CAUTION):
                self.sm.transition(State.READY)

    async def _process_signal(self, symbol: str, direction: str) -> None:
        if self.sm.is_symbol_cautioned(symbol):
            return  # cooling down after an error; clears automatically
        # A busy slot or a tripped session limit no longer discards the signal:
        # it is still priced and its outcome learned from, it just cannot trade.
        probe_only, why = False, ""
        if (self._open_count(symbol) >= self.cfg.max_concurrent_per_symbol
                or self._open_count() >= self.cfg.max_concurrent_contracts):
            probe_only, why = True, "position open"
        else:
            status, pnl = self._session()
            if status != "OK":
                probe_only, why = True, f"session {status}"
        if probe_only and not self.cfg.learn_from_rejected:
            return
        self._probe_only[symbol] = probe_only

        det = self.detectors[symbol]
        self.logger.info(f"{symbol} SIGNAL {direction}: EMA{self.cfg.ema_fast}={det.ema_fast:.4f} "
                         f"EMA{self.cfg.ema_slow}={det.ema_slow:.4f} price={det.last_price}"
                         + (f" [learning only: {why}]" if probe_only else ""))

        # fresh candles: the colour check needs the just-finished candle's final OHLC
        await self._refresh_candles(symbol)
        if self.cfg.require_candle_confirmation:
            candle = reference_candle(self.candles_by_symbol.get(symbol, []), time.time(),
                                      self.cfg.candle_granularity_seconds, self.cfg.candle_reference)
            if not candle_confirms(direction, candle):
                self.logger.info(f"{symbol}: signal {direction} skipped -- {self.cfg.candle_reference} "
                                 f"{self.cfg.candle_granularity_seconds}s candle does not confirm "
                                 f"(open={candle['open'] if candle else None} close={candle['close'] if candle else None})")
                return

        closes = np.array(self.closes_by_symbol.get(symbol, ()), dtype=float)
        if len(closes) == 0 or det.last_price is None:
            return
        current_price = float(det.last_price)

        shadow = self.cfg.shadow_mode
        # one bot, one ladder: the XML has a single global Loss Count
        consecutive_losses = self.storage.consecutive_losses(None, shadow=shadow)

        if not probe_only:
            self.sm.transition(State.SIMULATING, symbol)
        try:
            outcome = await evaluate_signal(
                symbol=symbol, direction=direction, closes=closes, current_price=current_price,
                client=self.client, calibration=self.calibration, staking=self.staking, cfg=self.cfg,
                logger=self.logger, consecutive_losses=consecutive_losses,
                stake_multiplier=self.sm.stake_multiplier(),
                extra_edge_requirement=self.sm.extra_edge_requirement(),
                probe_only=probe_only,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: decision cycle failed (non-fatal): {exc!r}")
            if not probe_only:
                self.sm.enter_caution(symbol, f"decision error: {exc!r}", self.cfg.caution_cooldown_seconds)
                self.sm.transition(State.READY)
            return

        if outcome.probes:
            try:
                self.storage.record_probes(outcome.probes)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"{symbol}: could not record probes (non-fatal): {exc!r}")
        if probe_only:
            return

        for rej in outcome.rejections:
            self.storage.record_rejection({
                k: rej.get(k) for k in (
                    "symbol", "duration_minutes", "lower_barrier", "upper_barrier", "raw_probability",
                    "calibrated_probability", "payout", "implied_probability", "edge", "expected_value",
                    "regime", "rejection_reason")
            })

        if not outcome.traded or outcome.trade_row is None:
            self.sm.transition(State.READY)
            return

        row = outcome.trade_row

        # MAX_DAILY_EXPOSURE existed in config but was never enforced.
        # Resets at 00:00 UTC -- temporary by design, like every other
        # restriction in this bot. Applied in shadow too, so shadow mirrors live.
        staked_today = self.storage.stake_since(utc_day_start(), shadow=shadow)
        if not exposure_allows(staked_today, row["stake"], self.cfg.max_daily_exposure):
            self.logger.warning(
                f"{symbol}: DAILY EXPOSURE CAP -- {staked_today:.2f} staked today + {row['stake']:.2f} "
                f"would exceed {self.cfg.max_daily_exposure:.2f}; skipping until 00:00 UTC"
            )
            return

        now = time.time()
        row = {**row, "opened_at": now, "expires_at": now + row["duration_minutes"] * 60}

        # SHADOW_MODE alone controls execution vs logging-only. ACCOUNT_MODE
        # (DEMO/LIVE) only controls which Deriv account the token authorizes.
        if shadow:
            self.storage.record_trade({**row, "shadow": 1, "contract_id": None, "balance_before": None})
            self.open_positions[row["trade_id"]] = symbol
            self.logger.info(f"{symbol} SHADOW TRADE (not executed): ev={row['expected_value']:+.4f} "
                             f"stake={row['stake']} resolves in {row['duration_minutes']}m")
            return

        self.sm.transition(State.EXECUTING, symbol)
        # No balance() round-trip before buying any more: every extra request
        # here ages the proposal we're about to buy. Deriv's buy response
        # carries balance_after, from which balance_before follows.
        buy_started = time.time()
        try:
            buy_resp = await self.client.buy_contract(row["proposal_id"], row["stake"])
        except BuyAmbiguousError as exc:
            self.logger.error(f"{symbol}: {exc} -- reconciling against portfolio")
            buy_resp = await self._reconcile_ambiguous_buy(symbol, since=buy_started - 5,
                                                         stake=row["stake"], payout=row["payout"])
            if buy_resp is None:
                self.sm.enter_caution(symbol, "ambiguous buy, no contract found", self.cfg.caution_cooldown_seconds)
                return
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: buy failed (non-fatal): {exc!r}")
            self.sm.enter_caution(symbol, f"buy error: {exc!r}", self.cfg.caution_cooldown_seconds)
            return

        contract_id = str(buy_resp.get("contract_id"))
        buy_price = _float_or_none(buy_resp.get("buy_price"))
        balance_after = _float_or_none(buy_resp.get("balance_after"))
        balance_before = balance_after + buy_price if (balance_after is not None and buy_price is not None) else None
        if buy_price is not None:
            row["stake"] = buy_price

        self.storage.record_trade({**row, "shadow": 0, "contract_id": contract_id, "balance_before": balance_before})
        self.open_positions[row["trade_id"]] = symbol
        self.sm.transition(State.POSITION_OPEN, symbol)
        self.logger.info(f"{symbol} TRADE EXECUTED: contract_id={contract_id} stake={row['stake']} ev={row['expected_value']:+.4f}")
        await self._track_contract(row["trade_id"], symbol, row["raw_probability"], contract_id)

    # --------------------------------------------------------- settlement
    def _apply_contract_update(self, trade_id, symbol, raw_probability, poc, key=None) -> bool:
        """Settles a trade from a proposal_open_contract payload if it is sold.
        Shared by the live stream callback and the settlement watchdog;
        _settle is idempotent so a duplicate is harmless."""
        if not poc or not poc.get("is_sold"):
            return False
        profit = float(poc.get("profit", 0.0) or 0.0)
        status = str(poc.get("status") or "").lower()
        won = status == "won" if status in ("won", "lost") else profit > 0
        self._settle(trade_id, symbol, raw_probability, won, profit,
                     exit_spot=_float_or_none(poc.get("exit_tick") or poc.get("exit_spot")),
                     source="live")
        if key:
            self._spawn(self.client.forget_subscription(key), f"forget-{key}")
        return True

    async def _track_contract(self, trade_id: str, symbol: str, raw_probability, contract_id: str):
        key = f"contract:{contract_id}"

        def _on_update(msg):
            self._apply_contract_update(trade_id, symbol, raw_probability,
                                        msg.get("proposal_open_contract") or {}, key)

        try:
            await self.client.subscribe_contract(contract_id, _on_update)
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"{symbol}: contract subscription failed for {contract_id} (non-fatal, "
                              f"the settlement watchdog will pick it up): {exc!r}")

    async def _live_settlement_watchdog(self):
        """Safety net for live trades whose settlement never arrived on the
        stream (e.g. the socket dropped at the moment of expiry). Any live
        trade past expiry is queried once per pass and settled if sold.
        Without this a missed settlement leaves the position slot occupied
        forever and the bot stops trading."""
        warned: dict[str, float] = {}
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=LIVE_WATCHDOG_INTERVAL_SECONDS)
                return
            except asyncio.TimeoutError:
                pass
            try:
                now = time.time()
                for t in self.storage.open_trades(shadow=False):
                    cid, exp = t.get("contract_id"), t.get("expires_at")
                    if not cid or not exp or now < float(exp) + LIVE_WATCHDOG_GRACE_SECONDS:
                        continue
                    try:
                        poc = await self.client.get_contract(cid)
                    except Exception as exc:  # noqa: BLE001
                        if now - warned.get(cid, 0) > 300:
                            warned[cid] = now
                            self.logger.warning(f"watchdog: could not query contract {cid}: {exc!r}")
                        continue
                    if self._apply_contract_update(t["trade_id"], t["symbol"], t.get("raw_probability"),
                                                   poc, f"contract:{cid}"):
                        self.logger.warning(f"{t['symbol']} watchdog settled contract {cid} that the stream missed")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"settlement watchdog pass failed (non-fatal): {exc!r}")

    def _settle(self, trade_id, symbol, raw_probability, won: bool, profit: float, *, exit_spot=None, source=""):
        """Single settlement path for live and shadow. The DB update is
        conditional on the trade still being OPEN, so a duplicate is_sold
        message (initial snapshot + stream, or a resubscribe after reconnect)
        cannot double-count an outcome into calibration."""
        result = "WIN" if won else "LOSS"
        first_time = self.storage.settle_trade(trade_id, result, profit, exit_spot=exit_spot)
        self.open_positions.pop(trade_id, None)
        self._session_cache = None  # P&L changed -- re-read on next check
        if not self.open_positions and self.sm.state != State.READY:
            self.sm.transition(State.READY)
        if not first_time:
            return
        if raw_probability is not None:
            self.calibration.record_outcome(float(raw_probability), won)
        self.logger.info(f"{symbol} SETTLED ({source}): {result} pnl={profit:+.4f}"
                         + (f" exit={exit_spot}" if exit_spot is not None else ""))

    async def _reconcile_ambiguous_buy(self, symbol: str, since: float,
                                       stake: float | None = None, payout: float | None = None) -> dict | None:
        """A timed-out buy may still have gone through. Never retry (that can
        open a second contract); look for it in the portfolio instead.

        When stake/payout are given, a contract must also match them (to the
        cent / within 1%). That stops this bot adopting a contract opened at
        the same moment by ANOTHER bot sharing the same Deriv account."""
        known = {t.get("contract_id") for t in self.storage.open_trades(shadow=False)}
        for attempt in range(3):
            try:
                contracts = await self.client.portfolio()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"{symbol}: portfolio check failed ({exc!r}), attempt {attempt + 1}/3")
                await asyncio.sleep(2)
                continue
            for c in contracts:
                cid = str(c.get("contract_id"))
                sym = c.get("symbol") or c.get("underlying_symbol")
                if (cid not in known and sym == symbol
                        and str(c.get("contract_type", "")).upper() == "EXPIRYRANGE"
                        and float(c.get("purchase_time") or 0) >= since
                        and _matches_order(c, stake, payout)):
                    self.logger.warning(f"{symbol}: ambiguous buy DID go through -> contract {cid}")
                    return {"contract_id": cid, "buy_price": c.get("buy_price")}
            return None
        return None

    async def _shadow_resolver_loop(self):
        """Shadow trades used to be logged and then left OPEN forever, so
        shadow mode -- the recommended way to validate the model -- never
        actually produced a win rate or P&L to validate against. This settles
        each one against the real tick at expiry, the same way Deriv decides
        an Ends Between contract: exit spot strictly inside the barriers."""
        while not self._stop.is_set():
            try:
                now = time.time()
                for t in self.storage.open_trades(shadow=True):
                    expires_at = t.get("expires_at")
                    if not expires_at or now < expires_at + SHADOW_RESOLVE_GRACE_SECONDS:
                        continue
                    spot = await self.client.get_spot_at(t["symbol"], int(expires_at))
                    if spot is None:
                        continue  # retry next pass
                    exit_price, _ = spot
                    won = float(t["lower_barrier"]) < exit_price < float(t["upper_barrier"])
                    stake = float(t["stake"] or 0)
                    profit = (float(t["payout"] or 0) - stake) if won else -stake
                    self._settle(t["trade_id"], t["symbol"], t.get("raw_probability"), won, profit,
                                 exit_spot=exit_price, source="shadow")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"shadow resolver pass failed (non-fatal): {exc!r}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=SHADOW_RESOLVE_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def _probe_resolver_loop(self):
        """Settles priced-but-not-bought candidates against the real tick at
        expiry (same rule as a contract: exit strictly inside the barriers).
        All candidates of one signal share an expiry, so one tick lookup
        settles the whole group. Fresh outcomes trigger a calibration refit."""
        last_prune = 0.0
        while not self._stop.is_set():
            try:
                now = time.time()
                due = self.storage.open_probes(now - SHADOW_RESOLVE_GRACE_SECONDS)
                groups: dict = {}
                for pr in due:
                    groups.setdefault((pr["symbol"], int(pr["expires_at"])), []).append(pr)
                resolved = 0
                for (symbol, expires_at), rows in groups.items():
                    if now - expires_at > 1800:           # tick history no longer reachable; stop retrying
                        for pr in rows:
                            self.storage.abandon_probe(pr["id"])
                        continue
                    spot = await self.client.get_spot_at(symbol, expires_at)
                    if spot is None:
                        continue                          # retry next pass
                    exit_price, _ = spot
                    for pr in rows:
                        won = float(pr["lower_barrier"]) < exit_price < float(pr["upper_barrier"])
                        if self.storage.resolve_probe(pr["id"], won, exit_price):
                            resolved += 1
                if resolved:
                    self.calibration.mark_dirty(resolved)
                if now - last_prune > 3600:
                    last_prune = now
                    self.storage.prune_probes(now - self.cfg.probe_keep_days * 86400)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"probe resolver pass failed (non-fatal): {exc!r}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=SHADOW_RESOLVE_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass

    # -------------------------------------------------------------- stats
    def stats(self) -> dict:
        shadow = self.cfg.shadow_mode
        streak = self.storage.consecutive_losses(None, shadow=shadow)
        status, pnl = self._session()
        return {
            "account_mode": self.cfg.account_mode,
            "shadow_mode": shadow,
            "state": self.sm.state.value,
            "uptime_seconds": int(time.time() - self.started_at),
            "open_positions": self._open_count(),
            "deriv": {
                "reconnects": getattr(self.client, "reconnect_count", None),
                "keepalive_pings": getattr(self.client, "ping_count", None),
                "last_ping_age_seconds": (round(time.time() - self.client.last_ping_ok, 1)
                                          if getattr(self.client, "last_ping_ok", 0) else None),
            },
            "active_cautions": {k: c.reason for k, c in self.sm.active_cautions().items()},
            "session": {"status": status, "day_pnl": round(pnl, 2),
                        "target_profit": self.cfg.target_profit, "stop_loss": self.cfg.stop_loss},
            "loss_streak": streak,
            "next_stake": self.staking.stake_for(consecutive_losses=streak),
            "signals_ready": {s: d.ready for s, d in self.detectors.items()},
            "staked_today": round(self.storage.stake_since(utc_day_start(), shadow=shadow), 2),
            "max_daily_exposure": self.cfg.max_daily_exposure,
            "shadow": self.storage.performance_summary(shadow=True),
            "executed": self.storage.performance_summary(shadow=False),
            "ladder": {m: self.storage.ladder_stats(m == "shadow", self.cfg.staking.martingale_start,
                                                    self.cfg.staking.martingale_level)
                       for m in ("executed", "shadow")},
            "calibration": {**self.calibration.summary(), "probes": self.storage.probe_counts()},
            "breakdown": {"executed": self.storage.breakdown(shadow=False),
                          "shadow": self.storage.breakdown(shadow=True)},
        }

    async def _stats_loop(self):
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=STATS_LOG_INTERVAL_SECONDS)
                return
            except asyncio.TimeoutError:
                pass
            try:
                s = self.stats()
                for mode in ("shadow", "executed"):
                    p = s[mode]
                    if p["settled"]:
                        self.logger.info(
                            f"PERFORMANCE[{mode}] settled={p['settled']} win_rate={p['win_rate']} "
                            f"avg_predicted={p['avg_predicted']} avg_implied={p['avg_implied']} "
                            f"pnl={p['pnl']:+.2f} roi={p['roi']}"
                            + self._ladder_suffix(s.get("ladder", {}).get(mode))
                        )
                c = s.get("calibration") or {}
                if c.get("n_trades", 0) + c.get("n_probes", 0):
                    self.logger.info(
                        f"CALIBRATION trades={c['n_trades']} probes={c['n_probes']} a={c['a']} b={c['b']} "
                        f"actual={c.get('actual_win_rate')} raw={c.get('mean_raw')} calibrated={c.get('mean_calibrated')} "
                        f"brier_raw={c.get('brier_raw')} brier_cal={c.get('brier_calibrated')} offsets={c['regime_offsets']}")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"stats logging failed (non-fatal): {exc!r}")

    @staticmethod
    def _ladder_suffix(ladder: dict | None) -> str:
        """Appended to the PERFORMANCE line so one pasted line carries the risk picture."""
        if not ladder:
            return ""
        ms = ladder.get("max_stake") or {}
        return (f" max_stake={ms.get('stake')} max_level={ladder['max_martingale_level']} "
                f"max_loss_streak={ladder['max_loss_streak']} max_drawdown={ladder['max_drawdown']}")

    def is_healthy(self) -> tuple[bool, str]:
        if self.client is None:
            return False, "not connected"
        age = time.time() - (self.client.last_message_at or 0)
        if age > HEALTH_STALE_AFTER_SECONDS:
            return False, f"no Deriv traffic for {int(age)}s"
        return True, "ok"

    async def shutdown(self):
        self.logger.info("Shutdown requested -- closing cleanly")
        self._stop.set()
        for task in list(self._background):
            task.cancel()
        if self.client:
            await self.client.close()


def _matches_order(contract: dict, stake, payout) -> bool:
    """Same stake (to the cent) and same payout (within 1%) as the order we
    placed. Missing values on either side are not used to reject."""
    bp, po = _float_or_none(contract.get("buy_price")), _float_or_none(contract.get("payout"))
    if stake is not None and bp is not None and abs(bp - float(stake)) > 0.005:
        return False
    if payout is not None and po is not None and po > 0 and abs(po - float(payout)) / po > 0.01:
        return False
    return True


def _float_or_none(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _start_health_server(port: int, bot: Bot):
    """`/` -> 200 while Deriv traffic is flowing, 503 once it goes stale
    (previously always 200, even with a dead connection). `/stats` -> JSON
    scorecard: shadow vs executed win rate, predicted vs realized, P&L."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/stats":
                try:
                    body = json.dumps(bot.stats(), indent=2, default=str).encode()
                    code = 200
                except Exception as exc:  # noqa: BLE001
                    body, code = json.dumps({"error": repr(exc)}).encode(), 500
                self._reply(code, "application/json", body)
                return
            ok, reason = bot.is_healthy()
            self._reply(200 if ok else 503, "text/plain", reason.encode())

        def _reply(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence default access logs
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bot.logger.info(f"Health endpoint listening on :{port} (/ and /stats)")
    return server


async def main():
    bot = Bot()
    await bot.start()
    _start_health_server(bot.cfg.health_port, bot)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.shutdown()))
        except NotImplementedError:
            pass  # Windows dev fallback; Railway runs Linux

    await bot.run_forever()


if __name__ == "__main__":
    asyncio.run(main())

