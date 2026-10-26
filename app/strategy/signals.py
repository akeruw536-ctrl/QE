"""
Entry signal -- a direct port of the XML's `before_purchase` block:

    EMA(7) < EMA(14)             (latest tick)
    AND EMA(7)* > EMA(14)*       (one tick earlier)   -> a fresh cross DOWN
    AND last tick direction == fall
    AND the candle at index 1 (60s) is black (close < open)
    -> purchase EXPIRYRANGE

The XML runs this on every tick, so the cross is an *event*. Here a
TickSignalDetector is fed every tick and reports the cross on the exact tick
it happens; the candle check is applied separately (it needs candle data).

EMA matches the Deriv Bot block: seeded with the SMA of the first `period`
values, then the usual recursion with k = 2 / (period + 1).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

DOWN = "DOWN"
UP = "UP"


def ema_series(values: Sequence[float], period: int) -> List[float]:
    """EMA list (length len(values) - period + 1), SMA-seeded."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return []
    k = 2.0 / (period + 1.0)
    ema = sum(values[:period]) / period
    out = [ema]
    for v in values[period:]:
        ema = ema + k * (v - ema)
        out.append(ema)
    return out


class TickSignalDetector:
    """Incremental EMA(fast)/EMA(slow) cross detector over a tick stream."""

    def __init__(self, fast: int = 7, slow: int = 14, require_tick_confirmation: bool = True):
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast, self.slow = fast, slow
        self.require_tick_confirmation = require_tick_confirmation
        self._kf = 2.0 / (fast + 1.0)
        self._ks = 2.0 / (slow + 1.0)
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.last_price: Optional[float] = None
        self.ready = False
        self._warm: List[float] = []

    def seed(self, prices: Sequence[float]) -> None:
        """Warm the state from tick history so the first live tick can already
        be evaluated. Needs at least `slow` + 1 prices to be ready."""
        f = ema_series(prices, self.fast)
        s = ema_series(prices, self.slow)
        if not f or not s:
            self.ready = False
            return
        self.ema_fast, self.ema_slow = f[-1], s[-1]
        self.last_price = float(prices[-1])
        self.ready = True

    def feed(self, price: float) -> Optional[str]:
        """update() once warm; until then collect live ticks and seed from them
        (used when tick history could not be fetched at startup)."""
        if self.ready:
            return self.update(price)
        self._warm.append(float(price))
        self._warm = self._warm[-(self.slow * 3):]
        if len(self._warm) >= self.slow + 1:
            self.seed(self._warm)
            self._warm = []
        return None

    def update(self, price: float) -> Optional[str]:
        """Feed one tick. Returns DOWN / UP on the tick a cross happens (with
        tick confirmation applied if enabled), otherwise None."""
        price = float(price)
        if not self.ready or self.ema_fast is None or self.ema_slow is None or self.last_price is None:
            return None
        prev_fast, prev_slow, prev_price = self.ema_fast, self.ema_slow, self.last_price
        self.ema_fast = prev_fast + self._kf * (price - prev_fast)
        self.ema_slow = prev_slow + self._ks * (price - prev_slow)
        self.last_price = price

        crossed_down = self.ema_fast < self.ema_slow and prev_fast > prev_slow
        crossed_up = self.ema_fast > self.ema_slow and prev_fast < prev_slow
        if crossed_down and (not self.require_tick_confirmation or price < prev_price):
            return DOWN
        if crossed_up and (not self.require_tick_confirmation or price > prev_price):
            return UP
        return None


# ------------------------------------------------------------------ candles
def last_closed_candle(candles: Sequence[dict], now: float, granularity_seconds: int) -> Optional[dict]:
    """Most recent COMPLETED candle. Deriv's history ends with the candle that
    is still forming (its epoch is the minute's open time)."""
    for c in reversed(candles):
        if c["epoch"] + granularity_seconds <= now:
            return c
    return None


def reference_candle(candles: Sequence[dict], now: float, granularity_seconds: int,
                     reference: str = "closed") -> Optional[dict]:
    if not candles:
        return None
    if reference == "forming":
        return candles[-1]
    return last_closed_candle(candles, now, granularity_seconds)


def is_black(candle: Optional[dict]) -> bool:
    """Deriv Bot's `is_candle_black`: the candle closed below its open."""
    return candle is not None and float(candle["close"]) < float(candle["open"])


def is_white(candle: Optional[dict]) -> bool:
    return candle is not None and float(candle["close"]) > float(candle["open"])


def candle_confirms(direction: str, candle: Optional[dict]) -> bool:
    return is_black(candle) if direction == DOWN else is_white(candle)


def direction_allowed(direction: str, allowed: str) -> bool:
    return allowed == "both" or allowed == direction.lower()
