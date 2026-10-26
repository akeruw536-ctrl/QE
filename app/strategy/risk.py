"""
Pure risk-limit helpers (no I/O) so they can be unit tested in isolation.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_day_start(now: float | None = None) -> float:
    """Epoch seconds of the most recent 00:00 UTC."""
    dt = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def exposure_allows(staked_today: float, next_stake: float, max_daily_exposure: float) -> bool:
    """True if placing `next_stake` keeps total stake for the UTC day within
    the cap. A cap of 0 or less disables the check."""
    if max_daily_exposure <= 0:
        return True
    return staked_today + next_stake <= max_daily_exposure + 1e-9


def session_status(pnl: float, target_profit: float, stop_loss: float) -> str:
    """The XML's end-of-trade check: 'TARGET' once profit >= Target Profit,
    'STOP' once profit <= -|Stop Loss|, else 'OK'. A limit of 0 (or less)
    disables that side."""
    if target_profit > 0 and pnl >= target_profit:
        return "TARGET"
    if stop_loss > 0 and pnl <= -abs(stop_loss):
        return "STOP"
    return "OK"
