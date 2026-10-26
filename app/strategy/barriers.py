"""
Barrier placement for the Ends Between (EXPIRYRANGE) contract.

The XML used fixed point offsets (+41.457 / -41.31 on 1HZ90V). That number is
only meaningful for one symbol at one volatility level (it is almost exactly
1 sigma of a 2-minute move on 1HZ90V at its usual volatility), so this module
supports three modes:

  fixed : the XML offsets from FIXED_BARRIERS
  vol   : multiple * 1-minute sigma * sqrt(duration) * price
  auto  : fixed for symbols that have an entry in FIXED_BARRIERS, else vol

Sigma for `vol` mode is the stable full-history pool sigma, never the noisy
20-bar estimate (same reasoning as ER's candidate search: sizing the barrier
and the Monte Carlo from the same noisy number is circular).

On top of the base width, `barrier_scales` produces a small set of widths to
price live, so the bot can pick whichever has the best EV right now.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class BarrierChoice:
    scale: float
    upper_offset: float   # positive points above the entry spot
    lower_offset: float   # negative points below the entry spot (stored negative)
    source: str           # "fixed" | "vol"


def base_offsets(symbol: str, mode: str, fixed: Dict[str, Tuple[float, float]],
                 vol_multiple: float, pool_sigma: float, price: float,
                 duration_minutes: int) -> Optional[Tuple[float, float, str]]:
    """(upper, lower) POSITIVE magnitudes plus the source, or None if no
    barrier can be derived (e.g. vol mode without a usable sigma)."""
    use_fixed = mode == "fixed" or (mode == "auto" and symbol in fixed)
    if use_fixed:
        if symbol not in fixed:
            return None
        up, lo = fixed[symbol]
        return up, lo, "fixed"
    if pool_sigma is None or pool_sigma <= 0 or price <= 0:
        return None
    half = vol_multiple * pool_sigma * math.sqrt(max(duration_minutes, 1)) * price
    return half, half, "vol"


def candidate_barriers(symbol: str, mode: str, fixed: Dict[str, Tuple[float, float]],
                       vol_multiple: float, pool_sigma: float, price: float,
                       duration_minutes: int, scales: List[float],
                       decimals: int = 2) -> List[BarrierChoice]:
    """Rounded, de-duplicated barrier candidates. Deriv rejects offsets with
    too many decimals (symbol-specific); 2 satisfies every symbol seen so far."""
    base = base_offsets(symbol, mode, fixed, vol_multiple, pool_sigma, price, duration_minutes)
    if base is None:
        return []
    up, lo, source = base
    out: List[BarrierChoice] = []
    seen = set()
    for s in scales:
        u = round(up * s, decimals)
        l = round(lo * s, decimals)
        if u <= 0 or l <= 0:
            continue
        key = (u, l)
        if key in seen:
            continue
        seen.add(key)
        out.append(BarrierChoice(scale=s, upper_offset=u, lower_offset=-l, source=source))
    return out
