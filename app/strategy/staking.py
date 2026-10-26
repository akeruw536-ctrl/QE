"""
Staking. `xml_martingale` is a faithful port of the XML's ladder; `fixed`
keeps the first stake every time.

XML behaviour (procedure "Ultimate Trading Bot Premium"):
  win  -> Stake = First Stake, Loss Count = 0
  loss -> Loss Count += 1; then, if Martingale Level > 0 and Do Martingale
          After > 0 and Loss Count >= Do Martingale After:
              if Loss Count <= Martingale Level:  Stake += Stake * Factor
              else:                               Stake = First Stake ("Martingale Reset")
Note `Stake += Stake * Factor` multiplies the stake by (1 + Factor), so a
factor of 1.5 is a 2.5x ladder. Loss Count is NOT cleared by the reset, so
after the level is exceeded the stake stays at First Stake until a win.

The ladder position is derived from `consecutive_losses` (read from the
trade journal by the caller), not from in-memory state, so it survives a
Railway restart.
"""
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import List

from app.config import StakingConfig


def to_cents(stake: float, minimum: float = 0.0) -> float:
    """Deriv rejects stakes with more than 2 decimals; round DOWN to cents,
    keep >= minimum."""
    cents = float(Decimal(str(stake)).quantize(Decimal("0.01"), rounding=ROUND_DOWN))
    floor = float(Decimal(str(minimum)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)) if minimum else 0.0
    return max(cents, floor)


def _clamp(cfg: StakingConfig, stake: float) -> float:
    return to_cents(max(cfg.minimum_stake, min(stake, cfg.maximum_stake)), cfg.minimum_stake)


class FixedStaking:
    def __init__(self, cfg: StakingConfig):
        self.cfg = cfg

    def stake_for(self, consecutive_losses: int = 0, **_) -> float:
        return _clamp(self.cfg, self.cfg.first_stake)


class XmlMartingaleStaking:
    def __init__(self, cfg: StakingConfig):
        self.cfg = cfg

    def raw_stake(self, consecutive_losses: int) -> float:
        c = self.cfg
        n = max(int(consecutive_losses), 0)
        if c.martingale_level > 0 and c.martingale_start > 0 and n >= c.martingale_start:
            if n <= c.martingale_level:
                return c.first_stake * (1.0 + c.martingale_factor) ** (n - c.martingale_start + 1)
            return c.first_stake  # "Martingale Reset"
        return c.first_stake

    def stake_for(self, consecutive_losses: int = 0, **_) -> float:
        return _clamp(self.cfg, self.raw_stake(consecutive_losses))

    def ladder(self, upto: int | None = None) -> List[float]:
        """Stake after 0, 1, 2 ... losses (after clamping) -- for logs/README."""
        top = (self.cfg.martingale_level + 1) if upto is None else upto
        return [self.stake_for(n) for n in range(0, top + 1)]


def build_staking_engine(cfg: StakingConfig):
    if cfg.mode == "fixed":
        return FixedStaking(cfg)
    if cfg.mode == "xml_martingale":
        return XmlMartingaleStaking(cfg)
    raise ValueError(f"Unknown staking mode: {cfg.mode}")
