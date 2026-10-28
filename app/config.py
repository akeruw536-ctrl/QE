"""
Central configuration. Everything secret or environment-specific comes from
env vars so the same image runs unmodified on Railway.

Defaults reproduce the original Deriv Bot (Blockly) strategy exactly where
the XML specified a value; every deviation or addition is marked
"[XML]" (taken from the XML) or "[ADDED]" (an intelligence layer / safety
rail that the XML did not have) so it is obvious what you are changing.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [s.strip() for s in val.split(",") if s.strip()]


def _env_float_list(name: str, default: List[float]) -> List[float]:
    val = os.getenv(name)
    if not val:
        return default
    return [float(s) for s in val.split(",") if s.strip()]


def parse_fixed_barriers(raw: str) -> Dict[str, Tuple[float, float]]:
    """'1HZ90V:41.457:41.31,R_100:1.2:1.2' -> {'1HZ90V': (41.457, 41.31), ...}
    Each entry is SYMBOL:UPPER_OFFSET:LOWER_OFFSET, both positive point
    distances from the entry spot (the XML used +41.457 / -41.31)."""
    out: Dict[str, Tuple[float, float]] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        sym, up, lo = part.split(":")
        out[sym.strip()] = (abs(float(up)), abs(float(lo)))
    return out


@dataclass
class StakingConfig:
    # "xml_martingale" (the XML's ladder) or "fixed"
    mode: str = field(default_factory=lambda: os.getenv("STAKING_MODE", "xml_martingale"))
    # [XML] First Stake = 1
    first_stake: float = float(os.getenv("FIRST_STAKE", "1.0"))
    minimum_stake: float = float(os.getenv("STAKING_MINIMUM_STAKE", "0.35"))
    # [ADDED] hard ceiling applied AFTER the ladder. The XML had none: with
    # first stake 1 / factor 1.5 / level 5 its top rung is 97.66. Set this to
    # 100 to reproduce the XML exactly.
    maximum_stake: float = float(os.getenv("STAKING_MAXIMUM_STAKE", "25.0"))
    # [XML] Martingale Factor = 1.5: each loss does Stake += Stake * factor,
    # i.e. the stake is multiplied by (1 + factor) = 2.5.
    martingale_factor: float = float(os.getenv("MARTINGALE_FACTOR", "1.5"))
    # [XML] Martingale Level = 5: escalate through the first 5 losses, then
    # drop back to the first stake until a win resets the streak.
    martingale_level: int = int(os.getenv("MARTINGALE_LEVEL", "5"))
    # [XML] Do Martingale After = 1: start escalating from the 1st loss.
    martingale_start: int = int(os.getenv("MARTINGALE_START", "1"))


@dataclass
class MonteCarloConfig:
    minimum_paths: int = 10_000
    default_paths: int = int(os.getenv("MC_PATHS", "25000"))
    maximum_paths: int = 100_000
    near_threshold_band: float = 0.03


@dataclass
class Config:
    # --- Deriv connection ---
    deriv_api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    deriv_app_id: str = field(default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    deriv_endpoint: str = field(
        default_factory=lambda: os.getenv("DERIV_WS_URL", "wss://ws.derivws.com/websockets/v3")
    )
    # DEMO is the only default that is safe; LIVE must be requested explicitly.
    account_mode: str = field(default_factory=lambda: os.getenv("DERIV_ACCOUNT_MODE", "DEMO").upper())

    # --- persistence ---
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", "data/quantumends.db"))
    # [ADDED] own Postgres schema by default so QE never collides with ER's
    # tables when both share one Supabase project. DB_SCHEMA= (empty) -> public.
    db_schema: str = field(default_factory=lambda: os.getenv("DB_SCHEMA", "qe").strip())

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # --- market / contract ---
    # [XML] 1HZ90V, EXPIRYRANGE ("Ends Between"), 2 minute duration
    symbols: List[str] = field(default_factory=lambda: _env_list("SYMBOLS", ["1HZ90V"]))
    duration_minutes: int = int(os.getenv("DURATION_MINUTES", "2"))

    # --- entry signal ---
    # [XML] EMA(7) vs EMA(14) over the tick stream
    ema_fast: int = int(os.getenv("EMA_FAST", "7"))
    ema_slow: int = int(os.getenv("EMA_SLOW", "14"))
    # [XML] only the bearish cross ("Crossing DOWN"). "down", "up" or "both".
    signal_directions: str = field(default_factory=lambda: os.getenv("SIGNAL_DIRECTIONS", "down").lower())
    # [XML] last tick must be a fall (check_direction == fall) / candle black
    require_tick_confirmation: bool = field(default_factory=lambda: _env_bool("REQUIRE_TICK_CONFIRMATION", True))
    require_candle_confirmation: bool = field(default_factory=lambda: _env_bool("REQUIRE_CANDLE_CONFIRMATION", True))
    # [XML] 60s candles. "closed" = most recent COMPLETED candle (default),
    # "forming" = the candle still being built.
    candle_granularity_seconds: int = int(os.getenv("CANDLE_GRANULARITY_SECONDS", "60"))
    candle_reference: str = field(default_factory=lambda: os.getenv("CANDLE_REFERENCE", "closed").lower())
    tick_buffer: int = int(os.getenv("TICK_BUFFER", "600"))

    # --- barriers ---
    # "fixed": XML points offsets, "vol": volatility-scaled, "auto": fixed for
    # symbols listed in FIXED_BARRIERS, vol-scaled for everything else.
    barrier_mode: str = field(default_factory=lambda: os.getenv("BARRIER_MODE", "auto").lower())
    # [XML] +41.457 / -41.31 on 1HZ90V
    fixed_barriers: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: parse_fixed_barriers(os.getenv("FIXED_BARRIERS", "1HZ90V:41.457:41.31"))
    )
    # vol mode half-width = multiple * 1-minute sigma * sqrt(duration) * price.
    # At 1HZ90V's usual sigma, 1.0 reproduces the XML's ~41 points.
    barrier_vol_multiple: float = float(os.getenv("BARRIER_VOL_MULTIPLE", "1.0"))
    # [ADDED] intelligence layer: also price barrier widths scaled around the
    # base (XML) width and take the best live-priced EV. "1.0" = XML only.
    # Widths are multiples of the base (XML / vol) width, so the search is
    # adaptive: every signal prices all of them live and keeps the best.
    barrier_scales: List[float] = field(
        default_factory=lambda: _env_float_list("BARRIER_SCALES", [0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0])
    )
    # What "best" means among the ACCEPTED candidates: "edge" (model probability
    # minus Deriv's implied probability) or "ev" (edge x payout, i.e. expected
    # profit per stake). They can disagree: a tight barrier has a big payout
    # and a small edge, a wide one the reverse.
    barrier_selection: str = field(default_factory=lambda: os.getenv("BARRIER_SELECTION", "edge").lower())

    # --- intelligence gates ([ADDED], each can be switched off) ---
    use_regime_gate: bool = field(default_factory=lambda: _env_bool("USE_REGIME_GATE", True))
    range_regimes: List[str] = field(
        default_factory=lambda: _env_list(
            "RANGE_REGIMES",
            ["LOW_VOLATILITY_RANGE", "NORMAL_RANGE", "VOLATILITY_CONTRACTION", "MEAN_REVERTING"],
        )
    )
    min_regime_confidence: float = float(os.getenv("MIN_REGIME_CONFIDENCE", "0.55"))

    use_edge_gate: bool = field(default_factory=lambda: _env_bool("USE_EDGE_GATE", True))
    min_payout_multiplier: float = float(os.getenv("MIN_PAYOUT_MULTIPLIER", "1.0"))
    min_edge: float = float(os.getenv("MIN_EDGE", "0.02"))
    min_ev: float = float(os.getenv("MIN_EV", "0.0"))
    max_probability_uncertainty: float = float(os.getenv("MAX_PROB_UNCERTAINTY", "0.06"))
    max_model_disagreement: float = float(os.getenv("MAX_MODEL_DISAGREEMENT", "0.05"))

    # --- session limits ---
    # [XML] Target Profit = 62, Stop Loss = 200. 0 disables either one.
    target_profit: float = float(os.getenv("TARGET_PROFIT", "62"))
    stop_loss: float = float(os.getenv("STOP_LOSS", "200"))
    # [ADDED] The XML simply stopped. A Railway service must not idle forever,
    # so by default the halt lasts until 00:00 UTC and trading resumes with a
    # fresh session. Set SESSION_HALT=false to ignore target/stop entirely.
    session_halt: bool = field(default_factory=lambda: _env_bool("SESSION_HALT", True))

    # --- risk ---
    # [XML] one contract at a time (trade_again only after the result)
    max_concurrent_contracts: int = int(os.getenv("MAX_CONCURRENT_CONTRACTS", "1"))
    max_concurrent_per_symbol: int = 1
    # 0 = disabled. Stake committed per UTC day.
    max_daily_exposure: float = float(os.getenv("MAX_DAILY_EXPOSURE", "0"))
    caution_cooldown_seconds: int = int(os.getenv("CAUTION_COOLDOWN_SECONDS", "120"))

    # --- runtime ---
    shadow_mode: bool = field(default_factory=lambda: _env_bool("SHADOW_MODE", True))
    # housekeeping cadence only -- signals are tick-driven, not polled
    scan_interval_seconds: float = float(os.getenv("SCAN_INTERVAL_SECONDS", "10"))
    health_port: int = int(os.getenv("PORT", "8080"))

    staking: StakingConfig = field(default_factory=StakingConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)

    def is_live(self) -> bool:
        return self.account_mode == "LIVE"

    def validate(self) -> List[str]:
        problems = []
        if self.is_live() and not self.deriv_api_token:
            problems.append("LIVE mode requested but DERIV_API_TOKEN is not set.")
        if self.staking.mode not in ("xml_martingale", "fixed"):
            problems.append("STAKING_MODE must be 'xml_martingale' or 'fixed'.")
        if self.staking.first_stake < self.staking.minimum_stake:
            problems.append("FIRST_STAKE below STAKING_MINIMUM_STAKE.")
        if self.staking.maximum_stake < self.staking.first_stake:
            problems.append("STAKING_MAXIMUM_STAKE below FIRST_STAKE.")
        if self.staking.martingale_factor < 0:
            problems.append("MARTINGALE_FACTOR must be >= 0.")
        if self.min_payout_multiplier < 1.0:
            problems.append("MIN_PAYOUT_MULTIPLIER must be >= 1.0.")
        if self.ema_fast >= self.ema_slow:
            problems.append("EMA_FAST must be smaller than EMA_SLOW.")
        if self.signal_directions not in ("down", "up", "both"):
            problems.append("SIGNAL_DIRECTIONS must be down, up or both.")
        if self.candle_reference not in ("closed", "forming"):
            problems.append("CANDLE_REFERENCE must be closed or forming.")
        if self.barrier_mode not in ("auto", "fixed", "vol"):
            problems.append("BARRIER_MODE must be auto, fixed or vol.")
        if self.barrier_selection not in ("edge", "ev"):
            problems.append("BARRIER_SELECTION must be edge or ev.")
        if not self.barrier_scales or any(s <= 0 for s in self.barrier_scales):
            problems.append("BARRIER_SCALES must be positive numbers.")
        if self.duration_minutes < 1:
            problems.append("DURATION_MINUTES must be >= 1.")
        if self.db_schema and not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", self.db_schema):
            problems.append("DB_SCHEMA must be lowercase letters, digits and _ (e.g. qe).")
        return problems


def load_config() -> Config:
    return Config()
