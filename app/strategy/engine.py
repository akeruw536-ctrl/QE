"""
Decision pipeline, run only when the entry signal fires:

    signal (EMA cross + tick + candle, checked by the caller)
      -> regime gate           [ADDED]  range contract needs a range-like market
      -> barrier candidates              XML offsets (+ scaled widths [ADDED])
      -> Monte Carlo + calibration [ADDED]  P(exit inside barriers)
      -> live Deriv proposal             never trade on theoretical payout
      -> edge / EV / uncertainty gate [ADDED]
      -> best accepted candidate (BARRIER_SELECTION: edge or ev) is returned for execution

Each [ADDED] layer can be switched off (USE_REGIME_GATE / USE_EDGE_GATE), and
with both off the bot is the plain XML strategy: signal -> buy at the XML
barriers whenever Deriv quotes a payout above the stake.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.deriv.client import DerivClient
from app.features.stats import detect_regime, log_returns, realized_volatility
from app.models.calibration import CalibrationTracker
from app.models.monte_carlo import MCResult, estimate_probability_in_range
from app.strategy.barriers import BarrierChoice, candidate_barriers
from app.strategy.filters import evaluate

MODEL_VERSION = "qe-v1"
MIN_HISTORY_CANDLES = 70


@dataclass
class ScanOutcome:
    symbol: str
    traded: bool
    trade_row: Optional[dict]
    rejections: List[dict]


def _rejection(symbol: str, duration: int, regime: str, reason: str, **extra) -> dict:
    row = {
        "symbol": symbol, "duration_minutes": duration, "lower_barrier": None, "upper_barrier": None,
        "raw_probability": None, "calibrated_probability": None, "payout": None,
        "implied_probability": None, "edge": None, "expected_value": None,
        "regime": regime, "rejection_reason": reason,
    }
    row.update(extra)
    return row


def _price_candidates(current_price: float, rets: np.ndarray, duration: int, current_vol: float,
                      choices: List[BarrierChoice], n_paths: int) -> List[MCResult]:
    """CPU-bound; called through asyncio.to_thread so the WebSocket pump and
    tick stream keep being serviced while we simulate."""
    return [
        estimate_probability_in_range(
            current_price, rets, duration,
            current_price + ch.lower_offset, current_price + ch.upper_offset,
            current_vol, n_paths,
        )
        for ch in choices
    ]


async def evaluate_signal(
    symbol: str,
    direction: str,
    closes: np.ndarray,
    current_price: float,
    client: DerivClient,
    calibration: CalibrationTracker,
    staking,
    cfg,
    logger,
    consecutive_losses: int = 0,
    stake_multiplier: float = 1.0,
    extra_edge_requirement: float = 0.0,
) -> ScanOutcome:
    duration = cfg.duration_minutes
    if len(closes) < MIN_HISTORY_CANDLES:
        logger.info(f"{symbol}: signal ignored, warming up ({len(closes)} candles)")
        return ScanOutcome(symbol, False, None, [])

    rets = log_returns(closes)
    current_vol = realized_volatility(rets[-20:]) if len(rets) >= 20 else realized_volatility(rets)
    pool_sigma = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    regime = detect_regime(closes)
    regime_name = regime.regime.value

    # ---- regime gate -------------------------------------------------
    if cfg.use_regime_gate and (regime_name not in cfg.range_regimes
                                or regime.confidence < cfg.min_regime_confidence):
        reason = (f"REGIME {regime_name}({regime.confidence:.2f}) NOT RANGE-FRIENDLY"
                  if regime_name not in cfg.range_regimes else "REGIME CONFIDENCE TOO LOW")
        logger.info(f"{symbol} | sig={direction} | regime={regime_name}({regime.confidence:.2f}) "
                    f"vol={current_vol:.5f} -> {reason}")
        return ScanOutcome(symbol, False, None, [_rejection(symbol, duration, regime_name, reason)])

    # ---- barrier candidates ------------------------------------------
    choices = candidate_barriers(
        symbol, cfg.barrier_mode, cfg.fixed_barriers, cfg.barrier_vol_multiple,
        pool_sigma, current_price, duration, cfg.barrier_scales,
    )
    if not choices:
        logger.warning(f"{symbol}: no usable barriers (mode={cfg.barrier_mode}) -- signal skipped")
        return ScanOutcome(symbol, False, None, [])

    mcs = await asyncio.to_thread(
        _price_candidates, current_price, rets, duration, current_vol, choices,
        cfg.monte_carlo.default_paths,
    )

    stake = staking.stake_for(consecutive_losses=consecutive_losses)
    if stake_multiplier != 1.0:
        stake = max(round(stake * stake_multiplier, 2), cfg.staking.minimum_stake)
    calibration_buckets = calibration.snapshot()

    # Edge gate off => only "payout must exceed stake" (and MIN_PAYOUT_MULTIPLIER) remain.
    if cfg.use_edge_gate:
        gate = dict(min_edge=cfg.min_edge, min_ev=cfg.min_ev,
                    max_probability_uncertainty=cfg.max_probability_uncertainty,
                    max_model_disagreement=cfg.max_model_disagreement,
                    extra_edge_requirement=extra_edge_requirement)
    else:
        gate = dict(min_edge=-1.0, min_ev=-1.0, max_probability_uncertainty=1.0,
                    max_model_disagreement=1.0, extra_edge_requirement=0.0)

    rejections: List[dict] = []
    best_decision = None
    best_row = None

    # Quote every width at once (the client rate-limits internally): one
    # round-trip of latency instead of N, and the proposals are all equally
    # fresh when the winner is bought.
    valid = [(ch, mc) for ch, mc in zip(choices, mcs)
             if ch.upper_offset > ch.lower_offset and ch.lower_offset < 0 < ch.upper_offset]
    quotes = await asyncio.gather(*[
        client.request_proposal(symbol=symbol, duration_minutes=duration, stake=stake,
                                lower_barrier=ch.lower_offset, upper_barrier=ch.upper_offset)
        for ch, _ in valid
    ], return_exceptions=True)

    def _score(d) -> float:
        return d.edge if cfg.barrier_selection == "edge" else d.expected_value

    for (ch, mc), proposal in zip(valid, quotes):
        calibrated = calibration.calibrate(mc.probability, buckets=calibration_buckets)
        lower_offset, upper_offset = ch.lower_offset, ch.upper_offset
        if isinstance(proposal, Exception):
            logger.warning(f"{symbol}: proposal failed for x{ch.scale:g} ({proposal!r}) -- skipping candidate")
            continue

        payout = float(proposal.get("payout", 0.0) or 0.0)
        proposal_id = proposal.get("id")
        if payout <= 0 or not proposal_id:
            continue

        # Settle shadow trades against what Deriv actually quoted: the
        # rounded offsets applied to Deriv's own reference spot.
        try:
            entry_spot = float(proposal.get("spot") or current_price)
        except (TypeError, ValueError):
            entry_spot = current_price

        decision = evaluate(
            calibrated_probability=calibrated,
            probability_uncertainty=mc.probability_uncertainty,
            payout=payout, stake=stake,
            min_payout_multiplier=cfg.min_payout_multiplier,
            min_edge=gate["min_edge"], min_ev=gate["min_ev"],
            max_probability_uncertainty=gate["max_probability_uncertainty"],
            extra_edge_requirement=gate["extra_edge_requirement"],
            model_disagreement=mc.model_disagreement,
            max_model_disagreement=gate["max_model_disagreement"],
            regime_confidence=regime.confidence, min_regime_confidence=0.0,
            duration_minutes=duration, edge_duration_scaling=0.0,
        )

        row = {
            "trade_id": str(uuid.uuid4()), "symbol": symbol, "entry_price": entry_spot,
            "duration_minutes": duration,
            "lower_barrier": entry_spot + lower_offset, "upper_barrier": entry_spot + upper_offset,
            "stake": stake, "proposal_id": proposal_id, "payout": payout,
            "payout_multiplier": decision.payout_multiplier,
            "raw_probability": mc.probability, "calibrated_probability": calibrated,
            "implied_probability": decision.implied_probability,
            "edge": decision.edge, "expected_value": decision.expected_value,
            "regime": regime_name, "regime_confidence": regime.confidence,
            "volatility": current_vol, "model_disagreement": mc.model_disagreement,
            "mc_path_count": mc.path_count, "decision_score": decision.expected_value,
            "model_version": MODEL_VERSION,
        }

        logger.info(
            f"{symbol} | sig={direction} | dur={duration}m | off=[{lower_offset:+.2f},{upper_offset:+.2f}] "
            f"({ch.source} x{ch.scale:g}) | regime={regime_name}({regime.confidence:.2f}) | vol={current_vol:.5f} | "
            f"raw_p={mc.probability:.3f} cal_p={calibrated:.3f} implied_p={decision.implied_probability:.3f} "
            f"edge={decision.edge:+.3f} payout_x={decision.payout_multiplier:.3f} ev={decision.expected_value:+.4f} "
            f"stake={stake:.2f} -> {decision.reason}"
        )

        if decision.accept:
            if best_decision is None or _score(decision) > _score(best_decision):
                best_decision, best_row = decision, row
        else:
            rejections.append({**row, "rejection_reason": decision.reason})

    if best_decision is not None and best_row is not None:
        return ScanOutcome(symbol, True, best_row, rejections)
    return ScanOutcome(symbol, False, None, rejections)
