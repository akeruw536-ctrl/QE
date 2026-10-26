"""
Explicit, restart-safe state machine.

Design rule (spec section 8/30): every state that restricts trading is
temporary and carries its own automatic recovery condition. There is no
"stopped forever" state anywhere in this machine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(Enum):
    INITIALIZING = "INITIALIZING"
    CONNECTING = "CONNECTING"
    SYNCING_DATA = "SYNCING_DATA"
    WARMING_UP = "WARMING_UP"
    READY = "READY"
    SCANNING = "SCANNING"
    SIMULATING = "SIMULATING"
    EVALUATING = "EVALUATING"
    PROPOSAL_PENDING = "PROPOSAL_PENDING"
    EXECUTING = "EXECUTING"
    POSITION_OPEN = "POSITION_OPEN"
    SETTLING = "SETTLING"
    LEARNING = "LEARNING"
    CAUTION = "CAUTION"
    RECOVERING = "RECOVERING"
    RECONNECTING = "RECONNECTING"


@dataclass
class CautionState:
    """A temporary, self-expiring restriction. Never permanent."""

    reason: str
    started_at: float = field(default_factory=time.time)
    cooldown_seconds: float = 120.0
    edge_penalty: float = 0.0   # added on top of min_edge while active
    stake_multiplier: float = 1.0

    def expired(self) -> bool:
        return (time.time() - self.started_at) >= self.cooldown_seconds


class StateMachine:
    def __init__(self, logger):
        self.state = State.INITIALIZING
        self.logger = logger
        self._active_cautions: dict[str, CautionState] = {}

    def transition(self, new_state: State, reason: str = "") -> None:
        old = self.state
        self.state = new_state
        self.logger.info(f"STATE {old.value} -> {new_state.value}" + (f" ({reason})" if reason else ""))

    def enter_caution(self, key: str, reason: str, cooldown_seconds: float,
                       edge_penalty: float = 0.02, stake_multiplier: float = 1.0) -> None:
        self._active_cautions[key] = CautionState(
            reason=reason, cooldown_seconds=cooldown_seconds,
            edge_penalty=edge_penalty, stake_multiplier=stake_multiplier,
        )
        self.logger.warning(f"CAUTION[{key}] entered: {reason} (cooldown={cooldown_seconds}s)")

    def reassess_cautions(self) -> None:
        """Call every loop tick. Automatically clears expired restrictions."""
        expired_keys = [k for k, c in self._active_cautions.items() if c.expired()]
        for k in expired_keys:
            self.logger.info(f"CAUTION[{k}] cleared automatically (cooldown elapsed)")
            del self._active_cautions[k]

    def is_symbol_cautioned(self, symbol: str) -> bool:
        return symbol in self._active_cautions

    def extra_edge_requirement(self) -> float:
        """Sum of active caution penalties -> raises the bar, never blocks forever."""
        return sum(c.edge_penalty for c in self._active_cautions.values())

    def stake_multiplier(self) -> float:
        if not self._active_cautions:
            return 1.0
        return min(c.stake_multiplier for c in self._active_cautions.values())

    def active_cautions(self) -> dict[str, CautionState]:
        return dict(self._active_cautions)
