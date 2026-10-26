from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("quantumends")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def format_no_trade(reason: str, **details) -> str:
    extra = " ".join(f"{k}={v}" for k, v in details.items())
    return f"NO TRADE — {reason} | {extra}" if extra else f"NO TRADE — {reason}"
