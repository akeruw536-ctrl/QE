"""
Persistence layer.

Uses SQLAlchemy Core so the same schema/queries work against SQLite
(local dev, or Railway fallback if DATABASE_URL isn't set) and Postgres
(Railway production, via DATABASE_URL). Tables are created automatically
on startup if missing -- no separate migration step required to get a
working deployment.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from sqlalchemy import (
    JSON, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, func as sa_func,
    delete, inspect, insert, select, text, update
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import func

metadata = MetaData()

trades = Table(
    "trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("trade_id", String, unique=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("symbol", String),
    Column("entry_price", Float),
    Column("duration_minutes", Integer),
    Column("lower_barrier", Float),
    Column("upper_barrier", Float),
    Column("stake", Float),
    Column("proposal_id", String),
    Column("contract_id", String),
    Column("payout", Float),
    Column("payout_multiplier", Float),
    Column("raw_probability", Float),
    Column("calibrated_probability", Float),
    Column("implied_probability", Float),
    Column("edge", Float),
    Column("expected_value", Float),
    Column("regime", String),
    Column("regime_confidence", Float),
    Column("volatility", Float),
    Column("model_disagreement", Float),
    Column("mc_path_count", Integer),
    Column("decision_score", Float),
    Column("shadow", Integer, default=0),
    Column("result", String, default="OPEN"),
    Column("profit_loss", Float, default=0.0),
    Column("balance_before", Float),
    Column("balance_after", Float),
    Column("model_version", String),
    # epoch seconds -- portable across SQLite/Postgres (unlike the tz-aware
    # created_at, which SQLite returns naive). Used for daily exposure and
    # for knowing when a shadow trade has expired and can be resolved.
    Column("opened_at", Float),
    Column("expires_at", Float),
    Column("exit_spot", Float),
    Column("signal_logit_mean", Float),
)

rejected_signals = Table(
    "rejected_signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("symbol", String),
    Column("duration_minutes", Integer),
    Column("lower_barrier", Float),
    Column("upper_barrier", Float),
    Column("raw_probability", Float),
    Column("calibrated_probability", Float),
    Column("payout", Float),
    Column("implied_probability", Float),
    Column("edge", Float),
    Column("expected_value", Float),
    Column("regime", String),
    Column("rejection_reason", String),
)

calibration_buckets = Table(
    "calibration_buckets", metadata,
    Column("bucket", String, primary_key=True),  # e.g. "0.70-0.75"
    Column("n_predictions", Integer, default=0),
    Column("n_wins", Integer, default=0),
    Column("updated_at", DateTime(timezone=True), onupdate=func.now(), server_default=func.now()),
)

# Outcomes of priced barrier candidates the bot did NOT trade ("probes"). Every
# signal prices several barrier widths but buys at most one; the rest used to
# be thrown away. Resolving them against the real expiry tick gives the
# calibration layer many more labelled outcomes -- including for the barrier
# widths and regimes it never trades, which removes the selection bias of
# learning only from its own trades.
calibration_samples = Table(
    "calibration_samples", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("symbol", String),
    Column("regime", String),
    Column("raw_probability", Float),
    Column("lower_barrier", Float),
    Column("upper_barrier", Float),
    Column("opened_at", Float),
    Column("expires_at", Float),
    Column("resolved", Integer, default=0),     # 0 open, 1 resolved, 2 gave up
    Column("won", Integer),
    Column("exit_spot", Float),
    # mean logit of the raw probability over ALL candidates priced for the same
    # signal -- lets calibration separate "how calm is the market" (between
    # signals) from "how wide is this barrier" (within a signal)
    Column("signal_logit_mean", Float),
)

system_events = Table(
    "system_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("level", String),
    Column("message", String),
    Column("context", JSON),
)


def _plain(v: Any) -> Any:
    """numpy scalars -> plain Python values, recursively through dicts/lists.

    psycopg2 renders a numpy float64 as the literal text np.float64(1.23),
    which Postgres parses as a function in a schema called "np":
    'schema "np" does not exist'. SQLite accepted them, so this only
    surfaced when ER moved to Supabase. Every write goes through this."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_plain(x) for x in v)
    return v


def normalize_db_url(url: str) -> str:
    """Any Postgres URL form -> the driver this bot ships with (psycopg2).

    A connection string copied from another service may name a different
    driver -- Reversal-System's uses postgresql+psycopg:// (psycopg 3), which
    is not installed here and crashed startup with "No module named
    'psycopg'". Heroku/Supabase-style postgres:// is also accepted."""
    url = url.strip()
    scheme, sep, rest = url.partition("://")
    if sep and scheme.split("+")[0] in ("postgres", "postgresql"):
        return "postgresql+psycopg2://" + rest
    return url


class Storage:
    def __init__(self, database_url: str, sqlite_path: str, logger, schema: str = ""):
        self.logger = logger
        self.schema = None
        self.sqlite_path = sqlite_path
        if database_url:
            try:
                self._connect_postgres(normalize_db_url(database_url), schema)
            except OperationalError as exc:
                # A database that is down, paused (Supabase free tier pauses
                # idle projects) or briefly unreachable through the pooler used
                # to raise straight out of Bot.__init__ and crash-loop the
                # container until Railway gave up. Retry first (done above),
                # then degrade to local SQLite so the bot keeps running.
                # Set DB_FALLBACK_TO_SQLITE=false to fail hard instead.
                first_line = str(exc.orig if getattr(exc, "orig", None) else exc).strip().splitlines()[0]
                if os.getenv("DB_FALLBACK_TO_SQLITE", "true").strip().lower() not in ("1", "true", "yes", "on"):
                    raise
                self.logger.error(
                    f"Storage: DATABASE_URL unreachable after retries ({first_line}). "
                    f"FALLING BACK to local SQLite at {sqlite_path} -- data is EPHEMERAL on "
                    f"Railway and will be lost on redeploy. Fix the database and redeploy."
                )
                self._use_sqlite()
        else:
            self._use_sqlite()

    def _use_sqlite(self) -> None:
        self.schema = None
        directory = os.path.dirname(self.sqlite_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.sqlite_path}", connect_args={"check_same_thread": False})
        self.logger.info(f"Storage: using local SQLite at {self.sqlite_path}")

    def _connect_postgres(self, url: str, schema: str) -> None:
        """Connect with bounded retry/backoff. Raises the last OperationalError
        if every attempt fails (caller decides whether to fall back).

        Tunables (env): DB_CONNECT_RETRIES (default 4), DB_CONNECT_TIMEOUT
        seconds per attempt (default 10)."""
        attempts = max(1, int(os.getenv("DB_CONNECT_RETRIES", "4")))
        timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))
        backoff = 2.0
        last: Optional[OperationalError] = None
        for attempt in range(1, attempts + 1):
            engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=300,  # poolers drop idle server connections
                connect_args={"connect_timeout": timeout},
            )
            try:
                with engine.begin() as conn:
                    if schema:
                        # every table this bot touches lives in its own schema, so it can
                        # share a database with other bots that use the same table names
                        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
                    else:
                        conn.execute(text("SELECT 1"))
            except OperationalError as exc:
                engine.dispose()
                last = exc
                reason = str(exc.orig if getattr(exc, "orig", None) else exc).strip().splitlines()[0]
                self.logger.warning(f"Storage: database connect attempt {attempt}/{attempts} failed: {reason}")
                if attempt < attempts:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                continue
            if schema:
                engine = engine.execution_options(schema_translate_map={None: schema})
                self.schema = schema
            self.engine = engine
            self.logger.info("Storage: connected to external database via DATABASE_URL"
                             + (f" (schema {schema})" if schema else ""))
            return
        assert last is not None
        raise last

    def init_schema(self) -> None:
        metadata.create_all(self.engine)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """create_all() never alters an existing table, so a column added to
        the schema in code would make every insert fail against a database
        created by an older version (e.g. your existing Railway Postgres).
        Adds any missing nullable columns in place. Additive only -- never
        drops or changes existing columns."""
        insp = inspect(self.engine)
        existing_tables = set(insp.get_table_names(schema=self.schema))
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in insp.get_columns(table.name, schema=self.schema)}
            for col in table.columns:
                if col.name in have or col.primary_key:
                    continue
                col_type = col.type.compile(dialect=self.engine.dialect)
                with self.engine.begin() as conn:
                    target = f'"{self.schema}".{table.name}' if self.schema else table.name
                    conn.execute(text(f'ALTER TABLE {target} ADD COLUMN {col.name} {col_type}'))
                self.logger.info(f"Storage: migrated {table.name}.{col.name} ({col_type})")

    # ------------------------------------------------------------- trades
    def record_trade(self, row: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(trades).values(**_plain(row)))

    def settle_trade(self, trade_id: str, result: str, profit_loss: float,
                     balance_after: Optional[float] = None, exit_spot: Optional[float] = None) -> bool:
        """Idempotent: only transitions a trade out of OPEN once. Returns True
        if this call did the settling, False if it was already settled.

        Deriv can deliver the final is_sold proposal_open_contract update more
        than once (initial snapshot + stream, or again after a reconnect
        resubscribe). Callers use the return value to make sure the outcome
        is fed to calibration exactly once."""
        with self.engine.begin() as conn:
            res = conn.execute(
                update(trades)
                .where(trades.c.trade_id == trade_id)
                .where(trades.c.result == "OPEN")
                .values(**_plain(dict(result=result, profit_loss=profit_loss, balance_after=balance_after,
                                     exit_spot=exit_spot)))
            )
        return (res.rowcount or 0) > 0

    def open_trades(self, shadow: bool) -> List[dict]:
        """Trades still marked OPEN. Used on startup to re-attach settlement
        tracking to live contracts opened before a restart, and by the shadow
        resolver to find hypothetical trades that have expired."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(trades).where(trades.c.result == "OPEN").where(trades.c.shadow == (1 if shadow else 0))
            ).mappings().all()
        return [dict(r) for r in rows]

    def retire_legacy_shadow_rows(self) -> int:
        """Shadow rows written before expires_at existed can never be
        resolved; left OPEN they would inflate the open count forever. Marks
        them UNRESOLVED (kept for the record, excluded from win-rate stats)."""
        with self.engine.begin() as conn:
            res = conn.execute(
                update(trades)
                .where(trades.c.shadow == 1).where(trades.c.result == "OPEN")
                .where(trades.c.expires_at.is_(None))
                .values(result="UNRESOLVED")
            )
        return res.rowcount or 0

    def stake_since(self, since_epoch: float, shadow: bool) -> float:
        """Total stake committed since `since_epoch` -- the basis for
        MAX_DAILY_EXPOSURE."""
        with self.engine.begin() as conn:
            total = conn.execute(
                select(sa_func.coalesce(sa_func.sum(trades.c.stake), 0.0))
                .where(trades.c.opened_at >= since_epoch)
                .where(trades.c.shadow == (1 if shadow else 0))
            ).scalar()
        return float(total or 0.0)

    def pnl_since(self, since_epoch: float, shadow: bool) -> float:
        """Net profit/loss of trades OPENED since `since_epoch` that have
        settled (WIN/LOSS). Basis for the session target-profit / stop-loss
        -- derived from the journal, so it survives a Railway restart."""
        with self.engine.begin() as conn:
            total = conn.execute(
                select(sa_func.coalesce(sa_func.sum(trades.c.profit_loss), 0.0))
                .where(trades.c.opened_at >= since_epoch)
                .where(trades.c.shadow == (1 if shadow else 0))
                .where(trades.c.result.in_(["WIN", "LOSS"]))
            ).scalar()
        return float(total or 0.0)

    def performance_summary(self, shadow: bool) -> Dict[str, Any]:
        """Settled-trade scorecard for one mode. The key calibration check is
        win_rate vs avg_predicted: if the model is honest these track each
        other; a persistent gap is the model overstating its own edge."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(trades.c.result, trades.c.profit_loss, trades.c.stake,
                       trades.c.calibrated_probability, trades.c.implied_probability)
                .where(trades.c.shadow == (1 if shadow else 0))
                .where(trades.c.result.in_(["WIN", "LOSS"]))
            ).all()
            open_count = conn.execute(
                select(sa_func.count()).select_from(trades)
                .where(trades.c.shadow == (1 if shadow else 0)).where(trades.c.result == "OPEN")
            ).scalar() or 0
        n = len(rows)
        wins = sum(1 for r in rows if r[0] == "WIN")
        staked = sum(float(r[2] or 0) for r in rows)
        pnl = sum(float(r[1] or 0) for r in rows)
        def avg(i):
            vals = [float(r[i]) for r in rows if r[i] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        return {
            "mode": "shadow" if shadow else "executed",
            "settled": n, "open": int(open_count), "wins": wins,
            "win_rate": round(wins / n, 4) if n else None,
            "avg_predicted": avg(3), "avg_implied": avg(4),
            "total_staked": round(staked, 2), "pnl": round(pnl, 2),
            "roi": round(pnl / staked, 4) if staked else None,
        }

    def ladder_stats(self, shadow: bool, martingale_start: int = 1, martingale_level: int = 5) -> Dict[str, Any]:
        """Risk side of the journal: highest stake, highest martingale level
        actually staked, longest losing streak and worst drawdown.

        "Level n" = the stake placed after n straight losses while the ladder
        was escalating (martingale_start <= n <= martingale_level). Level 0 is
        the first stake (including the reset after the ladder is exhausted).
        """
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(trades.c.id, trades.c.trade_id, trades.c.symbol, trades.c.opened_at, trades.c.stake,
                       trades.c.result, trades.c.profit_loss)
                .where(trades.c.shadow == (1 if shadow else 0))
                .order_by(trades.c.id.asc())
            ).all()

        streak = max_streak = max_level = 0
        cum = peak = max_dd = 0.0
        level_counts: Dict[int, int] = {}
        level_of: Dict[str, int] = {}
        for r in rows:
            if r[5] not in ("WIN", "LOSS"):
                continue
            n = streak
            level = n if (martingale_level > 0 and martingale_start > 0
                          and martingale_start <= n <= martingale_level) else 0
            level_of[r[1]] = level
            level_counts[level] = level_counts.get(level, 0) + 1
            max_level = max(max_level, level)
            cum += float(r[6] or 0.0)
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
            if r[5] == "WIN":
                streak = 0
            else:
                streak += 1
                max_streak = max(max_streak, streak)

        biggest = max((r for r in rows if r[4] is not None), key=lambda r: float(r[4]), default=None)
        max_stake = None
        if biggest is not None:
            max_stake = {
                "stake": float(biggest[4]), "symbol": biggest[2], "result": biggest[5],
                "level": level_of.get(biggest[1]),
                "opened_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(biggest[3])) if biggest[3] else None,
            }
        return {
            "max_stake": max_stake,
            "max_martingale_level": max_level,
            "max_loss_streak": max_streak,
            "current_loss_streak": streak,
            "max_drawdown": round(max_dd, 2),
            "trades_per_level": {str(k): v for k, v in sorted(level_counts.items())},
        }

    def breakdown(self, shadow: bool) -> Dict[str, Any]:
        """Settled trades split by the CLAIMED edge at entry, the regime, the
        stake size (= martingale rung) and the barrier width. Answers "which
        trades make the money?" -- in particular whether low-claimed-edge
        trades are really zero-edge (the model is optimistic by several points)."""
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(trades.c.edge, trades.c.regime, trades.c.stake, trades.c.result,
                       trades.c.profit_loss, trades.c.upper_barrier, trades.c.lower_barrier,
                       trades.c.entry_price)
                .where(trades.c.shadow == (1 if shadow else 0))
                .where(trades.c.result.in_(["WIN", "LOSS"]))
            ).all()

        def label(value, cuts, names):
            if value is None:
                return "unknown"
            for cut, name in zip(cuts, names):
                if value < cut:
                    return name
            return names[-1]

        def width_pct(r):
            try:
                return 100.0 * (float(r[5]) - float(r[6])) / float(r[7])
            except (TypeError, ValueError, ZeroDivisionError):
                return None

        def group(key_fn, order=None):
            acc: Dict[str, list] = {}
            for r in rows:
                acc.setdefault(key_fn(r), []).append(r)
            out = []
            for name in (order or sorted(acc)):
                grp = acc.get(name)
                if not grp:
                    continue
                n = len(grp)
                wins = sum(1 for r in grp if r[3] == "WIN")
                staked = sum(float(r[2] or 0) for r in grp)
                pnl = sum(float(r[4] or 0) for r in grp)
                out.append({"bucket": name, "n": n, "win_rate": round(wins / n, 3),
                            "pnl": round(pnl, 2), "roi": round(pnl / staked, 3) if staked else None})
            return out

        edge_names = ["<0.03", "0.03-0.06", "0.06-0.10", ">=0.10"]
        stake_names = ["<=1.0", "1.0-3.0", "3.0-10", ">10"]
        width_names = ["<0.3%", "0.3-0.4%", "0.4-0.5%", ">=0.5%"]
        return {
            "by_claimed_edge": group(lambda r: label(r[0], [0.03, 0.06, 0.10, 9e9], edge_names), edge_names + ["unknown"]),
            "by_regime": group(lambda r: r[1] or "unknown"),
            "by_stake": group(lambda r: label(r[2], [1.0000001, 3.0, 10.0, 9e9], stake_names), stake_names + ["unknown"]),
            "by_barrier_width": group(lambda r: label(width_pct(r), [0.3, 0.4, 0.5, 9e9], width_names), width_names + ["unknown"]),
        }

    # ------------------------------------------------------------- probes
    def record_probes(self, rows: list) -> int:
        rows = [r for r in rows if r.get("raw_probability") is not None]
        if not rows:
            return 0
        with self.engine.begin() as conn:
            conn.execute(insert(calibration_samples), [_plain(r) for r in rows])
        return len(rows)

    def open_probes(self, due_before: float, limit: int = 500) -> list:
        with self.engine.begin() as conn:
            res = conn.execute(
                select(calibration_samples)
                .where(calibration_samples.c.resolved == 0)
                .where(calibration_samples.c.expires_at <= due_before)
                .order_by(calibration_samples.c.expires_at.asc()).limit(limit)
            )
            return [dict(r._mapping) for r in res.all()]

    def resolve_probe(self, probe_id: int, won: bool, exit_spot: float) -> bool:
        with self.engine.begin() as conn:
            res = conn.execute(
                update(calibration_samples)
                .where(calibration_samples.c.id == probe_id)
                .where(calibration_samples.c.resolved == 0)
                .values(resolved=1, won=1 if won else 0, exit_spot=exit_spot)
            )
            return res.rowcount > 0

    def abandon_probe(self, probe_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(calibration_samples)
                         .where(calibration_samples.c.id == probe_id)
                         .where(calibration_samples.c.resolved == 0).values(resolved=2))

    def prune_probes(self, older_than_epoch: float) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(delete(calibration_samples)
                               .where(calibration_samples.c.resolved > 0)
                               .where(calibration_samples.c.opened_at < older_than_epoch))
            return res.rowcount or 0

    def probe_counts(self) -> Dict[str, int]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(calibration_samples.c.resolved, sa_func.count())
                                .group_by(calibration_samples.c.resolved)).all()
        c = {int(r[0] or 0): int(r[1]) for r in rows}
        return {"open": c.get(0, 0), "resolved": c.get(1, 0), "abandoned": c.get(2, 0)}

    def calibration_dataset(self, max_rows: int = 20000) -> list:
        """Everything the calibrator may learn from: settled trades (live and
        shadow) plus resolved probes. Newest first, capped."""
        out = []
        with self.engine.begin() as conn:
            for r in conn.execute(
                select(trades.c.raw_probability, trades.c.result, trades.c.regime, trades.c.opened_at,
                       trades.c.signal_logit_mean)
                .where(trades.c.result.in_(["WIN", "LOSS"]))
                .where(trades.c.raw_probability.is_not(None))
                .order_by(trades.c.id.desc()).limit(max_rows)
            ).all():
                out.append({"raw": float(r[0]), "won": 1 if r[1] == "WIN" else 0, "regime": r[2],
                            "opened_at": r[3], "source": "trade", "signal_mean": r[4]})
            for r in conn.execute(
                select(calibration_samples.c.raw_probability, calibration_samples.c.won,
                       calibration_samples.c.regime, calibration_samples.c.opened_at,
                       calibration_samples.c.symbol, calibration_samples.c.signal_logit_mean)
                .where(calibration_samples.c.resolved == 1)
                .order_by(calibration_samples.c.id.desc()).limit(max_rows)
            ).all():
                out.append({"raw": float(r[0]), "won": int(r[1] or 0), "regime": r[2],
                            "opened_at": r[3], "source": "probe", "symbol": r[4], "signal_mean": r[5]})
        return out

    def recent_trades(self, limit: int = 50) -> List[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(trades).order_by(trades.c.id.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]

    def consecutive_losses(self, symbol: Optional[str] = None, shadow: Optional[bool] = None) -> int:
        """shadow filters to one mode so a shadow losing streak doesn't size
        the first real stakes after switching SHADOW_MODE off (and vice
        versa)."""
        with self.engine.begin() as conn:
            q = select(trades.c.result).where(trades.c.result.in_(["WIN", "LOSS"]))
            if symbol:
                q = q.where(trades.c.symbol == symbol)
            if shadow is not None:
                q = q.where(trades.c.shadow == (1 if shadow else 0))
            q = q.order_by(trades.c.id.desc()).limit(20)
            rows = [r[0] for r in conn.execute(q).all()]
        count = 0
        for r in rows:
            if r == "LOSS":
                count += 1
            else:
                break
        return count

    # -------------------------------------------------------- rejections
    def record_rejection(self, row: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(rejected_signals).values(**_plain(row)))

    # -------------------------------------------------------- calibration
    def update_calibration(self, bucket: str, won: bool) -> None:
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(calibration_buckets).where(calibration_buckets.c.bucket == bucket)
            ).mappings().first()
            if existing:
                conn.execute(
                    update(calibration_buckets)
                    .where(calibration_buckets.c.bucket == bucket)
                    .values(
                        n_predictions=existing["n_predictions"] + 1,
                        n_wins=existing["n_wins"] + (1 if won else 0),
                    )
                )
            else:
                conn.execute(
                    insert(calibration_buckets).values(
                        bucket=bucket, n_predictions=1, n_wins=1 if won else 0
                    )
                )

    def get_calibration(self) -> Dict[str, tuple]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(calibration_buckets)).mappings().all()
        return {r["bucket"]: (r["n_predictions"], r["n_wins"]) for r in rows}

    # ------------------------------------------------------------- events
    def log_event(self, level: str, message: str, context: Optional[dict] = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(system_events).values(level=level, message=message, context=_plain(context or {})))
