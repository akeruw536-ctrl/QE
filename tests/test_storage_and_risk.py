import os
import tempfile

import numpy as np
from sqlalchemy import create_engine, text

from app.data.storage import Storage
from app.strategy.risk import exposure_allows, utc_day_start


class _NullLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def make_storage(path=None):
    s = Storage(database_url="", sqlite_path=path or _tmp_db(), logger=_NullLogger())
    s.init_schema()
    return s


def _trade(trade_id, **kw):
    row = {"trade_id": trade_id, "symbol": "RDBULL", "stake": 0.35, "payout": 0.55,
           "raw_probability": 0.8, "calibrated_probability": 0.8, "implied_probability": 0.64,
           "shadow": 0, "opened_at": 1_000.0, "expires_at": 1_120.0, "duration_minutes": 2,
           "lower_barrier": 99.0, "upper_barrier": 101.0}
    row.update(kw)
    return row


def test_settle_trade_is_idempotent():
    s = make_storage()
    s.record_trade(_trade("t1"))
    assert s.settle_trade("t1", "WIN", 0.2) is True
    assert s.settle_trade("t1", "LOSS", -0.35) is False  # duplicate is_sold must not overwrite
    assert s.recent_trades()[0]["result"] == "WIN"


def test_migration_adds_new_columns_to_existing_table():
    path = _tmp_db()
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as c:  # a trades table as created by the previous version
        c.execute(text("CREATE TABLE trades (id INTEGER PRIMARY KEY, trade_id VARCHAR, symbol VARCHAR, "
                       "stake FLOAT, shadow INTEGER, result VARCHAR)"))
    s = make_storage(path)
    s.record_trade({"trade_id": "x", "symbol": "R", "stake": 1.0, "shadow": 0, "result": "OPEN",
                    "opened_at": 5.0, "expires_at": 65.0})
    assert s.open_trades(shadow=False)[0]["expires_at"] == 65.0


def test_consecutive_losses_filters_by_mode_and_ignores_open():
    s = make_storage()
    for i, (res, sh) in enumerate([("LOSS", 1), ("LOSS", 1), ("WIN", 0), ("LOSS", 0), ("OPEN", 0)]):
        s.record_trade(_trade(f"t{i}", shadow=sh, result=res))
    assert s.consecutive_losses("RDBULL", shadow=False) == 1
    assert s.consecutive_losses("RDBULL", shadow=True) == 2


def test_stake_since_and_legacy_shadow_retirement():
    s = make_storage()
    s.record_trade(_trade("a", opened_at=100.0, stake=1.0))
    s.record_trade(_trade("b", opened_at=200.0, stake=2.0))
    s.record_trade(_trade("legacy", shadow=1, opened_at=None, expires_at=None))
    assert s.stake_since(150.0, shadow=False) == 2.0
    assert s.retire_legacy_shadow_rows() == 1
    assert s.open_trades(shadow=True) == []


def test_performance_summary():
    s = make_storage()
    s.record_trade(_trade("w", shadow=1, result="WIN", profit_loss=0.2))
    s.record_trade(_trade("l", shadow=1, result="LOSS", profit_loss=-0.35))
    p = s.performance_summary(shadow=True)
    assert p["settled"] == 2 and p["wins"] == 1 and p["win_rate"] == 0.5
    assert p["pnl"] == -0.15 and p["avg_predicted"] == 0.8


def test_exposure_cap():
    assert exposure_allows(49.0, 1.0, 50.0)
    assert not exposure_allows(49.8, 0.35, 50.0)
    assert exposure_allows(1e6, 1.0, 0)  # 0 disables


def test_utc_day_start():
    # 2026-09-28 10:24:00 UTC
    assert utc_day_start(1790591040.0) == 1790553600.0


def test_db_url_normalized_to_installed_driver():
    from app.data.storage import normalize_db_url
    tail = "user:pw@aws-0-eu.pooler.supabase.com:6543/postgres"
    for scheme in ("postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2", "postgresql+asyncpg"):
        assert normalize_db_url(f" {scheme}://{tail} ") == f"postgresql+psycopg2://{tail}"
    assert normalize_db_url("sqlite:///x.db") == "sqlite:///x.db"


def test_numpy_values_never_reach_the_database_driver(tmp_path):
    """Regression: Supabase/psycopg2 crashed on np.float64 ('schema "np" does
    not exist'). Every write must hand the driver plain Python numbers."""
    import logging
    import uuid

    import numpy as np
    from sqlalchemy import event

    from app.data.storage import Storage

    s = Storage("", str(tmp_path / "np.db"), logging.getLogger("t"))
    s.init_schema()
    seen = []

    @event.listens_for(s.engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, params, context, executemany):
        seen.append(params)

    tid = str(uuid.uuid4())
    s.record_trade({"trade_id": tid, "symbol": "R_10", "stake": np.float64(0.35),
                    "duration_minutes": np.int64(4), "edge": np.float32(0.06), "shadow": 0})
    s.settle_trade(tid, "WIN", np.float64(0.12), balance_after=np.float64(10.5), exit_spot=np.float64(1.2))
    s.record_rejection({"symbol": "R_10", "duration_minutes": np.int64(4), "lower_barrier": np.float64(1.0),
                        "edge": np.float64(-0.01), "rejection_reason": "INSUFFICIENT EDGE"})
    s.log_event("INFO", "x", {"v": np.float64(1.5), "n": [np.int64(2)]})

    flat = []
    for p in seen:
        if isinstance(p, dict):
            flat += list(p.values())
        elif isinstance(p, (list, tuple)):
            for q in p:
                flat += list(q.values()) if isinstance(q, dict) else (list(q) if isinstance(q, (list, tuple)) else [q])
    assert flat, "no parameters captured"
    assert not [v for v in flat if isinstance(v, np.generic)]
