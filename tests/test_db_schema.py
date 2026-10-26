"""DB_SCHEMA: ER's tables live in their own Postgres schema, so it can share a
Supabase project with another bot that has tables of the same names.
Needs a real Postgres: set TEST_PG_URL (skipped otherwise)."""
import logging
import os
import time
import uuid

import pytest
from sqlalchemy import create_engine, text

from app.data.storage import Storage

PG = os.getenv("TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG, reason="TEST_PG_URL not set")


@pytest.fixture
def clean_pg():
    eng = create_engine(PG)
    with eng.begin() as c:
        for s in ("er_t1", "er_t2"):
            c.execute(text(f"DROP SCHEMA IF EXISTS {s} CASCADE"))
        c.execute(text("DROP TABLE IF EXISTS public.trades"))
        # stand-in for Reversal-System's own trades table (different columns)
        c.execute(text("CREATE TABLE public.trades (id serial primary key, signal_id text NOT NULL, direction text)"))
        c.execute(text("INSERT INTO public.trades (signal_id, direction) VALUES ('rev-1', 'CALL')"))
    yield eng
    with eng.begin() as c:
        for s in ("er_t1", "er_t2"):
            c.execute(text(f"DROP SCHEMA IF EXISTS {s} CASCADE"))
        c.execute(text("DROP TABLE IF EXISTS public.trades"))


def _trade(symbol="R_10", result_stake=0.35):
    return {"trade_id": str(uuid.uuid4()), "symbol": symbol, "stake": result_stake, "shadow": 0,
            "duration_minutes": 2, "opened_at": time.time(), "expires_at": time.time() + 120}


def test_two_bots_and_reversal_share_one_database(clean_pg):
    log = logging.getLogger("t")
    a = Storage(PG, "", log, schema="er_t1")
    b = Storage(PG, "", log, schema="er_t2")
    a.init_schema()
    b.init_schema()
    t = _trade()
    a.record_trade(t)
    assert a.settle_trade(t["trade_id"], "LOSS", -0.35)
    b.record_trade(_trade("RDBEAR"))

    assert a.consecutive_losses("R_10", shadow=False) == 1
    assert b.consecutive_losses("R_10", shadow=False) == 0          # b never sees a's trades
    with clean_pg.begin() as c:
        assert c.execute(text("SELECT count(*) FROM er_t1.trades")).scalar() == 1
        assert c.execute(text("SELECT count(*) FROM er_t2.trades")).scalar() == 1
        # the other bot's table is untouched: same rows, no ER columns added
        cols = {r[0] for r in c.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='trades'"))}
        assert cols == {"id", "signal_id", "direction"}
        assert c.execute(text("SELECT count(*) FROM public.trades")).scalar() == 1


def test_missing_columns_migrate_inside_the_schema(clean_pg):
    log = logging.getLogger("t")
    s = Storage(PG, "", log, schema="er_t1")
    s.init_schema()
    with clean_pg.begin() as c:
        c.execute(text("ALTER TABLE er_t1.trades DROP COLUMN exit_spot"))
    s.init_schema()                                                    # restart: re-adds it in er_t1
    with clean_pg.begin() as c:
        n = c.execute(text("SELECT count(*) FROM information_schema.columns "
                           "WHERE table_schema='er_t1' AND table_name='trades' AND column_name='exit_spot'")).scalar()
    assert n == 1
