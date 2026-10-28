"""
Move the trade journal and calibration from the old (ephemeral SQLite) bot
into a persistent database, so a redeploy no longer loses them.

Step 1 -- on the OLD running Railway service, BEFORE redeploying:
    railway ssh
    <paste the command printed by:  python tools/import_journal.py --print-export-command>
  Copy everything between BEGIN and END into a file, e.g. journal.b64.

Step 2 -- from your own machine (needs: pip install sqlalchemy psycopg2-binary):
    set DATABASE_URL=<the Postgres PUBLIC url>      (PowerShell: $env:DATABASE_URL="...")
    python tools/import_journal.py journal.b64 --schema qe

Safe to re-run: trades already present (same trade_id) are skipped, and the
import refuses a target that already holds trades unless --force is given.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import DateTime, insert, select, update  # noqa: E402

from app.data.storage import Storage, calibration_buckets, trades  # noqa: E402

# Runs inside the old container (stdlib only). Tables copied: trades and
# calibration_buckets -- the rejected-signal log is large and not needed.
EXPORT_ONE_LINER = (
    "python3 -c 'import sqlite3,json,gzip,base64;"
    "c=sqlite3.connect(\"/app/data/quantumends.db\");c.row_factory=sqlite3.Row;"
    "d={t:[dict(r) for r in c.execute(\"select * from \"+t+\" order by 1\")] for t in (\"trades\",\"calibration_buckets\")};"
    "print(\"BEGIN\");print(base64.encodebytes(gzip.compress(json.dumps(d).encode())).decode());print(\"END\")'"
)


def decode_export(text: str) -> dict:
    """Accepts the raw terminal paste (BEGIN/END markers and line wraps are fine)."""
    if "BEGIN" in text and "END" in text:
        text = text.split("BEGIN", 1)[1].rsplit("END", 1)[0]
    blob = "".join(text.split())
    return json.loads(gzip.decompress(base64.b64decode(blob)).decode())


def _coerce(table, row: dict) -> dict:
    out = {}
    for col in table.columns:
        if col.name == "id" or col.name not in row:
            continue
        v = row[col.name]
        if v is not None and isinstance(col.type, DateTime) and isinstance(v, str):
            v = datetime.fromisoformat(v)
            if v.tzinfo is None:
                v = v.replace(tzinfo=timezone.utc)     # SQLite stored naive UTC
        out[col.name] = v
    return out


def import_journal(storage: Storage, payload: dict, force: bool = False) -> dict:
    storage.init_schema()
    with storage.engine.begin() as conn:
        existing = {r[0] for r in conn.execute(select(trades.c.trade_id)).all()}
        if existing and not force:
            raise SystemExit(f"Target already holds {len(existing)} trades; refusing to import "
                             f"(use --force to merge, existing trade_ids are skipped).")
        added = skipped = 0
        for row in sorted(payload.get("trades", []), key=lambda r: r.get("id") or 0):
            if row.get("trade_id") in existing:
                skipped += 1
                continue
            conn.execute(insert(trades).values(**_coerce(trades, row)))
            added += 1
        buckets = 0
        for row in payload.get("calibration_buckets", []):
            values = _coerce(calibration_buckets, row)
            hit = conn.execute(select(calibration_buckets.c.bucket)
                               .where(calibration_buckets.c.bucket == row["bucket"])).first()
            if hit:
                conn.execute(update(calibration_buckets)
                             .where(calibration_buckets.c.bucket == row["bucket"]).values(**values))
            else:
                conn.execute(insert(calibration_buckets).values(**values))
            buckets += 1
    return {"trades_added": added, "trades_skipped": skipped, "calibration_buckets": buckets}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_file", nargs="?", help="file holding the BEGIN/END export")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    ap.add_argument("--schema", default=os.getenv("DB_SCHEMA", "qe"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print-export-command", action="store_true")
    a = ap.parse_args()
    if a.print_export_command:
        print(EXPORT_ONE_LINER)
        return
    if not a.export_file or not a.database_url:
        ap.error("need the export file and a DATABASE_URL (the Postgres PUBLIC url)")
    payload = decode_export(open(a.export_file, encoding="utf-8").read())
    log = logging.getLogger("import"); logging.basicConfig(level=logging.INFO)
    storage = Storage(a.database_url, "data/_unused.db", log, schema=a.schema)
    print(f"Export holds {len(payload.get('trades', []))} trades, "
          f"{len(payload.get('calibration_buckets', []))} calibration buckets.")
    print(import_journal(storage, payload, force=a.force))


if __name__ == "__main__":
    main()
