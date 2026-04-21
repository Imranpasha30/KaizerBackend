"""One-shot SQLite -> Postgres migration.

Drops the stale Postgres schema, creates fresh tables via SQLAlchemy's
create_all (which reflects the current models.py), then copies every row
from the local kaizer.db preserving primary keys.  Finally resets every
sequence so the next auto-increment starts above the highest imported id.

Run once.  After verifying, flip DATABASE_URL in .env to the Postgres url and
delete (or rename) kaizer.db.

We intentionally DO NOT import main.py here — main.py runs _migrate_schema()
at module import time, which contains one SQLite-specific statement
(INSERT OR IGNORE) that Postgres rejects syntactically.  Since create_all
builds the schema directly from models.py, every column the runtime migrator
would add is already present in the fresh Postgres schema.
"""
from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

SQLITE_PATH = BACKEND / "kaizer.db"
PG_URL = os.environ["PG_URL"]

from sqlalchemy import create_engine, text, Boolean  # noqa: E402
import models  # noqa: E402 — registers Base.metadata


def _coerce(value, sqla_type):
    """SQLite stores booleans as 0/1 ints; Postgres enforces real bool type."""
    if value is None:
        return None
    if isinstance(sqla_type, Boolean):
        return bool(value)
    return value


def log(msg: str) -> None:
    print(f"[migrate] {msg}", flush=True)


def drop_all_public_tables(engine) -> None:
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )).fetchall()
        for (name,) in rows:
            conn.execute(text(f'DROP TABLE IF EXISTS "{name}" CASCADE'))
        log(f"dropped {len(rows)} existing public tables")


def create_schema(engine) -> None:
    models.Base.metadata.create_all(bind=engine)
    log(f"create_all done — {len(models.Base.metadata.sorted_tables)} tables")


def copy_rows(sqlite_conn: sqlite3.Connection, pg_engine) -> None:
    sqlite_conn.row_factory = sqlite3.Row

    with pg_engine.begin() as pg:
        pg.execute(text("SET CONSTRAINTS ALL DEFERRED"))

        for table in models.Base.metadata.sorted_tables:
            name = table.name
            try:
                src_rows = sqlite_conn.execute(f'SELECT * FROM "{name}"').fetchall()
            except sqlite3.OperationalError:
                log(f"skip {name} — not present in sqlite")
                continue
            if not src_rows:
                log(f"copy {name}: 0 rows")
                continue

            cols = [c.name for c in table.columns]
            col_types = {c.name: c.type for c in table.columns}
            col_list = ", ".join(f'"{c}"' for c in cols)
            placeholders = ", ".join([f":{c}" for c in cols])
            stmt = text(f'INSERT INTO "{name}" ({col_list}) VALUES ({placeholders})')

            inserted = 0
            for row in src_rows:
                rec = dict(row)
                for c in cols:
                    rec.setdefault(c, None)
                rec = {k: _coerce(v, col_types[k]) for k, v in rec.items() if k in cols}
                pg.execute(stmt, rec)
                inserted += 1
            log(f"copy {name}: {inserted} rows")


def reset_sequences(engine) -> None:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT c.table_name, c.column_name,
                   pg_get_serial_sequence(c.table_name, c.column_name) AS seq
            FROM information_schema.columns c
            WHERE c.table_schema='public'
              AND pg_get_serial_sequence(c.table_name, c.column_name) IS NOT NULL
        """)).fetchall()
        for t, col, seq in rows:
            max_id = conn.execute(text(f'SELECT COALESCE(MAX("{col}"), 0) FROM "{t}"')).scalar()
            if max_id and max_id > 0:
                conn.execute(text(f"SELECT setval('{seq}', :v, true)"), {"v": max_id})
            else:
                conn.execute(text(f"SELECT setval('{seq}', 1, false)"))
        log(f"reset {len(rows)} sequences")


def verify(sqlite_conn, pg_engine) -> None:
    log("verifying row counts:")
    mismatches = []
    with pg_engine.connect() as pg:
        for table in models.Base.metadata.sorted_tables:
            name = table.name
            try:
                s = sqlite_conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            except sqlite3.OperationalError:
                s = 0
            p = pg.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar()
            mark = "OK " if s == p else "!! "
            log(f"  {mark}{name}: sqlite={s} postgres={p}")
            if s != p:
                mismatches.append((name, s, p))
    if mismatches:
        raise RuntimeError(f"row-count mismatch: {mismatches}")


def run() -> None:
    if not SQLITE_PATH.exists():
        raise SystemExit(f"sqlite file not found: {SQLITE_PATH}")

    pg_engine = create_engine(PG_URL, pool_pre_ping=True)
    sqlite_conn = sqlite3.connect(str(SQLITE_PATH))

    log(f"source: {SQLITE_PATH}")
    log(f"target: {PG_URL.split('@')[-1]}")

    drop_all_public_tables(pg_engine)
    create_schema(pg_engine)
    copy_rows(sqlite_conn, pg_engine)
    reset_sequences(pg_engine)
    verify(sqlite_conn, pg_engine)

    sqlite_conn.close()
    pg_engine.dispose()
    log("DONE. Row counts match.")


if __name__ == "__main__":
    run()
