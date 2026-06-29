"""Wave 1.1 schema verification — runs the migration (via main import)
then asserts the durable-queue columns + partial indexes exist.

Run from KaizerBackend/:  python scripts/check_wave1_schema.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    import main as _main  # noqa: F401 — import runs _migrate_schema()
    from sqlalchemy import inspect, text

    from database import engine

    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("upload_jobs_v2")}
    need = {"next_attempt_at", "lease_expires_at", "claimed_by",
            "user_id", "priority"}
    missing = sorted(need - cols)
    print("columns present:", sorted(need & cols))
    if missing:
        print("MISSING COLUMNS:", missing)
        return 1

    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            idx = sorted(
                r[0] for r in conn.execute(text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'upload_jobs_v2' "
                    "AND indexname LIKE 'ix_ujv2%'"
                )).fetchall()
            )
        print("partial indexes:", idx)
        expected = {"ix_ujv2_claim", "ix_ujv2_user_active",
                    "ix_ujv2_lease", "ix_ujv2_parked"}
        if not expected.issubset(set(idx)):
            print("MISSING INDEXES:", sorted(expected - set(idx)))
            return 1

        with engine.connect() as conn:
            nulls = conn.execute(text(
                "SELECT count(*) FROM upload_jobs_v2 "
                "WHERE next_attempt_at IS NULL OR priority IS NULL"
            )).scalar()
        print("rows with NULL next_attempt_at/priority:", nulls)
        if nulls:
            return 1

    print("SCHEMA OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
