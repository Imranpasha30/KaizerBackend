import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./kaizer.db")
# Railway uses postgres:// but SQLAlchemy needs postgresql://
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )

    # Desktop runs TWO writers on this file at once — the serve process and
    # the spawned render orchestrator (its own engine, same URL). Default
    # sqlite journaling locks the whole DB per write and a concurrent writer
    # gets "database is locked" immediately. WAL lets readers and one writer
    # overlap across processes; busy_timeout makes a second writer WAIT (up
    # to 15s) instead of failing. Applied on every new connection (WAL is
    # sticky on the file, busy_timeout is per-connection); safe/no-op for
    # in-memory DBs and single-process use.
    from sqlalchemy import event as _sa_event

    @_sa_event.listens_for(engine, "connect")
    def _sqlite_tune(dbapi_conn, _record):
        try:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=15000")
            cur.close()
        except Exception:
            pass  # never block a connection over a pragma
else:
    # Wave 1.7 of the enterprise push — the default pool (5 + 10
    # overflow) starved under load: dispatch holds a session for the
    # full branding+upload duration, so N workers × concurrency must
    # fit in the pool. Sizing rule per process:
    #   pool_size ≥ KAIZER_WORKER_CONCURRENCY + web traffic + crons.
    # pool_pre_ping heals dropped connections (Railway/Cloud LBs kill
    # idle TCP); pool_recycle stays under common 30-min idle reapers;
    # statement_timeout stops one runaway query from wedging the app.
    _connect_args: dict = {}
    _stmt_ms = _int_env("KAIZER_DB_STATEMENT_TIMEOUT_MS", 60_000)
    if DATABASE_URL.startswith("postgresql") and _stmt_ms > 0:
        _connect_args["options"] = f"-c statement_timeout={_stmt_ms}"
    engine = create_engine(
        DATABASE_URL,
        pool_size=_int_env("KAIZER_DB_POOL_SIZE", 10),
        max_overflow=_int_env("KAIZER_DB_MAX_OVERFLOW", 10),
        pool_pre_ping=True,
        pool_recycle=_int_env("KAIZER_DB_POOL_RECYCLE_SECONDS", 1800),
        pool_timeout=_int_env("KAIZER_DB_POOL_TIMEOUT_SECONDS", 30),
        connect_args=_connect_args,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
