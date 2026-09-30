import time
from pathlib import Path

from loguru import logger

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.db"


def get_conn() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


def init_db() -> None:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fragments ("
            "hash TEXT PRIMARY KEY, source TEXT, text TEXT, created_at INTEGER)"
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_fragment(h: str) -> str | None:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        row = conn.execute("SELECT text FROM fragments WHERE hash=?", (h,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def save_fragment_cache(h: str, source: str, text: str) -> None:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT OR REPLACE INTO fragments(hash, source, text, created_at) VALUES(?,?,?,?)",
            (h, source, text, int(time.time())),
        )
        conn.commit()
    except Exception as exc:
        logger.warning(f"Не удалось сохранить кэш фрагмента: {exc}")
    finally:
        conn.close()
