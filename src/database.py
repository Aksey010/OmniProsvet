import json
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
        conn.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "id TEXT PRIMARY KEY, data TEXT, created_at INTEGER, updated_at INTEGER)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")
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


def save_job(job_id: str, data: dict) -> None:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        now = int(time.time())
        conn.execute(
            "INSERT OR REPLACE INTO jobs(id, data, created_at, updated_at) VALUES(?,?,?,?)",
            (job_id, json.dumps(data, ensure_ascii=False), now, now),
        )
        conn.commit()
    except Exception as exc:
        logger.warning(f"Не удалось сохранить задачу {job_id}: {exc}")
    finally:
        conn.close()


def get_job(job_id: str) -> dict | None:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        row = conn.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
    finally:
        conn.close()
    if row:
        return json.loads(row[0])
    return None


def list_jobs(limit: int = 50) -> list[dict]:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        # Check if jobs table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        )
        if not cursor.fetchone():
            return []
        rows = conn.execute(
            "SELECT id, data, created_at FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()
    jobs = []
    for row in rows:
        data = json.loads(row[1])
        jobs.append({
            "id": row[0],
            "state": data.get("state", "unknown"),
            "progress": data.get("progress", 0),
            "mode": data.get("mode", ""),
            "language": data.get("language", ""),
            "created_at": time.strftime("%d.%m.%Y %H:%M", time.localtime(row[2])),
        })
    return jobs
    return jobs


def delete_job(job_id: str) -> None:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        conn.commit()
    except Exception as exc:
        logger.warning(f"Не удалось удалить задачу {job_id}: {exc}")
    finally:
        conn.close()


def delete_old_jobs(days: int = 7) -> None:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        cutoff = int(time.time()) - days * 86400
        conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
        conn.commit()
    except Exception as exc:
        logger.warning(f"Не удалось удалить старые задачи: {exc}")
    finally:
        conn.close()
