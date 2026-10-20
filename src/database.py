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
        # Статистика скорости фаз: (единиц обработано, суммарные секунды фазы).
        # Используется для предварительной оценки времени следующих запусков.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS phase_timings ("
            "kind TEXT PRIMARY KEY, units INTEGER NOT NULL, seconds REAL NOT NULL,"
            " updated_at INTEGER)"
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


def save_job(job_id: str, data: dict) -> None:
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        now = int(time.time())
        # UPSERT: created_at остаётся от ПЕРВОЙ вставки. INSERT OR REPLACE
        # затирал его текущим временем при каждом сохранении — из-за этого
        # в истории задач «создана» показывала последний тик прогресса,
        # сортировка по created_at перемешивала список, а delete_old_jobs
        # не мог удалить долгоживущую задачу (её «дата создания» обновлялась).
        conn.execute(
            "INSERT INTO jobs(id, data, created_at, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
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
        # Check if jobs table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        )
        if not cursor.fetchone():
            return None
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


def mark_interrupted_jobs() -> int:
    """Помечает задачи в состояниях queued/working как interrupted.

    Вызывается при старте сервера: потоки обработки этих задач умерли
    вместе с прошлым процессом, поэтому честно показываем пользователю,
    что задача прервана. Возвращает количество помеченных задач.
    """
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    marked = 0
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        )
        if not cursor.fetchone():
            return 0
        conn.execute("PRAGMA journal_mode=WAL")
        rows = conn.execute("SELECT id, data FROM jobs").fetchall()
        now = int(time.time())
        for job_id, data_raw in rows:
            try:
                data = json.loads(data_raw)
            except (ValueError, TypeError):
                continue
            if data.get("state") not in ("queued", "working"):
                continue
            data["state"] = "interrupted"
            conn.execute(
                "UPDATE jobs SET data=?, updated_at=? WHERE id=?",
                (json.dumps(data, ensure_ascii=False), now, job_id),
            )
            marked += 1
        if marked:
            conn.commit()
    except Exception as exc:
        logger.warning(f"Не удалось пометить прерванные задачи: {exc}")
    finally:
        conn.close()
    if marked:
        logger.info(f"Помечено как прерванные: {marked} задач(а)")
    return marked


def record_phase_timing(kind: str, units: int, seconds: float) -> None:
    """Накопить статистику фазы: units обработанных единиц за seconds секунд.

    seconds — СТЕННОЕ время фазы (с учётом параллелизма и лимитов NIM):
    среднее seconds/units затем служит прогнозом скорости для оценки времени.
    """
    if units <= 0 or seconds <= 0:
        return
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO phase_timings(kind, units, seconds, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(kind) DO UPDATE SET "
            "units=units+excluded.units, seconds=seconds+excluded.seconds,"
            " updated_at=excluded.updated_at",
            (kind, units, seconds, int(time.time())),
        )
        conn.commit()
    except Exception as exc:
        logger.warning(f"Не удалось сохранить статистику фазы {kind}: {exc}")
    finally:
        conn.close()


def get_phase_timings() -> dict[str, tuple[int, float]]:
    """Накопленная статистика: kind -> (units, seconds). Пустая таблица не ошибка."""
    import sqlite3

    conn = sqlite3.connect(str(get_conn()))
    result: dict[str, tuple[int, float]] = {}
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='phase_timings'"
        )
        if not cursor.fetchone():
            return result
        for kind, units, seconds in conn.execute(
            "SELECT kind, units, seconds FROM phase_timings WHERE units > 0"
        ):
            result[kind] = (units, seconds)
    except Exception as exc:
        logger.warning(f"Не удалось прочитать статистику фаз: {exc}")
    finally:
        conn.close()
    return result
