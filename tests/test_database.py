import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.database as db


@pytest.fixture()
def db_path(monkeypatch, tmp_path):
    """Реальная SQLite-база в tmp: проверяем настоящий UPSERT."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "cache.db")
    db.init_db()
    return tmp_path / "cache.db"


class TestSaveJobKeepsCreatedAt:
    """INSERT OR REPLACE затирал created_at при каждом сохранении: задача,
    запущенная 12.10 15:01, в истории значилась «создана 13.10 17:24»."""

    def test_created_at_survives_updates(self, db_path):
        db.save_job("job1", {"state": "queued", "progress": 0})
        first_created = self._row(db_path)["created_at"]

        # Несколько сохранений подряд (тики прогресса) — с паузой ≥1с не нужна:
        # created_at не должен меняться даже в одну и ту же секунду.
        for i in range(5):
            db.save_job("job1", {"state": "working", "progress": i})
            row = self._row(db_path)
            assert row["created_at"] == first_created, "created_at перезаписан сохранением"
        assert row["updated_at"] >= first_created

    def test_updated_at_reflects_last_save(self, db_path, monkeypatch):
        db.save_job("job1", {"state": "queued"})
        t0 = self._row(db_path)["updated_at"]

        times = iter([t0 + 100])
        monkeypatch.setattr(db.time, "time", lambda: next(times))
        db.save_job("job1", {"state": "working", "progress": 50})

        row = self._row(db_path)
        assert row["updated_at"] == t0 + 100
        assert row["created_at"] == t0  # дата создания не тронута

    def test_data_overwritten_on_update(self, db_path):
        db.save_job("job1", {"state": "queued", "progress": 0})
        db.save_job("job1", {"state": "working", "progress": 77})
        import json

        data = json.loads(self._row(db_path)["data"])
        assert data["state"] == "working"
        assert data["progress"] == 77

    @staticmethod
    def _row(db_path):
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(
                "SELECT created_at, updated_at, data FROM jobs WHERE id=?", ("job1",)
            ).fetchone()
        finally:
            conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
