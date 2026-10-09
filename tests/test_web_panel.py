import io
import sys
import zipfile
from pathlib import Path

# Add src and repo root to path (web_panel живёт в корне)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import src.database as db


@pytest.fixture()
def panel(monkeypatch):
    """web_panel с подменённой БД задач (в памяти, per-test).

    Патчим src.database ДО первого импорта web_panel: его module-level
    init_db()/delete_old_jobs()/mark_interrupted_jobs() не должны трогать
    реальный data/cache.db в тестах. get_job/save_job перепривязываем
    на каждом тесте к свежему словарю.
    """
    jobs: dict = {}

    monkeypatch.setattr(db, "init_db", lambda: None)
    monkeypatch.setattr(db, "delete_old_jobs", lambda days=7: None)
    monkeypatch.setattr(db, "mark_interrupted_jobs", lambda: None)

    import web_panel

    monkeypatch.setattr(web_panel, "get_job", lambda jid: jobs.get(jid))
    monkeypatch.setattr(
        web_panel, "save_job", lambda jid, data: jobs.__setitem__(jid, data)
    )
    yield web_panel, jobs


def _make(p: Path, content: str = "текст") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestFilesEndpoint:
    def test_returns_only_job_files(self, panel, tmp_path):
        wp, jobs = panel
        f1 = _make(tmp_path / "a.transcript.txt")
        f2 = _make(tmp_path / "b.pdf.txt")
        f3 = _make(tmp_path / "c.images.txt")  # файл ДРУГОЙ задачи
        jobs["jobA"] = {
            "state": "done",
            "files": [
                {"name": f1.name, "path": str(f1)},
                {"name": f2.name, "path": str(f2)},
            ],
        }
        jobs["jobB"] = {"state": "done", "files": [{"name": f3.name, "path": str(f3)}]}

        client = wp.app.test_client()
        r = client.get("/api/files/jobA")
        assert r.status_code == 200
        names = [f["name"] for f in r.get_json()["files"]]
        assert names == [f1.name, f2.name]  # файлы jobB не подтекли

    def test_legacy_job_without_files_returns_empty(self, panel):
        """Старые задачи (до записи списка файлов) не вываливают всё за всё время."""
        wp, jobs = panel
        jobs["old"] = {"state": "done"}
        client = wp.app.test_client()
        r = client.get("/api/files/old")
        assert r.status_code == 200
        assert r.get_json()["files"] == []

    def test_unknown_job_404(self, panel):
        wp, _ = panel
        client = wp.app.test_client()
        assert client.get("/api/files/nope").status_code == 404


class TestZipEndpoint:
    def test_zip_contains_exactly_job_files(self, panel, tmp_path):
        wp, jobs = panel
        report = _make(tmp_path / "report_x.txt", "ОТЧЁТ")
        tr = _make(tmp_path / "v.transcript.txt", "[00:00] текст")
        sm = _make(tmp_path / "book.pdf.txt", "конспект")
        jobs["jobZ"] = {
            "state": "done",
            "files": [
                {"name": report.name, "path": str(report)},
                {"name": tr.name, "path": str(tr)},
                {"name": sm.name, "path": str(sm)},
            ],
        }
        client = wp.app.test_client()
        r = client.get("/api/zip/jobZ")
        assert r.status_code == 200
        assert r.mimetype == "application/zip"
        zf = zipfile.ZipFile(io.BytesIO(r.data))
        assert sorted(zf.namelist()) == [
            "report_x.txt",
            "summaries/book.pdf.txt",
            "transcripts/v.transcript.txt",
        ]

    def test_job_without_files_404(self, panel):
        wp, jobs = panel
        jobs["empty"] = {"state": "done"}
        client = wp.app.test_client()
        assert client.get("/api/zip/empty").status_code == 404

    def test_unknown_job_404(self, panel):
        wp, _ = panel
        client = wp.app.test_client()
        assert client.get("/api/zip/nope").status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
