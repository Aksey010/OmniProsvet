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


class TestPhaseStatus:
    """Отдельная надпись на каждую параллельную фазу.

    Раньше единое поле current/file_index/file_total перезаписывалось всеми
    фазами по очереди (транскрибация 36/4000 ↔ PDF 5/50) — фазы «перебивали»
    друг друга. Теперь статус хранится по фазам: phase_status[phase].
    """

    def _run_with_events(self, panel, monkeypatch, events):
        wp, jobs = panel
        snapshots: list[dict] = []

        def fake_emit(event_name, data=None, *a, **k):
            if event_name == "job_progress" and isinstance(data, dict):
                snapshots.append(dict(data))

        def fake_pipeline(paths, **kwargs):
            p = kwargs["progress"]
            for e in events:
                p(e)
            return {"transcripts": [], "report": None, "image_reports": []}

        monkeypatch.setattr(wp, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(wp.socketio, "emit", fake_emit)

        jobs["j1"] = {
            "state": "queued", "progress": 0, "lines": [], "phase_status": {},
            "error": "", "report": None,
        }
        wp.run_job("j1", ["x"], "full", False, "small", "auto")
        # Финальное состояние job очищено при завершении (state=done) —
        # статусы фаз смотрим в последнем снимке прогресса.
        return snapshots[-1]["phase_status"], jobs["j1"]

    def test_parallel_phases_have_separate_labels(self, panel, monkeypatch):
        """Две параллельные фазы сосуществуют, не затирая друг друга."""
        st, _ = self._run_with_events(panel, monkeypatch, [
            {"phase": "transcribe", "weight": 45},
            {"phase": "pdf", "weight": 15},
            {"file": "a.mp3", "file_index": 36, "file_total": 4000, "phase": "transcribe"},
            {"file": "doc.pdf", "file_index": 5, "file_total": 50, "phase": "pdf"},
        ])
        assert st["transcribe"]["index"] == 36 and st["transcribe"]["total"] == 4000
        assert st["transcribe"]["current"] == "a.mp3"
        assert st["pdf"]["index"] == 5 and st["pdf"]["total"] == 50
        assert st["pdf"]["current"] == "doc.pdf"

    def test_other_phase_event_does_not_overwrite(self, panel, monkeypatch):
        """Событие одной фазы не меняет надпись другой."""
        st, _ = self._run_with_events(panel, monkeypatch, [
            {"phase": "transcribe", "weight": 45},
            {"phase": "images", "weight": 25},
            {"file": "a.mp3", "file_index": 36, "file_total": 4000, "phase": "transcribe"},
            {"file": "батч 2 из 7 (3 шт.)", "phase": "images"},
        ])
        assert st["transcribe"]["total"] == 4000  # не сброшено событием images
        assert st["transcribe"]["current"] == "a.mp3"
        assert st["images"]["current"] == "батч 2 из 7 (3 шт.)"
        assert "total" not in st["images"]  # у батча нет счётчика

    def test_phase_done_removes_only_its_label(self, panel, monkeypatch):
        st, _ = self._run_with_events(panel, monkeypatch, [
            {"phase": "transcribe", "weight": 45},
            {"phase": "pdf", "weight": 15},
            {"file": "a.mp3", "file_index": 1, "file_total": 2, "phase": "transcribe"},
            {"file": "doc.pdf", "file_index": 1, "file_total": 3, "phase": "pdf"},
            {"phase_done": "pdf"},
        ])
        assert "pdf" not in st
        assert st["transcribe"]["total"] == 2  # чужая надпись осталась

    def test_all_phases_cleared_on_done(self, panel, monkeypatch):
        """Завершённая задача не хранит активных надписей фаз."""
        _, job = self._run_with_events(panel, monkeypatch, [
            {"phase": "transcribe", "weight": 100},
            {"file": "a.mp3", "file_index": 1, "file_total": 1, "phase": "transcribe"},
        ])
        assert job["state"] == "done"
        assert job["phase_status"] == {}

    def test_analysis_percent_label(self, panel, monkeypatch):
        """Анализ — тоже глобальная задача: показываем процент фазы."""
        st, _ = self._run_with_events(panel, monkeypatch, [
            {"phase": "analysis", "weight": 100},
            {"p": "analysis", "phase_progress": 0.42},
        ])
        assert st["analysis"]["percent"] == 42


class TestRerunMode:
    """rerun сохраняет режим и модель (раньше «mode» хранил имя модели,
    и любой повторный запуск терял анализ — режим full превращался в extract)."""

    def _capture_thread(self, wp, monkeypatch):
        calls: list[tuple] = []

        class FakeThread:
            def __init__(self, target=None, args=(), daemon=None, **kwargs):
                calls.append(tuple(args))

            def start(self):
                pass

        monkeypatch.setattr(wp.threading, "Thread", FakeThread)
        return calls

    def test_rerun_preserves_full_mode_and_model(self, panel, monkeypatch, tmp_path):
        wp, jobs = panel
        calls = self._capture_thread(wp, monkeypatch)
        jobs["old"] = {
            "state": "done", "mode": "full", "model": "medium",
            "language": "ru", "diarize": True, "paths": [str(tmp_path)],
        }
        client = wp.app.test_client()
        r = client.post("/api/rerun/old")
        assert r.status_code == 200
        new_id = r.get_json()["job_id"]
        assert jobs[new_id]["mode"] == "full"
        assert jobs[new_id]["model"] == "medium"
        # run_job(job_id, paths, mode, diarize, model, language)
        assert calls[0][2] == "full"
        assert calls[0][4] == "medium"

    def test_legacy_job_mode_field_held_model(self, panel, monkeypatch, tmp_path):
        wp, jobs = panel
        self._capture_thread(wp, monkeypatch)
        # старый формат: "mode" = имя модели, режим не сохранён
        jobs["legacy"] = {"state": "done", "mode": "medium", "paths": [str(tmp_path)]}
        client = wp.app.test_client()
        r = client.post("/api/rerun/legacy")
        assert r.status_code == 200
        new_id = r.get_json()["job_id"]
        assert jobs[new_id]["mode"] == "full"
        assert jobs[new_id]["model"] == "medium"


class TestPageRender:
    def test_index_serves_per_phase_markup(self, panel):
        """Страница содержит контейнер и логику отдельных надписей по фазам."""
        wp, _ = panel
        client = wp.app.test_client()
        r = client.get("/")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert 'id="phases"' in body
        assert "renderPhases" in body
        assert "phase_status" in body
        assert "fileStatus" not in body  # старая единая надпись удалена


class TestProgressPayload:
    """job_progress не тащит весь лог, lines ограничены, сохранения троттлятся.

    В сессии 13.10 job разросся до 7700+ строк / 500 КБ JSON: каждый тик
    прогресса сериализовал и отправлял его браузеру целиком, а каждый
    лог-строке — отдельная запись в SQLite.
    """

    def _run(self, panel, monkeypatch, events):
        wp, jobs = panel
        snapshots: list[dict] = []
        saves: list[int] = []

        def fake_emit(event_name, data=None, *a, **k):
            if event_name == "job_progress" and isinstance(data, dict):
                snapshots.append(dict(data))

        def fake_save(jid, data):
            saves.append(1)
            jobs[jid] = data

        def fake_pipeline(paths, **kwargs):
            p = kwargs["progress"]
            for e in events:
                p(e)
            return {"transcripts": [], "report": None, "image_reports": []}

        monkeypatch.setattr(wp, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(wp.socketio, "emit", fake_emit)
        monkeypatch.setattr(wp, "save_job", fake_save)

        jobs["j1"] = {
            "state": "queued", "progress": 0, "lines": [], "phase_status": {},
            "error": "", "report": None,
        }
        wp.run_job("j1", ["x"], "full", False, "small", "auto")
        return snapshots, jobs["j1"], saves

    def test_progress_snapshot_has_no_lines(self, panel, monkeypatch):
        events = [
            {"log": "строка 1"},
            {"file": "a.mp3", "file_index": 1, "file_total": 2, "phase": "transcribe"},
            {"log": "строка 2"},
        ]
        snapshots, _, _ = self._run(panel, monkeypatch, events)
        assert snapshots, "должен быть хотя бы один job_progress"
        assert all("lines" not in s for s in snapshots), (
            "job_progress не должен таскать строки лога (сотни КБ на каждый тик)"
        )

    def test_lines_capped_at_max(self, panel, monkeypatch):
        logs = [{"log": f"строка {i}"} for i in range(700)]
        _, job, _ = self._run(panel, monkeypatch, logs)
        cap = panel[0].MAX_JOB_LINES
        assert len(job["lines"]) == cap, "лимит строк не применён"
        # хвост сохраняется; последняя строка — итоговая (Готово…), перед ней —
        # последняя лог-строка, голова отрезана
        assert job["lines"][-1].startswith("Готово.")
        assert job["lines"][-2] == "строка 699"
        assert job["lines"][0] == f"строка {700 - cap + 1}"

    def test_final_summary_kept_after_cap(self, panel, monkeypatch):
        logs = [{"log": f"строка {i}"} for i in range(600)]
        _, job, _ = self._run(panel, monkeypatch, logs)
        assert job["lines"][-1].startswith("Готово."), "итоговая строка вытеснена лимитом"

    def test_log_saves_throttled(self, panel, monkeypatch):
        # 600 быстрых лог-событий подряд: реальных записей в БД — единицы,
        # а не 600 (монотонное время за цикл не успевает пройти 0.5с)
        logs = [{"log": f"строка {i}"} for i in range(600)]
        _, _, saves = self._run(panel, monkeypatch, logs)
        assert 1 <= len(saves) <= 10, f"слишком много записей: {len(saves)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
