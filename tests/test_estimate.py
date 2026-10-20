import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.database as db
import src.estimate as est
from src.estimate import build_estimate, fmt_duration


@pytest.fixture()
def empty_stats(monkeypatch):
    """Без накопленной статистики — используются первичные константы."""
    monkeypatch.setattr(est, "get_phase_timings", lambda: {})
    return {}


@pytest.fixture()
def fake_stats(monkeypatch):
    """Статистика прошлых запусков: транскрибация 100 файлов за 2000 с (20 с/файл),
    анализ 65 фрагментов за 78 с (1.2 с/фрагмент) и т.д."""
    data = {
        "transcribe": (100, 2000.0),
        "analysis": (65, 78.0),
        "images": (500, 5000.0),
        "pdf": (10, 1200.0),
    }
    monkeypatch.setattr(est, "get_phase_timings", lambda: data)
    return data


class TestFmtDuration:
    def test_seconds(self):
        assert fmt_duration(45) == "~45 с"

    def test_minutes(self):
        assert fmt_duration(300) == "~5 мин"

    def test_minutes_with_seconds(self):
        assert fmt_duration(230) == "~3 мин 50 с"

    def test_minutes_over_hour_kept_in_minutes(self):
        # 65 мин < порога 90 мин — так читабельнее
        assert fmt_duration(3900) == "~65 мин"

    def test_hours(self):
        assert fmt_duration(7200) == "~2 ч 00 мин"


class TestAvgFromStats:
    def test_uses_history_when_available(self, fake_stats):
        assert est._avg("transcribe") == pytest.approx(2000.0 / 100)
        assert est._avg("images") == pytest.approx(10.0)

    def test_fallback_without_history(self, empty_stats, monkeypatch):
        monkeypatch.setattr(est, "_transcribe_fallback_per_file", lambda: 30.0)
        assert est._avg("transcribe") == 30.0
        assert est._avg("pdf") == est.DEFAULT_AVG_SEC["pdf"]

    def test_transcribe_fallback_depends_on_cuda(self, empty_stats, monkeypatch):
        import types

        fake_ct2 = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
        monkeypatch.setitem(__import__("sys").modules, "ctranslate2", fake_ct2)
        assert est._transcribe_fallback_per_file() == est.TRANSCRIBE_GPU_SEC_PER_FILE
        monkeypatch.setitem(__import__("sys").modules, "ctranslate2", None)
        assert est._transcribe_fallback_per_file() == est.TRANSCRIBE_CPU_SEC_PER_FILE


class TestBuildEstimate:
    def test_empty_counts(self, empty_stats):
        lines, details = build_estimate(counts={}, extract=True, process=True)
        assert lines == ["Нет файлов для обработки — оценка не требуется"]
        assert details["total"] == 0.0

    def test_no_content_probing_needed_and_fast(self, fake_stats):
        """Оценка строится ТОЛЬКО по счётчикам и статистике — миллисекунды."""
        import time

        t0 = time.monotonic()
        build_estimate(
            counts={"transcribe": 3108, "images": 7533, "pdf": 50, "ppt": 9,
                    "spreadsheets": 11, "documents": 19},
            extract=True, process=True,
        )
        assert time.monotonic() - t0 < 0.5, "оценка не должна читать файлы"

    def test_history_drives_estimate(self, fake_stats):
        """100 новых файлов транскрибации по 20 с/файл из статистики = 2000 с."""
        _, d = build_estimate(counts={"transcribe": 100}, extract=True, process=False)
        assert d["transcribe"] == pytest.approx(2000.0)

    def test_full_plan_lines_and_parallel_total(self, fake_stats):
        lines, d = build_estimate(
            counts={"transcribe": 100, "images": 30, "pdf": 2, "ppt": 1,
                    "spreadsheets": 2, "documents": 5},
            extract=True, process=True,
        )
        text = "\n".join(lines)
        assert lines[0].startswith("Оценка времени задачи")
        assert "- Транскрибация: ~33 мин (100 файл(ов))" in text
        assert "- Картинки (OCR): ~5 мин (30 шт.)" in text
        assert "- PDF:" in text
        assert "- Анализ транскриптов:" in text and "фрагментов" in text
        # фрагменты из статистики: analysis_units/transcribe_units = 65/100 = 0.65
        assert d["fragments_est"] == 65
        assert lines[-1].startswith("ИТОГО:")
        assert "параллельно" in lines[-1]
        assert d["total"] >= max(d["transcribe"] + d["analysis"], d["attachments"]) - 1e-9
        assert d["total"] < d["transcribe"] + d["analysis"] + d["attachments"]

    def test_fragments_fallback_without_history(self, empty_stats):
        _, d = build_estimate(counts={"transcribe": 100}, extract=True, process=True)
        assert d["fragments_est"] == pytest.approx(65)  # 100 * 0.65

    def test_process_only(self, fake_stats):
        lines, d = build_estimate(counts={"transcribe": 100}, extract=False, process=True)
        text = "\n".join(lines)
        assert "Транскрибация" not in text
        assert "Анализ транскриптов" in text
        assert d["transcribe"] == 0.0

    def test_extract_only_skips_analysis(self, fake_stats):
        _, d = build_estimate(counts={"transcribe": 10}, extract=True, process=False)
        assert d["analysis"] == 0.0
        assert d["total"] == pytest.approx(d["transcribe"])

    def test_raw_mode_no_nim_phases(self, fake_stats):
        """Raw-режим: транскрибация + локальное извлечение, NIM-фаз нет."""
        lines, d = build_estimate(
            counts={"transcribe": 10, "raw": 100}, extract=True, process=True, raw=True,
        )
        text = "\n".join(lines)
        assert "режим без NIM" in text
        assert "- Транскрибация:" in text
        assert "- Локальное извлечение текста:" in text and "без NIM" in text
        assert "Картинки (OCR)" not in text and "Анализ транскриптов" not in text
        assert "PDF" not in text
        # total = max(фаз): параллельность
        assert d["total"] >= max(d["transcribe"], d["attachments"]) - 1e-9
        assert d["analysis"] == 0.0

    def test_raw_mode_empty(self, empty_stats):
        lines, d = build_estimate(counts={}, extract=True, process=False, raw=True)
        assert lines == ["Нет файлов для обработки — оценка не требуется"]
        assert d["total"] == 0.0


class TestEmitEstimate:
    def test_counts_from_plan_no_file_reads(self, empty_stats, tmp_path):
        jpg = tmp_path / "p.jpg"  # даже не существует — оценка не читает файлы
        events: list[dict] = []

        details = est.emit_estimate(
            media_files=[tmp_path / "a.mp3"],
            att_plan=[("images", [jpg], tmp_path)],
            extract=False,
            process=True,
            progress=events.append,
        )

        texts = [e.get("log", "") for e in events]
        assert any("Оценка времени" in t for t in texts)
        assert any("Картинки (OCR)" in t for t in texts)
        # отдельное событие с ИТОГО-строкой для постоянного отображения в панели
        estimates = [e["estimate"] for e in events if "estimate" in e]
        assert estimates and estimates[0].startswith("ИТОГО:")
        assert details["total"] > 0

    def test_logs_without_progress(self, empty_stats):
        logged: list[str] = []
        orig = est.logger.info
        est.logger.info = lambda msg, *a, **k: logged.append(msg)
        try:
            details = est.emit_estimate(media_files=[], att_plan=[], extract=False, process=False, progress=None)
        finally:
            est.logger.info = orig
        assert any("Нет файлов" in msg for msg in logged)
        assert details["total"] == 0.0


class TestPhaseTimingRecording:
    """Хуки записи статистики: фазы копят (units, seconds) в cache.db."""

    def test_transcribe_phase_recorded(self, monkeypatch, tmp_path):
        import src.transcription as tr

        recorded: list[tuple] = []
        monkeypatch.setattr(tr, "record_phase_timing", lambda k, u, s: recorded.append((k, u, s)))
        monkeypatch.setattr(tr, "settings", type("S", (), {"whisper_workers": 2})())
        monkeypatch.setattr(tr, "transcribe_file", lambda f, *a, **k: tmp_path / "t.txt")

        tr.run_extract(
            [(tmp_path / "a.mp3", tmp_path), (tmp_path / "b.mp3", tmp_path)],
            "small", "ru", False, {"transcribe": 100}, lambda e: None,
        )

        kinds = [r for r in recorded if r[0] == "transcribe"]
        assert len(kinds) == 1
        assert kinds[0][1] == 2, "units = число файлов фазы"
        assert kinds[0][2] >= 0

    def test_analysis_phase_recorded(self, tmp_path, monkeypatch):
        import src.analysis as analysis
        import src.utils

        recorded: list[tuple] = []
        monkeypatch.setattr(analysis, "record_phase_timing", lambda k, u, s: recorded.append((k, u, s)))
        fake = type("S", (), {"nim_workers": 2, "chunk_char_budget": 200,
                              "chunk_max_lines": 10, "nim_chunk_char_budget": 200,
                              "nim_reduce_group_size": 0, "nim_reduce_model": "m",
                              "nim_model": "m"})()
        monkeypatch.setattr(analysis, "settings", fake)
        monkeypatch.setattr(src.utils, "settings", fake)
        monkeypatch.setattr(analysis, "nim_client", type("N", (), {"send_prompt": staticmethod(lambda *a, **k: "конспект")})())
        monkeypatch.setattr(analysis, "get_cached_fragment", lambda h: None)
        monkeypatch.setattr(analysis, "save_fragment_cache", lambda *a: None)

        t = tmp_path / "big.transcript.txt"
        t.write_text("\n".join(f"[00:00] {'x' * 150}" for _ in range(4)) + "\n", encoding="utf-8")
        analysis._run_analysis_on_transcripts([t], tmp_path, {"analysis": 100}, lambda e: None)

        kinds = [r for r in recorded if r[0] == "analysis"]
        assert len(kinds) == 1
        assert kinds[0][1] == 4, "units = число фрагментов"

    def test_attachments_phase_recorded(self, monkeypatch, tmp_path):
        import src.pipeline as pipe

        recorded: list[tuple] = []
        monkeypatch.setattr(pipe, "record_phase_timing", lambda k, u, s: recorded.append((k, u, s)))
        monkeypatch.setattr(pipe, "run_attachments", lambda *a, **k: [])
        monkeypatch.setattr(pipe, "run_pdfs", lambda *a, **k: [])
        monkeypatch.setattr(pipe, "run_presentations", lambda *a, **k: [])
        monkeypatch.setattr(pipe, "run_spreadsheets", lambda *a, **k: [])

        from src.pipeline import _attachments_worker

        files = [tmp_path / f"p{i}.jpg" for i in range(3)]
        for f in files:
            f.write_bytes(b"jpg")
        _attachments_worker([("images", files, tmp_path)], {"images": 10}, lambda e: None, [])

        kinds = [r for r in recorded if r[0] == "images"]
        assert len(kinds) == 1
        assert kinds[0][1] == 3


class TestPhaseTimingsDb:
    def test_record_and_load_accumulates(self, monkeypatch, tmp_path):
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "cache.db")
        db.init_db()

        db.record_phase_timing("images", 10, 100.0)
        db.record_phase_timing("images", 5, 60.0)
        db.record_phase_timing("pdf", 2, 300.0)

        timings = db.get_phase_timings()
        assert timings["images"] == (15, pytest.approx(160.0))
        assert timings["pdf"] == (2, pytest.approx(300.0))
        assert db.get_phase_timings().get("transcribe") is None

    def test_invalid_records_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "cache.db")
        db.init_db()
        db.record_phase_timing("images", 0, 100.0)
        db.record_phase_timing("images", 5, -1.0)
        assert db.get_phase_timings() == {}

    def test_missing_table_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "no_init.db")
        assert db.get_phase_timings() == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
