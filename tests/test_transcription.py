import sys
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.transcription as tr
from src.transcription import run_extract


def _event_types(events):
    return [e for e in events if "file_index" in e]


@pytest.fixture()
def fast_workers(monkeypatch):
    """Транскрибация мокирована; воркеры из настроек подменены компактным числом."""
    from types import SimpleNamespace

    fake = SimpleNamespace(whisper_workers=3)
    monkeypatch.setattr(tr, "settings", fake)
    return fake


class TestFileCounters:
    def test_counters_monotonic_by_completion(self, tmp_path, fast_workers, monkeypatch):
        """«Файл n из N» — n это число ЗАВЕРШЁННЫХ файлов: индексы 1..N без
        повторов и пропусков, независимо от порядка завершения воркеров."""
        media = [(tmp_path / f"file{i}.mp3", tmp_path) for i in range(6)]

        def fake_transcribe(f, model, language, diarize=False, on_progress=None, group_dir=None):
            # эмуляция параллельной работы: файлы завершаются не по порядку
            idx = int(f.stem.replace("file", ""))
            if idx % 2 == 0:
                threading.Event().wait(0.01 * (6 - idx))  # чётные позже
            return tmp_path / f"{f.stem}.transcript.txt"

        monkeypatch.setattr(tr, "transcribe_file", fake_transcribe)

        events: list[dict] = []
        transcripts = run_extract(media, "small", "auto", False, {"transcribe": 100}, events.append)

        assert len(transcripts) == 6
        counter_events = _event_types(events)
        # ровно N событий счётчика, индексы строго 1..N (каждый по одному разу)
        assert [e["file_index"] for e in counter_events] == [1, 2, 3, 4, 5, 6]
        assert all(e["file_total"] == 6 for e in counter_events)
        assert all(e["phase"] == "transcribe" for e in counter_events)
        # фаза закрыта
        assert {"phase_done": "transcribe"} in events

    def test_start_events_have_no_counter(self, tmp_path, fast_workers, monkeypatch):
        """События старта файла не несут file_index — счётчик только по факту завершения."""
        media = [(tmp_path / "a.mp3", tmp_path), (tmp_path / "b.mp3", tmp_path)]

        monkeypatch.setattr(tr, "transcribe_file", lambda f, *a, **k: tmp_path / f"{f.stem}.transcript.txt")

        events: list[dict] = []
        run_extract(media, "small", "auto", False, {"transcribe": 100}, events.append)

        start_events = [e for e in events if "file" in e and "file_index" not in e and "log" not in e]
        assert len(start_events) == 2
        assert all(e.get("phase") == "transcribe" for e in start_events)

    def test_failed_file_still_advances_counter(self, tmp_path, fast_workers, monkeypatch):
        """Ошибка файла не ломает счётчик: n растёт по числу обработанных."""
        media = [(tmp_path / f"f{i}.mp3", tmp_path) for i in range(3)]

        def fake_transcribe(f, *a, **k):
            if "1" in f.name:
                raise RuntimeError("сломался")
            return tmp_path / f"{f.stem}.transcript.txt"

        monkeypatch.setattr(tr, "transcribe_file", fake_transcribe)

        events: list[dict] = []
        transcripts = run_extract(media, "small", "auto", False, {"transcribe": 100}, events.append)

        assert len(transcripts) == 2  # два успешных
        counter_events = _event_types(events)
        assert [e["file_index"] for e in counter_events] == [1, 2, 3]
        # ошибка залогирована
        assert any("Ошибка транскрибации" in e.get("log", "") for e in events)

    def test_extract_and_process_counters(self, tmp_path, fast_workers, monkeypatch):
        """Тот же счётчик в полном пайплайне (транскрибация + анализ)."""
        media = [(tmp_path / f"f{i}.mp3", tmp_path) for i in range(4)]
        monkeypatch.setattr(tr, "transcribe_file", lambda f, *a, **k: tmp_path / f"{f.stem}.transcript.txt")
        monkeypatch.setattr(
            tr,
            "run_analysis_on_transcripts",
            lambda transcripts, out_dir, weights, progress: (transcripts, tmp_path / "report.txt"),
        )

        events: list[dict] = []
        transcripts, report = tr.run_extract_and_process(
            media, "small", "auto", False, tmp_path, {"transcribe": 60, "analysis": 40}, events.append
        )

        assert len(transcripts) == 4
        counter_events = _event_types(events)
        assert [e["file_index"] for e in counter_events] == [1, 2, 3, 4]
        assert all(e["file_total"] == 4 and e["phase"] == "transcribe" for e in counter_events)
        assert {"phase_done": "transcribe"} in events
        assert {"phase": "analysis", "weight": 40} in events


class TestEmptyPhasesCloseProgress:
    """Фаза без данных обязана анонсировать вес и закрыться: иначе шкала
    прогресса не добирает до 100% (ползёт к 20% и скачет в конце)."""

    def test_run_extract_empty_closes_phase(self):
        events: list[dict] = []
        run_extract([], "small", "ru", False, {"transcribe": 80}, events.append)
        assert {"phase": "transcribe", "weight": 80} in events
        assert {"phase_done": "transcribe"} in events

    def test_run_extract_and_process_empty_closes_both(self):
        events: list[dict] = []
        tr.run_extract_and_process(
            [], "small", "ru", False, None, {"transcribe": 45, "analysis": 30}, events.append
        )
        assert {"phase": "transcribe", "weight": 45} in events
        assert {"phase_done": "transcribe"} in events
        assert {"phase": "analysis", "weight": 30} in events
        assert {"phase_done": "analysis"} in events

    def test_no_transcripts_closes_analysis(self, tmp_path, monkeypatch):
        """Транскрибация прошла, но ни одного транскрипта — анализ закрывается."""
        monkeypatch.setattr(tr, "transcribe_file", lambda f, *a, **k: None)
        events: list[dict] = []
        tr.run_extract_and_process(
            [(tmp_path / "a.mp3", tmp_path)], "small", "ru", False, None,
            {"transcribe": 45, "analysis": 30}, events.append,
        )
        assert {"phase": "transcribe", "weight": 45} in events
        assert {"phase_done": "transcribe"} in events
        assert {"phase": "analysis", "weight": 30} in events
        assert {"phase_done": "analysis"} in events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
