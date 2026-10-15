import sys
from pathlib import Path
from types import SimpleNamespace

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.attachments as att
import src.spreadsheets as sheets
from src.analysis import _FragProgress


@pytest.fixture()
def cfg(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        nim_api_key="test-key",
        nim_model="m",
        nim_vision_model="v",
        nim_rpm=600,
        nim_max_tokens=100,
        nim_temperature=0.7,
        nim_frequency_penalty=0.4,
        nim_presence_penalty=0.3,
        nim_workers=4,
        nim_max_continuations=0,
        nim_reduce_group_size=24,
        nim_images_per_batch=3,
        nim_chunk_char_budget=8000,
        chunk_char_budget=65000,
        chunk_max_lines=300,
        results_dir=tmp_path / "results",
        tmp_pdfs_dir=tmp_path / "tmp_pdfs",
    )
    fake.results_dir.mkdir(parents=True, exist_ok=True)
    fake.tmp_pdfs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(att, "settings", fake)
    monkeypatch.setattr(sheets, "settings", fake)
    return fake


class TestSharedTrackerPerPhase:
    """Один _FragProgress на всю фазу pdf/ppt/spreadsheets.

    Раньше трекер создавался на каждый файл: после завершения первого PDF
    доля фазы навсегда оставалась 1.0 (монотонность игнорировала откат
    следующего файла), и фаза досрочно показывала 100%.
    """

    def _collect_trackers(self, monkeypatch, module, process_name):
        trackers: list = []
        real_frag = module._FragProgress

        def spy(*args, **kwargs):
            t = real_frag(*args, **kwargs)
            trackers.append(t)
            return t

        monkeypatch.setattr(module, "_FragProgress", spy)
        return trackers

    def test_run_pdfs_one_tracker_for_all_files(self, cfg, monkeypatch, tmp_path):
        trackers = self._collect_trackers(monkeypatch, att, "_process_single_pdf")
        seen: list = []

        def fake_process(pdf, weights, progress, out_dir, quiet, tracker=None):
            seen.append(tracker)
            return []

        monkeypatch.setattr(att, "_process_single_pdf", fake_process)
        pdfs = [tmp_path / f"{i}.pdf" for i in range(4)]
        for p in pdfs:
            p.write_bytes(b"%PDF-")
        att.run_pdfs(pdfs, {"pdf": 10})
        assert len(trackers) == 1, f"трекеров: {len(trackers)}, ожидался один на фазу"
        assert all(t is trackers[0] for t in seen), "файлы получили разные трекеры"

    def test_run_presentations_one_tracker_for_all_files(self, cfg, monkeypatch, tmp_path):
        trackers: list = []
        real_frag = att._FragProgress

        def spy(*args, **kwargs):
            t = real_frag(*args, **kwargs)
            trackers.append(t)
            return t

        monkeypatch.setattr(att, "_FragProgress", spy)

        def fake_process(ppt, weights, progress, out_dir, quiet, tracker=None):
            return []

        monkeypatch.setattr(att, "_process_single_presentation", fake_process)
        ppts = [tmp_path / f"{i}.pptx" for i in range(3)]
        for p in ppts:
            p.write_bytes(b"PK")
        att.run_presentations(ppts, {"ppt": 10})
        assert len(trackers) == 1
        assert trackers[0]._phase == "ppt"

    def test_run_spreadsheets_one_tracker_for_all_files(self, cfg, monkeypatch, tmp_path):
        trackers: list = []
        real_frag = sheets._FragProgress

        def spy(*args, **kwargs):
            t = real_frag(*args, **kwargs)
            trackers.append(t)
            return t

        monkeypatch.setattr(sheets, "_FragProgress", spy)

        def fake_process(sheet, weights, progress, out_dir, quiet, tracker=None):
            return []

        monkeypatch.setattr(sheets, "_process_single_spreadsheet", fake_process)
        xls = [tmp_path / f"{i}.xlsx" for i in range(3)]
        for p in xls:
            p.write_bytes(b"PK")
        sheets.run_spreadsheets(xls, {"spreadsheets": 10})
        assert len(trackers) == 1
        assert trackers[0]._phase == "spreadsheets"

    def test_shared_tracker_progress_honest_across_files(self, cfg):
        """Семантика: с общим трекером доля 1.0 только когда ВСЕ фрагменты
        всех файлов готовы, и она не откатывается."""
        events: list[dict] = []
        tracker = _FragProgress("pdf", 1.0, events.append)

        # Файл A: 2 фрагмента
        tracker.add_total(2)
        tracker.tick()
        # Файл B: 1 фрагмент, ещё не готов
        tracker.add_total(1)
        fracs = [e["phase_progress"] for e in events]
        assert max(fracs) < 1.0, "фракция 1.0 при неготовых фрагментах"

        tracker.tick()  # A-2
        tracker.tick()  # B-1
        fracs = [e["phase_progress"] for e in events]
        assert fracs[-1] == pytest.approx(1.0)

    def test_per_file_tracker_stuck_at_one(self, cfg):
        """Демонстрация старого бага: отдельные трекеры на файлы дают
        frac=1.0 после первого файла, пока второй ещё даже не стартовал."""
        events: list[dict] = []
        first = _FragProgress("pdf", 1.0, events.append)
        first.add_total(1)
        first.tick()
        assert events[-1]["phase_progress"] == 1.0  # ← фаза «готова» слишком рано


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
