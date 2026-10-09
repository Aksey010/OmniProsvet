import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from src.progress import PhaseProgress, FilesProgress


class TestPhaseProgressParallel:
    """Фазы идут параллельно — процент это сумма по всем фазам, без «пинг-понга»."""

    def test_parallel_phases_sum(self):
        pp = PhaseProgress()
        pp.feed({"phase": "transcribe", "weight": 45})
        pp.feed({"phase": "pdf", "weight": 12.5})
        pp.feed({"phase_progress": 0.5, "p": "transcribe"})
        pp.feed({"phase_progress": 0.4, "p": "pdf"})
        # 45*0.5 + 12.5*0.4 = 22.5 + 5 = 27.5 → 28
        assert pp.percent() == 28

    def test_no_double_count_on_ping_pong(self):
        """Раньше каждое переключение фазы добавляло вес заново — инфляция процента."""
        pp = PhaseProgress()
        pp.feed({"phase": "transcribe", "weight": 45})
        pp.feed({"phase": "pdf", "weight": 12.5})
        for _ in range(5):
            pp.feed({"phase": "transcribe", "weight": 45})
            pp.feed({"phase": "pdf", "weight": 12.5})
        pp.feed({"phase_progress": 0.2, "p": "transcribe"})
        # вес учтён один раз: 45*0.2 = 9, вклад pdf = 0
        assert pp.percent() == 9

    def test_phase_done_counts_full_weight_once(self):
        pp = PhaseProgress()
        pp.feed({"phase": "transcribe", "weight": 45})
        pp.feed({"phase_progress": 0.3, "p": "transcribe"})
        pp.feed({"phase_done": "transcribe"})
        pp.feed({"phase_done": "transcribe"})  # повторное завершение — игнор
        assert pp.percent() == 45

    def test_frac_monotonic_ignores_backward(self):
        """Параллельные воркеры рапортуют вразнобой — откаты не двигают бар назад."""
        pp = PhaseProgress()
        pp.feed({"phase": "pdf", "weight": 10})
        pp.feed({"phase_progress": 0.9, "p": "pdf"})
        pp.feed({"phase_progress": 0.1, "p": "pdf"})  # отстающий трекер другого PDF
        assert pp.percent() == 9  # 10 * 0.9

    def test_frac_ignored_after_done(self):
        pp = PhaseProgress()
        pp.feed({"phase": "analysis", "weight": 30})
        pp.feed({"phase_done": "analysis"})
        pp.feed({"phase_progress": 0.5, "p": "analysis"})
        assert pp.percent() == 30

    def test_percent_capped_at_100(self):
        pp = PhaseProgress()
        pp.feed({"phase": "a", "weight": 60})
        pp.feed({"phase": "b", "weight": 60})
        pp.feed({"phase_done": "a"})
        pp.feed({"phase_done": "b"})
        assert pp.percent() == 100

    def test_unknown_phase_frac_ignored(self):
        """Продвижение фазы без анонса веса ничего не добавляет."""
        pp = PhaseProgress()
        pp.feed({"phase_progress": 0.5, "p": "ghost"})
        assert pp.percent() == 0

    def test_full_pipeline_scenario_monotonic(self):
        """Сквозной сценарий: транскрипция+анализ параллельно с pdf/images,
        события в перемешанном порядке — процент монотонен и в конце 100."""
        pp = PhaseProgress()
        seq = [
            {"phase": "transcribe", "weight": 45},
            {"phase": "analysis", "weight": 30},
            {"phase": "pdf", "weight": 12.5},
            {"phase": "images", "weight": 12.5},
        ]
        for e in seq:
            pp.feed(e)
        percents = []
        events = [
            {"phase_progress": 0.2, "p": "transcribe"},
            {"phase_progress": 0.1, "p": "pdf"},
            {"phase_progress": 0.5, "p": "transcribe"},
            {"phase_progress": 0.3, "p": "pdf"},
            {"phase_progress": 0.9, "p": "transcribe"},
            {"phase_progress": 0.7, "p": "pdf"},
            {"phase_progress": 0.2, "p": "pdf"},  # откат параллельного трекера
            {"phase_progress": 1.0, "p": "images"},
            {"phase_done": "transcribe"},
            {"phase_progress": 0.5, "p": "analysis"},
            {"phase_done": "pdf"},
            {"phase_progress": 1.0, "p": "analysis"},
            {"phase_done": "analysis"},
        ]
        for e in events:
            pp.feed(e)
            percents.append(pp.percent())
        assert percents == sorted(percents), f"прогресс убывал: {percents}"
        assert percents[-1] == 100
        # после транскрибации (45) + images (12.5) + pdf наполовину: 45+12.5+8.75 = 66 → 66
        idx_after_transcribe_done = events.index({"phase_done": "transcribe"})
        assert percents[idx_after_transcribe_done] >= 45


class TestFilesProgress:
    def test_average_across_files(self):
        events = []
        fp = FilesProgress(3, "transcribe")
        r1 = fp.reporter(1, events.append)
        r2 = fp.reporter(2, events.append)
        r1(100)
        r2(50)
        assert abs(events[-1]["phase_progress"] - (1.0 + 0.5) / 3) < 1e-9

    def test_no_sawtooth_under_reordering(self):
        """Файл 4 рапортует раньше файла 1 — доля фазы не убывает (старая
        индексная формула давала 'пилу': 0.95 → 0.025)."""
        events = []
        fp = FilesProgress(4, "transcribe")
        r1 = fp.reporter(1, events.append)
        r4 = fp.reporter(4, events.append)
        r4(80)
        after4 = events[-1]["phase_progress"]
        r1(10)
        after1 = events[-1]["phase_progress"]
        assert after1 >= after4  # (0.1 + 0.8) / 4 >= 0.8 / 4

    def test_reporter_clamps_percent(self):
        events = []
        fp = FilesProgress(2)
        fp.reporter(1, events.append)(150)  # Whisper может выдать >100
        assert events[-1]["phase_progress"] == 0.5

    def test_out_of_range_index(self):
        fp = FilesProgress(2)
        with pytest.raises(IndexError):
            fp.reporter(3, None)

    def test_zero_total_safe(self):
        fp = FilesProgress(0)
        assert fp.reporter(1, None) is not None  # не падает


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
