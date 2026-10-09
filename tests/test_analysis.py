import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.analysis
import src.utils
from src.analysis import _plan_fragments, _run_analysis_on_transcripts
from src.configuration import Settings


@pytest.fixture()
def small_budgets(monkeypatch):
    """Компактные бюджеты чанков, чтобы сценарии были предсказуемыми."""
    fake = Settings(chunk_char_budget=200, chunk_max_lines=10, nim_chunk_char_budget=200)
    monkeypatch.setattr(src.analysis, "settings", fake)
    monkeypatch.setattr(src.utils, "settings", fake)
    return fake


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestPlanFragments:
    def test_small_transcripts_merged_into_shared_batch(self, tmp_path, small_budgets):
        t1 = _write(tmp_path / "a.transcript.txt", ["[00:00] короткая строка"])
        t2 = _write(tmp_path / "b.transcript.txt", ["[00:00] другая строка"])
        units = _plan_fragments([t1, t2], budget=200)
        assert len(units) == 1
        source, chunks = units[0]
        assert source == "мелкие файлы"
        assert len(chunks) == 1
        assert len(chunks[0]) == 2  # обе строки в одном общем батче

    def test_large_transcript_keeps_own_unit(self, tmp_path, small_budgets):
        lines = [f"[00:00] {'x' * 150}" for _ in range(10)]
        t = _write(tmp_path / "big.transcript.txt", lines)
        units = _plan_fragments([t], budget=200)
        assert len(units) == 1
        source, chunks = units[0]
        assert source == "big.transcript.md"
        assert len(chunks) == 10  # каждая строка длиннее половины бюджета → свой чанк
        # чанки НЕ мелкие → не склеиваются с чужими
        for chunk in chunks:
            assert sum(len(l) for l in chunk) > 100

    def test_merge_flushes_when_budget_exceeded(self, tmp_path, small_budgets):
        # строка ~90 символов, бюджет склейки 200: 2 строки влезают, 3-я нет
        body = "[00:00] " + "т" * 67
        transcripts = [_write(tmp_path / f"t{i}.transcript.txt", [body]) for i in range(4)]
        units = _plan_fragments(transcripts, budget=200)
        merged = [(s, c) for s, c in units if s == "мелкие файлы"]
        assert len(merged) == 2
        assert [len(c[0]) for _, c in merged] == [2, 2]

    def test_empty_transcript_skipped(self, tmp_path, small_budgets):
        t = _write(tmp_path / "empty.transcript.txt", ["заголовок без таймстампа"])
        assert _plan_fragments([t], budget=200) == []

    def test_total_count_known_upfront(self, tmp_path, small_budgets):
        """Суммарное число фрагментов считается до отправки первого запроса."""
        body = "[00:00] " + "т" * 67
        smalls = [_write(tmp_path / f"s{i}.transcript.txt", [body]) for i in range(2)]
        big = _write(tmp_path / "big.transcript.txt", [f"[00:00] {'x' * 150}" for _ in range(10)])
        units = _plan_fragments(smalls + [big], budget=200)
        # 1 общий батч мелких + 10 чанков большого = 11 фрагментов
        assert sum(len(c) for _, c in units) == 11


class TestRunAnalysisIntegration:
    def test_full_map_reduce_flow(self, tmp_path, small_budgets, monkeypatch):
        """Сквозной прогон с фейковым NIM: фиксированный знаменатель прогресса,
        фаза закрыта phase_done, отчёт записан."""
        from src.analysis import REDUCE_FRACTION

        class FakeNim:
            def __init__(self):
                self.calls = []

            def send_prompt(self, prompt, **kwargs):
                self.calls.append(prompt)
                return "Конспект: событие 5 мая."

        fake_nim = FakeNim()
        monkeypatch.setattr(src.analysis, "nim_client", fake_nim)
        monkeypatch.setattr(src.analysis, "get_cached_fragment", lambda h: None)
        monkeypatch.setattr(src.analysis, "save_fragment_cache", lambda h, s, t: None)

        body = "[00:00] " + "т" * 67
        smalls = [_write(tmp_path / f"s{i}.transcript.txt", [body]) for i in range(2)]
        big = _write(tmp_path / "big.transcript.txt", [f"[00:00] {'x' * 150}" for _ in range(10)])
        transcripts = sorted(smalls + [big], key=lambda p: p.name)

        events: list[dict] = []
        result_transcripts, report = _run_analysis_on_transcripts(
            transcripts, tmp_path, {"analysis": 100}, events.append
        )

        # Фаза анонсирована и закрыта
        assert {"phase": "analysis", "weight": 100} in events
        assert {"phase_done": "analysis"} in events
        # phase_done — последнее событие фазы analysis
        analysis_events = [i for i, e in enumerate(events) if e.get("p") == "analysis"]
        done_idx = next(i for i, e in enumerate(events) if e.get("phase_done") == "analysis")
        assert done_idx > analysis_events[-1]

        # Знаменатель прогресса был известен до первого тика:
        # 11 фрагментов → последний MAP-тик даёт ровно REDUCE_FRACTION (11/11 * 0.9)
        fracs = [e["phase_progress"] for e in events if e.get("p") == "analysis"]
        map_fracs = [f for f in fracs if f <= REDUCE_FRACTION + 1e-9]
        assert map_fracs == sorted(map_fracs), "прогресс фазы убывал"
        assert any(abs(f - REDUCE_FRACTION) < 1e-9 for f in map_fracs)

        # Отчёт записан, текст конспектов вошёл целиком
        assert report is not None and Path(report).exists()
        text = Path(report).read_text(encoding="utf-8")
        assert "Конспект: событие 5 мая." in text
        # reduce + 11 map запросов
        assert len(fake_nim.calls) == 12

    def test_no_text_returns_none_and_closes_phase(self, tmp_path, small_budgets, monkeypatch):
        monkeypatch.setattr(src.analysis, "get_cached_fragment", lambda h: None)
        monkeypatch.setattr(src.analysis, "save_fragment_cache", lambda h, s, t: None)
        t = _write(tmp_path / "empty.transcript.txt", ["без таймстампов"])
        events: list[dict] = []
        transcripts, report = _run_analysis_on_transcripts([t], tmp_path, {"analysis": 100}, events.append)
        assert report is None
        assert {"phase_done": "analysis"} in events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
