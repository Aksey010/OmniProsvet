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


class TestHierarchicalReduce:
    def _prepare(self, tmp_path, monkeypatch, group_size: int):
        import src.analysis as analysis
        from src.configuration import Settings

        fake = Settings(
            chunk_char_budget=200,
            chunk_max_lines=10,
            nim_chunk_char_budget=200,
            nim_reduce_group_size=group_size,
        )
        monkeypatch.setattr(src.analysis, "settings", fake)
        monkeypatch.setattr(src.utils, "settings", fake)

        class FakeNim:
            def __init__(self):
                self.calls: list[str] = []

            def send_prompt(self, prompt, **kwargs):
                self.calls.append(prompt)
                return "Сводка: факт 5 мая."

        fake_nim = FakeNim()
        monkeypatch.setattr(analysis, "nim_client", fake_nim)
        monkeypatch.setattr(analysis, "get_cached_fragment", lambda h: None)
        monkeypatch.setattr(analysis, "save_fragment_cache", lambda h, s, t: None)
        return fake_nim

    def _make_transcripts(self, tmp_path, chunks: int):
        # каждый фрагмент — отдельный «файл» длиннее половины бюджета
        lines = [f"[00:00] {'x' * 150}" for _ in range(chunks)]
        return [_write(tmp_path / "big.transcript.txt", lines)]

    def test_small_job_has_no_intermediates(self, tmp_path, monkeypatch):
        """Фрагментов не больше group_size — прежний одиночный reduce (регресс)."""
        fake_nim = self._prepare(tmp_path, monkeypatch, group_size=10)
        transcripts = self._make_transcripts(tmp_path, chunks=5)

        _, report = _run_analysis_on_transcripts(transcripts, tmp_path, {"analysis": 100}, lambda e: None)

        assert report is not None
        # 5 map + 1 reduce = 6 запросов, промежуточных сводок нет
        assert len(fake_nim.calls) == 6

    def test_large_job_uses_group_summaries(self, tmp_path, monkeypatch):
        """Фрагментов больше group_size — групповые сводки + финальный reduce."""
        fake_nim = self._prepare(tmp_path, monkeypatch, group_size=10)
        transcripts = self._make_transcripts(tmp_path, chunks=25)

        events: list[dict] = []
        _, report = _run_analysis_on_transcripts(transcripts, tmp_path, {"analysis": 100}, events.append)

        assert report is not None
        # 25 map + 3 промежуточных (10+10+5) + 1 финальный = 29
        assert len(fake_nim.calls) == 29
        # промежуточные сводки содержат шапку с диапазоном фрагментов
        intermediates = [c for c in fake_nim.calls if "промежуточный конспект" in c.lower()]
        assert len(intermediates) == 3
        assert "1-10" in intermediates[0] or "1-10" in intermediates[1] or "1-10" in intermediates[2]
        # финальный reduce получил уже СВОДКИ, а не 25 сырых конспектов
        final = fake_nim.calls[-1]
        assert "Сводка: факт 5 мая." in final
        # прогресс не откатывался
        fracs = [e["phase_progress"] for e in events if e.get("p") == "analysis"]
        assert fracs == sorted(fracs)

    def test_group_size_zero_disables_hierarchy(self, tmp_path, monkeypatch):
        fake_nim = self._prepare(tmp_path, monkeypatch, group_size=0)
        transcripts = self._make_transcripts(tmp_path, chunks=25)
        _, report = _run_analysis_on_transcripts(transcripts, tmp_path, {"analysis": 100}, lambda e: None)
        assert report is not None
        assert len(fake_nim.calls) == 26  # 25 map + 1 reduce


class TestAnalysisResilience:
    """Анализ обязан завершаться при неисправимых ошибках NIM.

    В сессии 14.10 (01:27, задача e72938ef) один ReadTimeout в
    промежуточной сводке убил весь многочасовой анализ — отчёт не был
    создан вообще. Теперь сбой фрагмента/группы даёт placeholder, а
    отказ финального reduce — частичный отчёт.
    """

    def _prepare(self, tmp_path, monkeypatch, group_size: int = 0,
                 nim_model: str = "default-model", nim_reduce_model: str = "reduce-model"):
        import src.analysis as analysis
        from src.configuration import Settings

        fake = Settings(
            chunk_char_budget=200,
            chunk_max_lines=10,
            nim_chunk_char_budget=200,
            nim_reduce_group_size=group_size,
            nim_model=nim_model,
            nim_reduce_model=nim_reduce_model,
        )
        monkeypatch.setattr(analysis, "settings", fake)
        monkeypatch.setattr(src.utils, "settings", fake)

        from src.nim_client import NimError

        class FakeNim:
            def __init__(self, fail_on=None):
                self.calls: list[tuple[str, dict]] = []
                self.fail_on = fail_on

            def send_prompt(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                if self.fail_on and self.fail_on(prompt, kwargs):
                    raise NimError("NIM: не удалось получить ответ после 5 попыток. Сеть: ReadTimeout")
                return "Конспект: событие 5 мая."

        return fake, FakeNim

    def _install(self, monkeypatch, fake_nim):
        monkeypatch.setattr(src.analysis, "nim_client", fake_nim)
        monkeypatch.setattr(src.analysis, "get_cached_fragment", lambda h: None)
        monkeypatch.setattr(src.analysis, "save_fragment_cache", lambda h, s, t: None)

    def _transcripts(self, tmp_path, chunks: int) -> list[Path]:
        lines = [f"[00:00] {'x' * 150}" for _ in range(chunks)]
        return [_write(tmp_path / "big.transcript.txt", lines)]

    def test_map_failure_does_not_kill_analysis(self, tmp_path, monkeypatch):
        """Сбой одного MAP-фрагмента: отчёт всё равно собирается, пропуск
        виден в тексте, остальные фрагменты обработаны."""
        _, FakeNim = self._prepare(tmp_path, monkeypatch)
        state = {"n": 0}

        def fail_first(prompt, kwargs):
            state["n"] += 1
            return state["n"] == 1

        fake_nim = FakeNim(fail_on=fail_first)
        self._install(monkeypatch, fake_nim)
        transcripts = self._transcripts(tmp_path, chunks=5)

        events: list[dict] = []
        _, report = _run_analysis_on_transcripts(transcripts, tmp_path, {"analysis": 100}, events.append)

        assert report is not None and Path(report).exists()
        # placeholder виден на входе финального reduce (отчёт — его выход)
        final_input = fake_nim.calls[-1][0]
        assert "не обработан" in final_input, "пропущенный фрагмент не помечен"
        assert "Конспект: событие 5 мая." in Path(report).read_text(encoding="utf-8")
        # 5 map + 1 reduce: сбой не породил лишних попыток
        assert len(fake_nim.calls) == 6
        assert {"phase_done": "analysis"} in events

    def test_group_summary_failure_does_not_kill_analysis(self, tmp_path, monkeypatch):
        """Точка падения задачи e72938ef: сбой промежуточной сводки больше
        не убивает анализ — группа помечается, финальный reduce проходит."""
        _, FakeNim = self._prepare(tmp_path, monkeypatch, group_size=2)

        def fail_group_1_2(prompt, kwargs):
            # «фрагментов 1-2 » с пробелом: иначе совпадёт с «21-22»
            return "промежуточный конспект" in prompt and "фрагментов 1-2 " in prompt

        fake_nim = FakeNim(fail_on=fail_group_1_2)
        self._install(monkeypatch, fake_nim)
        transcripts = self._transcripts(tmp_path, chunks=25)

        events: list[dict] = []
        _, report = _run_analysis_on_transcripts(transcripts, tmp_path, {"analysis": 100}, events.append)

        assert report is not None and Path(report).exists()
        # сбойная группа помечена на входе финального reduce
        assert "группа фрагментов 1-2" in fake_nim.calls[-1][0], "сбойная группа не помечена"
        assert {"phase_done": "analysis"} in events
        # финальный reduce состоялся: последним вызовом был НЕ промежуточный
        last_prompt = fake_nim.calls[-1][0]
        assert "промежуточный конспект" not in last_prompt

    def test_final_reduce_falls_back_to_default_model(self, tmp_path, monkeypatch):
        """Основная reduce-модель недоступна → повтор дефолтной, отчёт есть."""
        _, FakeNim = self._prepare(
            tmp_path, monkeypatch, group_size=0,
            nim_model="default-model", nim_reduce_model="reduce-model",
        )
        fake_nim = FakeNim(fail_on=lambda p, kw: kw.get("model") == "reduce-model")
        self._install(monkeypatch, fake_nim)
        transcripts = self._transcripts(tmp_path, chunks=5)

        _, report = _run_analysis_on_transcripts(transcripts, tmp_path, {"analysis": 100}, lambda e: None)

        assert report is not None and Path(report).exists()
        models_used = [kw.get("model") for _, kw in fake_nim.calls if kw.get("model")]
        assert models_used == ["reduce-model", "default-model"]
        text = Path(report).read_text(encoding="utf-8")
        assert "ФИНАЛЬНАЯ СВОДКА НЕ УДАЛАСЬ" not in text

    def test_total_reduce_failure_writes_partial_report(self, tmp_path, monkeypatch):
        """Обе reduce-модели мертвы: отчёт всё равно записан — из сырых
        конспектов, с честной пометкой (анализ не теряется целиком)."""
        _, FakeNim = self._prepare(
            tmp_path, monkeypatch, group_size=0,
            nim_model="reduce-model", nim_reduce_model="reduce-model",
        )
        fake_nim = FakeNim(fail_on=lambda p, kw: kw.get("model") is not None)
        self._install(monkeypatch, fake_nim)
        transcripts = self._transcripts(tmp_path, chunks=5)

        events: list[dict] = []
        _, report = _run_analysis_on_transcripts(transcripts, tmp_path, {"analysis": 100}, events.append)

        assert report is not None and Path(report).exists()
        text = Path(report).read_text(encoding="utf-8")
        assert "ФИНАЛЬНАЯ СВОДКА НЕ УДАЛАСЬ" in text
        assert "Конспект: событие 5 мая." in text, "конспекты фрагментов потеряны"
        assert {"phase_done": "analysis"} in events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
