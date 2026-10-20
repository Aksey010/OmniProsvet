import sys
from pathlib import Path
from types import SimpleNamespace

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.attachments as att
from src.nim_client import NimError


@pytest.fixture()
def cfg(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        nim_api_key="test-key",
        nim_model="text-model",
        nim_vision_model="vision-model",
        nim_rpm=600,
        nim_max_tokens=100,
        nim_temperature=0.7,
        nim_frequency_penalty=0.4,
        nim_presence_penalty=0.3,
        nim_workers=4,
        nim_max_continuations=0,
        nim_reduce_group_size=24,
        nim_images_per_batch=5,
        images_per_batch=25,
        chunk_char_budget=65000,
        chunk_max_lines=300,
        results_dir=tmp_path / "results",
        tmp_pdfs_dir=tmp_path / "tmp_pdfs",
    )
    fake.results_dir.mkdir(parents=True, exist_ok=True)
    fake.tmp_pdfs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(att, "settings", fake)
    return fake


@pytest.fixture()
def no_cache(monkeypatch):
    monkeypatch.setattr(att, "get_cached_fragment", lambda h: None)
    saved: list = []
    monkeypatch.setattr(att, "save_fragment_cache", lambda h, s, t: saved.append((h, s)))
    return saved


class TestManyImagesInstantSplit:
    """Детерминированный 400 «At most 1 image(s)»: батч режется сразу на
    одиночки, без половинаний и повторных 400-х.

    В сессии 13.10 каждый батч из 5 картинок к llama-3.2-90b-vision проходил
    цепочку 5→3+2→2+1+1+1 — по полному циклу ретраев на каждом шаге.
    """

    def test_batch_of_five_split_to_singletons(self, cfg, no_cache, monkeypatch, tmp_path):
        calls: list[int] = []

        def fake_send(prompt, **kwargs):
            n = len(kwargs["attachments"])
            calls.append(n)
            if n > 1:
                raise NimError(
                    "NIM: HTTP 400: At most 1 image(s) may be provided in one request."
                )
            return "конспект"

        monkeypatch.setattr(att.nim_client, "send_prompt", fake_send)
        files = [tmp_path / f"photo_{i}.jpg" for i in range(5)]
        for p in files:
            p.write_bytes(b"jpg")

        events: list[dict] = []
        results = att.run_attachments(files, "images", {"images": 10}, tmp_path / "out", events.append)

        assert len(results) == 5, "все файлы должны быть обработаны"
        # Ровно ОДНА попытка много-картиночного батча (лимит ещё не известен),
        # дальше — сразу одиночки, без цепочки половинаний 5→3→2→...
        assert calls.count(5) == 1, f"много-картиночных попыток больше одной: {calls}"
        assert calls.count(1) == 5, f"одиночных запросов не 5: {calls}"

    def test_no_halving_cascade_for_deterministic_400(self, cfg, no_cache, monkeypatch, tmp_path):
        """Гарантия: каждая пачка файлов даёт РОВНО ОДНУ много-картиночную
        попытку (первую), дальше — одиночки. Старое поведение гоняло цепочку
        половинаний 5→3+2→... с полным циклом ретраев на каждом шаге."""

        calls: list[int] = []

        def fake_send(prompt, **kwargs):
            n = len(kwargs["attachments"])
            calls.append(n)
            if n > 1:
                raise NimError("NIM: HTTP 400: At most 1 image(s) may be provided.")
            return "конспект"

        monkeypatch.setattr(att.nim_client, "send_prompt", fake_send)
        files = [tmp_path / f"p{i}.jpg" for i in range(8)]
        for p in files:
            p.write_bytes(b"jpg")
        # batch_size = min(25, 5) = 5 → батчи [5, 3]
        att.run_attachments(files, "images", {"images": 10}, tmp_path / "out")
        multi = [c for c in calls if c > 1]
        singles = [c for c in calls if c == 1]
        assert len(multi) == 2, f"много-картиночных попыток: {multi}"
        assert len(singles) == 8, f"одиночных запросов: {len(singles)}"


    def test_progress_not_leaked_by_subcalls(self, cfg, no_cache, monkeypatch, tmp_path):
        """Под-вызовы тихие: только верхний уровень эмитит phase_progress.

        Утечка phase_progress=1/1=1.0 из под-батчей — корень бага «100% при
        17/1585 батчей» в сессии 13.10.
        """

        def fake_send(prompt, **kwargs):
            return "конспект"

        monkeypatch.setattr(att.nim_client, "send_prompt", fake_send)
        files = [tmp_path / f"p{i}.jpg" for i in range(4)]
        for p in files:
            p.write_bytes(b"jpg")

        events: list[dict] = []
        # батчи по 2 → 2 батча; оба успешны
        cfg.nim_images_per_batch = 2
        att.run_attachments(files, "images", {"images": 10}, tmp_path / "out", events.append)

        fracs = [e["phase_progress"] for e in events if e.get("p") == "images"]
        assert fracs == [pytest.approx(0.5), pytest.approx(1.0)]
        # нет события с frac=1.0, за которым следует фаза ещё в работе:
        # последнее событие фракции — ровно по завершении последнего батча

    def test_failed_singletons_counted_in_stats(self, cfg, no_cache, monkeypatch, tmp_path):
        def fake_send(prompt, **kwargs):
            if any("bad" in a for a in kwargs["attachments"]):
                raise NimError("NIM: не удалось получить ответ после 5 попыток")
            return "конспект"

        monkeypatch.setattr(att.nim_client, "send_prompt", fake_send)
        files = [tmp_path / "good.jpg", tmp_path / "bad.jpg"]
        for p in files:
            p.write_bytes(b"jpg")

        events: list[dict] = []
        stats: dict = {}
        att.run_attachments(files, "images", {"images": 10}, tmp_path / "out", events.append, stats=stats)
        assert stats["failed"] == 1
        # total выставляет _attachments_worker (план фазы), не run_attachments —
        # рекурсия не задваивает счётчик


class TestStatsSummary:
    @staticmethod
    def _patch_worker_fakes(monkeypatch, fake_run_attachments):
        import src.pipeline as pipe

        monkeypatch.setattr(pipe, "run_attachments", fake_run_attachments)
        monkeypatch.setattr(pipe, "run_pdfs", lambda *a, **k: [])
        monkeypatch.setattr(pipe, "run_presentations", lambda *a, **k: [])
        monkeypatch.setattr(pipe, "run_spreadsheets", lambda *a, **k: [])

    def test_worker_reports_failed_files(self, cfg, monkeypatch, tmp_path):
        """_attachments_worker после закрытия фазы честно сообщает число
        необработанных файлов (раньше оно терялось в потоке лога)."""

        def fake_run(files, kind, weights, out_dir, progress, quiet, stats=None):
            if stats is not None:
                stats["failed"] = 3
                stats["total"] = 10
            return []

        self._patch_worker_fakes(monkeypatch, fake_run)

        from src.pipeline import _attachments_worker

        events: list[dict] = []
        f = tmp_path / "a.jpg"
        f.write_bytes(b"jpg")
        _attachments_worker([("images", [f], tmp_path)], {"images": 10}, events.append, [])

        msgs = [e["log"] for e in events if "не обработано" in e.get("log", "")]
        assert msgs, "сводка о неудачных файлах не выведена"
        assert "3 из 10" in msgs[0]

    def test_worker_silent_when_no_failures(self, cfg, monkeypatch, tmp_path):
        self._patch_worker_fakes(monkeypatch, lambda *a, **k: [])

        from src.pipeline import _attachments_worker

        events: list[dict] = []
        f = tmp_path / "a.jpg"
        f.write_bytes(b"jpg")
        _attachments_worker([("images", [f], tmp_path)], {"images": 10}, events.append, [])
        assert not any("не обработано" in e.get("log", "") for e in events)


class TestOcrCacheVersion:
    """Переработка механики картинок (описание сцены -> OCR): старые
    кэшированные описания не должны подменять новые OCR-результаты."""

    def test_images_hash_includes_ocr_version(self, cfg, tmp_path):
        files = [tmp_path / "p.jpg"]
        files[0].write_bytes(b"jpg")
        key = "|".join(f"{p.name}:{p.stat().st_size}" for p in files)
        plain_hash = __import__("hashlib").sha256(key.encode("utf-8")).hexdigest()
        h = att._cache_hash("images", files)
        assert h != plain_hash, "версия OCR не вошла в ключ кэша"
        # та же картинка в kind="documents" — версия не нужна (промпт не менялся)
        assert att._cache_hash("documents", files) == plain_hash
        # стабильность
        assert att._cache_hash("images", files) == h

    def test_ocr_header_in_output(self, cfg, no_cache, monkeypatch, tmp_path):
        monkeypatch.setattr(att.nim_client, "send_prompt", lambda prompt, **kw: "текста нет")
        f = tmp_path / "p.jpg"
        f.write_bytes(b"jpg")
        results = att.run_attachments([f], "images", {"images": 10}, tmp_path / "out")
        text = results[0].read_text(encoding="utf-8")
        assert "OCR КАРТИНОК" in text


class TestBatchCounterEvent:
    """Счётчик «N из M» в событиях фазы: при тысячах батчей целые проценты
    долго показывают 0% — счётчик показывает, что очередь движется."""

    def test_progress_events_carry_done_total(self, cfg, no_cache, monkeypatch, tmp_path):
        monkeypatch.setattr(att.nim_client, "send_prompt", lambda prompt, **kw: "текста нет")
        files = [tmp_path / f"p{i}.jpg" for i in range(4)]
        for p in files:
            p.write_bytes(b"jpg")

        events: list[dict] = []
        cfg.nim_images_per_batch = 2  # батчи по 2 → 2 батча
        att.run_attachments(files, "images", {"images": 10}, tmp_path / "out", events.append)

        counters = [
            (e["done"], e["total"])
            for e in events
            if e.get("p") == "images" and "done" in e
        ]
        assert counters == [(1, 2), (2, 2)], f"счётчик должен расти по завершении батчей: {counters}"
        # парные события всегда с phase_progress
        assert all("phase_progress" in e for e in events if e.get("p") == "images" and "done" in e)

    def test_single_batch_carries_done_total(self, cfg, no_cache, monkeypatch, tmp_path):
        monkeypatch.setattr(att.nim_client, "send_prompt", lambda prompt, **kw: "текста нет")
        f = tmp_path / "p.jpg"
        f.write_bytes(b"jpg")

        events: list[dict] = []
        att.run_attachments([f], "images", {"images": 10}, tmp_path / "out", events.append)

        counters = [(e["done"], e["total"]) for e in events if e.get("p") == "images" and "done" in e]
        assert counters == [(1, 1)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
