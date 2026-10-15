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
        nim_images_per_batch=2,
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


def _no_cache(monkeypatch):
    monkeypatch.setattr(att, "get_cached_fragment", lambda h: None)
    monkeypatch.setattr(att, "save_fragment_cache", lambda h, s, t: None)


def _fake_pdf_pages(monkeypatch, tmp_path, n: int) -> list[Path]:
    pages = []
    for i in range(1, n + 1):
        p = tmp_path / "tmp_pdfs" / f"scan_{i:04d}.png"
        p.write_bytes(b"png")
        pages.append(p)
    monkeypatch.setattr(att, "_pdf_to_images", lambda pdf, dpi=200: list(pages))
    return pages


class TestScanBatchIsolation:
    """Vision fallback сканов обязан переживать сбои NIM.

    В сессии 14.10 (00:09–00:30) три сканированных PDF гибли ЦЕЛИКОМ: один
    ReadTimeout на батче страниц отменял все остальные страницы («ошибка
    Vision fallback … return []»).
    """

    def test_failed_batch_does_not_kill_rest(self, cfg, monkeypatch, tmp_path):
        _no_cache(monkeypatch)
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"pdf")
        _fake_pdf_pages(monkeypatch, tmp_path, 3)

        def fake_send(prompt, **kwargs):
            if len(kwargs["attachments"]) > 1:
                raise NimError("NIM: не удалось получить ответ после 5 попыток. Сеть: ReadTimeout")
            return f"текст {Path(kwargs['attachments'][0]).name}"

        monkeypatch.setattr(att.nim_client, "send_prompt", fake_send)
        events: list[dict] = []
        results = att._process_pdf_as_images(pdf, {"pdf": 10}, events.append, tmp_path / "out", False)

        # батчи по 2: [1,2] не распознан, [3] успешен
        assert len(results) == 1, "успешная страница потеряна из-за сбойного батча"
        assert any("не распознано страниц: 2 из 3" in e.get("log", "") for e in events)

    def test_many_images_400_splits_to_single_pages(self, cfg, monkeypatch, tmp_path):
        """400 «At most 1 image»: батч страниц распиливается на одиночки —
        все страницы распознаны (раньше терялся весь скан, сессия 09.10)."""
        _no_cache(monkeypatch)
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"pdf")
        _fake_pdf_pages(monkeypatch, tmp_path, 4)

        calls: list[int] = []

        def fake_send(prompt, **kwargs):
            n = len(kwargs["attachments"])
            calls.append(n)
            if n > 1:
                raise NimError(
                    "NIM: HTTP 400: {'message': 'At most 1 image(s) may be provided in one request.'}"
                )
            return "ок"

        monkeypatch.setattr(att.nim_client, "send_prompt", fake_send)
        results = att._process_pdf_as_images(pdf, {"pdf": 10}, None, tmp_path / "out", False)

        assert len(results) == 4, "все 4 страницы должны быть распознаны"
        # батчи [1,2] и [3,4]: одна много-картиночная попытка на батч, затем одиночки
        assert calls.count(2) == 2
        assert calls.count(1) == 4

    def test_single_page_failure_only_loses_that_page(self, cfg, monkeypatch, tmp_path):
        cfg.nim_images_per_batch = 1
        _no_cache(monkeypatch)
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"pdf")
        _fake_pdf_pages(monkeypatch, tmp_path, 3)

        def fake_send(prompt, **kwargs):
            if Path(kwargs["attachments"][0]).name.endswith("0002.png"):
                raise NimError("NIM: HTTP 500 (лимит запросов?)")
            return "ок"

        monkeypatch.setattr(att.nim_client, "send_prompt", fake_send)
        events: list[dict] = []
        results = att._process_pdf_as_images(pdf, {"pdf": 10}, events.append, tmp_path / "out", False)

        assert len(results) == 2, "соседние страницы не должны пострадать"
        assert any("не распознано страниц: 1 из 3" in e.get("log", "") for e in events)

    def test_conversion_failure_returns_empty_without_crash(self, cfg, monkeypatch, tmp_path):
        """Сбой конвертации PDF → пустой результат и понятный лог, не исключение."""
        _no_cache(monkeypatch)
        pdf = tmp_path / "broken.pdf"
        pdf.write_bytes(b"pdf")

        def broken_convert(pdf, dpi=200):
            raise RuntimeError("poppler не найден")

        monkeypatch.setattr(att, "_pdf_to_images", broken_convert)
        events: list[dict] = []
        results = att._process_pdf_as_images(pdf, {"pdf": 10}, events.append, tmp_path / "out", False)
        assert results == []
        assert any("конвертация в изображения не удалась" in e.get("log", "") for e in events)


class TestStableScanPageNames:
    def test_page_names_are_stable_and_numbered(self, cfg, monkeypatch, tmp_path):
        """Имена страниц содержат хэш PDF и номер: ключ кэша одинаков между
        перезапусками — распознанные страницы берутся из кэша."""
        from pdf2image.generators import counter_generator

        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"pdf")
        key = att.hashlib.sha256(f"{pdf.name}|{pdf.stat().st_size}".encode("utf-8")).hexdigest()[:8]
        prefix = f"{pdf.stem}_{key}_"
        gen = counter_generator(prefix=prefix)
        names = [next(gen) for _ in range(3)]
        assert names[0] == f"scan_{key}_0001"
        assert names[2] == f"scan_{key}_0003"
        # повторный вызов даёт те же имена (стабильность между запусками)
        gen2 = counter_generator(prefix=prefix)
        assert [next(gen2) for _ in range(3)] == names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
