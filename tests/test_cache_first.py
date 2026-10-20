import sys
from pathlib import Path
from types import SimpleNamespace

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.attachments as att


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
        nim_chunk_char_budget=8000,
        results_dir=tmp_path / "results",
        tmp_pdfs_dir=tmp_path / "tmp_pdfs",
    )
    fake.results_dir.mkdir(parents=True, exist_ok=True)
    fake.tmp_pdfs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(att, "settings", fake)
    return fake


def _make_pdf(path: Path):
    path.write_bytes(b"%PDF-1.4 fake")


class TestPdfCacheFirst:
    def test_cache_hit_skips_extraction(self, cfg, monkeypatch, tmp_path):
        """Попадание в кэш: PdfReader вообще не открывается — раньше он
        перемалывал все страницы (сканы — десятки секунд) перед проверкой кэша."""
        pdf = tmp_path / "book.pdf"
        _make_pdf(pdf)

        def no_reader(*a, **k):
            raise AssertionError("PdfReader не должен открываться при попадании в кэш")

        monkeypatch.setattr("pypdf.PdfReader", no_reader)
        monkeypatch.setattr(att, "get_cached_fragment", lambda h: "Кэшированный консПЕКТ")

        events: list[dict] = []
        results = att._process_single_pdf(pdf, {"pdf": 10}, events.append, tmp_path / "out", True)
        assert len(results) == 1
        text = results[0].read_text(encoding="utf-8")
        assert "Кэшированный консПЕКТ" in text
        assert "КОНСПЕКТ PDF" in text

    def test_scan_marker_skips_extraction_and_goes_to_vision(self, cfg, monkeypatch, tmp_path):
        """Маркер «это скан» — извлечение пропускается, сразу Vision fallback."""
        pdf = tmp_path / "scan.pdf"
        _make_pdf(pdf)

        def no_reader(*a, **k):
            raise AssertionError("извлечение текста не нужно для известного скана")

        monkeypatch.setattr("pypdf.PdfReader", no_reader)

        def fake_cache(h: str):
            return "1" if h.endswith("|scan") else None

        monkeypatch.setattr(att, "get_cached_fragment", fake_cache)
        sentinel = [tmp_path / "vision.txt"]
        monkeypatch.setattr(att, "_process_pdf_as_images", lambda *a, **k: sentinel)

        results = att._process_single_pdf(pdf, {"pdf": 10}, None, tmp_path / "out", True)
        assert results is sentinel

    def test_scan_verdict_is_cached_on_discovery(self, cfg, monkeypatch, tmp_path):
        """Первичное обнаружение скана сохраняет маркер: следующий запуск
        не будет заново извлекать текст из всех страниц."""
        pdf = tmp_path / "scan2.pdf"
        _make_pdf(pdf)
        saved: list[tuple] = []
        monkeypatch.setattr(att, "get_cached_fragment", lambda h: None)
        monkeypatch.setattr(att, "save_fragment_cache", lambda h, s, t: saved.append((h, s)))

        class _EmptyReader:
            def __init__(self, path):
                self.pages = []

        monkeypatch.setattr("pypdf.PdfReader", _EmptyReader)
        sentinel = [tmp_path / "vision.txt"]
        monkeypatch.setattr(att, "_process_pdf_as_images", lambda *a, **k: sentinel)

        att._process_single_pdf(pdf, {"pdf": 10}, None, tmp_path / "out", True)

        markers = [(h, s) for h, s in saved if h.endswith("|scan")]
        assert markers, "маркер скана не сохранён"
        assert markers[0][1] == "pdf_scan_marker"


class TestScanRenderReuse:
    def test_existing_pages_not_rerendered(self, cfg, monkeypatch, tmp_path):
        """PNG страниц уже отрендерены прошлым запуском — poppler не вызывается."""
        pdf = tmp_path / "book.pdf"
        _make_pdf(pdf)

        from pypdf import PdfReader

        class _FakeReader:
            def __init__(self, path):
                self.pages = [object(), object(), object()]  # 3 страницы

        monkeypatch.setattr("pypdf.PdfReader", _FakeReader)

        def no_convert(*a, **k):
            raise AssertionError("готовые страницы не должны рендериться заново")

        monkeypatch.setattr(att, "convert_from_path", no_convert)

        pages = []
        key = att.hashlib.sha256(f"{pdf.name}|{pdf.stat().st_size}".encode("utf-8")).hexdigest()[:8]
        for i in range(1, 4):
            p = cfg.tmp_pdfs_dir / f"book_{key}_{i:04d}.png"
            p.write_bytes(b"png")
            pages.append(p)

        out = att._pdf_to_images(pdf, dpi=200)
        assert [p.name for p in out] == [p.name for p in pages]
        assert all(p.exists() for p in out)


class TestPptCacheFirst:
    def test_cache_hit_skips_conversion_and_extraction(self, cfg, monkeypatch, tmp_path):
        """Кэшированная презентация: ни COM-конвертации, ни разбора слайдов."""
        ppt = tmp_path / "deck.pptx"
        ppt.write_bytes(b"PK")

        def no_extract(path):
            raise AssertionError("извлечение не нужно при попадании в кэш")

        def no_convert(ppt_path):
            raise AssertionError("COM-конвертация не нужна при попадании в кэш")

        monkeypatch.setattr(att, "_extract_pptx_lines", no_extract)
        monkeypatch.setattr(att, "_convert_ppt_to_pptx_via_com", no_convert)
        monkeypatch.setattr(att, "get_cached_fragment", lambda h: "Кэш-консПЕКТ слайдов")

        results = att._process_single_presentation(ppt, {"ppt": 10}, None, tmp_path / "out", True)
        assert len(results) == 1
        assert "Кэш-консПЕКТ слайдов" in results[0].read_text(encoding="utf-8")


class TestSheetCacheFirst:
    def test_cache_hit_skips_extraction(self, cfg, monkeypatch, tmp_path):
        """Кэшированная таблица: openpyxl не открывается (прайсы с десятками
        тысяч строк разбирались заново при каждом запуске)."""
        import src.spreadsheets as sheets

        xlsx = tmp_path / "price.xlsx"
        xlsx.write_bytes(b"PK")

        def no_extract(path):
            raise AssertionError("извлечение не нужно при попадании в кэш")

        monkeypatch.setattr(sheets, "_extract_xlsx_lines", no_extract)
        monkeypatch.setattr(sheets, "get_cached_fragment", lambda h: "Кэш-консПЕКТ таблицы")

        results = sheets._process_single_spreadsheet(xlsx, {"spreadsheets": 10}, None, tmp_path / "out", True)
        assert len(results) == 1
        assert "Кэш-консПЕКТ таблицы" in results[0].read_text(encoding="utf-8")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
