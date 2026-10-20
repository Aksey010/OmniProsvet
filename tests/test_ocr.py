import sys
from pathlib import Path
from types import SimpleNamespace

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.attachments as att
import src.ocr as ocr


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
        ocr_engine="rapidocr",
        ocr_workers=2,
        ocr_min_confidence=0.5,
        results_dir=tmp_path / "results",
        tmp_pdfs_dir=tmp_path / "tmp_pdfs",
    )
    fake.results_dir.mkdir(parents=True, exist_ok=True)
    fake.tmp_pdfs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(att, "settings", fake)
    return fake


def _imgs(tmp_path, n=2):
    files = [tmp_path / f"p{i}.jpg" for i in range(n)]
    for p in files:
        p.write_bytes(b"jpg")
    return files


class TestGroupLines:
    def test_scrambled_boxes_read_in_order(self):
        """Боксы в перемешанном порядке -> строки сверху вниз, слева направо."""
        boxes = [
            [[200, 100], [300, 100], [300, 140], [200, 140]],  # строка 2, справа
            [[100, 0], [150, 0], [150, 40], [100, 40]],        # строка 1, слева
            [[100, 100], [180, 100], [180, 140], [100, 140]],  # строка 2, слева
            [[100, 200], [150, 200], [150, 240], [100, 240]],  # строка 3
        ]
        txts = ("мир", "привет", "снова", "конец")
        scores = (0.9, 0.9, 0.9, 0.9)
        lines = ocr._group_lines(boxes, txts, scores, 0.5)
        assert lines == ["привет", "снова мир", "конец"]

    def test_low_confidence_dropped(self):
        boxes = [[[100, 0], [150, 0], [150, 40], [100, 40]]]
        lines = ocr._group_lines(boxes, ("шум",), (0.2,), 0.5)
        assert lines == []

    def test_empty_result(self):
        assert ocr._group_lines(None, None, None, 0.5) == []

    def test_line_merges_by_height_not_absolute_gap(self):
        """Толерантность пропорциональна высоте строки: крупный текст с
        большим межстрочным зазором не склеивается в одну строку."""
        h = 100
        boxes = [
            [[100, 0], [200, 0], [200, h], [100, h]],
            [[100, h * 2], [200, h * 2], [200, h * 3], [100, h * 3]],
        ]
        lines = ocr._group_lines(boxes, ("верх", "низ"), (0.9, 0.9), 0.5)
        assert lines == ["верх", "низ"]


class TestOcrImage:
    def test_no_text_returns_empty(self, monkeypatch):
        class _EmptyOut:
            boxes = None
            txts = None
            scores = None

        monkeypatch.setattr(ocr, "get_engine", lambda: (lambda p: _EmptyOut()))
        assert ocr.ocr_image(Path("x.png")) == ""

    def test_engine_error_raises_to_caller(self, monkeypatch):
        def boom(p):
            raise RuntimeError("битая картинка")

        monkeypatch.setattr(ocr, "get_engine", lambda: boom)
        with pytest.raises(RuntimeError):
            ocr.ocr_image(Path("x.png"))


class TestWiringImages:
    def test_local_ocr_skips_nim(self, cfg, monkeypatch, tmp_path):
        """OCR_ENGINE=rapidocr: NIM не вызывается вовсе, файл и кэш на месте."""
        files = _imgs(tmp_path, 2)

        def no_nim(prompt, **kw):
            raise AssertionError("NIM не должен вызываться при локальном OCR")

        monkeypatch.setattr(att.nim_client, "send_prompt", no_nim)
        monkeypatch.setattr(att.local_ocr, "RAPIDOCR_AVAILABLE", True)
        texts = iter(["текст один", ""])
        monkeypatch.setattr(
            att.local_ocr, "ocr_images_parallel",
            lambda paths, workers=None, min_confidence=None: list(texts),
        )
        saved: list[tuple] = []
        monkeypatch.setattr(att, "save_fragment_cache", lambda h, s, t: saved.append((h, s)))

        results = att.run_attachments(files, "images", {"images": 10}, tmp_path / "out")
        assert len(results) == 1
        text = results[0].read_text(encoding="utf-8")
        assert "Картинка 1:\nтекст один" in text
        assert "Картинка 2:\nтекста нет" in text  # пустой OCR -> честная пометка
        assert "OCR КАРТИНОК" in text
        assert saved and saved[0][1] == "images"

    def test_fallback_to_nim_when_package_missing(self, cfg, monkeypatch, tmp_path):
        """Пакет rapidocr не установлен -> прежний NIM-путь с предупреждением."""
        files = _imgs(tmp_path, 1)
        monkeypatch.setattr(att.local_ocr, "RAPIDOCR_AVAILABLE", False)

        def local_forbidden(paths, **kw):
            raise AssertionError("локальный OCR недоступен, но вызван")

        monkeypatch.setattr(att.local_ocr, "ocr_images_parallel", local_forbidden)
        monkeypatch.setattr(att.nim_client, "send_prompt", lambda prompt, **kw: "nim ответ")

        results = att.run_attachments(files, "images", {"images": 10}, tmp_path / "out")
        assert "nim ответ" in results[0].read_text(encoding="utf-8")

    def test_engine_nim_uses_nim_path(self, cfg, monkeypatch, tmp_path):
        """OCR_ENGINE=nim (дефолт) — прежнее поведение без локального OCR."""
        cfg.ocr_engine = "nim"
        files = _imgs(tmp_path, 1)
        monkeypatch.setattr(att.local_ocr, "RAPIDOCR_AVAILABLE", True)

        def local_forbidden(paths, **kw):
            raise AssertionError("локальный OCR не должен вызываться при engine=nim")

        monkeypatch.setattr(att.local_ocr, "ocr_images_parallel", local_forbidden)
        monkeypatch.setattr(att.nim_client, "send_prompt", lambda prompt, **kw: "nim ответ")

        results = att.run_attachments(files, "images", {"images": 10}, tmp_path / "out")
        assert "nim ответ" in results[0].read_text(encoding="utf-8")


class TestWiringScans:
    def test_scan_pages_local_ocr_with_page_markers(self, cfg, monkeypatch, tmp_path):
        """Страницы сканов локальным OCR: маркеры [Страница N], без NIM."""
        pages = _imgs(tmp_path, 3)

        def no_nim(prompt, **kw):
            raise AssertionError("NIM не должен вызываться для сканов при локальном OCR")

        monkeypatch.setattr(att.nim_client, "send_prompt", no_nim)
        monkeypatch.setattr(att.local_ocr, "RAPIDOCR_AVAILABLE", True)
        monkeypatch.setattr(
            att.local_ocr, "ocr_images_parallel",
            lambda paths, workers=None, min_confidence=None: ["текст стр.1", "", "текст стр.3"],
        )

        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"pdf")
        out = att._recognize_scan_batch(pages, pdf, first_page=7, total_pages=10,
                                        out_dir=tmp_path / "out", progress=None)
        text = out.read_text(encoding="utf-8")
        assert "[Страница 7]\nтекст стр.1" in text
        assert "[Страница 8]\nтекста нет" in text
        assert "[Страница 9]\nтекст стр.3" in text


class TestRealEngine:
    @pytest.mark.skipif(not ocr.RAPIDOCR_AVAILABLE, reason="rapidocr не установлен")
    def test_russian_text_recognized(self, tmp_path):
        """Интеграционная проверка: кириллическая модель PP-OCRv5 читает
        русский текст (дефолтная ch-модель PP-OCRv6 его ТЕРЯЛА)."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (900, 200), "white")
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 48)
        d.text((30, 30), "Скидка 25% до 31 декабря", font=font, fill="black")
        d.text((30, 120), "Каталог 16 2025", font=font, fill="black")
        p = tmp_path / "rus.png"
        img.save(p)

        text = ocr.ocr_image(p)
        assert "25" in text, f"цифры не распознаны: {text!r}"
        assert "декабря" in text.lower(), f"кириллица не распознана: {text!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
