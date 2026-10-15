import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from src.prompts import load_sections


class TestPromptsPlainText:
    """Контракт: выходной файл — txt, промпты обязывают LLM писать без markdown."""

    def test_map_requires_plain_text(self):
        s = load_sections()
        assert "Никакой markdown" in s["map"]
        assert "звёздочек" in s["map"]
        # старое требование «маркированный список» убрано
        assert "маркированный список" not in s["map"]

    def test_reduce_requires_plain_text(self):
        s = load_sections()
        assert "Никакого markdown" in s["reduce"]
        assert "звёздочек" in s["reduce"]
        # заголовки разделов больше не «##»
        assert "##" not in s["reduce"]
        assert "КЛЮЧЕВЫЕ СОБЫТИЯ И ОБЪЯВЛЕНИЯ" in s["reduce"]

    def test_images_section_is_ocr_only(self):
        """Механика картинок переработана: вместо распознавания сцены —
        только OCR (описание содержимого признано избыточным)."""
        s = load_sections()
        assert "без markdown" in s["images"]
        # OCR-семантика
        assert "ТОЛЬКО OCR" in s["images"]
        assert "дословно" in s["images"]
        assert "«текста нет»" in s["images"]
        # описание сцены убрано
        assert "Подробно опиши" not in s["images"]
        assert "опиши содержимое" not in s["images"]
        # плейсхолдер количества картинок на месте
        assert "{count}" in s["images"]

    def test_images_and_documents_require_plain_text(self):
        s = load_sections()
        assert "без markdown" in s["images"]
        assert "без markdown" in s["documents"]

    def test_spreadsheets_section(self):
        s = load_sections()
        assert s.get("spreadsheets"), "секция spreadsheets пуста"
        # промпт объясняет формат строк и требует plain text
        assert "КОЛОНКИ" in s["spreadsheets"]
        assert "без markdown" in s["spreadsheets"]
        # плейсхолдеры map-типа на месте
        for ph in ("{i}", "{n}", "{source}", "{body}"):
            assert ph in s["spreadsheets"]

    def test_no_emoji_or_md_tables_rule(self):
        """Модель склонна вставлять emoji и markdown-таблицы — промпты запрещают явно."""
        s = load_sections()
        for name in ("map", "reduce", "spreadsheets"):
            sec = s[name]
            assert "emoji" in sec, f"в секции {name} нет запрета emoji"
            assert "markdown-таблиц" in sec, f"в секции {name} нет запрета markdown-таблиц"
            assert "дефис" in sec, f"в секции {name} нет указания на дефис-перечисления"

    def test_all_sections_present(self):
        s = load_sections()
        for name in ("map", "reduce", "images", "documents", "spreadsheets"):
            assert s.get(name), f"секция {name} пуста"


    def test_default_and_prompt_md_agree_on_ocr(self):
        """Runtime-копия prompt.md имеет приоритет — она тоже должна быть OCR."""
        s = load_sections()
        assert "ТОЛЬКО OCR" in s["images"]
        assert "Подробно опиши" not in s["images"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
