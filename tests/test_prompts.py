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

    def test_images_and_documents_require_plain_text(self):
        s = load_sections()
        assert "без markdown" in s["images"]
        assert "без markdown" in s["documents"]

    def test_all_sections_present(self):
        s = load_sections()
        for name in ("map", "reduce", "images", "documents"):
            assert s.get(name), f"секция {name} пуста"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
