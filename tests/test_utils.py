import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from src.utils import _chunk_lines, _chunk_hash, _is_small_chunk
from src.configuration import Settings


class TestChunkLines:
    def test_basic_chunking(self):
        lines = ["line1", "line2", "line3"]
        chunks = _chunk_lines(lines, budget=20)
        assert len(chunks) >= 1
        assert all(isinstance(c, list) for c in chunks)

    def test_respects_budget(self):
        lines = ["a" * 100, "b" * 100, "c" * 100]
        chunks = _chunk_lines(lines, budget=150)
        # Each line is 100 chars, budget 150 -> each line in separate chunk
        assert len(chunks) == 3

    def test_respects_max_lines(self):
        lines = ["x"] * 10
        # _chunk_lines uses settings.chunk_max_lines, not a parameter
        chunks = _chunk_lines(lines, budget=10000)
        # Should split into chunks of max settings.chunk_max_lines
        for chunk in chunks:
            assert len(chunk) <= 300  # default chunk_max_lines

    def test_empty_input(self):
        chunks = _chunk_lines([], budget=100)
        assert chunks == []


class TestChunkHash:
    def test_deterministic(self):
        source = "test_source"
        chunk = ["line1", "line2"]
        h1 = _chunk_hash(source, chunk)
        h2 = _chunk_hash(source, chunk)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_different_for_different_content(self):
        h1 = _chunk_hash("src", ["a"])
        h2 = _chunk_hash("src", ["b"])
        assert h1 != h2

    def test_different_for_different_source(self):
        h1 = _chunk_hash("src1", ["same"])
        h2 = _chunk_hash("src2", ["same"])
        assert h1 != h2


class TestIsSmallChunk:
    def test_small_chunk_detected(self):
        # Small chunk: half budget (50 chars << 5000) and few lines (50 << chunk_max_lines/2)
        chunk = ["x"] * 50
        assert _is_small_chunk(chunk, budget=10000) is True

    def test_large_chunk_not_small(self):
        # size 1_000_000 > budget/2 (5000)
        chunk = ["x" * 5000] * 200
        assert _is_small_chunk(chunk, budget=10000) is False


class TestConfiguration:
    def test_default_settings(self):
        s = Settings()
        assert s.whisper_model == "small"
        assert s.language == "auto"
        assert s.nim_workers == 8

    def test_validation_errors(self):
        s = Settings(nim_api_key="", nim_rpm=0, nim_max_tokens=-1, nim_temperature=3.0, nim_workers=0)
        errors = s.validate()
        assert len(errors) == 5
        assert any("NIM_API_KEY" in e for e in errors)
        assert any("NIM_RPM" in e for e in errors)
        assert any("NIM_MAX_TOKENS" in e for e in errors)
        assert any("NIM_TEMPERATURE" in e for e in errors)
        assert any("NIM_WORKERS" in e for e in errors)

    def test_valid_settings_no_errors(self):
        s = Settings(nim_api_key="test-key", nim_rpm=40, nim_max_tokens=16384, nim_temperature=0.7, nim_workers=8)
        errors = s.validate()
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])