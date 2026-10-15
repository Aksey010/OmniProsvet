import sys
from pathlib import Path
from types import SimpleNamespace

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

import src.whisper_extract as we
from src.whisper_extract import get_transcriber, resolve_whisper_device


@pytest.fixture()
def cfg(monkeypatch):
    """Настройки Whisper с управляемым устройством; кэш моделей чистый."""
    fake = SimpleNamespace(
        whisper_device="auto",
        whisper_compute_type="int8",
        whisper_fast=False,
        whisper_batch_size=8,
    )
    monkeypatch.setattr(we, "settings", fake)
    monkeypatch.setattr(we, "_transcribers", {})
    return fake


class TestResolveDevice:
    def test_auto_without_cuda_is_cpu(self, cfg, monkeypatch):
        monkeypatch.setattr(we, "_cuda_available", lambda: False)
        assert resolve_whisper_device() == ("cpu", "int8")

    def test_auto_with_cuda_upgrades_int8_to_float16(self, cfg, monkeypatch):
        monkeypatch.setattr(we, "_cuda_available", lambda: True)
        assert resolve_whisper_device() == ("cuda", "float16")

    def test_explicit_compute_type_kept_on_gpu(self, cfg, monkeypatch):
        cfg.whisper_compute_type = "int8_float16"
        monkeypatch.setattr(we, "_cuda_available", lambda: True)
        assert resolve_whisper_device() == ("cuda", "int8_float16")

    def test_forced_cpu_even_with_cuda(self, cfg, monkeypatch):
        cfg.whisper_device = "cpu"
        cfg.whisper_compute_type = "int8"
        monkeypatch.setattr(we, "_cuda_available", lambda: True)
        assert resolve_whisper_device() == ("cpu", "int8")

    def test_forced_cuda_without_gpu_still_cuda(self, cfg, monkeypatch):
        """Резолвер честно возвращает cuda — фолбэк на CPU делает get_transcriber."""
        cfg.whisper_device = "cuda"
        monkeypatch.setattr(we, "_cuda_available", lambda: False)
        assert resolve_whisper_device() == ("cuda", "float16")

    def test_garbage_value_treated_as_auto(self, cfg, monkeypatch):
        cfg.whisper_device = "GPU pls"
        monkeypatch.setattr(we, "_cuda_available", lambda: False)
        assert resolve_whisper_device() == ("cpu", "int8")

    def test_cuda_detection_survives_errors(self, monkeypatch):
        """Нет ctranslate2/кривой драйвер — детекция возвращает False, не падает."""
        import types

        def boom():
            raise RuntimeError("CUDA driver version is insufficient")

        monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=boom))
        assert we._cuda_available() is False

    def test_cuda_detection_without_module(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ctranslate2", None)  # import упадёт
        assert we._cuda_available() is False


class TestGetTranscriber:
    def test_cpu_passes_int8(self, cfg, monkeypatch):
        monkeypatch.setattr(we, "_cuda_available", lambda: False)
        captured = {}

        class FakeModel:
            def __init__(self, size, device="cpu", compute_type="int8"):
                captured["args"] = (size, device, compute_type)

        monkeypatch.setattr(we, "WhisperModel", FakeModel)
        get_transcriber("small", False)
        assert captured["args"] == ("small", "cpu", "int8")

    def test_gpu_passes_float16(self, cfg, monkeypatch):
        monkeypatch.setattr(we, "_cuda_available", lambda: True)
        captured = {}

        class FakeModel:
            def __init__(self, size, device="cpu", compute_type="int8"):
                captured["args"] = (size, device, compute_type)

        monkeypatch.setattr(we, "WhisperModel", FakeModel)
        get_transcriber("small", False)
        assert captured["args"] == ("small", "cuda", "float16")

    def test_forced_cuda_falls_back_to_cpu_on_error(self, cfg, monkeypatch, caplog):
        """cuda без рабочего драйвера → warning + модель на CPU, без исключения."""
        cfg.whisper_device = "cuda"
        calls = []

        class FlakyModel:
            def __init__(self, size, device="cpu", compute_type="int8"):
                calls.append((device, compute_type))
                if device == "cuda":
                    raise RuntimeError("CUDA driver version is insufficient")

        monkeypatch.setattr(we, "WhisperModel", FlakyModel)
        model = get_transcriber("small", False)
        assert model is not None
        assert calls[0] == ("cuda", "float16")  # попытка на GPU
        assert calls[1] == ("cpu", "int8")  # фолбэк с CPU-типом
        assert len(calls) == 2

    def test_cpu_error_raises(self, cfg, monkeypatch):
        """На CPU ошибка загрузки — честное исключение (фолбэк не нужен)."""

        class BrokenModel:
            def __init__(self, *a, **k):
                raise RuntimeError("model download failed")

        monkeypatch.setattr(we, "WhisperModel", BrokenModel)
        with pytest.raises(RuntimeError, match="model download failed"):
            get_transcriber("small", False)

    def test_model_cached_by_key(self, cfg, monkeypatch):
        monkeypatch.setattr(we, "_cuda_available", lambda: False)
        created = []

        class FakeModel:
            def __init__(self, size, device="cpu", compute_type="int8"):
                created.append(1)

        monkeypatch.setattr(we, "WhisperModel", FakeModel)
        m1 = get_transcriber("small", False)
        m2 = get_transcriber("small", False)
        assert m1 is m2
        assert len(created) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
