import sys
from pathlib import Path
from types import SimpleNamespace

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from src.whisper_extract import _transcribe_with_fallback


class _Seg:
    def __init__(self, text, start=0.0, end=1.0):
        self.text = text
        self.start = start
        self.end = end


class _Info:
    duration = 10.0


class _BatchedBroken:
    """Имитация BatchedInferencePipeline: падает на коротком аудио
    («tuple index out of range»), хранит обычную модель в .model."""

    def __init__(self, plain):
        self.model = plain
        self.calls: list[dict] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(dict(kwargs))
        raise RuntimeError("tuple index out of range")


class _Plain:
    def __init__(self, segments=None):
        self.segments = segments or [_Seg(" привет", 0.0, 1.0)]
        self.calls: list[dict] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(dict(kwargs))
        return iter(self.segments), _Info()


class TestBatchedFallback:
    def test_falls_back_to_plain_model(self):
        plain = _Plain()
        batched = _BatchedBroken(plain)

        segments, info = _transcribe_with_fallback(batched, "samples", {"beam_size": 1, "batch_size": 8}, None)

        assert len(segments) == 1
        assert segments[0].text.strip() == "привет"
        assert info.duration == 10.0

    def test_fallback_drops_batch_size_kwarg(self):
        plain = _Plain()
        batched = _BatchedBroken(plain)

        _transcribe_with_fallback(
            batched, "samples",
            {"vad_filter": True, "beam_size": 1, "batch_size": 8},
            None,
        )

        assert "batch_size" not in plain.calls[0], "batch_size передан обычному пайплайну"
        assert plain.calls[0]["beam_size"] == 1
        assert plain.calls[0]["vad_filter"] is True

    def test_no_fallback_without_plain_model(self):
        """Обычный WhisperModel (без .model) не зацикливается — ошибка наружу."""

        class _Broken:
            def transcribe(self, samples, **kwargs):
                raise RuntimeError("tuple index out of range")

        broken = _Broken()
        with pytest.raises(RuntimeError):
            _transcribe_with_fallback(broken, "samples", {}, None)

    def test_self_referencing_model_no_infinite_loop(self):
        class _SelfModel:
            model = None  # будет присвоен сам объект

        obj = _SelfModel()
        obj.model = obj

        def boom(samples, **kwargs):
            raise RuntimeError("x")

        obj.transcribe = boom
        with pytest.raises(RuntimeError):
            _transcribe_with_fallback(obj, "samples", {}, None)

    def test_progress_reported_from_fallback(self):
        plain = _Plain([_Seg(" один", 0, 2), _Seg(" два", 2, 4)])
        batched = _BatchedBroken(plain)
        ticks: list[int] = []

        _transcribe_with_fallback(batched, "samples", {}, ticks.append)

        assert ticks == [20, 40]  # 2/10 и 4/10 длительности


class TestNoVadFallback:
    """Сессия 14.10 08:40: 60+ файлов (стикеры-mp4, гифки) гибли с «tuple
    index out of range» и в batched, и в plain режиме — VAD вырезал всю
    дорожку на немом клипе. Повтор без VAD спасает такие файлы."""

    def test_vad_empty_audio_retried_without_vad(self):
        class _Batched:
            def __init__(self, plain):
                self.model = plain

            def transcribe(self, samples, **kwargs):
                raise RuntimeError("tuple index out of range")

        calls: list[dict] = []

        class _Plain:
            def transcribe(self, samples, **kwargs):
                calls.append(dict(kwargs))
                if kwargs.get("vad_filter"):
                    raise RuntimeError("tuple index out of range")
                return iter([_Seg(" эхо", 0.0, 1.0)]), _Info()

        batched = _Batched(_Plain())
        segments, info = _transcribe_with_fallback(
            batched, "samples", {"vad_filter": True, "beam_size": 1, "batch_size": 8}, None
        )

        assert len(segments) == 1
        assert segments[0].text.strip() == "эхо"
        assert len(calls) == 2
        assert calls[0].get("vad_filter") is True
        assert "vad_filter" not in calls[1], "повтор должен идти без VAD"
        assert "batch_size" not in calls[1]

    def test_plain_failure_without_vad_raises(self):
        """VAD и так выключен — повторять нечего, ошибка наружу."""

        class _Broken:
            def transcribe(self, samples, **kwargs):
                raise RuntimeError("tuple index out of range")

        with pytest.raises(RuntimeError):
            _transcribe_with_fallback(_Broken(), "samples", {"beam_size": 1}, None)


class TestShortAudioGraceful:
    """Клипы короче 3 с не транскрибируются ни в одном режиме: вместо
    исключения (ошибка на файл) пишется транскрипт-заглушка — файл не роняет
    задачу и не перекачивается при каждом перезапуске."""

    @staticmethod
    def _patch_transcribe_file(monkeypatch, tmp_path, samples, transcriber):
        import numpy as np

        import src.whisper_extract as we

        monkeypatch.setattr(we, "find_existing_transcript", lambda media, group_dir: None)
        monkeypatch.setattr(
            we, "transcript_path_for", lambda media, group_dir: tmp_path / "clip.transcript.txt"
        )
        monkeypatch.setattr(we, "get_transcriber", lambda size, fast: transcriber)
        monkeypatch.setattr(
            "faster_whisper.audio.decode_audio",
            lambda path, sampling_rate=16000: np.zeros(samples, dtype=np.float32),
        )
        return we

    def test_short_audio_writes_no_speech_transcript(self, tmp_path, monkeypatch):
        class _Broken:
            def transcribe(self, samples, **kwargs):
                raise RuntimeError("tuple index out of range")

        we = self._patch_transcribe_file(monkeypatch, tmp_path, 8000, _Broken())

        out = we.transcribe_file(tmp_path / "sticker.mp4", "small", "ru", diarize=False)
        text = out.read_text(encoding="utf-8")
        assert "Речь не распознана" in text
        assert "tuple index out of range" in text, "причина должна быть видна"
        assert "Сегментов:" in text

    def test_long_audio_failure_still_raises(self, tmp_path, monkeypatch):
        """Длинное аудио не должно «глотаться» молча — это реальная потеря."""

        class _Broken:
            def transcribe(self, samples, **kwargs):
                raise RuntimeError("модель сломалась")

        we = self._patch_transcribe_file(monkeypatch, tmp_path, 10 * 16000, _Broken())

        with pytest.raises(RuntimeError):
            we.transcribe_file(tmp_path / "talk.mp3", "small", "ru", diarize=False)

    def test_diarization_not_started_when_disabled(self, tmp_path, monkeypatch):
        """diarize=False: диаризация не запускалась (раньше флаг игнорировался)."""

        class _Ok:
            def transcribe(self, samples, **kwargs):
                return iter([_Seg(" привет", 0.0, 1.0)]), _Info()

        we = self._patch_transcribe_file(monkeypatch, tmp_path, 5 * 16000, _Ok())

        def _boom(samples):
            raise AssertionError("диаризация не должна запускаться при diarize=False")

        monkeypatch.setattr(we, "_diarize_samples", _boom)
        out = we.transcribe_file(tmp_path / "voice.mp3", "small", "ru", diarize=False)
        assert "привет" in out.read_text(encoding="utf-8")


class TestNoAudioStream:
    """Диагноз сессии 15.10: 90 mp4-мемов падали с «tuple index out of
    range» — файлы ТОЛЬКО с видео (h264), без аудиодорожки. Падал сам
    decode_audio (PyAV), до whisper и его фолбэков дело не доходило."""

    @staticmethod
    def _patch_decode_fail(monkeypatch, tmp_path, exc=None):
        import src.whisper_extract as we

        monkeypatch.setattr(we, "find_existing_transcript", lambda media, group_dir: None)
        monkeypatch.setattr(
            we, "transcript_path_for", lambda media, group_dir: tmp_path / "clip.transcript.txt"
        )
        monkeypatch.setattr(
            "faster_whisper.audio.decode_audio",
            lambda path, sampling_rate=16000: (_ for _ in ()).throw(exc or IndexError("tuple index out of range")),
        )
        return we

    def test_video_only_file_gets_no_speech_stub(self, tmp_path, monkeypatch):
        """Файл без аудиопотока → заглушка «без речи», whisper не грузится."""
        we = self._patch_decode_fail(monkeypatch, tmp_path)
        monkeypatch.setattr(we, "_has_audio_stream", lambda media: False)
        model_never_loaded: list = []

        def _no_model(size, fast):
            model_never_loaded.append(1)
            raise AssertionError("модель не должна грузиться для файла без аудио")

        monkeypatch.setattr(we, "get_transcriber", _no_model)

        out = we.transcribe_file(tmp_path / "meme.mp4", "small", "ru", diarize=False)
        text = out.read_text(encoding="utf-8")
        assert "Речь не распознана" in text
        assert "аудиодорожки" in text
        assert "Сегментов:" in text
        assert model_never_loaded == []

    def test_decode_failure_with_audio_stream_raises(self, tmp_path, monkeypatch):
        """Аудио в файле есть, но декодер сломался — реальная потеря, ошибка наружу."""
        we = self._patch_decode_fail(monkeypatch, tmp_path)
        monkeypatch.setattr(we, "_has_audio_stream", lambda media: True)
        monkeypatch.setattr(we, "get_transcriber", lambda size, fast: (_ for _ in ()).throw(AssertionError("не должен вызываться")))

        with pytest.raises(IndexError):
            we.transcribe_file(tmp_path / "broken.mp3", "small", "ru", diarize=False)

    def test_has_audio_stream_true_when_probe_fails(self, tmp_path, monkeypatch):
        """Не смогли открыть файл для проверки — считаем, что аудио есть
        (лучше честная ошибка, чем ложная заглушка)."""
        import src.whisper_extract as we

        def broken_open(path):
            raise RuntimeError("нет кодеков")

        import builtins

        monkeypatch.setattr("av.open", broken_open)
        assert we._has_audio_stream(tmp_path / "whatever.mp4") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
