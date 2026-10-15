import threading
import time
from pathlib import Path

from faster_whisper import WhisperModel
from loguru import logger

from src.collect import find_existing_transcript, transcript_path_for
from src.configuration import settings

_transcribers: dict[tuple[str, bool], object] = {}
_model_lock = threading.Lock()

_diar_pipelines: dict[str, object] = {}
_diar_lock = threading.Lock()

DIAR_MAX_SAMPLES = 57_600_000
DIAR_OVERLAP_SEC = 30

# Клипы короче 3 секунд (стикеры-mp4, гифки без речи) не транскрибируются
# ни в одном режиме — вместо исключения для них пишется транскрипт без речи.
SHORT_AUDIO_SAMPLES = 3 * 16_000


def _has_audio_stream(media: Path) -> bool:
    """Есть ли в файле аудиопоток.

    Не смогли открыть/проверить — отвечаем True: лучше честная ошибка
    транскрибации, чем ложная заглушка «без речи» для битого файла с аудио.
    """
    try:
        import av

        with av.open(str(media)) as container:
            return any(s.type == "audio" for s in container.streams)
    except Exception:
        return True


def _write_no_speech_transcript(
    out_path: Path, media: Path, reason: str, duration_sec: float | None = None
) -> Path:
    """Транскрипт-заглушка для файла без распознаваемой речи.

    Пишется вместо исключения: файл не роняет задачу, помечается
    обработанным и не перекачивается при каждом перезапуске
    (existing transcript -> skip).
    """
    meta = []
    if duration_sec is not None:
        meta.append(f"{'Длительность:':<14}{duration_sec:.0f} сек")
    meta.append(f"{'Сегментов:':<14}0")
    header = (
        f"{'=' * 64}\n"
        "  ТРАНСКРИПТ\n"
        f"{'=' * 64}\n"
        f"{'Файл:':<14}{media.name}\n"
        f"{'Дата:':<14}{time.strftime('%d.%m.%Y %H:%M')}\n"
        + "\n".join(meta)
        + f"\n{'-' * 64}\n\n"
    )
    note = f"  Речь не распознана: {reason}\n"
    footer = f"{'-' * 64}\n  Конец транскрипта · OmniProsvet\n{'=' * 64}\n"
    out_path.write_text(header + note + "\n" + footer, encoding="utf-8")
    logger.info(f"Транскрипт-заглушка (без речи): {out_path.name}")
    return out_path


def _cuda_available() -> bool:
    """Есть ли CUDA-устройства (без падения на машинах без драйверов)."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_whisper_device() -> tuple[str, str]:
    """Разрешить (device, compute_type) из настроек и наличия CUDA.

    WHISPER_DEVICE: auto (cuda при наличии GPU, иначе cpu) | cuda | cpu.
    На GPU CPU-дефолт int8 повышается до float16 — быстрее; если нужна
    экономия VRAM, задайте int8_float16 явно.
    """
    wanted = settings.whisper_device
    if wanted not in ("auto", "cuda", "cpu"):
        logger.warning(f"WHISPER_DEVICE={settings.whisper_device!r} не понят — трактую как auto")
        wanted = "auto"
    if wanted == "cuda" or (wanted == "auto" and _cuda_available()):
        device = "cuda"
        compute = settings.whisper_compute_type
        if compute == "int8":  # CPU-дефолт на GPU не оптимален
            compute = "float16"
    else:
        device = "cpu"
        compute = settings.whisper_compute_type
    return device, compute


def get_transcriber(size: str, fast: bool) -> object:
    key = (size, fast)
    with _model_lock:
        if key not in _transcribers:
            device, compute = resolve_whisper_device()
            logger.info(f"Загрузка модели whisper '{size}' (первый раз — скачивается)…")
            try:
                model = WhisperModel(size, device=device, compute_type=compute)
            except Exception as exc:
                if device != "cuda":
                    raise
                # Принудительный cuda без рабочего драйвера — не падаем, уходим на CPU
                logger.warning(
                    f"CUDA недоступна ({type(exc).__name__}: {exc}) — fallback на CPU "
                    f"с compute_type={settings.whisper_compute_type}"
                )
                model = WhisperModel(size, device="cpu", compute_type=settings.whisper_compute_type)
                device, compute = "cpu", settings.whisper_compute_type
            logger.info(f"Whisper: device={device}, compute_type={compute}")
            if device == "cuda":
                logger.info(
                    "GPU-режим: на одной карте оптимально WHISPER_WORKERS=1..2 "
                    "и WHISPER_BATCH_SIZE=16..32 (в .env)"
                )
            if fast:
                from faster_whisper import BatchedInferencePipeline

                logger.info("Быстрый режим: BatchedInferencePipeline + beam_size=1")
                _transcribers[key] = BatchedInferencePipeline(model=model)
            else:
                _transcribers[key] = model
        return _transcribers[key]


def get_diar_pipeline(variant: str) -> object:
    import polyvoice

    key = variant if variant in ("balanced", "mobile") else "balanced"
    with _diar_lock:
        if key not in _diar_pipelines:
            logger.info(f"Загрузка модели диаризации polyvoice ({key}, первый раз — скачивается)…")
            factory = getattr(polyvoice.Pipeline, key)
            _diar_pipelines[key] = factory()
        return _diar_pipelines[key]


def _prettify_speaker(label) -> str:
    if label is None or isinstance(label, bool):
        return "Спикер ?"
    if isinstance(label, int):
        return f"Спикер {label + 1}"
    text = str(label)
    if text.startswith("SPEAKER_"):
        try:
            return f"Спикер {int(text.split('_')[1]) + 1}"
        except (IndexError, ValueError):
            return text
    return text


def _assign_speakers(segments: list, turns: list[dict]) -> None:
    for seg in segments:
        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg.end, turn["end"]) - max(seg.start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg.speaker = best_speaker


def _collect_segments(segments_iter, info, on_progress) -> list:
    segments = []
    for seg in segments_iter:
        if not seg.text.strip():
            continue
        segments.append(seg)
        if on_progress and info.duration:
            on_progress(min(100, round(seg.end / info.duration * 100)))
    return segments


def _transcribe_with_fallback(transcriber, samples, kwargs: dict, on_progress) -> tuple[list, object]:
    """Транскрибация + сбор сегментов с цепочкой фолбэков.

    1) BatchedInferencePipeline (WHISPER_FAST=1) падает с «tuple index out of
       range» на очень коротком аудио — повтор обычной моделью без батчинга
       (BatchedInferencePipeline хранит исходный WhisperModel в .model).
    2) Если упал и обычный режим при включённом VAD — повтор БЕЗ VAD: на
       немых/шумовых клипах (стикеры-mp4, гифки) VAD вырезает всю дорожку,
       мел-спектрограмма пустая, и «tuple index out of range» возникает в
       самом whisper (сессия 14.10 08:40: 60+ файлов погибли в обоих режимах).
    """
    try:
        segments_iter, info = transcriber.transcribe(samples, **kwargs)
        return _collect_segments(segments_iter, info, on_progress), info
    except Exception as exc:
        plain_model = getattr(transcriber, "model", None)
        if plain_model is None or plain_model is transcriber:
            raise
        logger.warning(
            f"Пакетный режим не справился ({type(exc).__name__}: {exc}) — "
            f"повторяю без батчинга"
        )
        plain_kwargs = {k: v for k, v in kwargs.items() if k != "batch_size"}
        try:
            segments_iter, info = plain_model.transcribe(samples, **plain_kwargs)
            return _collect_segments(segments_iter, info, on_progress), info
        except Exception as exc2:
            if not plain_kwargs.get("vad_filter"):
                raise
            logger.warning(
                f"Обычный режим тоже упал ({type(exc2).__name__}: {exc2}) — "
                f"повторяю без VAD (вероятно, VAD вырезал всю дорожку)"
            )
            no_vad_kwargs = {k: v for k, v in plain_kwargs.items() if k != "vad_filter"}
            segments_iter, info = plain_model.transcribe(samples, **no_vad_kwargs)
            return _collect_segments(segments_iter, info, on_progress), info


def _diarize_samples(samples) -> list[dict]:
    pipeline = get_diar_pipeline(settings.polyvoice_variant)
    sr = 16000
    if len(samples) <= DIAR_MAX_SAMPLES:
        result = pipeline.run(samples, sample_rate=sr)
        return [
            {"speaker": t["speaker"], "start": float(t["start"]), "end": float(t["end"])}
            for t in result.get("turns", [])
        ]
    return _diarize_long(samples, pipeline, sr)


def _diarize_long(samples, pipeline, sr: int) -> list[dict]:
    logger.info("Аудио длиннее часа — диаризую по частям (лимит polyvoice)")
    step = DIAR_MAX_SAMPLES - DIAR_OVERLAP_SEC * sr
    all_turns: list[dict] = []
    known: dict = {}
    local_to_global: dict = {}
    next_label = 0
    offset = 0
    total = len(samples)
    while offset < total:
        chunk = samples[offset:offset + DIAR_MAX_SAMPLES]
        result = pipeline.run(chunk, sample_rate=sr)
        raw = sorted(result.get("turns", []), key=lambda t: t["start"])
        local_to_global = {}
        for t in raw:
            local = t["speaker"]
            g_start = float(t["start"]) + offset / sr
            g_end = float(t["end"]) + offset / sr
            g = local_to_global.get(local)
            if g is None:
                best_g = None
                best_ov = 0.0
                for gl, span in known.items():
                    ov = min(g_end, span[1]) - max(g_start, span[0])
                    if ov > best_ov:
                        best_ov = ov
                        best_g = gl
                if best_g is None or best_ov < 0.5 * max(1e-9, g_end - g_start):
                    best_g = next_label
                    next_label += 1
                local_to_global[local] = best_g
                g = best_g
            all_turns.append({"speaker": g, "start": g_start, "end": g_end})
            known[g] = [g_start, g_end]
        if offset + DIAR_MAX_SAMPLES >= total:
            break
        offset += step
    return all_turns


def transcribe_file(
    media: Path,
    model_size: str,
    language: str,
    diarize: bool = False,
    on_progress=None,
    group_dir: Path | None = None,
) -> Path:
    existing = find_existing_transcript(media, group_dir)
    if existing:
        logger.info(f"Транскрипт уже есть: {existing.name} — пропускаю")
        return existing

    out_path = transcript_path_for(media, group_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Декодирование ДО загрузки модели: файл без аудио не должен грузить whisper.
    from faster_whisper.audio import decode_audio

    try:
        samples = decode_audio(str(media), sampling_rate=16000)
    except Exception as exc:
        if _has_audio_stream(media):
            # Аудиодорожка есть, но декодер не справился — реальная потеря
            raise
        # Telegram-гифки и видеомемы (.gif.mp4, мем-mp4) НЕ имеют аудиопотока:
        # только h264-видео, decode_audio падает с IndexError (PyAV) до
        # всякого whisper — «tuple index out of range» приходил отсюда
        # (сессия 15.10: 90 файлов). Речи в файле нет в принципе — заглушка.
        logger.warning(
            f"{media.name}: в файле нет аудиодорожки "
            f"({type(exc).__name__}: {exc}) — записываю транскрипт без речи"
        )
        return _write_no_speech_transcript(
            out_path, media, f"в файле нет аудиодорожки ({type(exc).__name__}: {exc})"
        )

    transcriber = get_transcriber(model_size, settings.whisper_fast)
    kwargs = dict(
        vad_filter=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        beam_size=1 if settings.whisper_fast else 5,
    )
    if language and language != "auto":
        kwargs["language"] = language
    if settings.whisper_fast:
        kwargs["batch_size"] = settings.whisper_batch_size

    diar_errors: list[str] = []
    turns_box: list[list[dict]] = []

    def _diar_worker() -> None:
        try:
            turns_box.append(_diarize_samples(samples))
        except Exception as exc:
            diar_errors.append(str(exc))

    # Диаризация — самая тяжёлая часть пайплайна: при diarize=False она
    # раньше всё равно запускалась (флаг игнорировался, GPU/CPU впустую).
    diar_thread = None
    if diarize:
        diar_thread = threading.Thread(target=_diar_worker, daemon=True, name="omni-diarization")
        diar_thread.start()

    try:
        segments, info = _transcribe_with_fallback(transcriber, samples, kwargs, on_progress)
    except Exception as exc:
        if diar_thread is not None:
            diar_thread.join()
        if len(samples) >= SHORT_AUDIO_SAMPLES:
            # Длинное аудио не должно «успешно» глотаться: это реальная потеря
            raise
        # Стикеры-mp4/гифки без речи: whisper падает на любой цепочке фолбэков.
        logger.warning(
            f"{media.name}: аудио короче 3 с не транскрибируется "
            f"({type(exc).__name__}: {exc}) — записываю транскрипт без речи"
        )
        return _write_no_speech_transcript(
            out_path,
            media,
            f"аудио короче 3 с или без звуковой дорожки ({type(exc).__name__}: {exc})",
            duration_sec=len(samples) / 16000.0,
        )

    if diar_thread is not None:
        diar_thread.join()
    turns = turns_box[0] if turns_box else []
    if diarize:
        if diar_errors:
            logger.warning(f"Диаризация не удалась ({diar_errors[0]}) — транскрипт будет без спикеров")
        else:
            logger.info(f"Диаризация: найдено спикеров: {len({t['speaker'] for t in turns})}")

    _assign_speakers(segments, turns)

    meta = []
    if info.duration:
        meta.append(f"{'Длительность:':<14}{info.duration:.0f} сек")
    if turns:
        speakers = sorted({_prettify_speaker(t["speaker"]) for t in turns})
        meta.append(f"{'Спикеры:':<14}{', '.join(speakers)}")
    meta.append(f"{'Сегментов:':<14}{len(segments)}")

    body = []
    for seg in segments:
        speaker = getattr(seg, "speaker", None)
        who = f"{_prettify_speaker(speaker)}: " if speaker is not None else ""
        body.append(f"[{seg.start:6.1f}s] {who}{seg.text.strip()}")

    footer = f"{'-' * 64}\n  Конец транскрипта · OmniProsvet\n{'=' * 64}\n"
    header = (
        f"{'=' * 64}\n"
        "  ТРАНСКРИПТ\n"
        f"{'=' * 64}\n"
        f"{'Файл:':<14}{media.name}\n"
        f"{'Дата:':<14}{time.strftime('%d.%m.%Y %H:%M')}\n"
        + "\n".join(meta)
        + f"\n{'-' * 64}\n\n"
    )
    out_path.write_text(header + "\n".join(body).rstrip() + "\n\n" + footer, encoding="utf-8")
    logger.info(f"Транскрипт готов: {out_path.name} ({len(segments)} сегментов)")
    return out_path
