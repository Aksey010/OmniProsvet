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


def get_transcriber(size: str, fast: bool) -> object:
    key = (size, fast)
    with _model_lock:
        if key not in _transcribers:
            logger.info(f"Загрузка модели whisper '{size}' (первый раз — скачивается)…")
            model = WhisperModel(size, device="auto", compute_type="int8")
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

    from faster_whisper.audio import decode_audio

    samples = decode_audio(str(media), sampling_rate=16000)
    segments_iter, info = transcriber.transcribe(samples, **kwargs)

    diar_errors: list[str] = []
    turns_box: list[list[dict]] = []

    def _diar_worker() -> None:
        try:
            turns_box.append(_diarize_samples(samples))
        except Exception as exc:
            diar_errors.append(str(exc))

    diar_thread = threading.Thread(target=_diar_worker, daemon=True, name="omni-diarization")
    diar_thread.start()

    segments = []
    for seg in segments_iter:
        if not seg.text.strip():
            continue
        segments.append(seg)
        if on_progress and info.duration:
            on_progress(min(100, round(seg.end / info.duration * 100)))

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
