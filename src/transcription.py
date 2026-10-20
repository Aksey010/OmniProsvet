from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from loguru import logger

from src.analysis import run_analysis_on_transcripts
from src.configuration import settings
from src.database import record_phase_timing
from src.progress import FilesProgress
from src.whisper_extract import transcribe_file


def _transcribe_single(media_file: tuple[Path, Path], model: str, language: str, diarize: bool, progress, on_progress) -> Path | None:
    """Transcribe a single file with progress reporting."""
    f, gdir = media_file
    if progress:
        # Событие старта: только имя, без счётчика — индекс присваивается
        # по факту ЗАВЕРШЕНИЯ файла, чтобы «Файл n из N» не скакал между
        # параллельными воркерами.
        progress({"file": f.name, "phase": "transcribe"})
    try:
        out = transcribe_file(
            f,
            model,
            language,
            diarize=diarize,
            on_progress=on_progress,
            group_dir=gdir,
        )
        if progress:
            progress({"log": f"Транскрипт готов: {out.name}"})
        return out
    except Exception as exc:
        logger.error(f"Ошибка транскрибации {f.name}: {exc}")
        if progress:
            progress({"log": f"Ошибка транскрибации {f.name}: {exc}"})
        return None


def _transcribe_parallel(
    media_files: list[tuple[Path, Path]],
    model: str,
    language: str,
    diarize: bool,
    max_workers: int,
    progress=None,
) -> list[Path]:
    """Общий параллельный цикл транскрибации: пул воркеров, монотонный
    счётчик «файл N из M», статистика фазы. Используется run_extract и
    фазой 1 run_extract_and_process (раньше — две копии цикла)."""
    total = len(media_files)
    files_progress = FilesProgress(total, "transcribe")
    phase_started = time.monotonic()
    results: list[Path] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _transcribe_single,
                mf, model, language, diarize,
                progress, files_progress.reporter(i, progress) if progress else None,
            ): i
            for i, mf in enumerate(media_files, 1)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            out = fut.result()
            if out:
                results.append(out)
            # Счётчик n из N — по числу файлов с ЗАВЕРШЁННОЙ обработкой
            # (успехом или ошибкой): монотонный, не зависит от порядка воркеров.
            done += 1
            if progress:
                name = out.name if out else media_files[i - 1][0].name
                progress({"file": name, "file_index": done, "file_total": total, "phase": "transcribe"})

    # Статистика для оценок времени следующих запусков: стенное время фазы
    record_phase_timing("transcribe", total, time.monotonic() - phase_started)
    return results


def run_extract(
    media_files: list[tuple[Path, Path]],
    model: str,
    language: str,
    diarize: bool,
    weights: dict,
    progress=None,
    max_workers: int | None = None,
) -> list[Path]:
    if not media_files:
        if progress:
            # Пустая фаза всё равно анонсируется и закрывается: иначе её вес
            # (в extract-режиме — 80%) никогда не засчитывается, и шкала
            # прогресса не добирает до 100% (ползёт к 20% и скачет в конце)
            progress({"phase": "transcribe", "weight": weights.get("transcribe", 0)})
            progress({"phase_done": "transcribe"})
        return []
    
    max_workers = max_workers or min(settings.whisper_workers, len(media_files))
    logger.info(
        f"Транскрибация: {len(media_files)} файлов, модель {model}, "
        f"язык {language}, диаризация: {'вкл' if diarize else 'выкл'}, "
        f"воркеров: {max_workers}"
    )
    if progress:
        progress({"log": f"Транскрибация: {len(media_files)} файлов (параллельно: {max_workers})"})
        progress({"phase": "transcribe", "weight": weights.get("transcribe", 100)})

    results = _transcribe_parallel(media_files, model, language, diarize, max_workers, progress)

    if progress:
        progress({"phase_done": "transcribe"})
    return results


def run_extract_and_process(
    media_files: list[tuple[Path, Path]],
    model: str,
    language: str,
    diarize: bool,
    out_dir: Path | None,
    weights: dict,
    progress=None,
) -> tuple[list[Path], Path | None]:
    """
    Transcribe media files in parallel, then run Map/Reduce analysis on transcripts.
    Returns (transcripts, report_path).
    """
    if not media_files:
        if progress:
            # Фазы без данных закрываются сразу с анонсом веса — иначе шкала
            # не добирает до 100% (см. комментарий в run_extract)
            progress({"phase": "transcribe", "weight": weights.get("transcribe", 0)})
            progress({"phase_done": "transcribe"})
            progress({"phase": "analysis", "weight": weights.get("analysis", 0)})
            progress({"phase_done": "analysis"})
        return [], None

    max_workers = min(settings.whisper_workers, len(media_files))
    logger.info(
        f"Транскрибация + обработка: {len(media_files)} файлов, модель {model}, "
        f"язык {language}, диаризация: {'вкл' if diarize else 'выкл'}, "
        f"воркеров: {max_workers}"
    )
    if progress:
        progress({"log": f"Транскрибация: {len(media_files)} файлов (параллельно: {max_workers})"})
        progress({"phase": "transcribe", "weight": weights.get("transcribe", 50)})

    transcripts_order = _transcribe_parallel(media_files, model, language, diarize, max_workers, progress)

    if progress:
        progress({"phase_done": "transcribe"})

    if not transcripts_order:
        logger.warning("Ни один транскрипт не создан")
        if progress:
            progress({"log": "Ни один транскрипт не создан"})
            # Анализ не состоится — закрываем фазу с весом, шкала добирает до 100%
            progress({"phase": "analysis", "weight": weights.get("analysis", 0)})
            progress({"phase_done": "analysis"})
        return [], None

# Phase 2: Map/Reduce analysis on transcripts
    logger.info(f"Начинаю анализ {len(transcripts_order)} транскриптов")
    if progress:
        progress({"phase": "analysis", "weight": weights.get("analysis", 50)})

    _, report = run_analysis_on_transcripts(transcripts_order, out_dir, weights, progress)
    
    return transcripts_order, report