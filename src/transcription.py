import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

from loguru import logger

from src.collect import find_existing_transcript, transcript_path_for
from src.configuration import settings
from src.whisper_extract import transcribe_file
from src.utils import (
    _chunk_lines,
    _footer_block,
    _header_block,
    _is_small_chunk,
    _prepare_transcript,
    _transcript_source_label,
)


def _transcribe_single(media_file: tuple[Path, Path], model: str, language: str, diarize: bool, progress, idx: int, total: int) -> Path | None:
    """Transcribe a single file with progress reporting."""
    f, gdir = media_file
    if progress:
        progress({"file": f.name, "file_index": idx, "file_total": total})
    try:
        out = transcribe_file(
            f,
            model,
            language,
            diarize=diarize,
            on_progress=(
                lambda pct, _i=idx, _t=total: progress(
                    {"p": "transcribe", "phase_progress": (_i - 1 + pct / 100) / _t}
                )
            )
            if progress
            else None,
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
        return []
    
    max_workers = max_workers or min(4, len(media_files))
    logger.info(
        f"Транскрибация: {len(media_files)} файлов, модель {model}, "
        f"язык {language}, диаризация: {'вкл' if diarize else 'выкл'}, "
        f"воркеров: {max_workers}"
    )
    if progress:
        progress({"log": f"Транскрибация: {len(media_files)} файлов (параллельно: {max_workers})"})
        progress({"phase": "transcribe", "weight": weights.get("transcribe", 100)})

    results: list[Path] = []
    total = len(media_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_transcribe_single, mf, model, language, diarize, progress, i, total): i
            for i, mf in enumerate(media_files, 1)
        }
        for fut in as_completed(futures):
            out = fut.result()
            if out:
                results.append(out)

    return results


def run_extract_and_process(
    media_files: list[tuple[Path, Path]],
    model: str,
    language: str,
    diarize: bool,
    out_dir: Path | None,
    weights: dict,
    progress=None,
) -> tuple[list[Path], list[Path]]:
    if not media_files:
        return [], []

    max_workers = min(4, len(media_files))
    logger.info(
        f"Транскрибация + обработка: {len(media_files)} файлов, модель {model}, "
        f"язык {language}, диаризация: {'вкл' if diarize else 'выкл'}, "
        f"воркеров: {max_workers}"
    )
    if progress:
        progress({"log": f"Транскрибация: {len(media_files)} файлов (параллельно: {max_workers})"})
        progress({"phase": "transcribe", "weight": weights.get("transcribe", 50)})

    transcripts_order: list[Path] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_transcribe_single, mf, model, language, diarize, progress, i, len(media_files)): i
            for i, mf in enumerate(media_files, 1)
        }
        for fut in as_completed(futures):
            out = fut.result()
            if out:
                with lock:
                    transcripts_order.append(out)

    return transcripts_order, transcripts_order