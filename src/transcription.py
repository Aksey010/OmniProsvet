import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def run_extract(
    media_files: list[tuple[Path, Path]],
    model: str,
    language: str,
    diarize: bool,
    weights: dict,
    progress=None,
) -> list[Path]:
    if not media_files:
        return []
    logger.info(
        f"Транскрибация: {len(media_files)} файлов, модель {model}, "
        f"язык {language}, диаризация: {'вкл' if diarize else 'выкл'}"
    )
    if progress:
        progress({"log": f"Транскрибация: {len(media_files)} файлов"})
        progress({"phase": "transcribe", "weight": weights.get("transcribe", 100)})

    results: list[Path] = []
    total = len(media_files)
    for idx, (f, gdir) in enumerate(media_files, 1):
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
            results.append(out)
            if progress:
                progress({"log": f"Транскрипт готов: {out.name}"})
        except Exception as exc:
            logger.error(f"Ошибка транскрибации {f.name}: {exc}")
            if progress:
                progress({"log": f"Ошибка транскрибации {f.name}: {exc}"})
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
    import queue as q_mod

    if not media_files:
        return [], []

    logger.info(
        f"Транскрибация + обработка: {len(media_files)} файлов, модель {model}, "
        f"язык {language}, диаризация: {'вкл' if diarize else 'выкл'}"
    )
    if progress:
        progress({"log": f"Транскрибация: {len(media_files)} файлов (обработка идёт параллельно)"})
        progress({"phase": "transcribe", "weight": weights.get("transcribe", 50)})

    q: q_mod.Queue = q_mod.Queue()
    transcripts_order: list[Path] = []
    lock = threading.Lock()

    def transcribe_worker() -> None:
        total = len(media_files)
        for idx, (f, gdir) in enumerate(media_files, 1):
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
                with lock:
                    transcripts_order.append(out)
                if progress:
                    progress({"log": f"Транскрипт готов: {out.name}"})
                q.put(out)
            except Exception as exc:
                logger.error(f"Ошибка транскрибации {f.name}: {exc}")
                if progress:
                    progress({"log": f"Ошибка транскрибации {f.name}: {exc}"})
        q.put(None)

    t_thread = threading.Thread(target=transcribe_worker, name="omni-transcriber")
    t_thread.start()
    t_thread.join()

    # Collect all transcripts from queue
    all_transcripts: list[Path] = []
    while True:
        t = q.get()
        if t is None:
            break
        all_transcripts.append(t)
        q.task_done()

    return transcripts_order, all_transcripts