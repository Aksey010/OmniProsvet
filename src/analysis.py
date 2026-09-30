import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from src.configuration import settings
from src.database import get_cached_fragment, save_fragment_cache
from src.nim_client import NimError, client as nim_client
from src.prompts import build_map_prompt, build_reduce_prompt
from src.utils import (
    _chunk_hash,
    _chunk_lines,
    _footer_block,
    _header_block,
    _is_small_chunk,
)

REDUCE_FRACTION = 0.9


class _FragProgress:
    def __init__(self, phase: str, share: float, emit) -> None:
        self._lock = threading.Lock()
        self._phase = phase
        self._share = share
        self._emit = emit
        self._done = 0
        self._total = 0

    def add_total(self, n: int) -> None:
        with self._lock:
            self._total += n
            self._emit_progress()

    def tick(self, _future=None) -> None:
        with self._lock:
            self._done += 1
            self._emit_progress()

    def _emit_progress(self) -> None:
        if not self._emit:
            return
        frac = min(1.0, self._done / max(1, self._total)) * self._share
        self._emit({"p": self._phase, "phase_progress": frac})


def _chunk_hash(source: str, chunk: list[str]) -> str:
    return hashlib.sha256((source + "|" + "\n".join(chunk)).encode("utf-8")).hexdigest()


def _is_small_chunk(chunk: list[str], budget: int | None = None) -> bool:
    b = budget or settings.chunk_char_budget
    size = sum(len(line) for line in chunk)
    return (
        size <= b // 2
        and len(chunk) <= settings.chunk_max_lines // 2
    )


class _SmallBuffer:
    def __init__(self, executor: ThreadPoolExecutor, tracker: _FragProgress, progress) -> None:
        self.lines: list[str] = []
        self.size = 0
        self.futures: list = []
        self.executor = executor
        self.tracker = tracker
        self.progress = progress

    def add_chunk(self, name: str, chunk: list[str]) -> None:
        need = sum(len(line) for line in chunk)
        if self.lines and (
            self.size + need > settings.chunk_char_budget
            or len(self.lines) + len(chunk) > settings.chunk_max_lines
        ):
            self.flush()
        if self.progress:
            self.progress({"log": f"«{name}» — мелкий, копится в общий батч"})
        self.lines += chunk
        self.size += need

    def flush(self) -> None:
        if not self.lines:
            return
        chunk = self.lines
        self.lines = []
        self.size = 0
        self.tracker.add_total(1)
        fut = self.executor.submit(
            _process_fragment, 1, 1, chunk, "мелкие файлы", self.progress
        )
        fut.add_done_callback(self.tracker.tick)
        self.futures.append(fut)


def _process_fragment(idx: int, total: int, chunk: list[str], source: str, progress) -> str:
    body = "\n".join(chunk)
    h = _chunk_hash(source, chunk)
    cached = get_cached_fragment(h)
    if cached:
        logger.info(f"Конспект фрагмента {idx}/{total} «{source}» — из кэша")
        if progress:
            progress({"log": f"Фрагмент {idx}/{total} «{source}» — из кэша"})
        return cached
    if progress:
        progress({"log": f"Фрагмент {idx}/{total} «{source}» — отправляю в NIM…"})
    text = nim_client.send_prompt(
        build_map_prompt(idx, total, source, body),
        deepthink=False,
        expert=False,
        stable_rounds=6,
    )
    save_fragment_cache(h, source, text)
    logger.info(f"Конспект фрагмента {idx}/{total} «{source}» готов ({len(text)} симв.)")
    return text


def _submit_fragments(
    name: str,
    chunks: list[list[str]],
    executor: ThreadPoolExecutor,
    tracker: _FragProgress,
    progress,
) -> list:
    tracker.add_total(len(chunks))
    futures = []
    for i, chunk in enumerate(chunks):
        fut = executor.submit(_process_fragment, i + 1, len(chunks), chunk, name, progress)
        fut.add_done_callback(tracker.tick)
        futures.append(fut)
    return futures


def _collect_results(futures_map: dict[str, list], transcripts_order: list[Path]) -> list:
    all_futures: list = []
    for t in transcripts_order:
        all_futures += futures_map.get(str(t), [])
    all_futures += futures_map.get("мелкие файлы", [])
    return all_futures


def _feed_transcript(
    transcript: Path,
    futures_map: dict[str, list],
    executor: ThreadPoolExecutor,
    small_buffer: _SmallBuffer,
    tracker: _FragProgress,
    progress,
    budget: int | None = None,
) -> None:
    prepared = _prepare_transcript(transcript, budget)
    if prepared is None:
        return
    source, chunks = prepared
    if len(chunks) == 1 and _is_small_chunk(chunks[0], budget):
        small_buffer.add_chunk(source, chunks[0])
        return
    futures_map[str(transcript)] = _submit_fragments(source, chunks, executor, tracker, progress)


def _reduce_and_save(
    transcripts_order: list[Path],
    all_futures: list,
    out_dir: Path | None,
    progress,
) -> Path:
    if progress:
        progress({"log": "Фрагменты готовы, собираю финальный отчёт…"})
        progress({"p": "deepseek", "phase_progress": 0.95})
    parts = [fut.result() for fut in all_futures]
    parts_block = "\n\n".join(f"### Фрагмент {i}\n{p}" for i, p in enumerate(parts, 1))
    final = nim_client.send_prompt(
        build_reduce_prompt(len(transcripts_order), len(all_futures), parts_block),
        deepthink=settings.nim_workers > 1,
        expert=settings.nim_workers > 1,
        stable_rounds=6,
        model=settings.nim_reduce_model,
    )
    target_dir = out_dir or settings.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    header = _header_block(
        "ЕДИНЫЙ ОТЧЁТ",
        [
            ("Транскриптов", len(transcripts_order)),
            ("Фрагментов", len(all_futures)),
            ("Дата", time.strftime("%d.%m.%Y %H:%M")),
        ],
    )
    out_path.write_text(header + final.strip() + "\n\n" + _footer_block("отчёта"), encoding="utf-8")
    if progress:
        progress({"p": "deepseek", "phase_progress": 1.0})
    logger.info(f"Единый отчёт готов: {out_path}")
    return out_path


def run_process_only(
    media_files: list[tuple[Path, Path]],
    out_dir: Path | None,
    weights: dict,
    progress=None,
) -> tuple[list[Path], Path | None]:
    from src.collect import find_existing_transcript

    transcripts: list[Path] = []
    for media, gdir in media_files:
        tp = find_existing_transcript(media, gdir)
        if tp:
            transcripts.append(tp)
    uniq = {str(t): t for t in transcripts}
    transcripts = sorted(uniq.values(), key=lambda x: x.name)
    if not transcripts:
        logger.warning("Транскрипты не найдены")
        return [], None

    logger.info(f"Обработка: {len(transcripts)} транскриптов, NIM workers: {settings.nim_workers}")
    if progress:
        progress({"log": f"NIM: {len(transcripts)} транскриптов, workers: {settings.nim_workers}"})
        progress({"phase": "deepseek", "weight": weights.get("deepseek", 100)})

    executor = ThreadPoolExecutor(max_workers=settings.nim_workers)
    tracker = _FragProgress("deepseek", REDUCE_FRACTION, progress)
    budget = settings.nim_chunk_char_budget
    futures_map: dict[str, list] = {}
    small_buffer = _SmallBuffer(executor, tracker, progress)
    for t in transcripts:
        _feed_transcript(t, futures_map, executor, small_buffer, tracker, progress, budget)
    small_buffer.flush()
    futures_map["мелкие файлы"] = small_buffer.futures
    executor.shutdown(wait=True)

    all_futures = _collect_results(futures_map, transcripts)
    if not all_futures:
        logger.warning("В транскриптах нет распознанного текста")
        return transcripts, None
    report = _reduce_and_save(transcripts, all_futures, out_dir, progress)
    return transcripts, report