import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from src.configuration import settings
from src.database import get_cached_fragment, save_fragment_cache
from src.nim_client import NimError, client as nim_client
from src.outputs import plain_text
from src.prompts import build_map_prompt, build_reduce_prompt
from src.utils import (
    _chunk_hash,
    _chunk_lines,
    _footer_block,
    _header_block,
    _is_small_chunk,
    _prepare_transcript,
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


def _process_fragment(idx: int, total: int, chunk: list[str], source: str, progress) -> str:
    body = "\n".join(chunk)
    h = _chunk_hash(source, chunk)
    cached = get_cached_fragment(h)
    if cached:
        # Старый кэш мог быть записан с markdown-разметкой — приводим к plain text
        text = plain_text(cached)
        logger.info(f"Конспект фрагмента {idx}/{total} «{source}» — из кэша")
        if progress:
            progress({"log": f"Фрагмент {idx}/{total} «{source}» — из кэша"})
        return text
    if progress:
        progress({"log": f"Фрагмент {idx}/{total} «{source}» — отправляю в NIM…"})
    text = plain_text(
        nim_client.send_prompt(
            build_map_prompt(idx, total, source, body),
            deepthink=False,
            expert=False,
            stable_rounds=6,
        )
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
    add_total: bool = True,
) -> list:
    """Засабмитить чанки источника в пул.

    add_total=False — когда суммарный знаменатель уже посчитан заранее
    (two-pass планирование в _run_analysis_on_transcripts).
    """
    if add_total:
        tracker.add_total(len(chunks))
    futures = []
    for i, chunk in enumerate(chunks):
        fut = executor.submit(_process_fragment, i + 1, len(chunks), chunk, name, progress)
        fut.add_done_callback(tracker.tick)
        futures.append(fut)
    return futures


def _plan_fragments(
    transcripts: list[Path],
    budget: int | None = None,
    progress=None,
) -> list[tuple[str, list[list[str]]]]:
    """Спроектировать все фрагменты заранее (первый проход two-pass).

    Чтение транскриптов и чанкинг дешёвые, поэтому считаем их до старта
    NIM-запросов: суммарное число фрагментов известно до первого «тика»
    прогресса, и знаменатель не растёт на глазах у пользователя.

    Мелкие одиночные чанки склеиваются в общие батчи «мелкие файлы» —
    так же, как раньше делал _SmallBuffer.
    """
    units: list[tuple[str, list[list[str]]]] = []
    small_lines: list[str] = []
    small_size = 0
    for t in transcripts:
        prepared = _prepare_transcript(t, budget)
        if prepared is None:
            continue
        source, chunks = prepared
        if len(chunks) == 1 and _is_small_chunk(chunks[0], budget):
            if progress:
                progress({"log": f"«{source}» — мелкий, копится в общий батч"})
            chunk = chunks[0]
            need = sum(len(line) for line in chunk)
            if small_lines and (
                small_size + need > settings.chunk_char_budget
                or len(small_lines) + len(chunk) > settings.chunk_max_lines
            ):
                units.append(("мелкие файлы", [small_lines]))
                small_lines = []
                small_size = 0
            small_lines += chunk
            small_size += need
        else:
            units.append((source, chunks))
    if small_lines:
        units.append(("мелкие файлы", [small_lines]))
    return units


def _reduce_and_save(
    transcripts_order: list[Path],
    all_futures: list,
    out_dir: Path | None,
    progress,
) -> Path:
    if progress:
        progress({"log": "Фрагменты готовы, собираю финальный отчёт…"})
        progress({"p": "analysis", "phase_progress": 0.95})
    parts = [fut.result() for fut in all_futures]
    parts_block = "\n\n".join(f"### Фрагмент {i}\n{p}" for i, p in enumerate(parts, 1))
    final = plain_text(
        nim_client.send_prompt(
            build_reduce_prompt(len(transcripts_order), len(all_futures), parts_block),
            deepthink=settings.nim_workers > 1,
            expert=settings.nim_workers > 1,
            stable_rounds=6,
            model=settings.nim_reduce_model,
        )
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
        progress({"p": "analysis", "phase_progress": 1.0})
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

    return _run_analysis_on_transcripts(transcripts, out_dir, weights, progress)


def run_analysis_on_transcripts(
    transcripts: list[Path],
    out_dir: Path | None,
    weights: dict,
    progress=None,
) -> tuple[list[Path], Path | None]:
    """Run Map/Reduce analysis on pre-existing transcript files."""
    uniq = {str(t): t for t in transcripts}
    transcripts = sorted(uniq.values(), key=lambda x: x.name)
    if not transcripts:
        logger.warning("Транскрипты не найдены")
        return [], None

    return _run_analysis_on_transcripts(transcripts, out_dir, weights, progress)


def _run_analysis_on_transcripts(
    transcripts: list[Path],
    out_dir: Path | None,
    weights: dict,
    progress=None,
) -> tuple[list[Path], Path | None]:
    logger.info(f"Обработка: {len(transcripts)} транскриптов, NIM workers: {settings.nim_workers}")
    if progress:
        progress({"log": f"NIM: {len(transcripts)} транскриптов, workers: {settings.nim_workers}"})
        progress({"phase": "analysis", "weight": weights.get("analysis", 100)})

    budget = settings.nim_chunk_char_budget

    # Two-pass: сначала дешёвый проход по файлам — чтение и чанкинг.
    # Знаменатель прогресса фиксируется до первого NIM-запроса.
    units = _plan_fragments(transcripts, budget, progress)
    total_fragments = sum(len(chunks) for _, chunks in units)
    if total_fragments == 0:
        logger.warning("В транскриптах нет распознанного текста")
        if progress:
            progress({"phase_done": "analysis"})
        return transcripts, None

    executor = ThreadPoolExecutor(max_workers=settings.nim_workers)
    tracker = _FragProgress("analysis", REDUCE_FRACTION, progress)
    tracker.add_total(total_fragments)

    all_futures: list = []
    for source, chunks in units:
        all_futures += _submit_fragments(source, chunks, executor, tracker, progress, add_total=False)
    executor.shutdown(wait=True)

    if not all_futures:
        logger.warning("В транскриптах нет распознанного текста")
        if progress:
            progress({"phase_done": "analysis"})
        return transcripts, None
    report = _reduce_and_save(transcripts, all_futures, out_dir, progress)
    if progress:
        progress({"phase_done": "analysis"})
    return transcripts, report