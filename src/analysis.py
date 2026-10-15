import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Промежуточная сводка группы конспектов (иерархический reduce).
# Включается, когда фрагментов больше NIM_REDUCE_GROUP_SIZE: группа конспектов
# сжимается параллельно, финальный reduce получает компактный вход — выход
# каждого запроса ограничен, вероятность упереться в max_tokens падает.
INTERMEDIATE_REDUCE_PROMPT = (
    "Ниже — конспекты фрагментов {first}-{last} одного массива материалов.\n"
    "Сведь их в один компактный промежуточный конспект:\n"
    "- дедуплицируй повторяющиеся события, контакты и ссылки (каждый факт — один раз);\n"
    "- сохрани все датированные факты: даты, числа и суммы, имена, названия, ссылки;\n"
    "- разделяй датированные события и нематериальные обсуждения;\n"
    "- пиши по-русски, обычным текстом без markdown, объём — не более 3000 символов.\n\n"
    "# Конспекты фрагментов\n{parts}\n\nПромежуточный конспект:"
)


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


# Подставка вместо неполученного конспекта: финальный reduce видит пропуск
# честно, а весь многочасовой анализ не умирает из-за одного сбоя NIM.
FRAGMENT_FAILED_TEMPLATE = "(Фрагмент {idx} из «{source}» не обработан: ошибка NIM — {reason})"


def _failed_fragment_note(idx: int, source: str, reason: str) -> str:
    return FRAGMENT_FAILED_TEMPLATE.format(idx=idx, source=source, reason=reason)


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
    try:
        text = plain_text(
            nim_client.send_prompt(
                build_map_prompt(idx, total, source, body),
                deepthink=False,
                expert=False,
                stable_rounds=6,
            )
        )
    except NimError as exc:
        # В сессии 13–14.10 один ReadTimeout здесь убивал весь анализ
        # (задача e72938ef: сотни файлов — в мусор из-за одного фрагмента).
        # Теперь фрагмент честно помечается пропущенным, анализ продолжается.
        logger.error(f"Фрагмент {idx}/{total} «{source}» не обработан: {exc}")
        if progress:
            progress({"log": f"Ошибка фрагмента {idx}/{total} «{source}» — пропущен ({exc})"})
        return _failed_fragment_note(idx, source, str(exc)[:200])
    save_fragment_cache(h, source, text)
    logger.info(f"Конспект фрагмента {idx}/{total} «{source}» готов ({len(text)} симв.)")
    return text


def _result_or_failed_note(fut, idx: int, total: int, source: str) -> str:
    """Результат будущего фрагмента; любой сбой — заметка-подставка.

    Страховка вокруг _process_fragment (там ловится NimError): неожиданные
    исключения тоже не должны убивать сборку отчёта.
    """
    try:
        return fut.result()
    except Exception as exc:  # noqa: BLE001 — страховка всей фазы
        logger.error(f"Фрагмент {idx}/{total} «{source}» упал: {type(exc).__name__}: {exc}")
        return _failed_fragment_note(idx, source, f"{type(exc).__name__}: {exc}"[:200])


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


def _intermediate_group_summaries(
    parts: list[str],
    group_size: int,
    progress,
) -> list[str]:
    """Параллельно сжать конспекты фрагментов группами (иерархический reduce).

    Возвращает список промежуточных сводок — по одной на группу.
    """
    groups = [parts[i:i + group_size] for i in range(0, len(parts), group_size)]
    logger.info(f"Иерархический reduce: {len(parts)} конспектов -> {len(groups)} групп (по {group_size})")
    if progress:
        progress({"log": f"Сборка отчёта: {len(groups)} промежуточных сводок (по {group_size} фрагментов)…"})

    def _summarize_group(start_index: int, group: list[str]) -> str:
        block = "\n\n".join(f"### Фрагмент {j}\n{p}" for j, p in enumerate(group, 1))
        prompt = INTERMEDIATE_REDUCE_PROMPT.format(
            first=start_index, last=start_index + len(group) - 1, parts=block
        )
        return plain_text(
            nim_client.send_prompt(
                prompt,
                deepthink=False,
                expert=False,
                stable_rounds=6,
            )
        )

    summaries: list[str | None] = [None] * len(groups)
    starts = [i * group_size + 1 for i in range(len(groups))]
    span = 0.95 - REDUCE_FRACTION
    done = 0
    with ThreadPoolExecutor(max_workers=min(settings.nim_workers, len(groups))) as executor:
        futures = {
            executor.submit(_summarize_group, starts[gi], group): gi
            for gi, group in enumerate(groups)
        }
        for fut in as_completed(futures):
            gi = futures[fut]
            try:
                summaries[gi] = fut.result()
            except NimError as exc:
                # Точка падения задачи e72938ef (14.10 01:27): один ReadTimeout
                # в промежуточной сводке убивал весь анализ. Группа помечается
                # пропущенной — финальный reduce получает отчёт с пробелом.
                first, last = starts[gi], starts[gi] + len(groups[gi]) - 1
                logger.error(f"Промежуточная сводка фрагментов {first}-{last} не удалась: {exc}")
                if progress:
                    progress({"log": f"Промежуточная сводка {first}-{last} не удалась — пропускаю ({exc})"})
                summaries[gi] = _failed_fragment_note(first, f"группа фрагментов {first}-{last}", str(exc)[:200])
            except Exception as exc:  # noqa: BLE001 — неожиданное тоже не роняет фазу
                first, last = starts[gi], starts[gi] + len(groups[gi]) - 1
                logger.error(f"Промежуточная сводка {first}-{last} упала: {type(exc).__name__}: {exc}")
                summaries[gi] = _failed_fragment_note(first, f"группа фрагментов {first}-{last}", f"{type(exc).__name__}: {exc}"[:200])
            done += 1
            if progress:
                # Строго ниже 0.95: финальный reduce эмитит ровно 0.95 (монотонность)
                progress({"p": "analysis", "phase_progress": REDUCE_FRACTION + span * done / (len(groups) + 1)})
    return [s for s in summaries if s is not None]


def _final_reduce_with_fallback(parts_block: str, n_transcripts: int, total_fragments: int) -> str:
    """Финальный reduce с fallback-моделью; обе мертвы — частичный отчёт.

    Возвращает текст отчёта в любом случае: либо сводка, либо сырые
    промежуточные части с пометкой, что финальная сводка не удалась.
    Анализ обязан доводиться до файла-отчёта при любых ошибках NIM.
    """
    try:
        return plain_text(
            nim_client.send_prompt(
                build_reduce_prompt(n_transcripts, total_fragments, parts_block),
                deepthink=settings.nim_workers > 1,
                expert=settings.nim_workers > 1,
                stable_rounds=6,
                model=settings.nim_reduce_model,
            )
        )
    except NimError as exc:
        logger.warning(f"Финальный reduce моделью {settings.nim_reduce_model} не удался: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Финальный reduce упал: {type(exc).__name__}: {exc}")

    fallback_model = settings.nim_model
    if fallback_model and fallback_model != settings.nim_reduce_model:
        try:
            logger.warning(f"Повтор финального reduce моделью {fallback_model}…")
            return plain_text(
                nim_client.send_prompt(
                    build_reduce_prompt(n_transcripts, total_fragments, parts_block),
                    deepthink=False,
                    expert=False,
                    stable_rounds=6,
                    model=fallback_model,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Финальный reduce моделью {fallback_model} тоже не удался: {exc}")

    return (
        "ФИНАЛЬНАЯ СВОДКА НЕ УДАЛАСЬ (ошибка NIM после повторов) — ниже "
        "промежуточные конспекты фрагментов без объединения.\n\n" + parts_block
    )


def _reduce_and_save(
    transcripts_order: list[Path],
    all_futures: list,
    out_dir: Path | None,
    progress,
) -> Path:
    if progress:
        progress({"log": "Фрагменты готовы, собираю финальный отчёт…"})
    parts = []
    failed_parts = 0
    for i, fut in enumerate(all_futures, 1):
        try:
            parts.append(fut.result())
        except Exception as exc:  # noqa: BLE001 — сбой одного фрагмента не убирает отчёт
            failed_parts += 1
            logger.error(f"Фрагмент {i}/{len(all_futures)} недоступен при сборке: {type(exc).__name__}: {exc}")
            parts.append(_failed_fragment_note(i, "фрагмент", f"{type(exc).__name__}: {exc}"[:200]))
    total_fragments = len(parts)

    group_size = max(0, settings.nim_reduce_group_size)
    if group_size and total_fragments > group_size:
        # Промежуточные сводки сами эмитят прогресс REDUCE_FRACTION..0.95
        parts = _intermediate_group_summaries(parts, group_size, progress)
    if progress:
        progress({"p": "analysis", "phase_progress": 0.95})

    parts_block = "\n\n".join(f"### Фрагмент {i}\n{p}" for i, p in enumerate(parts, 1))
    final = _final_reduce_with_fallback(parts_block, len(transcripts_order), total_fragments)
    target_dir = out_dir or settings.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    header_rows = [
        ("Транскриптов", len(transcripts_order)),
        ("Фрагментов", len(all_futures)),
        ("Дата", time.strftime("%d.%m.%Y %H:%M")),
    ]
    if failed_parts:
        header_rows.append(("Не обработано", failed_parts))
    header = _header_block("ЕДИНЫЙ ОТЧЁТ", header_rows)
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