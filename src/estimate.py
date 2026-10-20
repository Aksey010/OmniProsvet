"""Предварительная оценка времени обработки задачи.

Считается ДО запуска фаз и выводится пользователю. Никакого разбора
содержимого файлов (проб PDF/аудио/презентаций) НЕ делается — расчёт
мгновенный, по счётчикам файлов.

Источник скорости — накопленная статистика прошлых запусков (таблица
phase_timings в cache.db): среднее СТЕННОЕ время фазы на единицу уже
включает параллелизм, лимиты NIM и реальную длину файлов этого хранилища.
Пока статистики нет, используются первичные константы, откалиброванные
по логам сессий 13–15.10 (omniprosvet.log).
"""

import math
from pathlib import Path

from loguru import logger

from src.database import get_phase_timings

# --- первичные константы (сек на единицу; логи сессий 13–15.10) -------------
# transcribe зависит от устройства:
TRANSCRIBE_CPU_SEC_PER_FILE = 30.0   # whisper small, int8, ~4 воркера
TRANSCRIBE_GPU_SEC_PER_FILE = 8.0    # whisper small, float16, ~2 воркера
DEFAULT_AVG_SEC = {
    "images": 10.0,       # OCR-запрос на картинку (с учётом параллельных фаз)
    "pdf": 120.0,         # текстовые ~10 с, сканы — минуты
    "ppt": 60.0,
    "spreadsheets": 150.0,  # прайсы с ретраями
    "documents": 10.0,
    "analysis": 1.2,      # сек на MAP-фрагмент (rate-limit bound)
}
# Фрагментов анализа на файл медиа (≈2007 фрагментов / 3108 файлов, сессия 15.10)
FRAGMENTS_PER_FILE = 0.65
# Локальное извлечение (raw): OCR ~1 с/файл + PDF-разбор, с запасом
RAW_SEC_PER_FILE = 2.0


def fmt_duration(seconds: float) -> str:
    """Секунды -> человекочитаемо: «45 с», «5 мин», «1 ч 05 мин»."""
    seconds = int(round(seconds))
    if seconds < 90:
        return f"~{seconds} с"
    minutes, sec = divmod(seconds, 60)
    if minutes < 90:
        return f"~{minutes} мин" + (f" {sec} с" if sec and minutes < 10 else "")
    hours, minutes = divmod(minutes, 60)
    return f"~{hours} ч {minutes:02d} мин"


def _transcribe_fallback_per_file() -> float:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return TRANSCRIBE_GPU_SEC_PER_FILE
    except Exception:
        pass
    return TRANSCRIBE_CPU_SEC_PER_FILE


def _avg(kind: str) -> float:
    """Среднее стенное время на единицу: статистика прошлых запусков или константа."""
    timings = get_phase_timings()
    units, seconds = timings.get(kind, (0, 0.0))
    if units > 0 and seconds > 0:
        return seconds / units
    if kind == "transcribe":
        return _transcribe_fallback_per_file()
    return DEFAULT_AVG_SEC[kind]


def build_estimate(
    *,
    counts: dict[str, int],
    extract: bool,
    process: bool,
    raw: bool = False,
) -> tuple[list[str], dict]:
    """Оценка по счётчикам файлов. counts: transcribe/images/pdf/ppt/spreadsheets/documents/raw.

    Возвращает (строки для вывода, детали по фазам).
    """
    counts = {k: max(0, int(v)) for k, v in counts.items()}
    if raw:
        # raw-режим: только локальная работа, NIM-фаз нет
        has_any = bool(counts.get("transcribe") or counts.get("raw"))
        if not has_any:
            return ["Нет файлов для обработки — оценка не требуется"], {"total": 0.0}
        lines: list[str] = ["Оценка времени задачи (режим без NIM, локальное извлечение):"]
        t_transcribe = 0.0
        if extract and counts.get("transcribe"):
            t_transcribe = counts["transcribe"] * _avg("transcribe")
            lines.append(f"- Транскрибация: {fmt_duration(t_transcribe)} ({counts['transcribe']} файл(ов))")
        t_raw = counts.get("raw", 0) * RAW_SEC_PER_FILE
        if counts.get("raw"):
            lines.append(
                f"- Локальное извлечение текста: {fmt_duration(t_raw)} "
                f"({counts['raw']} файл(ов), без NIM)"
            )
        total = max(t_transcribe, t_raw)
        lines.append(
            f"ИТОГО: {fmt_duration(total)}–{fmt_duration(total * 1.6)} "
            "(фазы работают параллельно; зависит от скорости диска и CPU)"
        )
        return lines, {
            "total": total,
            "transcribe": t_transcribe,
            "analysis": 0.0,
            "attachments": t_raw,
            "fragments_est": 0,
        }

    has_any = any(
        (extract and counts.get("transcribe")) or counts.get(kind, 0)
        for kind in ("images", "pdf", "ppt", "spreadsheets", "documents")
    ) or (process and (counts.get("transcribe") or counts.get("documents")))
    if not has_any:
        return ["Нет файлов для обработки — оценка не требуется"], {"total": 0.0}

    lines: list[str] = ["Оценка времени задачи (ориентировочно, по скорости прошлых запусков):"]

    # --- транскрибация ------------------------------------------------------
    t_transcribe = 0.0
    if extract and counts.get("transcribe"):
        t_transcribe = counts["transcribe"] * _avg("transcribe")
        lines.append(f"- Транскрибация: {fmt_duration(t_transcribe)} ({counts['transcribe']} файл(ов))")

    # --- NIM-фазы (идут параллельно с транскрибацией) ------------------------
    t_attachments = 0.0
    labels = {
        "images": "Картинки (OCR)",
        "pdf": "PDF",
        "ppt": "Презентации",
        "spreadsheets": "Таблицы",
        "documents": "Документы",
    }
    for kind in ("images", "pdf", "ppt", "spreadsheets", "documents"):
        n = counts.get(kind, 0)
        if not n:
            continue
        t_kind = n * _avg(kind)
        t_attachments += t_kind
        lines.append(f"- {labels[kind]}: {fmt_duration(t_kind)} ({n} шт.)")

    # --- анализ транскриптов -------------------------------------------------
    t_analysis = 0.0
    fragments_est = 0
    if process and (counts.get("transcribe") or counts.get("documents")):
        # Фрагменты ≈ из статистики прошлых запусков (фрагментов на файл)
        fpp = FRAGMENTS_PER_FILE
        timings = get_phase_timings()
        units_t, _ = timings.get("transcribe", (0, 0.0))
        units_a, _ = timings.get("analysis", (0, 0.0))
        if units_t > 0 and units_a > 0:
            fpp = units_a / units_t
        n_sources = counts.get("transcribe", 0) or counts.get("documents", 0)
        fragments_est = max(1, math.ceil(n_sources * fpp))
        t_analysis = fragments_est * _avg("analysis") + 180.0  # + финальная сводка
        lines.append(f"- Анализ транскриптов: {fmt_duration(t_analysis)} (≈{fragments_est} фрагментов + сводка)")

    # --- итог с учётом параллельности фаз -----------------------------------
    if extract and process:
        total = max(t_transcribe + t_analysis, t_attachments)
    elif extract:
        total = t_transcribe
    elif process:
        total = max(t_analysis, t_attachments)
    else:
        total = t_attachments

    lines.append(
        f"ИТОГО: {fmt_duration(total)}–{fmt_duration(total * 1.6)} "
        "(фазы работают параллельно; зависит от загрузки NIM)"
    )
    details = {
        "total": total,
        "transcribe": t_transcribe,
        "analysis": t_analysis,
        "attachments": t_attachments,
        "fragments_est": fragments_est,
    }
    return lines, details


def emit_estimate(
    media_files: list[Path],
    att_plan: list[tuple[str, list[Path], Path]],
    extract: bool,
    process: bool,
    progress=None,
    raw: bool = False,
) -> dict:
    """Посчитать и вывести оценку до запуска фаз: в лог задачи (панель) и в logger (CLI).

    Никакого чтения содержимого файлов — только счётчики и накопленная статистика.
    """
    kind_files: dict[str, list[Path]] = {}
    for kind, files, _ in att_plan:
        kind_files.setdefault(kind, []).extend(files)

    counts = {
        "transcribe": len(media_files) if extract else 0,
        "images": 0 if raw else len(kind_files.get("images", [])),
        "pdf": 0 if raw else len(kind_files.get("pdf", [])),
        "ppt": 0 if raw else len(kind_files.get("ppt", [])),
        "spreadsheets": 0 if raw else len(kind_files.get("spreadsheets", [])),
        "documents": 0 if raw else len(kind_files.get("documents", [])),
        "raw": sum(len(files) for _, files, _ in att_plan) if raw else 0,
    }

    lines, details = build_estimate(counts=counts, extract=extract, process=process, raw=raw)

    logger.info("\n".join(lines))
    if progress:
        for line in lines:
            progress({"log": line})
        # Отдельное событие с итоговой строкой: веб-панель показывает её
        # постоянной строкой в карточке прогресса (не уезжает с прокруткой лога).
        if len(lines) > 1:
            progress({"estimate": lines[-1]})
    return details