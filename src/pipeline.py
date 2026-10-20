import threading
import time
from pathlib import Path

from loguru import logger

from src.collect import collect_media
from src.configuration import settings
from src.database import record_phase_timing
from src.estimate import emit_estimate
from src.transcription import run_extract, run_extract_and_process
from src.analysis import run_process_only
from src.attachments import run_attachments, run_pdfs, run_presentations
from src.spreadsheets import run_spreadsheets


def _group_names(paths: list[Path]) -> dict[str, str]:
    import hashlib
    names: dict[str, str] = {}
    out: dict[str, str] = {}
    for p in paths:
        base = p.stem if p.is_file() else p.name
        base = "".join(c if c.isalnum() or c in " _-" else "_" for c in base).strip() or "source"
        full = str(p.resolve())
        if base in names and names[base] != full:
            h = hashlib.sha256(full.encode("utf-8")).hexdigest()[:8]
            base = f"{base}_{h}"
        names[base] = full
        out[full] = base
    return out


def _weights_for(extract: bool, process: bool, att_kinds: list[str]) -> dict:
    if extract and process:
        weights = {"transcribe": 45, "analysis": 30}
    elif extract:
        weights = {"transcribe": 80}
    else:
        weights = {"analysis": 75}
    if att_kinds:
        base = 100 - sum(weights.values())
        each = round(base / len(att_kinds), 1)
        for kind in att_kinds[:-1]:
            weights[kind] = each
        weights[att_kinds[-1]] = round(base - each * (len(att_kinds) - 1), 1)
    elif extract and process:
        weights = {"transcribe": 60, "analysis": 40}
    elif extract:
        weights = {"transcribe": 100}
    else:
        weights = {"analysis": 100}
    return weights


def _attachments_worker(att_plan: list, weights: dict, progress, reports: list) -> None:
    """Process all attachment types in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = []
    for kind, files, gdir in att_plan:
        if not files:
            continue
        tasks.append((kind, files, gdir))

    if not tasks:
        return

    # Анонсируем фазы с весами здесь: внутренние функции запускаются с quiet=True
    # и своих анонсов не делают. Так проценты фаз известны с самого старта.
    if progress:
        for kind, _, _ in tasks:
            progress({"phase": kind, "weight": weights.get(kind, 0)})

    # Счётчики неудач по видам вложений: в финальной сводке задачи честно
    # показать, сколько файлов не обработано (раньше они терялись в логе).
    kind_stats: dict[str, dict] = {kind: {"total": len(files)} for kind, files, _ in tasks}
    labels = {"images": "Картинки", "documents": "Документы"}

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {}
        started: dict[str, float] = {}
        for kind, files, gdir in tasks:
            if kind == "pdf":
                fn = run_pdfs
            elif kind == "ppt":
                fn = run_presentations
            elif kind == "spreadsheets":
                fn = run_spreadsheets
            else:
                fn = run_attachments
            started[kind] = time.monotonic()
            if fn is run_attachments:
                futures[executor.submit(
                    fn, files, kind, weights, gdir, progress, True, kind_stats[kind]
                )] = (kind, len(files))
            else:
                futures[executor.submit(fn, files, weights, progress, gdir, True)] = (kind, len(files))

        for fut in as_completed(futures):
            kind, n_files = futures[fut]
            try:
                reports.extend(fut.result())
            except Exception as exc:
                logger.error(f"Обработка {kind} упала: {type(exc).__name__}: {exc}")
                if progress:
                    progress({"log": f"Ошибка обработки {kind}: {exc}"})
            finally:
                # Статистика скорости фазы для оценок времени будущих запусков
                record_phase_timing(kind, n_files, time.monotonic() - started[kind])
                # Фаза закончилась (успехом или ошибкой) — засчитываем вес целиком
                if progress:
                    progress({"phase_done": kind})
                st = kind_stats.get(kind) or {}
                failed, total = st.get("failed", 0), st.get("total", 0)
                if failed:
                    name = labels.get(kind, kind)
                    msg = f"{name}: не обработано файлов: {failed} из {total} (ошибки NIM — см. лог)"
                    logger.warning(msg)
                    if progress:
                        progress({"log": msg})


def run_pipeline(
    paths,
    model: str | None = None,
    language: str | None = None,
    diarize: bool | None = None,
    extract: bool = True,
    process: bool = True,
    images: bool = True,
    raw: bool = False,
    out_dir: Path | None = None,
    progress=None,
) -> dict:
    from src.collect import collect_media

    model = model or settings.whisper_model
    language = language or settings.language
    use_diar = settings.diarization if diarize is None else diarize

    clean_paths = [Path(str(p).strip().strip('"')) for p in paths]
    group_names = _group_names(clean_paths)

    media_with_groups: list[tuple[Path, Path]] = []
    att_plan: list[tuple[str, list[Path], Path]] = []
    found_any = False

    def _notify_skipped(skipped: list[Path]) -> None:
        # collect_media пишет в omniprosvet.log; дублируем в лог задачи,
        # чтобы число выброшенных стикеров было видно в веб-панели.
        if progress:
            progress({"log": f"Пропущено файлов-стикеров/анимаций: {len(skipped)} (по шаблонам имён)"})

    for p in clean_paths:
        a, v, im, d = collect_media(str(p), notify=_notify_skipped)
        gdir = settings.output_dir / group_names[str(p.resolve())]
        gdir.mkdir(parents=True, exist_ok=True)
        found_any = found_any or bool(a or v or im or d)
        for m in a + v:
            media_with_groups.append((m, gdir))
        pdfs = [f for f in d if f.suffix.lower() == ".pdf"]
        ppts = [f for f in d if f.suffix.lower() in (".pptx", ".ppt")]
        sheets = [f for f in d if f.suffix.lower() in (".xlsx", ".xlsm", ".xls")]
        others = [
            f for f in d
            if f.suffix.lower() not in (".pdf", ".pptx", ".ppt", ".xlsx", ".xlsm", ".xls")
        ]
        if im:
            att_plan.append(("images", im, gdir))
        if pdfs:
            att_plan.append(("pdf", pdfs, gdir))
        if ppts:
            att_plan.append(("ppt", ppts, gdir))
        if sheets:
            att_plan.append(("spreadsheets", sheets, gdir))
        if others:
            att_plan.append(("documents", others, gdir))

    if not found_any:
        logger.warning(f"Поддерживаемых файлов не найдено: {[str(p) for p in clean_paths]}")
        if progress:
            progress({"log": "Поддерживаемых файлов (аудио/видео/картинки/документы) не найдено"})

    kinds = [k for k, files, _ in att_plan if files]
    if not images:
        # Флаг «не обрабатывать картинки»: раньше был мёртвым (объявлен,
        # передавался из UI/CLI — и игнорировался). Теперь честно исключает
        # картинки из плана обработки во всех режимах.
        att_plan = [(k, f, g) for k, f, g in att_plan if k != "images"]
        kinds = [k for k in kinds if k != "images"]
    if raw:
        weights = _weights_for(extract, False, ["raw"])
    else:
        weights = _weights_for(extract, process, kinds)
    out_dir = settings.output_dir

    # Предварительная оценка времени — ДО запуска фаз: пользователь видит
    # ориентир (в логе задачи веб-панели и в консоли CLI).
    emit_estimate(
        media_files=[m for m, _ in media_with_groups],
        att_plan=att_plan,
        extract=extract,
        process=process,
        raw=raw,
        progress=progress,
    )

    att_reports: list[Path] = []
    raw_files: list[Path] = []
    att_thread = None
    raw_thread = None
    if raw:
        # Режим «без умной обработки»: все вложения — локальное извлечение,
        # NIM не вызывается; анализ/отчёт отсутствуют (process=False ниже).
        # Фильтр images применён выше — картинки исключены, если запрещены.
        raw_all = [f for _, files, _ in att_plan for f in files]
        if raw_all:
            raw_thread = threading.Thread(
                target=_raw_worker,
                args=(raw_all, weights, progress, raw_files, out_dir),
                name="omni-raw",
            )
            raw_thread.start()
        elif progress:
            # Нечего извлекать — фаза закрывается сразу с анонсом веса:
            # иначе 20% raw никогда не засчитываются, шкала не добирает до 100%
            progress({"phase": "raw", "weight": weights.get("raw", 0)})
            progress({"phase_done": "raw"})
    elif (extract or process) and kinds:
        att_thread = threading.Thread(
            target=_attachments_worker,
            args=(att_plan, weights, progress, att_reports),
            name="omni-attachments",
        )
        att_thread.start()
        if progress:
            progress({"log": "Картинки, PDF, презентации, таблицы и документы обрабатываются параллельно"})
    elif kinds:
        # Process sequentially but each kind in parallel internally
        for kind, files, gdir in att_plan:
            started = time.monotonic()
            try:
                if kind == "pdf":
                    att_reports += run_pdfs(files, weights, progress, gdir)
                elif kind == "ppt":
                    att_reports += run_presentations(files, weights, progress, gdir)
                elif kind == "spreadsheets":
                    att_reports += run_spreadsheets(files, weights, progress, gdir)
                else:
                    att_reports += run_attachments(files, kind, weights, gdir, progress)
            except Exception as exc:
                logger.error(f"Обработка {kind} упала: {type(exc).__name__}: {exc}")
            finally:
                record_phase_timing(kind, len(files), time.monotonic() - started)
                if progress:
                    progress({"phase_done": kind})

    transcripts: list[Path] = []
    report = None
    if extract and process:
        transcripts, report = run_extract_and_process(
            media_with_groups, model, language, use_diar, out_dir, weights, progress
        )
    elif extract:
        transcripts = run_extract(media_with_groups, model, language, use_diar, weights, progress)
    elif process:
        transcripts, report = run_process_only(media_with_groups, out_dir, weights, progress)

    if att_thread is not None:
        att_thread.join()
    if raw_thread is not None:
        raw_thread.join()

    return {
        "transcripts": transcripts,
        "report": report,
        "image_reports": att_reports,
        "raw_files": raw_files,
    }


def _raw_worker(files: list[Path], weights: dict, progress, raw_files: list, out_dir: Path) -> None:
    """Фоновый воркер raw-режима: локальное извлечение всех файлов в текст."""
    from src.raw_extract import run_raw_extract

    try:
        raw_files.extend(run_raw_extract(files, weights, progress, out_dir))
    except Exception as exc:
        logger.error(f"Raw-извлечение упало: {type(exc).__name__}: {exc}")
        if progress:
            progress({"log": f"Ошибка извлечения: {exc}"})
    finally:
        if progress:
            progress({"phase_done": "raw"})