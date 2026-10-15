import hashlib
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from pypdf import PdfReader

from src.configuration import settings
from src.database import get_cached_fragment, save_fragment_cache
from src.nim_client import NimError, client as nim_client
from src.outputs import plain_text
from src.prompts import build_documents_prompt, build_images_prompt
from src.utils import _chunk_lines, _footer_block, _header_block, MAX_LINE_CHARS
from src.analysis import _FragProgress, _submit_fragments

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image не установлен — fallback для сканированных PDF недоступен")

# PowerPoint COM для конвертации legacy .ppt -> .pptx (только Windows + Office)
try:
    import pythoncom  # noqa: F401
    import win32com.client  # noqa: F401
    COM_AVAILABLE = True
except ImportError:
    COM_AVAILABLE = False

PP_SAVE_AS_OPEN_XML = 24  # ppSaveAsOpenXMLPresentation

# Версия OCR-семантики для картинок и страниц сканов. Раньше vision-модель
# «распознавала» картинки (описывала сцену) — решено, что это избыточно:
# теперь только OCR. Версия в ключе кэша инвалидирует старые кэшированные
# описания: при перезапуске картинки переобрабатываются по новому промпту.
OCR_CACHE_VERSION = "ocr-v2"


def _cache_hash(kind: str, batch: list[Path]) -> str:
    """Хэш кэша батча. Для картинок — с версией OCR-промпта: старые
    кэшированные описания сцен не должны подменять новые OCR-результаты."""
    key = "|".join(f"{p.name}:{p.stat().st_size}" for p in batch)
    if kind == "images":
        key = f"{OCR_CACHE_VERSION}|{key}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _pdf_to_images(pdf: Path, dpi: int = 200) -> list[Path]:
    """Конвертирует страницы PDF в изображения (PNG) со СТАБИЛЬНЫМИ именами.

    Имена вида `{hash}_0001.png` (номер страницы внутри имени) делают ключ
    кэша Vision-батча одинаковым между перезапусками: упавшие в прошлый раз
    страницы дораспознаются, а удачные возьмутся из кэша.
    """
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("pdf2image не установлен")

    from pdf2image.generators import counter_generator

    key = hashlib.sha256(f"{pdf.name}|{pdf.stat().st_size}".encode("utf-8")).hexdigest()[:8]
    prefix = f"{pdf.stem}_{key}_"
    settings.tmp_pdfs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        images = convert_from_path(
            str(pdf),
            dpi=dpi,
            output_folder=tmpdir,
            fmt="png",
            paths_only=True,
            output_file=counter_generator(prefix=prefix),
        )
        # Копируем в постоянное место (tmpdir удалится после выхода)
        saved = []
        for img_path in images:
            src = Path(img_path)
            dst = settings.tmp_pdfs_dir / src.name
            dst.write_bytes(src.read_bytes())
            saved.append(dst)
        return saved


def _recognize_scan_batch(
    batch: list[Path],
    pdf: Path,
    first_page: int,
    total_pages: int,
    out_dir: Path | None,
    progress,
) -> Path:
    """Распознать один батч страниц скана через Vision-модель, записать .txt.

    Страницы скана — тоже картинки: обрабатываются OCR-промптом (только
    текст, без описания сцены), кэш — с версией OCR-промпта.
    Бросает NimError наружу — изоляция сбоев решает вызывающая сторона.
    """
    # Кэш для этого батча: стабильные имена страниц -> попадает и при перезапуске
    key = "|".join(f"{p.name}:{p.stat().st_size}" for p in batch)
    h = hashlib.sha256(f"{OCR_CACHE_VERSION}|{key}".encode("utf-8")).hexdigest()
    text = get_cached_fragment(h)

    if not text:
        prompt = build_images_prompt(len(batch))
        text = plain_text(
            nim_client.send_prompt(
                prompt,
                deepthink=False,
                expert=False,
                stable_rounds=6,
                attachments=[str(p.resolve()) for p in batch],
            )
        )
        save_fragment_cache(h, "pdf_scan", text)
    else:
        text = plain_text(text)  # старый кэш мог быть с markdown-разметкой

    last_page = first_page + len(batch) - 1
    target_dir = out_dir or settings.results_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{pdf.stem}_scan_{h[:8]}.txt"
    header = _header_block(
        f"PDF СКАН — OCR (страницы {first_page}-{last_page})",
        [
            ("Документ", pdf.name),
            ("Страниц", len(batch)),
            ("Дата", time.strftime("%d.%m.%Y %H:%M")),
        ],
    )
    out_path.write_text(header + text.strip() + "\n\n" + _footer_block("распознавания"), encoding="utf-8")
    logger.info(f"Vision PDF готов: {out_path.name}")
    if progress:
        progress({"log": f"Vision PDF готов: {out_path.name}"})
    return out_path


def _process_pdf_as_images(pdf: Path, weights: dict, progress, out_dir: Path | None, quiet: bool) -> list[Path]:
    """Fallback: конвертирует PDF в изображения и обрабатывает через Vision модель.

    Изоляция сбоев: не распознанный батч страниц (ReadTimeout/HTTP 400) не
    отменяет остальные страницы. Ошибка «At most 1 image» режет батч на
    одиночные страницы — как в run_attachments.
    """
    if not PDF2IMAGE_AVAILABLE:
        logger.error(f"PDF {pdf.name}: pdf2image не установлен, невозможно обработать скан")
        if progress:
            progress({"log": f"PDF {pdf.name}: нужен pdf2image для сканированных PDF"})
        return []

    if progress and not quiet:
        progress({"file": f"{pdf.name}: скан", "phase": "pdf"})
    if progress:
        progress({"log": f"PDF {pdf.name}: конвертирую страницы в изображения…"})

    try:
        image_paths = _pdf_to_images(pdf, dpi=200)
    except Exception as exc:
        logger.error(f"PDF {pdf.name}: не удалось конвертировать в изображения — {type(exc).__name__}: {exc}")
        if progress:
            progress({"log": f"Ошибка: {pdf.name} — конвертация в изображения не удалась ({exc})"})
        return []

    if not image_paths:
        logger.warning(f"PDF {pdf.name}: не удалось извлечь страницы")
        return []

    logger.info(f"PDF {pdf.name}: извлечено {len(image_paths)} страниц как изображения")
    if progress:
        progress({"log": f"PDF {pdf.name}: {len(image_paths)} страниц → Vision модель"})

    results: list[Path] = []
    failed_pages = 0
    total_pages = len(image_paths)
    batch_size = max(1, min(settings.nim_images_per_batch, total_pages))

    def _single_pages(batch: list[Path], reason: str) -> None:
        nonlocal failed_pages
        for single in batch:
            page_no = image_paths.index(single) + 1
            try:
                results.append(
                    _recognize_scan_batch([single], pdf, page_no, total_pages, out_dir, progress)
                )
            except NimError as single_exc:
                failed_pages += 1
                logger.error(f"PDF {pdf.name}: страница {page_no} не распознана: {single_exc}")
                if progress:
                    progress({"log": f"PDF {pdf.name}: страница {page_no} пропущена ({reason})"})

    for bi in range(0, total_pages, batch_size):
        batch = image_paths[bi:bi + batch_size]
        first_page = bi + 1
        if progress and not quiet:
            progress({
                "file": f"{pdf.name}: скан, страницы {first_page}-{min(first_page + batch_size - 1, total_pages)} из {total_pages}",
                "phase": "pdf",
            })
        if progress:
            progress({"log": f"Vision: страница {first_page}-{min(first_page + batch_size - 1, total_pages)} из {total_pages}"})

        try:
            results.append(_recognize_scan_batch(batch, pdf, first_page, total_pages, out_dir, progress))
        except NimError as exc:
            msg = str(exc)
            if _MANY_IMAGES_MARK in msg and len(batch) > 1:
                logger.warning(
                    f"PDF {pdf.name}: модель принимает 1 картинку за запрос — "
                    f"страницы {first_page}-{first_page + len(batch) - 1} обрабатываю по одной"
                )
                if progress:
                    progress({"log": f"PDF {pdf.name}: батч страниц по одной (модель принимает 1 картинку)"})
                _single_pages(batch, "модель принимает 1 картинку")
            else:
                failed_pages += len(batch)
                logger.error(
                    f"PDF {pdf.name}: страницы {first_page}-{first_page + len(batch) - 1} не распознаны: {exc}"
                )
                if progress:
                    progress({"log": f"Ошибка Vision: {pdf.name} страницы {first_page}-{first_page + len(batch) - 1} — {exc}"})

    if failed_pages:
        logger.warning(
            f"PDF {pdf.name}: не распознано страниц: {failed_pages} из {total_pages} (ошибки NIM — см. лог)"
        )
        if progress:
            progress({"log": f"PDF {pdf.name}: не распознано страниц: {failed_pages} из {total_pages}"})

    return results


def _process_attachment_batch(batch: list[Path], kind: str, out_dir: Path | None = None) -> Path:
    h = _cache_hash(kind, batch)
    text = get_cached_fragment(h)
    if not text:
        prompt = build_images_prompt(len(batch)) if kind == "images" else build_documents_prompt(len(batch))
        text = plain_text(
            nim_client.send_prompt(
                prompt,
                deepthink=False,
                expert=False,
                stable_rounds=6,
                attachments=[str(p.resolve()) for p in batch],
            )
        )
        save_fragment_cache(h, kind, text)
    else:
        text = plain_text(text)  # старый кэш мог быть с markdown-разметкой
    target_dir = out_dir or settings.results_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{batch[0].stem}_{h[:8]}.{kind}.txt"
    label = "КАРТИНОК" if kind == "images" else "ДОКУМЕНТОВ"
    header = _header_block(
        f"OCR {label}" if kind == "images" else f"РАСПОЗНАВАНИЕ {label}",
        [
            ("Файлов", len(batch)),
            ("Дата", time.strftime("%d.%m.%Y %H:%M")),
        ],
    )
    out_path.write_text(header + text.strip() + "\n\n" + _footer_block("распознавания"), encoding="utf-8")
    logger.info(f"Обработка {len(batch)} {label.lower()} готова: {out_path.name}")
    return out_path


# Детерминированная ошибка NIM «модель принимает не больше N картинок за
# запрос» (llama-3.2-90b-vision: ровно 1). Резать батч надо сразу по одной,
# не тратя таймауты на повторные 400.
_MANY_IMAGES_MARK = "At most 1 image"


def run_attachments(
    files: list[Path],
    kind: str,
    weights: dict,
    out_dir: Path | None = None,
    progress=None,
    quiet: bool = False,
    stats: dict | None = None,
) -> list[Path]:
    if not files:
        return []
    batch_size = max(1, settings.images_per_batch)
    if settings.nim_workers > 0:
        batch_size = max(1, min(batch_size, settings.nim_images_per_batch))
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    label = "Картинки" if kind == "images" else "Документы"
    logger.info(f"{label}: {len(files)} шт., батчей: {len(batches)} (по {batch_size})")
    if progress and not quiet:
        progress({"phase": kind, "weight": weights.get(kind, 0)})

    results: list[Path] = []
    for bi, batch in enumerate(batches, 1):
        if progress and not quiet:
            # Событие батча: без полей file_index/file_total — счётчик «Файл n
            # из N» в UI показывается только для файловых фаз (транскрибация,
            # PDF, презентации), у батчей свой прогресс в тексте.
            # Имя без префикса фазы: метку («Картинки»/«Документы») добавляет UI/CLI
            progress({"file": f"батч {bi} из {len(batches)} ({len(batch)} шт.)", "phase": kind})
        if progress:
            progress({"log": f"{label}: батч {bi}/{len(batches)} ({len(batch)} шт.)…"})
        try:
            results.append(_process_attachment_batch(batch, kind, out_dir))
            if progress:
                progress({"p": kind, "phase_progress": bi / len(batches)})
        except NimError as exc:
            msg = str(exc)
            if _MANY_IMAGES_MARK in msg and len(batch) > 1:
                logger.warning(
                    f"{label}: модель принимает 1 картинку за запрос — "
                    f"батч {bi} ({len(batch)} шт.) обрабатываю по одной"
                )
                if progress:
                    progress({"log": f"{label}: модель принимает 1 картинку за запрос — батч {bi} по одной"})
                for single in batch:
                    results += run_attachments(
                        [single], kind, weights, out_dir, progress, True, stats
                    )
                continue
            if len(batch) > 1:
                logger.warning(f"Батч {bi} не прошёл ({exc}) — режу пополам и повторяю")
                if progress:
                    progress({"log": f"Батч {bi} не прошёл — режу батч пополам"})
                mid = len(batch) // 2
                results += run_attachments(batch[:mid], kind, weights, out_dir, progress, True, stats)
                results += run_attachments(batch[mid:], kind, weights, out_dir, progress, True, stats)
            else:
                logger.error(f"{label[:-1]} {batch[0].name}: ошибка {exc}")
                if progress:
                    progress({"log": f"Ошибка: {batch[0].name} — {exc}"})
                if stats is not None:
                    stats["failed"] = stats.get("failed", 0) + 1
    return results


def _process_single_pdf(
    pdf: Path,
    weights: dict,
    progress,
    out_dir: Path | None,
    quiet: bool,
    tracker: "_FragProgress | None" = None,
) -> list[Path]:
    """Process a single PDF file.

    tracker — ОБЩИЙ на всю фазу (создаётся в run_pdfs): доля фазы считается
    по всем фрагментам всех файлов, а не сбрасывается на каждый файл.
    Раньше у каждого PDF был свой трекер, и после завершения первого файла
    фаза навсегда показывала 100% (монотонность игнорировала откат к 0
    следующего файла).
    """
    try:
        if progress and not quiet:
            progress({"file": pdf.name, "phase": "pdf"})
        if progress:
            progress({"log": f"PDF: извлекаю текст из {pdf.name}…"})

        from pypdf import PdfReader
        reader = PdfReader(str(pdf))
        lines = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                lines.append(text[:500])

        if not lines:
            logger.warning(f"PDF {pdf.name}: текст не извлечён — пробуем Vision fallback")
            if progress:
                progress({"log": f"PDF {pdf.name}: нет текстового слоя — Vision fallback…"})
            return _process_pdf_as_images(pdf, weights, progress, out_dir, quiet)

        # Check cache
        h = hashlib.sha256(f"{pdf.stem}.pdf|{pdf.stat().st_size}".encode("utf-8")).hexdigest()
        text = get_cached_fragment(h)

        if not text:
            budget = settings.nim_chunk_char_budget
            chunks = _chunk_lines(lines, budget)
            if not chunks:
                return []

            from concurrent.futures import ThreadPoolExecutor

            if tracker is None:
                tracker = _FragProgress("pdf", 1.0, progress)
            executor = ThreadPoolExecutor(max_workers=min(settings.nim_workers, len(chunks)))

            futures = _submit_fragments(f"{pdf.stem}.pdf", chunks, executor, tracker, progress)
            executor.shutdown(wait=True)

            texts = [fut.result() for fut in futures]
            text = "\n\n".join(texts)
            save_fragment_cache(h, "pdf", text)
        else:
            text = plain_text(text)  # старый кэш мог быть с markdown-разметкой
        
        target_dir = out_dir or settings.results_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{pdf.stem}_{h[:8]}.pdf.txt"
        header = _header_block(
            "КОНСПЕКТ PDF",
            [
                ("Документ", pdf.name),
                ("Фрагментов", len(text.split("\n\n")) if text else 0),
                ("Дата", time.strftime("%d.%m.%Y %H:%M")),
            ],
        )
        out_path.write_text(header + text.strip() + "\n\n" + _footer_block("конспекта"), encoding="utf-8")
        logger.info(f"Конспект PDF готов: {out_path.name}")
        if progress:
            progress({"log": f"Конспект PDF готов: {out_path.name}"})
        return [out_path]
    except NimError as exc:
        logger.error(f"PDF {pdf.name}: ошибка {exc}")
        if progress:
            progress({"log": f"Ошибка: {pdf.name} — {exc}"})
        return []
    except Exception as exc:
        logger.error(f"PDF {pdf.name}: ошибка {type(exc).__name__}: {exc}")
        if progress:
            progress({"log": f"Ошибка: {pdf.name} — {exc}"})
        return []


def run_pdfs(pdfs: list[Path], weights: dict, progress=None, out_dir: Path | None = None, quiet: bool = False) -> list[Path]:
    if not pdfs:
        return []
    
    logger.info(f"PDF: {len(pdfs)} шт. (параллельно по {min(settings.nim_workers, len(pdfs))} файлов)")
    if progress and not quiet:
        progress({"phase": "pdf", "weight": weights.get("pdf", 0)})

    results: list[Path] = []
    max_workers = min(settings.nim_workers, len(pdfs), 4)
    total = len(pdfs)
    # Один трекер на всю фазу: доля = готовые фрагменты / все фрагменты всех PDF
    tracker = _FragProgress("pdf", 1.0, progress)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_pdf, pdf, weights, progress, out_dir, quiet, tracker): pdf
            for pdf in pdfs
        }
        done = 0
        for fut in as_completed(futures):
            pdf = futures[fut]
            try:
                results.extend(fut.result())
            except Exception as exc:
                logger.error(f"PDF {pdf.name}: ошибка в потоке {exc}")
                if progress:
                    progress({"log": f"Ошибка: {pdf.name} — {exc}"})
            # Счётчик n из N — по числу файлов с ЗАВЕРШЁННОЙ обработкой
            # (успехом или ошибкой): монотонный, не смешивается с другими фазами.
            done += 1
            if progress:
                progress({"file": pdf.name, "file_index": done, "file_total": total, "phase": "pdf"})

    return results


# ---------------------------------------------------------------------------
# Презентации (.pptx / .ppt) — логика как у PDF: извлечение текста -> чанкинг
# -> MAP через NIM -> кэш -> конспект. Отличие: нет Vision-fallback (рендер
# слайдов требует Office/LibreOffice) — файлы без текста пропускаются.
# ---------------------------------------------------------------------------


def _extract_pptx_lines(path: Path) -> list[str]:
    """Строки текста из .pptx: надписи, таблицы, заметки — по аналогии со
    страницами PDF. Каждая строка обрезана до MAX_LINE_CHARS."""
    from pptx import Presentation

    lines: list[str] = []
    prs = Presentation(str(path))
    for si, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                t = (shape.text_frame.text or "").strip()
                if t:
                    lines.append(f"[Слайд {si}] {t}"[:MAX_LINE_CHARS])
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        lines.append((f"[Слайд {si}] " + " | ".join(cells))[:MAX_LINE_CHARS])
        if getattr(slide, "has_notes_slide", False):
            nt = (slide.notes_slide.notes_text_frame.text or "").strip()
            if nt:
                lines.append(f"[Слайд {si}. Заметки] {nt}"[:MAX_LINE_CHARS])
    return lines


def _ppt_conversion_target(ppt: Path) -> Path:
    key = hashlib.sha256(f"{ppt.name}|{ppt.stat().st_size}".encode("utf-8")).hexdigest()[:8]
    return settings.tmp_pdfs_dir / f"{ppt.stem}_{key}.pptx"


def _convert_ppt_to_pptx_via_com(ppt: Path) -> Path:
    """Legacy .ppt -> .pptx через PowerPoint COM (нужен установленный Office).

    Результат кэшируется в tmp-каталоге: повторные запуски не конвертируют
    заново (кэш работает даже без pywin32/Office). DispatchEx создаёт
    отдельный процесс PowerPoint — окно не открывается и чужой запущенный
    PowerPoint не затрагивается.
    """
    target = _ppt_conversion_target(ppt)
    if target.exists():
        return target

    if not COM_AVAILABLE:
        raise RuntimeError(
            "pywin32 не установлен — legacy .ppt недоступен; конвертируйте файл в .pptx"
        )

    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    app = None
    pres = None
    try:
        app = win32com.client.DispatchEx("PowerPoint.Application")
        pres = app.Presentations.Open(str(ppt.resolve()), ReadOnly=True, Untitled=False, WithWindow=False)
        try:
            pres.SaveAs(str(target), PP_SAVE_AS_OPEN_XML)
        finally:
            pres.Close()
    finally:
        try:
            if app is not None:
                app.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()

    if not target.exists():
        raise RuntimeError("PowerPoint не создал .pptx — конвертация не удалась")
    return target


def _process_single_presentation(
    ppt: Path,
    weights: dict,
    progress,
    out_dir: Path | None,
    quiet: bool,
    tracker: "_FragProgress | None" = None,
) -> list[Path]:
    """Конспект одной презентации (.pptx/.ppt) — по аналогии с _process_single_pdf.

    tracker — общий на всю фазу (см. комментарий в _process_single_pdf).
    """
    try:
        if progress and not quiet:
            progress({"file": ppt.name, "phase": "ppt"})
        if progress:
            progress({"log": f"Презентация: извлекаю текст из {ppt.name}…"})

        if ppt.suffix.lower() == ".pptx":
            lines = _extract_pptx_lines(ppt)
        else:
            lines = _extract_pptx_lines(_convert_ppt_to_pptx_via_com(ppt))

        if not lines:
            logger.warning(f"Презентация {ppt.name}: текст не извлечён (слайды без текста)")
            if progress:
                progress({"log": f"Презентация {ppt.name}: нет текста — пропускаю (рендер слайдов не поддерживается)"})
            return []

        h = hashlib.sha256(f"{ppt.name}|{ppt.stat().st_size}".encode("utf-8")).hexdigest()
        text = get_cached_fragment(h)

        if not text:
            budget = settings.nim_chunk_char_budget
            chunks = _chunk_lines(lines, budget)
            if not chunks:
                return []

            from concurrent.futures import ThreadPoolExecutor

            if tracker is None:
                tracker = _FragProgress("ppt", 1.0, progress)
            executor = ThreadPoolExecutor(max_workers=min(settings.nim_workers, len(chunks)))

            futures = _submit_fragments(ppt.name, chunks, executor, tracker, progress)
            executor.shutdown(wait=True)

            texts = [fut.result() for fut in futures]
            text = "\n\n".join(texts)
            save_fragment_cache(h, "ppt", text)
        else:
            text = plain_text(text)  # старый кэш мог быть с markdown-разметкой

        target_dir = out_dir or settings.results_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{ppt.stem}_{h[:8]}.ppt.txt"
        header = _header_block(
            "КОНСПЕКТ ПРЕЗЕНТАЦИИ",
            [
                ("Документ", ppt.name),
                ("Фрагментов", len(text.split("\n\n")) if text else 0),
                ("Дата", time.strftime("%d.%m.%Y %H:%M")),
            ],
        )
        out_path.write_text(header + text.strip() + "\n\n" + _footer_block("конспекта"), encoding="utf-8")
        logger.info(f"Конспект презентации готов: {out_path.name}")
        if progress:
            progress({"log": f"Конспект презентации готов: {out_path.name}"})
        return [out_path]
    except RuntimeError as exc:
        # pywin32/PowerPoint недоступны — сообщение уже человекочитаемое
        logger.error(f"Презентация {ppt.name}: {exc}")
        if progress:
            progress({"log": f"Ошибка: {ppt.name} — {exc}"})
        return []
    except NimError as exc:
        logger.error(f"Презентация {ppt.name}: ошибка {exc}")
        if progress:
            progress({"log": f"Ошибка: {ppt.name} — {exc}"})
        return []
    except Exception as exc:
        logger.error(f"Презентация {ppt.name}: ошибка {type(exc).__name__}: {exc}")
        if progress:
            progress({"log": f"Ошибка: {ppt.name} — {exc}"})
        return []


def run_presentations(ppts: list[Path], weights: dict, progress=None, out_dir: Path | None = None, quiet: bool = False) -> list[Path]:
    """Параллельная обработка презентаций — по аналогии с run_pdfs."""
    if not ppts:
        return []

    logger.info(f"Презентации: {len(ppts)} шт. (параллельно по {min(settings.nim_workers, len(ppts), 4)} файлов)")
    if progress and not quiet:
        progress({"phase": "ppt", "weight": weights.get("ppt", 0)})

    results: list[Path] = []
    max_workers = min(settings.nim_workers, len(ppts), 4)
    total = len(ppts)
    # Один трекер на всю фазу — как в run_pdfs
    tracker = _FragProgress("ppt", 1.0, progress)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_presentation, ppt, weights, progress, out_dir, quiet, tracker): ppt
            for ppt in ppts
        }
        done = 0
        for fut in as_completed(futures):
            ppt = futures[fut]
            try:
                results.extend(fut.result())
            except Exception as exc:
                logger.error(f"Презентация {ppt.name}: ошибка в потоке {exc}")
                if progress:
                    progress({"log": f"Ошибка: {ppt.name} — {exc}"})
            done += 1
            if progress:
                progress({"file": ppt.name, "file_index": done, "file_total": total, "phase": "ppt"})

    return results