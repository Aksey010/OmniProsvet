import hashlib
import time
from pathlib import Path

from loguru import logger
from pypdf import PdfReader, PdfWriter

from src.configuration import settings
from src.database import get_cached_fragment, save_fragment_cache
from src.nim_client import NimError, client as nim_client
from src.prompts import build_documents_prompt, build_images_prompt
from src.utils import _chunk_lines, _footer_block, _header_block


def _process_attachment_batch(batch: list[Path], kind: str, out_dir: Path | None = None) -> Path:
    key = "|".join(f"{p.name}:{p.stat().st_size}" for p in batch)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    text = get_cached_fragment(h)
    if not text:
        prompt = build_images_prompt(len(batch)) if kind == "images" else build_documents_prompt(len(batch))
        text = nim_client.send_prompt(
            prompt,
            deepthink=False,
            expert=False,
            stable_rounds=6,
            attachments=[str(p.resolve()) for p in batch],
        )
        save_fragment_cache(h, kind, text)
    target_dir = out_dir or settings.results_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{batch[0].stem}_{h[:8]}.{kind}.txt"
    label = "КАРТИНОК" if kind == "images" else "ДОКУМЕНТОВ"
    header = _header_block(
        f"РАСПОЗНАВАНИЕ {label}",
        [
            ("Файлов", len(batch)),
            ("Дата", time.strftime("%d.%m.%Y %H:%M")),
        ],
    )
    out_path.write_text(header + text.strip() + "\n\n" + _footer_block("распознавания"), encoding="utf-8")
    logger.info(f"Обработка {len(batch)} {label.lower()} готова: {out_path.name}")
    return out_path


def run_attachments(files: list[Path], kind: str, weights: dict, out_dir: Path | None = None, progress=None, quiet: bool = False) -> list[Path]:
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
            progress({"file": f"{label}: {len(batch)} шт.", "file_index": bi, "file_total": len(batches)})
        if progress:
            progress({"log": f"{label}: батч {bi}/{len(batches)} ({len(batch)} шт.)…"})
        try:
            results.append(_process_attachment_batch(batch, kind, out_dir))
            if progress:
                progress({"p": kind, "phase_progress": bi / len(batches)})
        except NimError as exc:
            if len(batch) > 1:
                logger.warning(f"Батч {bi} не прошёл ({exc}) — режу пополам и повторяю")
                if progress:
                    progress({"log": f"Батч {bi} не прошёл — режу батч пополам"})
                mid = len(batch) // 2
                results += run_attachments(batch[:mid], kind, weights, out_dir, progress, quiet)
                results += run_attachments(batch[mid:], kind, weights, out_dir, progress, quiet)
            else:
                logger.error(f"{label[:-1]} {batch[0].name}: ошибка {exc}")
                if progress:
                    progress({"log": f"Ошибка: {batch[0].name} — {exc}"})
    return results


PDF_MERGE_DIR = Path(__file__).resolve().parent.parent / "tmp_pdfs"


def _pdf_groups(pdfs: list[Path]) -> list[list[Path]]:
    budget = settings.pdf_max_size_mb * 1024 * 1024
    groups: list[list[Path]] = []
    current: list[Path] = []
    size = 0
    for pdf in pdfs:
        psz = pdf.stat().st_size
        if current and size + psz > budget:
            groups.append(current)
            current = []
            size = 0
        current.append(pdf)
        size += psz
    if current:
        groups.append(current)
    return groups


def _merge_pdf_group(pdfs: list[Path]) -> Path | None:
    key = "|".join(f"{p.name}:{p.stat().st_size}" for p in pdfs)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    out_path = PDF_MERGE_DIR / f"{h[:8]}.pdf"
    if out_path.exists():
        return out_path
    PDF_MERGE_DIR.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()
    for pdf in pdfs:
        try:
            reader = PdfReader(str(pdf))
            for page in reader.pages:
                writer.add_page(page)
        except Exception as exc:
            logger.warning(f"PDF {pdf.name} пропущен при склейке (не читается: {type(exc).__name__}: {exc})")
    if len(writer.pages) == 0:
        logger.error(f"Ни один PDF из группы не склеился ({len(pdfs)} шт.)")
        return None
    with open(out_path, "wb") as f:
        writer.write(f)
    logger.info(f"PDF склеены: {len(pdfs)} шт. → {out_path.name} ({out_path.stat().st_size // 1048576} МБ)")
    return out_path


def _split_pdf_by_pages(pdf: Path) -> list[Path]:

    budget = settings.pdf_max_size_mb * 1024 * 1024
    reader = PdfReader(str(pdf))
    total_pages = len(reader.pages)
    if total_pages < 2:
        return [pdf]
    size_per_page = max(1, pdf.stat().st_size // total_pages)
    max_pages = max(1, budget // size_per_page)
    parts: list[Path] = []
    for start in range(0, total_pages, max_pages):
        writer = PdfWriter()
        for i in range(start, min(start + max_pages, total_pages)):
            writer.add_page(reader.pages[i])
        out_path = PDF_MERGE_DIR / f"{pdf.stem}_part{len(parts) + 1}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        parts.append(out_path)
        logger.info(f"PDF разбит: часть {len(parts)} — страницы {start + 1}-{min(start + max_pages, total_pages)} ({out_path.stat().st_size // 1048576} МБ)")
    return parts


def _process_single_pdf(pdf: Path, weights: dict, progress, out_dir: Path | None, quiet: bool) -> list[Path]:
    """Process a single PDF file."""
    try:
        if progress and not quiet:
            progress({"file": f"PDF: {pdf.name}", "file_index": 1, "file_total": 1})
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
            logger.warning(f"PDF {pdf.name}: текст не извлечён (скан без текстового слоя?)")
            if progress:
                progress({"log": f"PDF {pdf.name}: нет текстового слоя — пропущен"})
            return []
        
        # Check cache
        h = hashlib.sha256(f"{pdf.stem}.pdf|{pdf.stat().st_size}".encode("utf-8")).hexdigest()
        text = get_cached_fragment(h)
        
        if not text:
            budget = settings.nim_chunk_char_budget
            chunks = _chunk_lines(lines, budget)
            if not chunks:
                return []
            
            from concurrent.futures import ThreadPoolExecutor
            from src.analysis import _FragProgress, _process_fragment, _submit_fragments, _collect_results
            
            executor = ThreadPoolExecutor(max_workers=min(settings.nim_workers, len(chunks)))
            tracker = _FragProgress("pdf", 1.0, progress)
            
            futures = _submit_fragments(f"{pdf.stem}.pdf", chunks, [nim_client] * min(settings.nim_workers, len(chunks)), executor, tracker, progress)
            executor.shutdown(wait=True)
            
            texts = [fut.result() for fut in futures]
            text = "\n\n".join(texts)
            save_fragment_cache(h, "pdf", text)
        
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
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_pdf, pdf, weights, progress, out_dir, quiet): pdf
            for pdf in pdfs
        }
        for fut in as_completed(futures):
            try:
                results.extend(fut.result())
            except Exception as exc:
                pdf = futures[fut]
                logger.error(f"PDF {pdf.name}: ошибка в потоке {exc}")
                if progress:
                    progress({"log": f"Ошибка: {pdf.name} — {exc}"})
    
    return results