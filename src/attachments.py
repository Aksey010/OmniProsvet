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


def run_pdfs(pdfs: list[Path], weights: dict, progress=None, out_dir: Path | None = None, quiet: bool = False) -> list[Path]:
    if not pdfs:
        return []
    groups = _pdf_groups(pdfs)
    budget = settings.pdf_max_size_mb * 1024 * 1024
    logger.info(
        f"PDF: {len(pdfs)} шт. → {len(groups)} объединённых файлов (лимит {settings.pdf_max_size_mb} МБ)"
    )
    if progress and not quiet:
        progress({"phase": "pdf", "weight": weights.get("pdf", 0)})

    results: list[Path] = []
    for gi, group in enumerate(groups, 1):
        if progress and not quiet:
            progress({"file": f"PDF: {len(group)} шт. в одном", "file_index": gi, "file_total": len(groups)})
        if progress:
            progress({"log": f"PDF: группа {gi}/{len(groups)} ({len(group)} шт. в одном файле)…"})
        try:
            h = hashlib.sha256(("|".join(f"{p.name}:{p.stat().st_size}" for p in group)).encode("utf-8")).hexdigest()
            text = get_cached_fragment(h)
            merged = None
            if not text:
                merged = _merge_pdf_group(group)
                if merged is None:
                    continue
                if merged.stat().st_size > budget:
                    if len(group) > 1:
                        logger.warning(
                            f"Склеенный файл {merged.name} ({merged.stat().st_size // 1048576} МБ) превышает лимит — режу группу пополам"
                        )
                        mid = len(group) // 2
                        results += run_pdfs(group[:mid], weights, progress, quiet=True)
                        results += run_pdfs(group[mid:], weights, progress, quiet=True)
                        continue
                    else:
                        parts = _split_pdf_by_pages(group[0])
                        if len(parts) > 1:
                            results += run_pdfs(parts, weights, progress, quiet=True)
                            continue
                text = nim_client.send_prompt(
                    build_documents_prompt(len(group)),
                    deepthink=False,
                    expert=False,
                    stable_rounds=6,
                    attachments=[str(merged.resolve())],
                )
                save_fragment_cache(h, "pdf", text)
            target_dir = out_dir or settings.results_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / f"{group[0].stem}_{h[:8]}.pdf.txt"
            header = _header_block(
                "КОНСПЕКТ PDF",
                [
                    ("Документов", len(group)),
                    ("Объединено", f"{merged.name} ({merged.stat().st_size // 1048576} МБ)" if merged else ""),
                    ("Дата", time.strftime("%d.%m.%Y %H:%M")),
                ],
            )
            out_path.write_text(header + text.strip() + "\n\n" + _footer_block("конспекта"), encoding="utf-8")
            results.append(out_path)
            if progress:
                progress({"p": "pdf", "phase_progress": gi / len(groups)})
        except NimError as exc:
            if len(group) > 1:
                logger.warning(f"Группа {gi} из {len(group)} PDF не прошла ({exc}) — режу пополам и повторяю")
                if progress:
                    progress({"log": f"Группа PDF не прошла — режу пополам"})
                mid = len(group) // 2
                results += run_pdfs(group[:mid], weights, progress, quiet=True)
                results += run_pdfs(group[mid:], weights, progress, quiet=True)
            else:
                logger.error(f"PDF {group[0].name}: ошибка {exc}")
                if progress:
                    progress({"log": f"Ошибка: {group[0].name} — {exc}"})
    return results