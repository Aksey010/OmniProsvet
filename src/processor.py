import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from src.collect import collect_transcripts
from src.configuration import settings
from src.database import get_cached_fragment, save_fragment_cache
from src.deepseek_agent import GenerationError, agent, agent2
from src.nim_client import NimError
from src.prompts import build_documents_prompt, build_images_prompt, build_map_prompt, build_reduce_prompt
from src.whisper_extract import transcribe_file

MAX_LINE_CHARS = 500
SMALL_SOURCE = "мелкие файлы"
REDUCE_FRACTION = 0.9
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
LINE = "=" * 64
THIN = "-" * 64


def _header_block(title: str, fields: list[tuple[str, str]]) -> str:
    lines = [LINE, f"  {title}", LINE]
    for name, value in fields:
        if value:
            lines.append(f"{name + ':':<14}{value}")
    lines.append(THIN)
    lines.append("")
    return "\n".join(lines)


def _footer_block(label: str) -> str:
    return f"{THIN}\n  Конец {label} · OmniProsvet\n{LINE}\n"


def _transcript_source_label(transcript: Path) -> str:
    stem = transcript.stem
    if stem.endswith(".transcript"):
        stem = stem[: -len(".transcript")]
    return stem + ".transcript.md"


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


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip() + "…"


def _chunk_hash(source: str, chunk: list[str]) -> str:
    return hashlib.sha256((source + "|" + "\n".join(chunk)).encode("utf-8")).hexdigest()


def _agents_pool(platform: str | None = None) -> list:
    eff = platform or settings.analysis_platform
    if eff == "nim":
        from src.nim_client import client

        return [client] * max(1, settings.nim_workers)
    n = max(1, settings.deepseek_workers)
    if n >= 2:
        return [agent, agent2]
    return [agent]


def _read_lines(transcript: Path) -> list[str]:
    lines = []
    for raw in transcript.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line.startswith("["):
            continue
        lines.append(_truncate(line, MAX_LINE_CHARS))
    return lines


def _chunk_lines(lines: list[str], budget: int | None = None) -> list[list[str]]:
    b = budget or settings.chunk_char_budget
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and (
            size + len(line) > b
            or len(current) >= settings.chunk_max_lines
        ):
            chunks.append(current)
            current = []
            size = 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append(current)
    return chunks


def _process_fragment(idx: int, total: int, chunk: list[str], source: str, ag, progress) -> str:
    body = "\n".join(chunk)
    h = _chunk_hash(source, chunk)
    cached = get_cached_fragment(h)
    if cached:
        logger.info(f"Конспект фрагмента {idx}/{total} «{source}» — из кэша")
        if progress:
            progress({"log": f"Фрагмент {idx}/{total} «{source}» — из кэша"})
        return cached
    if progress:
        progress({"log": f"Фрагмент {idx}/{total} «{source}» — отправляю в DeepSeek…"})
    text = ag.send_prompt(
        build_map_prompt(idx, total, source, body),
        login=settings.deepseek_login,
        password=settings.deepseek_password,
        deepthink=False,
        expert=False,
        stable_rounds=6,
    )
    save_fragment_cache(h, source, text)
    logger.info(f"Конспект фрагмента {idx}/{total} «{source}» готов ({len(text)} симв.)")
    return text


def _prepare_transcript(transcript: Path, budget: int | None = None) -> tuple[str, list[list[str]]] | None:
    lines = _read_lines(transcript)
    if not lines:
        return None
    source = _transcript_source_label(transcript)
    entries = [f"[{transcript.stem}] {line}" for line in lines]
    return source, _chunk_lines(entries, budget)


def _submit_fragments(
    name: str,
    chunks: list[list[str]],
    agents: list,
    executor: ThreadPoolExecutor,
    tracker: _FragProgress,
    progress,
) -> list:
    tracker.add_total(len(chunks))
    futures = []
    for i, chunk in enumerate(chunks):
        ag = agents[i % len(agents)]
        fut = executor.submit(_process_fragment, i + 1, len(chunks), chunk, name, ag, progress)
        fut.add_done_callback(tracker.tick)
        futures.append(fut)
    return futures


def _is_small_chunk(chunk: list[str], budget: int | None = None) -> bool:
    b = budget or settings.chunk_char_budget
    size = sum(len(line) for line in chunk)
    return (
        size <= b // 2
        and len(chunk) <= settings.chunk_max_lines // 2
    )


class _SmallBuffer:
    def __init__(self, agents: list, executor: ThreadPoolExecutor, tracker: _FragProgress, progress) -> None:
        self.lines: list[str] = []
        self.size = 0
        self.futures: list = []
        self.agents = agents
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
            _process_fragment, 1, 1, chunk, SMALL_SOURCE, self.agents[0], self.progress
        )
        fut.add_done_callback(self.tracker.tick)
        self.futures.append(fut)


def _reduce_and_save(
    transcripts_order: list[Path],
    all_futures: list,
    out_dir: Path | None,
    progress,
    platform: str | None = None,
) -> Path:
    if progress:
        progress({"log": "Фрагменты готовы, собираю финальный отчёт…"})
        progress({"p": "deepseek", "phase_progress": 0.95})
    parts = [fut.result() for fut in all_futures]
    parts_block = "\n\n".join(f"### Фрагмент {i}\n{p}" for i, p in enumerate(parts, 1))
    ag = _agents_pool(platform)[0]
    reduce_model = settings.nim_reduce_model if (platform or settings.analysis_platform) == "nim" else None
    final = ag.send_prompt(
        build_reduce_prompt(len(transcripts_order), len(all_futures), parts_block),
        login=settings.deepseek_login,
        password=settings.deepseek_password,
        deepthink=settings.deepseek_deepthink,
        expert=settings.deepseek_expert,
        stable_rounds=6,
        model=reduce_model,
    )
    target_dir = out_dir or OUTPUT_DIR
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


def _collect_results(futures_map: dict[str, list], transcripts_order: list[Path]) -> list:
    all_futures: list = []
    for t in transcripts_order:
        all_futures += futures_map.get(str(t), [])
    all_futures += futures_map.get(SMALL_SOURCE, [])
    return all_futures


def _feed_transcript(
    transcript: Path,
    futures_map: dict[str, list],
    agents: list,
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
    futures_map[str(transcript)] = _submit_fragments(source, chunks, agents, executor, tracker, progress)


def run_extract_and_process(
    media_files: list[tuple[Path, Path]],
    model: str,
    language: str,
    diarize: bool,
    out_dir: Path | None,
    weights: dict,
    progress=None,
    platform: str | None = None,
) -> tuple[list[Path], Path | None]:
    import queue as q_mod

    if not media_files:
        return [], None

    logger.info(
        f"Транскрибация + обработка: {len(media_files)} файлов, модель {model}, "
        f"язык {language}, диаризация: {'вкл' if diarize else 'выкл'}"
    )
    if progress:
        progress({"log": f"Транскрибация: {len(media_files)} файлов (обработка идёт параллельно)"})
        progress({"phase": "transcribe", "weight": weights.get("transcribe", 50)})

    agents = _agents_pool(platform)
    executor = ThreadPoolExecutor(max_workers=len(agents))
    tracker = _FragProgress("deepseek", REDUCE_FRACTION, progress)
    q: q_mod.Queue = q_mod.Queue()
    budget = settings.nim_chunk_char_budget if (platform or settings.analysis_platform) == "nim" else None
    transcripts_order: list[Path] = []
    futures_map: dict[str, list] = {}
    small_buffer = _SmallBuffer(agents, executor, tracker, progress)
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

    def feeder_worker() -> None:
        while True:
            t = q.get()
            try:
                if t is None:
                    return
                try:
                    _feed_transcript(t, futures_map, agents, executor, small_buffer, tracker, progress, budget)
                except Exception as exc:
                    logger.error(f"Ошибка обработки фрагментов {t.name}: {exc}")
                    if progress:
                        progress({"log": f"Ошибка обработки {t.name}: {exc}"})
            finally:
                q.task_done()

    t_thread = threading.Thread(target=transcribe_worker, name="omni-transcriber")
    f_thread = threading.Thread(target=feeder_worker, name="omni-feeder")
    t_thread.start()
    f_thread.start()
    t_thread.join()
    if progress:
        progress({"phase": "deepseek", "weight": weights.get("deepseek", 35)})
    f_thread.join()
    small_buffer.flush()
    futures_map[SMALL_SOURCE] = small_buffer.futures
    executor.shutdown(wait=True)

    all_futures = _collect_results(futures_map, transcripts_order)
    if not all_futures:
        logger.warning("В транскриптах нет распознанного текста")
        return transcripts_order, None
    report = _reduce_and_save(transcripts_order, all_futures, out_dir, progress, platform)
    return transcripts_order, report


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


def run_process_only(
    media_files: list[tuple[Path, Path]],
    out_dir: Path | None,
    weights: dict,
    progress=None,
    platform: str | None = None,
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

    agents = _agents_pool(platform)
    logger.info(f"Обработка: {len(transcripts)} транскриптов, агентов: {len(agents)}")
    if progress:
        label = "DeepSeek" if (platform or settings.analysis_platform) == "deepseek" else "NIM"
        progress({"log": f"{label}: {len(transcripts)} транскриптов, агентов: {len(agents)}"})
        progress({"phase": "deepseek", "weight": weights.get("deepseek", 100)})

    executor = ThreadPoolExecutor(max_workers=len(agents))
    tracker = _FragProgress("deepseek", REDUCE_FRACTION, progress)
    budget = settings.nim_chunk_char_budget if (platform or settings.analysis_platform) == "nim" else None
    futures_map: dict[str, list] = {}
    small_buffer = _SmallBuffer(agents, executor, tracker, progress)
    for t in transcripts:
        _feed_transcript(t, futures_map, agents, executor, small_buffer, tracker, progress, budget)
    small_buffer.flush()
    futures_map[SMALL_SOURCE] = small_buffer.futures
    executor.shutdown(wait=True)

    all_futures = _collect_results(futures_map, transcripts)
    if not all_futures:
        logger.warning("В транскриптах нет распознанного текста")
        return transcripts, None
    report = _reduce_and_save(transcripts, all_futures, out_dir, progress, platform)
    return transcripts, report


def _process_attachment_batch(batch: list[Path], kind: str, out_dir: Path | None = None, platform: str | None = None) -> Path:
    key = "|".join(f"{p.name}:{p.stat().st_size}" for p in batch)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    text = get_cached_fragment(h)
    if not text:
        prompt = build_images_prompt(len(batch)) if kind == "images" else build_documents_prompt(len(batch))
        ag = _agents_pool(platform)[0]
        text = ag.send_prompt(
            prompt,
            login=settings.deepseek_login,
            password=settings.deepseek_password,
            deepthink=False,
            expert=False,
            stable_rounds=6,
            attachments=[str(p.resolve()) for p in batch],
        )
        save_fragment_cache(h, kind, text)
    target_dir = out_dir or RESULTS_DIR
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


def run_attachments(files: list[Path], kind: str, weights: dict, out_dir: Path | None = None, progress=None, quiet: bool = False, platform: str | None = None) -> list[Path]:
    if not files:
        return []
    batch_size = max(1, settings.images_per_batch)
    if settings.analysis_platform == "nim":
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
            results.append(_process_attachment_batch(batch, kind, out_dir, platform))
            if progress:
                progress({"p": kind, "phase_progress": bi / len(batches)})
        except (GenerationError, NimError) as exc:
            if len(batch) > 1:
                logger.warning(f"Батч {bi} не прошёл ({exc}) — режу пополам и повторяю")
                if progress:
                    progress({"log": f"Батч {bi} не прошёл — режу батч пополам"})
                mid = len(batch) // 2
                results += run_attachments(batch[:mid], kind, weights, out_dir, progress, quiet, platform)
                results += run_attachments(batch[mid:], kind, weights, out_dir, progress, quiet, platform)
            else:
                logger.error(f"{label[:-1]} {batch[0].name}: ошибка {exc}")
                if progress:
                    progress({"log": f"Ошибка: {batch[0].name} — {exc}"})
    return results


def _attachments_worker(att_plan: list, weights: dict, progress, reports: list, platform: str | None = None) -> None:
    for kind, files, gdir in att_plan:
        if not files:
            continue
        try:
            if kind == "pdf":
                reports += run_pdfs(files, weights, progress, gdir, quiet=True, platform=platform)
            else:
                reports += run_attachments(files, kind, weights, gdir, progress, quiet=True, platform=platform)
        except Exception as exc:
            logger.error(f"Обработка {kind} упала: {type(exc).__name__}: {exc}")
            if progress:
                progress({"log": f"Ошибка обработки {kind}: {exc}"})


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
    from pypdf import PdfReader, PdfWriter

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


def run_pdfs_nim(pdfs: list[Path], weights: dict, progress=None, out_dir: Path | None = None, quiet: bool = False, platform: str | None = None) -> list[Path]:
    if progress and not quiet:
        progress({"phase": "pdf", "weight": weights.get("pdf", 0)})
    agents = _agents_pool(platform)
    executor = ThreadPoolExecutor(max_workers=len(agents))
    tracker = _FragProgress("pdf", 1.0, progress)
    target_dir = out_dir or RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    futures_by_pdf: dict[str, list] = []
    futures_by_pdf = {}
    order: list[Path] = []
    total = len(pdfs)

    for idx, pdf in enumerate(pdfs, 1):
        if progress and not quiet:
            progress({"file": f"PDF: {pdf.name}", "file_index": idx, "file_total": total})
        if progress:
            progress({"log": f"PDF: извлекаю текст из {pdf.name}…"})
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf))
            lines = []
            for page in reader.pages:
                text = (page.extract_text() or "").strip()
                if text:
                    lines.append(_truncate(text, MAX_LINE_CHARS))
            if not lines:
                logger.warning(f"PDF {pdf.name}: текст не извлечён (скан без текстового слоя?)")
                if progress:
                    progress({"log": f"PDF {pdf.name}: нет текстового слоя — пропущен"})
                continue
            source = f"{pdf.stem}.pdf"
            budget = settings.nim_chunk_char_budget if (platform or settings.analysis_platform) == "nim" else None
            chunks = _chunk_lines(lines, budget)
            futures_by_pdf[source] = _submit_fragments(source, chunks, agents, executor, tracker, progress)
            order.append(pdf)
        except GenerationError:
            raise
        except Exception as exc:
            logger.error(f"PDF {pdf.name}: ошибка {type(exc).__name__}: {exc}")
            if progress:
                progress({"log": f"Ошибка: {pdf.name} — {exc}"})

    executor.shutdown(wait=True)

    for pdf in order:
        source = f"{pdf.stem}.pdf"
        futures = futures_by_pdf.get(source, [])
        try:
            texts = [fut.result() for fut in futures]
        except Exception as exc:
            logger.error(f"PDF {pdf.name}: ошибка фрагментов {exc}")
            continue
        h = hashlib.sha256(f"{pdf.stem}.pdf|{pdf.stat().st_size}".encode("utf-8")).hexdigest()
        out_path = target_dir / f"{pdf.stem}_{h[:8]}.pdf.txt"
        header = _header_block(
            "КОНСПЕКТ PDF",
            [
                ("Документ", pdf.name),
                ("Фрагментов", len(texts)),
                ("Дата", time.strftime("%d.%m.%Y %H:%M")),
            ],
        )
        out_path.write_text(header + "\n\n".join(texts).strip() + "\n\n" + _footer_block("конспекта"), encoding="utf-8")
        results.append(out_path)
        logger.info(f"Конспект PDF готов: {out_path.name} ({len(texts)} фрагментов)")
        if progress:
            progress({"log": f"Конспект PDF готов: {out_path.name}"})
    return results


def _split_pdf_by_pages(pdf: Path) -> list[Path]:
    from pypdf import PdfReader, PdfWriter

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


def run_pdfs(pdfs: list[Path], weights: dict, progress=None, out_dir: Path | None = None, quiet: bool = False, platform: str | None = None) -> list[Path]:
    if not pdfs:
        return []
    eff = platform or settings.analysis_platform
    if eff == "nim":
        return run_pdfs_nim(pdfs, weights, progress, out_dir, quiet)
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
                    else:
                        parts = _split_pdf_by_pages(group[0])
                        if len(parts) > 1:
                            results += run_pdfs(parts, weights, progress, quiet=True)
                        continue
                ag = _agents_pool(platform)[0]
                text = ag.send_prompt(
                    build_documents_prompt(len(group)),
                    login=settings.deepseek_login,
                    password=settings.deepseek_password,
                    deepthink=False,
                    expert=False,
                    stable_rounds=6,
                    attachments=[str(merged.resolve())],
                )
                save_fragment_cache(h, "pdf", text)
            target_dir = out_dir or RESULTS_DIR
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
        except GenerationError as exc:
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


def _group_names(paths: list[Path]) -> dict[str, str]:
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


def _att_kinds(img_files: list[Path], doc_files: list[Path]) -> list[tuple[str, list[Path]]]:
    pdfs = [f for f in doc_files if f.suffix.lower() == ".pdf"]
    other_docs = [f for f in doc_files if f.suffix.lower() != ".pdf"]
    return [("images", img_files), ("pdf", pdfs), ("documents", other_docs)]


def _weights_for(extract: bool, process: bool, att_kinds: list[str]) -> dict:
    if extract and process:
        weights = {"transcribe": 45, "deepseek": 30}
    elif extract:
        weights = {"transcribe": 80}
    else:
        weights = {"deepseek": 75}
    if att_kinds:
        base = 100 - sum(weights.values())
        each = round(base / len(att_kinds), 1)
        for kind in att_kinds[:-1]:
            weights[kind] = each
        weights[att_kinds[-1]] = round(base - each * (len(att_kinds) - 1), 1)
    elif extract and process:
        weights = {"transcribe": 60, "deepseek": 40}
    elif extract:
        weights = {"transcribe": 100}
    else:
        weights = {"deepseek": 100}
    return weights


def run_pipeline(
    paths,
    model: str | None = None,
    language: str | None = None,
    diarize: bool | None = None,
    extract: bool = True,
    process: bool = True,
    images: bool = True,
    out_dir: Path | None = None,
    progress=None,
    platform: str | None = None,
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
    for p in clean_paths:
        a, v, im, d = collect_media(str(p))
        gdir = OUTPUT_DIR / group_names[str(p.resolve())]
        gdir.mkdir(parents=True, exist_ok=True)
        found_any = found_any or bool(a or v or im or d)
        for m in a + v:
            media_with_groups.append((m, gdir))
        pdfs = [f for f in d if f.suffix.lower() == ".pdf"]
        others = [f for f in d if f.suffix.lower() != ".pdf"]
        if im:
            att_plan.append(("images", im, gdir))
        if pdfs:
            att_plan.append(("pdf", pdfs, gdir))
        if others:
            att_plan.append(("documents", others, gdir))

    if not found_any:
        logger.warning(f"Поддерживаемых файлов не найдено: {[str(p) for p in clean_paths]}")
        if progress:
            progress({"log": "Поддерживаемых файлов (аудио/видео/картинки/документы) не найдено"})

    kinds = [k for k, files, _ in att_plan if files]
    weights = _weights_for(extract, process, kinds)
    out_dir = OUTPUT_DIR

    att_reports: list[Path] = []
    att_thread = None
    if (extract or process) and kinds:
        att_thread = threading.Thread(
            target=_attachments_worker,
            args=(att_plan, weights, progress, att_reports, platform),
            name="omni-attachments",
        )
        att_thread.start()
        if progress:
            progress({"log": "Картинки, PDF и документы обрабатываются параллельно"})
    elif kinds:
        for kind, files, gdir in att_plan:
            try:
                if kind == "pdf":
                    att_reports += run_pdfs(files, weights, progress, gdir, platform=platform)
                else:
                    att_reports += run_attachments(files, kind, weights, gdir, progress, platform=platform)
            except Exception as exc:
                logger.error(f"Обработка {kind} упала: {type(exc).__name__}: {exc}")

    transcripts: list[Path] = []
    report = None
    if extract and process:
        transcripts, report = run_extract_and_process(
            media_with_groups, model, language, use_diar, out_dir, weights, progress, platform
        )
    elif extract:
        transcripts = run_extract(media_with_groups, model, language, use_diar, weights, progress)
    elif process:
        transcripts, report = run_process_only(media_with_groups, out_dir, weights, progress, platform)

    if att_thread is not None:
        if progress:
            for kind, files, _ in att_plan:
                if files:
                    progress({"phase": kind, "weight": weights.get(kind, 0)})
        att_thread.join()

    return {"transcripts": transcripts, "report": report, "image_reports": att_reports}
