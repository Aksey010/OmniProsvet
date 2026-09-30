import threading
from pathlib import Path

from loguru import logger

from src.collect import collect_media
from src.configuration import settings
from src.database import init_db
from src.transcription import run_extract, run_extract_and_process
from src.analysis import run_process_only
from src.attachments import run_attachments, run_pdfs, _att_kinds


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


def _attachments_worker(att_plan: list, weights: dict, progress, reports: list) -> None:
    for kind, files, gdir in att_plan:
        if not files:
            continue
        try:
            if kind == "pdf":
                reports += run_pdfs(files, weights, progress, gdir, quiet=True)
            else:
                reports += run_attachments(files, kind, weights, gdir, progress, quiet=True)
        except Exception as exc:
            logger.error(f"Обработка {kind} упала: {type(exc).__name__}: {exc}")
            if progress:
                progress({"log": f"Ошибка обработки {kind}: {exc}"})


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
        gdir = settings.output_dir / group_names[str(p.resolve())]
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
    out_dir = settings.output_dir

    att_reports: list[Path] = []
    att_thread = None
    if (extract or process) and kinds:
        att_thread = threading.Thread(
            target=_attachments_worker,
            args=(att_plan, weights, progress, att_reports),
            name="omni-attachments",
        )
        att_thread.start()
        if progress:
            progress({"log": "Картинки, PDF и документы обрабатываются параллельно"})
    elif kinds:
        for kind, files, gdir in att_plan:
            try:
                if kind == "pdf":
                    att_reports += run_pdfs(files, weights, progress, gdir)
                else:
                    att_reports += run_attachments(files, kind, weights, gdir, progress)
            except Exception as exc:
                logger.error(f"Обработка {kind} упала: {type(exc).__name__}: {exc}")

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
        if progress:
            for kind, files, _ in att_plan:
                if files:
                    progress({"phase": kind, "weight": weights.get(kind, 0)})
        att_thread.join()

    return {"transcripts": transcripts, "report": report, "image_reports": att_reports}