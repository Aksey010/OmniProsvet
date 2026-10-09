"""Выходные файлы задачи: сбор списка, ZIP-архив, plain-text санитайзер LLM-ответов."""

import io
import re
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Сбор выходных файлов задачи
# ---------------------------------------------------------------------------

def collect_job_files(result: dict) -> list[dict]:
    """Собрать уникальные выходные файлы из результата run_pipeline.

    Порядок: общий отчёт → транскрипты → отчёты вложений (картинки/PDF/документы).
    Возвращает [{"name": ..., "path": ...}, ...] без дубликатов.
    """
    ordered: list[Path] = []
    report = result.get("report")
    if report:
        ordered.append(Path(report))
    for t in result.get("transcripts") or []:
        ordered.append(Path(t))
    for r in result.get("image_reports") or []:
        ordered.append(Path(r))

    seen: set[str] = set()
    files: list[dict] = []
    for p in ordered:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        files.append({"name": p.name, "path": str(p)})
    return files


def _classify_arcname(name: str) -> str:
    """Относительный путь внутри ZIP по имени файла.

    - *.transcript.txt          → transcripts/<имя>
    - *.pdf.txt / *.images.txt / *.documents.txt / *_scan_*.txt → summaries/<имя>
    - прочее (report_*.txt)     → <имя> (корень архива)
    """
    if name.endswith(".transcript.txt"):
        return f"transcripts/{name}"
    if name.endswith((".pdf.txt", ".images.txt", ".documents.txt")) or "_scan_" in name:
        return f"summaries/{name}"
    return name


def _unique_arcname(arc: str, seen: set) -> str:
    if arc not in seen:
        seen.add(arc)
        return arc
    base = arc
    dot = base.rfind(".")
    stem, ext = (base[:dot], base[dot:]) if dot > 0 else (base, "")
    i = 2
    while True:
        cand = f"{stem}_{i}{ext}"
        if cand not in seen:
            seen.add(cand)
            return cand
        i += 1


def build_zip_bytes(files: list[dict]) -> bytes:
    """Собрать ZIP из списка [{"name","path"}, ...] в памяти.

    Несуществующие файлы пропускаются. Имена внутри архива уникальны.
    """
    buf = io.BytesIO()
    seen: set = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in files:
            p = Path(item["path"])
            if not p.is_file():
                continue
            arc = _unique_arcname(_classify_arcname(p.name), seen)
            zf.write(p, arc)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Plain-text санитайзер LLM-ответов
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_MD_HR = re.compile(r"^ {0,3}([-*_])\1{2,}\s*$")
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?")
_ASTERISKS = re.compile(r"\*+")
_BACKTICKS = re.compile(r"`+")
_UNDER_EMPH = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")


def plain_text(text: str) -> str:
    """Привести LLM-ответ к читаемому plain text (выходной файл — .txt).

    Убирает markdown-разметку: звёздочки, обратные апострофы, решётки в
    заголовках, горизонтальные линии, html-теги, ссылки [текст](url) → «текст (url)».
    Подчёркивания внутри слов (имена файлов, snake_case) сохраняются.
    """
    if not text:
        return ""
    out = _MD_IMAGE.sub("", text)
    out = _MD_LINK.sub(r"\1 (\2)", out)
    out = _HTML_TAG.sub("", out)
    out = _ASTERISKS.sub("", out)
    out = _BACKTICKS.sub("", out)
    out = _UNDER_EMPH.sub(r"\1", out)

    lines = []
    for raw in out.splitlines():
        line = raw.rstrip()
        if _MD_HR.match(line):
            continue
        line = _MD_HEADING.sub("", line)
        line = _MD_QUOTE.sub("", line)
        lines.append(line)
    out = "\n".join(lines)

    out = _MULTI_BLANK.sub("\n\n", out)
    return out.strip()
