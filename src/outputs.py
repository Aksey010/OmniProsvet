"""Выходные файлы задачи: сбор списка, ZIP-архив, plain-text санитайзер LLM-ответов."""

import json
import os
import re
import threading
import time
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Сбор выходных файлов задачи
# ---------------------------------------------------------------------------

def collect_job_files(result: dict) -> list[dict]:
    """Собрать уникальные выходные файлы из результата run_pipeline.

    Порядок: общий отчёт → транскрипты → raw-извлечения → отчёты вложений
    (картинки/PDF/документы). Возвращает [{"name": ..., "path": ...}, ...]
    без дубликатов.
    """
    ordered: list[Path] = []
    report = result.get("report")
    if report:
        ordered.append(Path(report))
    for t in result.get("transcripts") or []:
        ordered.append(Path(t))
    for r in result.get("raw_files") or []:
        ordered.append(Path(r))
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
    """Относительный путь внутри ZIP: двухуровневое дерево категорий
    (стадия обработки -> тип источника).

    Первичная обработка — сырой извлечённый текст (транскрипты, OCR картинок
    и сканов). Вторичная — консПЕКТы NIM и распознанные документы. Отчёт —
    отдельно. Raw-извлечение (файлы режима «без NIM») — в собственной папке
    верхнего уровня, отдельно от остальных результатов. Неопознанное — «прочее».
    """
    if name.endswith((".transcript.txt", ".transcript.md")):
        return f"первичная обработка/транскрипты/{name}"
    if name.endswith(".raw.txt"):
        return f"raw-извлечение/{name}"
    if name.endswith(".images.txt"):
        return f"первичная обработка/картинки OCR/{name}"
    if "_scan_" in name and name.endswith(".txt"):
        return f"первичная обработка/сканы PDF OCR/{name}"
    if name.endswith(".pdf.txt"):
        return f"вторичная обработка/конспекты PDF/{name}"
    if name.endswith(".ppt.txt"):
        return f"вторичная обработка/конспекты презентаций/{name}"
    if name.endswith((".xlsx.txt", ".xls.txt")):
        return f"вторичная обработка/конспекты таблиц/{name}"
    if name.endswith(".documents.txt"):
        return f"вторичная обработка/распознанные документы/{name}"
    if name.startswith("report_") and name.endswith(".txt"):
        return f"отчёты/{name}"
    return f"прочее/{name}"


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


# ---------------------------------------------------------------------------
# Кэшированный ZIP на диске: большие задачи (11k+ файлов) собираются
# синхронно ~10-60 с — собираем ОДИН раз (в фоне после завершения задачи),
# повторные скачивания отдают готовый файл мгновенно.
# ---------------------------------------------------------------------------

def build_zip_to_file(files: list[dict], dest: Path) -> tuple[Path, int]:
    """Собрать архив прямо в файл (без буфера в памяти), атомарно.

    compresslevel=1: сборка ~2x быстрее дефолта, архив чуть больше —
    для кэшируемого архива скорость важнее степени сжатия.
    Возвращает (путь, число файлов в архиве).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Уникальный tmp: параллельные сборки одной задачи не затирают друг друга
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    count = 0
    seen: set = set()
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for item in files:
                p = Path(item["path"])
                if not p.is_file():
                    continue
                arc = _unique_arcname(_classify_arcname(p.name), seen)
                zf.write(p, arc)
                count += 1
        os.replace(tmp, dest)  # атомарно: читатель видит либо старый, либо новый
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    sidecar = dest.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"files": count, "built_at": int(time.time())}, ensure_ascii=False),
        encoding="utf-8",
    )
    return dest, count


def get_or_build_zip(files: list[dict], dest: Path) -> Path | None:
    """Отдать кэшированный архив или собрать. Пересборка — только если
    число существующих файлов изменилось (дозапуск задачи добавил файлы)."""
    expected = sum(1 for item in files if Path(item["path"]).is_file())
    sidecar = dest.with_suffix(".json")
    if dest.is_file() and sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("files") == expected:
                return dest
        except (ValueError, OSError):
            pass
    if expected == 0:
        return None
    build_zip_to_file(files, dest)
    return dest


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

# Markdown-таблицы: строка-разделитель — только |, :, -, пробелы (и >=3 дефисов)
_TABLE_SEP = re.compile(r"^[\s|:\-]*\-{3,}[\s|:\-]*$")
# Крайние трубы строки данных: «| a | b |» -> «a | b»
_TABLE_EDGE = re.compile(r"^\s*\|+|\|+\s*$")
# Оставшиеся трубы -> тире (в выводе LLM труб быть не должно)
_TABLE_PIPE = re.compile(r"\s*\|+\s*")
_DASH_RUN = re.compile(r"(?:\s*—\s*){2,}")
# Эмодзи (пиктограммы, dingbats, флаги, keycap-декорации «1️⃣», variation selector).
# Буквы, цифры, ₽ (U+20BD), № (U+2116), ≈, тире и кавычки в диапазоны не входят.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u20E3]")
# Буллеты в начале строки -> дефис (промпт требует перечисления через «- »)
_BULLET = re.compile(r"^(\s*)[•‣●▪○◦·]\s*")


def plain_text(text: str) -> str:
    """Привести LLM-ответ к читаемому plain text (выходной файл — .txt).

    Убирает markdown-разметку: звёздочки, обратные апострофы, решётки в
    заголовках, горизонтальные линии, html-теги, ссылки [текст](url) → «текст (url)».
    Дополнительно: markdown-таблицы (| a | b |, |---|---|) → строки «a — b»,
    эмодзи удаляются (keycap «1️⃣» → «1»), буллеты • в начале строки → «- ».
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
    out = _EMOJI.sub("", out)

    lines = []
    for raw in out.splitlines():
        line = raw.rstrip()
        if _TABLE_SEP.match(line):
            continue
        if _MD_HR.match(line):
            continue
        if "|" in line:
            line = _TABLE_EDGE.sub("", line)
            line = _TABLE_PIPE.sub(" — ", line)
            line = _DASH_RUN.sub(" — ", line)
        line = _BULLET.sub(r"\1- ", line)
        line = _MD_HEADING.sub("", line)
        line = _MD_QUOTE.sub("", line)
        lines.append(line)
    out = "\n".join(lines)

    out = _MULTI_BLANK.sub("\n\n", out)
    return out.strip()
