import hashlib
from pathlib import Path
from typing import Any

from src.configuration import settings


MAX_LINE_CHARS = 500

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


def _prepare_transcript(transcript: Path, budget: int | None = None) -> tuple[str, list[list[str]]] | None:
    lines = _read_lines(transcript)
    if not lines:
        return None
    source = _transcript_source_label(transcript)
    entries = [f"[{transcript.stem}] {line}" for line in lines]
    return source, _chunk_lines(entries, budget)


def _is_small_chunk(chunk: list[str], budget: int | None = None) -> bool:
    b = budget or settings.chunk_char_budget
    size = sum(len(line) for line in chunk)
    return (
        size <= b // 2
        and len(chunk) <= settings.chunk_max_lines // 2
    )