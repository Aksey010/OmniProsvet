import hashlib
import os
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aac"}
VIDEO_EXT = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v",
    ".wmv", ".flv", ".ts", ".mpg", ".mpeg",
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
DOCUMENT_EXT = {
    ".pdf", ".docx", ".doc", ".txt", ".rtf", ".csv",
    ".xlsx", ".xls", ".pptx", ".ppt", ".odt",
}

TRANSCRIPT_SUFFIX = ".transcript.txt"
TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "transcripts"

SKIP_DIRS = {
    "venv", ".venv", "__pycache__", ".git", "node_modules",
    "site-packages", ".idea", ".pytest_cache",
}


def _clean(path_str: str) -> Path:
    return Path(path_str.strip().strip('"'))


def _iter_files(path_str: str) -> list[Path]:
    p = _clean(path_str)
    if not p.exists():
        return []
    if p.is_file():
        return [p]
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
        for name in filenames:
            files.append(Path(root) / name)
    return sorted(files)


def is_document(f: Path) -> bool:
    name = f.name.lower()
    if name.endswith(TRANSCRIPT_SUFFIX):
        return False
    if name.startswith("report_") and name.endswith(".txt"):
        return False
    if name.endswith(".images.txt") or name.endswith(".documents.txt"):
        return False
    return f.suffix.lower() in DOCUMENT_EXT


def collect_media(path_str: str) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    files = _iter_files(path_str)
    audio = [f for f in files if f.suffix.lower() in AUDIO_EXT]
    video = [f for f in files if f.suffix.lower() in VIDEO_EXT]
    images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    documents = [f for f in files if is_document(f)]
    return audio, video, images, documents


def transcript_path_for(media: Path, base: Path | None = None) -> Path:
    h = hashlib.sha256(str(media.resolve()).encode("utf-8")).hexdigest()[:8]
    root = (base / "transcripts") if base else TRANSCRIPTS_DIR
    return root / h / f"{media.stem}.transcript.txt"


def find_existing_transcript(media: Path, base: Path | None = None) -> Path | None:
    new = transcript_path_for(media, base)
    if new.exists():
        return new
    if base is not None:
        legacy = transcript_path_for(media)
        if legacy.exists():
            return legacy
    old_md = media.with_name(media.stem + ".transcript.md")
    if old_md.exists():
        return old_md
    old_txt = media.with_name(media.stem + ".transcript.txt")
    if old_txt.exists():
        return old_txt
    return None


def collect_transcripts(path_str: str) -> list[Path]:
    p = _clean(path_str)
    if not p.exists():
        return []
    if p.is_file():
        tp = find_existing_transcript(p)
        return [tp] if tp else []
    return [
        f
        for f in _iter_files(path_str)
        if f.name.endswith((".transcript.txt", ".transcript.md"))
    ]
