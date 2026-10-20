"""Локальный OCR (RapidOCR/onnxruntime) — для картинок и страниц сканов.

С картинок теперь только извлекается текст (описание сцены признано
избыточным), поэтому специализированный OCR-движок выгоднее LLM-vision:
без сети, лимитов RPM и таймаутов NIM.

Модель распознавания — cyrillic_PP-OCRv5_rec_mobile (словарь кириллицы;
дефолтная ch-модель PP-OCRv6 русский текст ТЕРЯЕТ — проверено:
«Скидка 25% до 31 декабря» → « 25 31 »). Детектор остаётся универсальным.

Включается настройкой OCR_ENGINE=rapidocr (.env); пакет не установлен —
автоматический fallback на NIM-vision путь (см. attachments._local_ocr_active).
"""

import threading
from pathlib import Path

from loguru import logger

try:
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    RAPIDOCR_AVAILABLE = True
except ImportError:  # pragma: no cover — зависит от окружения
    RAPIDOCR_AVAILABLE = False

_engine = None
_engine_lock = threading.Lock()

DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_WORKERS = 4


def get_engine():
    """Ленивый синглтон движка. Модель rec — кириллическая PP-OCRv5 mobile:
    баланс скорости/качества для смешанного русско-латинского контента."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                logger.info("Загрузка OCR-движка RapidOCR (PP-OCRv5 cyrillic, onnxruntime)…")
                _engine = RapidOCR(params={
                    "Rec.lang_type": LangRec.CYRILLIC,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                })
    return _engine


def _group_lines(boxes, txts, scores, min_confidence: float) -> list[str]:
    """Боксы -> читабельные строки: группировка по вертикали (центры боксов
    в пределах высоты строки — одна строка), внутри строки — слева направо,
    строки — сверху вниз. Порядок обхода детектора не гарантирован."""
    items = []
    # boxes/txts/scores приходят numpy-массивами: truth-value проверка запрещена
    if boxes is None or txts is None or scores is None:
        return []
    for box, text, score in zip(boxes, txts, scores):
        if float(score) < min_confidence:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        text = (text or "").strip()
        if not text:
            continue
        items.append({"y0": min(ys), "y1": max(ys), "x": min(xs), "text": text})
    if not items:
        return []
    items.sort(key=lambda it: ((it["y0"] + it["y1"]) / 2, it["x"]))

    lines: list[dict] = []
    for it in items:
        placed = False
        if lines:
            last = lines[-1]
            last_mid = (last["y0"] + last["y1"]) / 2
            it_mid = (it["y0"] + it["y1"]) / 2
            height = max(1.0, last["h"], it["y1"] - it["y0"])
            if abs(it_mid - last_mid) <= 0.6 * height:
                last["items"].append(it)
                last["y0"] = min(last["y0"], it["y0"])
                last["y1"] = max(last["y1"], it["y1"])
                last["h"] = max(last["h"], it["y1"] - it["y0"])
                placed = True
        if not placed:
            lines.append({
                "y0": it["y0"], "y1": it["y1"],
                "h": it["y1"] - it["y0"], "items": [it],
            })

    out: list[str] = []
    for line in lines:
        line["items"].sort(key=lambda it: it["x"])
        out.append(" ".join(it["text"] for it in line["items"]))
    return out


def ocr_image(path: Path, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> str:
    """Распознать текст на одной картинке. Пусто -> '' (нет текста)."""
    result = get_engine()(str(path))
    if result is None or result.txts is None or len(result.txts) == 0:
        return ""
    lines = _group_lines(result.boxes, result.txts, result.scores, min_confidence)
    return "\n".join(lines)


def ocr_images_parallel(paths: list[Path], workers: int | None = None, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> list[str]:
    """OCR списка файлов в потоках (onnxruntime отпускает GIL в Run()).
    Ошибка одного файла -> пустой текст для него, остальные продолжаются."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not paths:
        return []
    workers = max(1, workers or DEFAULT_WORKERS)

    results: list[str] = [""] * len(paths)

    def _one(i: int, p: Path) -> tuple[int, str]:
        try:
            return i, ocr_image(p, min_confidence)
        except Exception as exc:
            logger.warning(f"OCR {p.name}: ошибка {type(exc).__name__}: {exc}")
            return i, ""

    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as executor:
        futures = [executor.submit(_one, i, p) for i, p in enumerate(paths)]
        for fut in as_completed(futures):
            i, text = fut.result()
            results[i] = text
    return results
