import argparse
from pathlib import Path

from loguru import logger

from src.configuration import settings
from src.database import init_db
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OmniProsvet — экстрактор информации из аудио, видео и картинок"
    )
    parser.add_argument(
        "command",
        choices=["extract", "process", "all"],
        help="extract — транскрибация; process — NIM по транскриптам и картинкам; all — всё",
    )
    parser.add_argument("paths", nargs="+", help="Файл или папка с материалами")
    parser.add_argument(
        "-m", "--model", default=None,
        help="Модель whisper: tiny, base, small, medium, large-v3 (по умолчанию из .env)",
    )
    parser.add_argument("--language", default=None, help="Код языка речи (по умолчанию из .env)")
    parser.add_argument("--no-diarize", action="store_true", help="Отключить диаризацию спикеров")
    parser.add_argument("--no-images", action="store_true", help="Не обрабатывать картинки")
    args = parser.parse_args()

    logger.add("omniprosvet.log", rotation="1 week", encoding="utf-8")
    init_db()

    # Валидация настроек
    errors = settings.validate()
    if errors:
        for err in errors:
            logger.error(f"Ошибка конфигурации: {err}")
        return

    def progress(event: dict) -> None:
        if "log" in event:
            logger.info(event["log"])
        elif "file" in event:
            pct = event.get("progress")
            suffix = f" {pct}%" if pct else ""
            print(
                f"[{event.get('file_index', 0)}/{event.get('file_total', 0)}] "
                f"{event['file']}{suffix}",
                flush=True,
            )

    result = run_pipeline(
        [Path(p) for p in args.paths],
        model=args.model,
        language=args.language,
        diarize=False if args.no_diarize else None,
        extract=args.command in ("extract", "all"),
        process=args.command in ("process", "all"),
        images=args.command != "extract" and not args.no_images,
        progress=progress,
    )

    print("\n=== Итоги ===")
    for t in result["transcripts"]:
        print(f"Транскрипт: {t}")
    for r in result["image_reports"]:
        print(f"Картинки: {r}")
    if result["report"]:
        print(f"Единый отчёт: {result['report']}")


if __name__ == "__main__":
    main()
