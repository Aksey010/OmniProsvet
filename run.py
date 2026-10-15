#!/usr/bin/env python
"""
OmniProsvet — единая точка входа.
Запуск:
  python run.py              # веб-интерфейс (по умолчанию)
  python run.py web          # веб-интерфейс
  python run.py cli all /path/to/files  # CLI режим
  python run.py cli extract /path
  python run.py cli process /path
"""
import sys
import argparse
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_web():
    """Запуск веб-панели."""
    from web_panel import socketio, app
    import threading
    import webbrowser

    url = "http://127.0.0.1:5000"
    print(f"🌐 Открываю веб-интерфейс: {url}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)


def run_cli(args):
    """Запуск CLI."""
    from src.pipeline import run_pipeline
    from src.configuration import settings
    from src.database import init_db
    from src.progress import format_file_event
    from loguru import logger

    logger.add("omniprosvet.log", rotation="1 week", encoding="utf-8")
    init_db()

    errors = settings.validate()
    if errors:
        for err in errors:
            logger.error(f"Ошибка конфигурации: {err}")
        return 1

    def progress(event: dict) -> None:
        if "log" in event:
            logger.info(event["log"])
        else:
            # Метка фазы разграничивает параллельные процессы в потоке вывода
            line = format_file_event(event)
            if line:
                print(line, flush=True)

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
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="OmniProsvet — экстрактор информации из аудио, видео и документов",
        usage="%(prog)s [web|cli ...]",
    )
    subparsers = parser.add_subparsers(dest="mode", help="Режим запуска")

    # Web режим
    web_parser = subparsers.add_parser("web", help="Запустить веб-интерфейс (по умолчанию)")

    # CLI режим
    cli_parser = subparsers.add_parser("cli", help="Командная строка")
    cli_sub = cli_parser.add_subparsers(dest="command", required=True)

    for cmd in ("extract", "process", "all"):
        sp = cli_sub.add_parser(cmd, help=f"{cmd} — {'транскрибация' if cmd=='extract' else 'анализ' if cmd=='process' else 'всё'}")
        sp.add_argument("paths", nargs="+", help="Файл или папка с материалами")
        sp.add_argument("-m", "--model", default=None, help="Модель whisper: tiny, base, small, medium, large-v3")
        sp.add_argument("--language", default=None, help="Код языка речи (по умолчанию из .env)")
        sp.add_argument("--no-diarize", action="store_true", help="Отключить диаризацию спикеров")
        sp.add_argument("--no-images", action="store_true", help="Не обрабатывать картинки")

    # Если аргументов нет — запускаем веб
    if len(sys.argv) == 1:
        run_web()
        return

    args = parser.parse_args()

    if args.mode == "web" or args.mode is None:
        run_web()
    elif args.mode == "cli":
        sys.exit(run_cli(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()