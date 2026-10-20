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
    """Запуск CLI — делегирует в main.main (единый парсер, лог и итоги).

    Раньше здесь была вторая копия argparse/progress/итогов — любые правки
    CLI приходилось делать дважды."""
    from main import main as run_main

    argv = [args.command, *args.paths]
    if args.model:
        argv += ["-m", args.model]
    if args.language:
        argv += ["--language", args.language]
    if args.no_diarize:
        argv += ["--no-diarize"]
    if args.no_images:
        argv += ["--no-images"]
    return run_main(argv)


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

    for cmd in ("extract", "process", "all", "raw"):
        help_text = {
            "extract": "транскрибация",
            "process": "анализ",
            "all": "всё",
            "raw": "транскрибация + локальное извлечение (без NIM)",
        }[cmd]
        sp = cli_sub.add_parser(cmd, help=f"{cmd} — {help_text}")
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