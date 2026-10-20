"""Веб-панель OmniProsvet: запуск задач, прогресс по фазам, скачивание результатов.

UI — статический файл static/index.html (без Jinja: правки UI делаются в нём).
Состояние задач живёт в SQLite (src.database), прогресс — через SocketIO.
"""

import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_socketio import SocketIO
from loguru import logger

from src.configuration import settings
from src.database import init_db, get_job, save_job, delete_old_jobs, mark_interrupted_jobs
from src.outputs import collect_job_files
from src.progress import PhaseProgress
from src.pipeline import run_pipeline

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = settings.output_dir
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

logger.add(BASE_DIR / "omniprosvet.log", rotation="1 week", encoding="utf-8")

init_db()

# Сколько последних строк лога задачи хранить в job (и в SQLite).
# Полный лог всё равно пишется в omniprosvet.log; хранить тысячи строк в
# job-записи — значит сериализовать мегабайты JSON на каждый тик прогресса
# (в сессии 13.10 job дошёл до 500+ КБ и 7700 строк).
MAX_JOB_LINES = 500

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
LANGUAGES = {"auto", "zh", "ru", "en", "ja", "ko", "de", "fr", "es"}
MODES = ("full", "extract", "raw")
ZIPS_DIR = OUTPUT_DIR / "zips"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Очистка старых задач и пометка прерванных (их потоки умерли с прошлым процессом)
delete_old_jobs(days=7)
mark_interrupted_jobs()


def _prebuild_zip(job_id: str, files: list[dict]) -> None:
    """Фоновая сборка ZIP задачи: к моменту клика архив готов на диске."""
    from src.outputs import get_or_build_zip

    try:
        get_or_build_zip(files, ZIPS_DIR / f"{job_id}.zip")
    except Exception as exc:
        logger.warning(f"Предсборка ZIP {job_id[:8]} не удалась: {exc}")


def run_job(job_id: str, paths: list[str], mode: str, diarize: bool, model: str, language: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    pp = PhaseProgress()
    # Троттлинг записи состояния в SQLite: тики прогресса приходят чаще
    # двух раз в секунду, и каждая запись — сериализация всего job в JSON.
    last_save = {"t": 0.0}

    def progress(event: dict) -> None:
        if "log" in event:
            job["lines"].append(event["log"])
            if len(job["lines"]) > MAX_JOB_LINES:
                del job["lines"][:-MAX_JOB_LINES]
            socketio.emit("job_log", {"job_id": job_id, "log": event["log"]})
            # Лог-события приходят пачками (по строке на каждый фрагмент/файл):
            # писать весь job в SQLite на каждый — квадратичная нагрузка.
            now = time.monotonic()
            if now - last_save["t"] >= 0.5:
                last_save["t"] = now
                save_job(job_id, job)
            return
        if "estimate" in event:
            # Предварительная оценка времени — отдельное поле: показывается
            # в карточке прогресса постоянно, не уезжает с прокруткой лога.
            job["estimate"] = event["estimate"]
        pp.feed(event)
        pct = pp.percent()
        if pct > job.get("progress", 0):
            job["progress"] = pct

        # Статус по фазам: у каждой параллельной задачи — своя надпись.
        # Раньше единое поле current/file_index перезаписывалось всеми фазами
        # по очереди (36/4000 → 5/50 → обратно), и фазы «перебивали» друг друга.
        status = job.setdefault("phase_status", {})
        if "file" in event and event.get("phase"):
            entry = status.setdefault(event["phase"], {})
            entry["current"] = event["file"]
            if event.get("file_index") is not None:
                entry["index"] = event["file_index"]
            if event.get("file_total") is not None:
                entry["total"] = event["file_total"]
        if "phase_progress" in event and event.get("p"):
            entry = status.setdefault(event["p"], {})
            entry["percent"] = min(
                100, round(float(event["phase_progress"]) * 100)
            )
            # Счётчик «N из M» от батчевых фаз (Картинки/Документы)
            if "done" in event and "total" in event:
                entry["done"] = int(event["done"])
                entry["total"] = int(event["total"])
        if "phase_done" in event:
            # Фаза закрыта — её надпись убирается, чужие надписи не трогаются
            status.pop(event["phase_done"], None)

        # В сокет — только состояние без строк лога: браузер получает лог
        # инкрементально через job_log, а не 0.5 МБ JSON на каждый тик.
        snapshot = {k: v for k, v in job.items() if k != "lines"}
        socketio.emit("job_progress", {"job_id": job_id, **snapshot})
        now = time.monotonic()
        if now - last_save["t"] >= 0.5:
            last_save["t"] = now
            save_job(job_id, job)

    try:
        init_db()
        job["state"] = "working"
        job["progress"] = 0
        job["phase_status"] = {}
        job["estimate"] = ""
        save_job(job_id, job)
        socketio.emit("job_status", {"job_id": job_id, "state": "working"})

        result = run_pipeline(
            [Path(p) for p in paths],
            model=model,
            language=language,
            diarize=diarize,
            extract=True,
            process=(mode == "full"),
            images=(mode == "full"),
            raw=(mode == "raw"),
            out_dir=OUTPUT_DIR,
            progress=progress,
        )
        job["state"] = "done"
        job["progress"] = 100
        job["phase_status"] = {}  # фазы закрыты — надписи не висят на завершённой задаче
        job["files"] = collect_job_files(result)
        if result.get("report"):
            job["report"] = str(result["report"])
        job["lines"].append(
            f"Готово. Транскриптов: {len(result['transcripts'])}, "
            f"извлечённых текстов: {len(result.get('raw_files') or [])}, "
            f"отчётов по картинкам: {len(result['image_reports'])}"
        )
        if len(job["lines"]) > MAX_JOB_LINES:
            del job["lines"][:-MAX_JOB_LINES]
        save_job(job_id, job)
        socketio.emit("job_done", {"job_id": job_id, **job})
        # Предсборка ZIP в фоне: к моменту клика по «Скачать всё архивом»
        # файл уже готов (сборка 11k+ файлов занимает ~10-60 с)
        threading.Thread(
            target=_prebuild_zip,
            args=(job_id, job.get("files") or []),
            daemon=True,
            name="omni-zip",
        ).start()
    except Exception as e:
        job["state"] = "error"
        job["error"] = str(e)
        job["phase_status"] = {}
        save_job(job_id, job)
        logger.exception(f"Задача {job_id} упала с ошибкой")
        socketio.emit("job_error", {"job_id": job_id, "error": str(e)})


@app.post("/api/start")
def start():
    files = []
    job_dir = UPLOADS_DIR / uuid.uuid4().hex
    uploaded = request.files.getlist("files")
    if uploaded:
        job_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            name = Path(f.filename).name
            dest = job_dir / name
            f.save(dest)
            files.append(dest)

    paths: list[str] = []
    if files:
        paths.append(str(job_dir))
    disk_path = request.form.get("path", "").strip().strip('"')
    if disk_path:
        paths.append(disk_path)
    if not paths:
        return jsonify(error="Не выбран ни один файл и путь не найден"), 400

    mode = request.form.get("mode", "full")
    if mode not in MODES:
        mode = "full"
    diarize = request.form.get("diarize", "1") == "1"
    model = request.form.get("model", settings.whisper_model)
    if model not in MODEL_SIZES:
        model = settings.whisper_model
    language = request.form.get("language", settings.language)
    if language not in LANGUAGES:
        language = settings.language

    job_id = uuid.uuid4().hex
    job_data = {
        "state": "queued", "progress": 0, "lines": [], "phase_status": {},
        "error": "", "report": None,
        "mode": mode, "model": model, "language": language, "diarize": diarize,
        "paths": paths,  # Save paths for rerun
    }
    save_job(job_id, job_data)

    threading.Thread(
        target=run_job, args=(job_id, paths, mode, diarize, model, language), daemon=True
    ).start()
    return jsonify(job_id=job_id)


@app.post("/api/rerun/<job_id>")
def rerun(job_id):
    """Повторный запуск существующей задачи: новая задача с теми же путями.

    Раньше «mode» в записи задачи хранил имя модели — любой повтор терял
    анализ (full превращался в extract). Теперь режим и модель сохраняются
    раздельно и переносятся в новую задачу.
    """
    old_job = get_job(job_id)
    if not old_job:
        return jsonify(error="Задача не найдена"), 404

    existing_paths = old_job.get("paths") or []
    paths = [p for p in existing_paths if Path(p).exists()]

    # Из записи задачи "mode" хранит РЕЖИМ ("mode" legacy-задач — имя модели)
    stored_mode = old_job.get("mode", "full")
    if stored_mode not in MODES:
        stored_mode = "full"
    language = old_job.get("language", settings.language)
    diarize = old_job.get("diarize", True)
    model = old_job.get("model") or (
        old_job.get("mode") if old_job.get("mode") in MODEL_SIZES else settings.whisper_model
    )
    if model not in MODEL_SIZES:
        model = settings.whisper_model

    if not paths:
        return jsonify(error="В исходной задаче нет путей для повторного запуска"), 400

    # Validate paths still exist
    job_id_new = uuid.uuid4().hex
    job_data = {
        "state": "queued", "progress": 0, "lines": [], "phase_status": {},
        "error": "", "report": None,
        "mode": stored_mode, "model": model, "language": language, "diarize": diarize,
        "paths": paths,
    }
    save_job(job_id_new, job_data)

    threading.Thread(
        target=run_job,
        args=(job_id_new, paths, stored_mode, diarize, model, language),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id_new)


@app.get("/api/status/<job_id>")
def job_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify(error="Задача не найдена"), 404
    return jsonify(job)


@app.get("/api/files/<job_id>")
def job_files(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify(error="Задача не найдена"), 404
    # Старые задачи (до записи списка файлов) — пустой список, а не всё подряд
    return jsonify(files=job.get("files") or [])


@app.get("/api/zip/<job_id>")
def zip_archive(job_id):
    """Все выходные файлы задачи одним ZIP-архивом (кэшируется на диске)."""
    job = get_job(job_id)
    if not job:
        return jsonify(error="Задача не найдена"), 404
    files = job.get("files") or []
    if not files:
        return "Файлы этой задачи не записаны", 404
    from src.outputs import get_or_build_zip

    zip_path = get_or_build_zip(files, ZIPS_DIR / f"{job_id}.zip")
    if zip_path is None:
        return "Файлы этой задачи не записаны", 404
    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"omniprosvet_{job_id[:8]}.zip",
    )


@app.get("/api/result/<job_id>")
def result(job_id):
    job = get_job(job_id)
    if not job or not job.get("report"):
        return "Not ready", 404
    return send_file(job["report"], as_attachment=True)


@app.get("/api/download")
def download():
    p = Path(request.args.get("path", ""))
    try:
        p.resolve().relative_to(BASE_DIR.resolve())
    except ValueError:
        return "Forbidden", 403
    if not p.exists() or not p.is_file():
        return "Not found", 404
    return send_file(p, as_attachment=True)


@app.get("/api/jobs")
def list_jobs():
    from src.database import list_jobs as db_list_jobs
    try:
        jobs = db_list_jobs(limit=50)
    except Exception as e:
        logger.warning(f"Ошибка получения списка задач: {e}")
        jobs = []
    return jsonify(jobs=jobs)


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id):
    from src.database import delete_job as db_delete_job
    db_delete_job(job_id)
    # Clean up upload directory
    upload_dir = UPLOADS_DIR / job_id
    if upload_dir.exists():
        import shutil
        shutil.rmtree(upload_dir, ignore_errors=True)
    # Кэш архива задачи — тоже в мусор (иначе копится вечно)
    for suffix in (".zip", ".json"):
        cached = ZIPS_DIR / f"{job_id}{suffix}"
        cached.unlink(missing_ok=True)
    return jsonify(ok=True)


@app.get("/")
def index():
    # Статический файл без Jinja-шаблонизации: правки UI делаются
    # прямо в static/index.html (раньше HTML+JS жили в Python-строке).
    resp = send_file(BASE_DIR / "static" / "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@socketio.on("join_job")
def on_join_job(data):
    pass  # Client joins room for this job


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)
