import io
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file
from flask_socketio import SocketIO, emit
from loguru import logger

from src.configuration import settings
from src.database import init_db, get_job, save_job, delete_old_jobs, mark_interrupted_jobs
from src.outputs import build_zip_bytes, collect_job_files
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

ACCEPT = (
    ".wav,.mp3,.m4a,.flac,.ogg,.opus,.wma,.aac,"
    ".mp4,.mkv,.avi,.mov,.webm,.m4v,.wmv,.flv,.ts,"
    ".jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff,"
    ".pdf,.docx,.doc,.txt,.rtf,.csv,.xlsx,.xls,.pptx,.ppt,.odt"
)

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
LANGUAGES = {"auto", "zh", "ru", "en", "ja", "ko", "de", "fr", "es"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Очистка старых задач и пометка прерванных (их потоки умерли с прошлым процессом)
delete_old_jobs(days=7)
mark_interrupted_jobs()


def run_job(job_id: str, paths: list[str], mode: str, diarize: bool, model: str, language: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    pp = PhaseProgress()

    def progress(event: dict) -> None:
        if "log" in event:
            job["lines"].append(event["log"])
            socketio.emit("job_log", {"job_id": job_id, "log": event["log"]})
            save_job(job_id, job)
            return
        pp.feed(event)
        pct = pp.percent()
        if pct > job.get("progress", 0):
            job["progress"] = pct
        if "file" in event:
            job["current"] = event["file"]
            if event.get("file_index"):
                job["file_index"] = event["file_index"]
            if event.get("file_total"):
                job["file_total"] = event["file_total"]
        socketio.emit("job_progress", {"job_id": job_id, **job})
        save_job(job_id, job)

    try:
        init_db()
        job["state"] = "working"
        job["progress"] = 0
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
            out_dir=OUTPUT_DIR,
            progress=progress,
        )
        job["state"] = "done"
        job["progress"] = 100
        job["files"] = collect_job_files(result)
        if result["report"]:
            job["report"] = str(result["report"])
        job["lines"].append(
            f"Готово. Транскриптов: {len(result['transcripts'])}, "
            f"отчётов по картинкам: {len(result['image_reports'])}"
        )
        save_job(job_id, job)
        socketio.emit("job_done", {"job_id": job_id, **job})
    except Exception as e:
        job["state"] = "error"
        job["error"] = str(e)
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
    if mode not in ("full", "extract"):
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
        "state": "queued", "progress": 0, "lines": [], "current": "",
        "file_index": 0, "file_total": 0, "error": "", "report": None,
        "mode": model, "language": language, "diarize": diarize,
        "paths": paths,  # Save paths for rerun
    }
    save_job(job_id, job_data)

    threading.Thread(
        target=run_job, args=(job_id, paths, mode, diarize, model, language), daemon=True
    ).start()
    return jsonify(job_id=job_id)


@app.post("/api/rerun/<job_id>")
def rerun(job_id):
    """Rerun a job with the same parameters."""
    old_job = get_job(job_id)
    if not old_job:
        return jsonify(error="Задача не найдена"), 404
    
    # Extract parameters from old job
    paths = old_job.get("paths", [])
    mode = old_job.get("mode", "full")
    language = old_job.get("language", settings.language)
    diarize = old_job.get("diarize", True)
    model = old_job.get("mode", settings.whisper_model)  # model is stored in "mode" field
    
    if not paths:
        return jsonify(error="В исходной задаче нет путей для повторного запуска"), 400
    
    # Validate paths still exist
    existing_paths = [p for p in paths if Path(p).exists()]
    if not existing_paths:
        return jsonify(error="Файлы из исходной задачи не найдены на диске"), 400
    
    # Create new job with same parameters
    new_job_id = uuid.uuid4().hex
    job_data = {
        "state": "queued", "progress": 0, "lines": [], "current": "",
        "file_index": 0, "file_total": 0, "error": "", "report": None,
        "mode": model, "language": language, "diarize": diarize,
        "paths": existing_paths,
    }
    save_job(new_job_id, job_data)
    
    threading.Thread(
        target=run_job, args=(new_job_id, existing_paths, mode, diarize, model, language), daemon=True
    ).start()
    return jsonify(job_id=new_job_id)


@app.get("/api/status/<job_id>")
def status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify(error="Задача не найдена"), 404
    return jsonify(**job)


@app.get("/api/files/<job_id>")
def files_list(job_id):
    """Только файлы, созданные этой задачей (не все файлы за всё время)."""
    job = get_job(job_id)
    if not job:
        return jsonify(error="Задача не найдена"), 404
    return jsonify(files=job.get("files") or [])


@app.get("/api/zip/<job_id>")
def zip_archive(job_id):
    """Все выходные файлы задачи одним ZIP-архивом."""
    job = get_job(job_id)
    if not job:
        return jsonify(error="Задача не найдена"), 404
    files = job.get("files") or []
    if not files:
        return "Файлы этой задачи не записаны", 404
    data = build_zip_bytes(files)
    return send_file(
        io.BytesIO(data),
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
    return jsonify(ok=True)


PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>OmniProsvet — экстрактор информации</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: "Segoe UI", sans-serif; background: #f0f2f5; margin: 0; padding: 30px; }
  .wrap { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 22px; color: #1a1a2e; }
  .card { background: #fff; border-radius: 12px; padding: 22px; margin-bottom: 16px;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .drop { border: 2px dashed #aab; border-radius: 10px; padding: 26px; text-align: center;
          color: #667; cursor: pointer; transition: .15s; }
  .drop.over { border-color: #c62828; background: #fff5f5; }
  input[type=file] { display: none; }
  label { display: block; font-size: 13px; color: #555; margin: 12px 0 4px; }
  input[type=text], select { width: 100%; padding: 9px 11px; border: 1px solid #ccd;
          border-radius: 8px; font-size: 14px; }
  button { background: #c62828; color: #fff; border: 0; border-radius: 8px;
          padding: 12px 26px; font-size: 15px; cursor: pointer; margin-top: 14px; }
  button:hover { background: #a91f1f; }
  button:disabled { background: #bbb; cursor: default; }
  #files { font-size: 13px; color: #444; margin-top: 8px; }
  #bar { height: 14px; background: #e5e7eb; border-radius: 7px; overflow: hidden; margin: 10px 0; }
  #fill { height: 100%; width: 0; background: linear-gradient(90deg,#e53935,#f77062); transition: width .5s; }
  #log { font-size: 13px; line-height: 1.7; background: #fafafa; border: 1px solid #eee;
          border-radius: 8px; padding: 12px; max-height: 340px; overflow-y: auto; white-space: pre-wrap; }
  #status { font-size: 14px; color: #555; }
  .srcbtn { background: #e5e7eb; color: #333; padding: 9px 16px; font-size: 14px; margin-top: 0; margin-right: 8px; }
  .srcbtn:hover { background: #d5d8dd; }
  .srcbtn.active { background: #c62828; color: #fff; }
  .srcbtn.active:hover { background: #a91f1f; }
  .err { color: #c62828; }
  .chk { font-size: 14px; color: #333; margin-top: 12px; display: flex; align-items: center; gap: 8px; }
  .hint { font-size: 12px; color: #777; margin-top: 8px; }
  a.dl { display: inline-block; margin-top: 8px; margin-right: 14px; color: #1565c0; font-size: 14px; }
  a.dl.zip { font-size: 16px; font-weight: 600; padding: 10px 18px; background: #e3f2fd;
             border-radius: 8px; text-decoration: none; margin-bottom: 10px; }
  a.dl.zip:hover { background: #bbdefb; }
  .job-item { padding: 10px; border-bottom: 1px solid #eee; cursor: pointer; }
  .job-item:hover { background: #f5f5f5; }
  .job-item:last-child { border-bottom: none; }
  .job-meta { font-size: 12px; color: #777; }
  .job-status { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .status-queued { background: #e3f2fd; color: #1565c0; }
  .status-working { background: #fff3e0; color: #e65100; }
  .status-done { background: #e8f5e9; color: #2e7d32; }
  .status-error { background: #ffebee; color: #c62828; }
  .status-interrupted { background: #eceff1; color: #546e7a; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🔎 OmniProsvet — экстрактор информации</h1>

  <div class="card">
    <div class="drop" id="drop">Перетащите аудио, видео, картинки или документы сюда<br>или нажмите, чтобы выбрать</div>
    <input type="file" id="file" multiple accept="__ACCEPT__">
    <div id="files"></div>
    <label>…или укажите путь к файлу / папке на диске</label>
    <input type="text" id="path" placeholder="Например: D:\\Downloads\\медиа">
    <label>Режим</label>
    <div>
      <button type="button" class="srcbtn active" id="btnFull">Полная обработка (транскрибация + анализ)</button>
      <button type="button" class="srcbtn" id="btnExtract">Только транскрибация</button>
    </div>
    <label>Модель whisper</label>
    <select id="model">
      <option value="tiny">tiny — очень быстро, грубо</option>
      <option value="base">base — быстро</option>
      <option value="small">small — баланс (рекомендуется)</option>
      <option value="medium">medium — точнее, медленнее</option>
      <option value="large-v3">large-v3 — максимум качества</option>
    </select>
    <label>Язык речи</label>
    <select id="language">
      <option value="auto">определять автоматически</option>
      <option value="ru">русский</option>
      <option value="en">английский</option>
      <option value="zh">китайский</option>
      <option value="ja">японский</option>
      <option value="ko">корейский</option>
      <option value="de">немецкий</option>
      <option value="fr">французский</option>
      <option value="es">испанский</option>
    </select>
    <div class="chk"><input type="checkbox" id="diarize" checked> Диаризация спикеров (polyvoice, модели качаются автоматически)</div>
    <br>
    <button id="go">Начать</button>
  </div>

  <div class="card">
    <label>История задач</label>
    <div id="jobList"></div>
  </div>

  <div class="card hidden" id="progress_card">
    <div id="status">Подготовка…</div>
    <div id="bar"><div id="fill"></div></div>
    <div id="log"></div>
    <div id="downloads"></div>
  </div>
</div>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<script>
const $ = id => document.getElementById(id);
let jobId = null, socket = null;
let mode = "full";

const drop = $("drop"), fileInput = $("file");
drop.onclick = () => fileInput.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over");
  fileInput.files = e.dataTransfer.files; showFiles(); };
fileInput.onchange = showFiles;
function showFiles() {
  $("files").textContent = [...fileInput.files].map(f => f.name).join(", ");
}

$("btnFull").onclick = () => setMode("full");
$("btnExtract").onclick = () => setMode("extract");
function setMode(m) {
  mode = m;
  $("btnFull").classList.toggle("active", m === "full");
  $("btnExtract").classList.toggle("active", m === "extract");
}

$("go").onclick = async () => {
  const fd = new FormData();
  for (const f of fileInput.files) fd.append("files", f);
  fd.append("path", $("path").value);
  fd.append("mode", mode);
  fd.append("model", $("model").value);
  fd.append("language", $("language").value);
  fd.append("diarize", $("diarize").checked ? "1" : "0");
  const r = await fetch("/api/start", { method: "POST", body: fd });
  const data = await r.json();
  if (!r.ok) { alert(data.error); return; }
  jobId = data.job_id;
  $("progress_card").classList.remove("hidden");
  $("downloads").innerHTML = "";
  const rw = $("resume_wrap");
  if (rw) rw.remove();
  $("go").disabled = true;
  connectSocket();
};

function connectSocket() {
  if (socket) { socket.disconnect(); }
  socket = io();
  socket.on("connect", () => {
    console.log("WebSocket connected");
    socket.emit("join_job", {job_id: jobId});
  });
  socket.on("job_progress", (data) => {
    if (data.job_id !== jobId) return;
    $("fill").style.width = (data.progress || 0) + "%";
    if (data.state === "queued") $("status").textContent = "В очереди…";
    if (data.state === "working")
      $("status").textContent = (data.file_total > 1 ? "Файл " + data.file_index + " из " + data.file_total + ": " : "") + (data.current || "работаю…");
    if (data.error) $("status").innerHTML = '<span class="err">Ошибка: ' + data.error + "</span>";
    if (data.lines && data.lines.length) { $("log").textContent = data.lines.join("\\n"); $("log").scrollTop = 1e9; }
  });
  socket.on("job_log", (data) => {
    if (data.job_id !== jobId) return;
    if (data.log) { $("log").textContent += "\\n" + data.log; $("log").scrollTop = 1e9; }
  });
  socket.on("job_done", (data) => {
    if (data.job_id !== jobId) return;
    $("go").disabled = false;
    $("status").textContent = "Готово ✔";
    loadFiles();
  });
  socket.on("job_error", (data) => {
    if (data.job_id !== jobId) return;
    $("go").disabled = false;
    $("status").innerHTML = '<span class="err">Ошибка: ' + data.error + "</span>";
  });
  socket.on("disconnect", () => {
    console.log("WebSocket disconnected");
  });
}

async function loadFiles() {
  const r = await fetch("/api/files/" + jobId);
  if (!r.ok) return;
  const j = await r.json();
  const files = j.files || [];
  if (!files.length) {
    $("downloads").innerHTML = '<div class="job-meta">Файлы этой задачи не записаны</div>';
    return;
  }
  const zipBtn = '<a class="dl zip" href="/api/zip/' + jobId + '">&#128230; Скачать всё одним архивом (ZIP)</a>';
  const parts = files.map(f =>
    '<a class="dl" href="/api/download?path=' + encodeURIComponent(f.path) + '">&#11015; ' + f.name + "</a>");
  $("downloads").innerHTML = zipBtn + "<br>" + parts.join("<br>");
}

async function loadJobList() {
  const r = await fetch("/api/jobs");
  if (!r.ok) return;
  const j = await r.json();
  const html = j.jobs.map(job => `
    <div class="job-item">
      <div onclick="showJob('${job.id}')" style="cursor:pointer">
        <strong>${job.id.slice(0,8)}</strong> <span class="job-status status-${job.state}">${job.state}</span>
      </div>
      <div class="job-meta">${job.created_at} • ${job.mode || 'N/A'} • ${job.progress || 0}%</div>
      ${job.state === 'interrupted' ? `
        <button class="srcbtn" onclick="event.stopPropagation(); rerunJob('${job.id}', this, '▶ Продолжить')">▶ Продолжить</button>
      ` : (job.state === 'done' || job.state === 'error') ? `
        <button class="srcbtn" onclick="event.stopPropagation(); rerunJob('${job.id}', this, '🔁 Повторить')">🔁 Повторить</button>
      ` : ''}
    </div>
  `).join("");
  $("jobList").innerHTML = html || '<div class="job-meta">Нет задач</div>';
}

function showJob(id) {
  jobId = id;
  $("progress_card").classList.remove("hidden");
  $("downloads").innerHTML = "";
  const rw = $("resume_wrap");
  if (rw) rw.remove();
  $("go").disabled = true;
  connectSocket();
  // Load existing progress and restore UI state
  fetch("/api/status/" + id).then(r => r.json()).then(job => {
    if (!job) { alert("Задача не найдена"); return; }
    
    // Restore progress bar
    if (job.progress !== undefined) $("fill").style.width = job.progress + "%";
    
    // Restore status text based on state
    const statusEl = $("status");
    if (job.state === "queued") {
      statusEl.textContent = "В очереди…";
    } else if (job.state === "working") {
      statusEl.textContent = (job.file_total > 1 ? "Файл " + job.file_index + " из " + job.file_total + ": " : "") + (job.current || "работаю…");
    } else if (job.state === "done") {
      statusEl.textContent = "Готово ✔";
    } else if (job.state === "error") {
      statusEl.innerHTML = '<span class="err">Ошибка: ' + (job.error || "неизвестная") + "</span>";
    } else if (job.state === "interrupted") {
      statusEl.innerHTML = '<span class="err">Прервана (сервер перезапускался) — нажмите «Продолжить»</span>';
    }
    
    // Restore log
    if (job.lines && job.lines.length) { 
      $("log").textContent = job.lines.join("\\n"); 
      $("log").scrollTop = 1e9; 
    }
    
    // Load files if done
    if (job.state === "done") loadFiles();

    // Resume button inside progress card for interrupted jobs
    const resumeWrap = $("resume_wrap");
    if (resumeWrap) resumeWrap.remove();
    if (job.state === "interrupted") {
      const btn = document.createElement("button");
      btn.id = "resume_wrap";
      btn.className = "srcbtn";
      btn.textContent = "▶ Продолжить задачу";
      btn.onclick = () => rerunJob(id, btn, "▶ Продолжить задачу");
      statusEl.after(btn);
    }
  });
}

async function rerunJob(id, btn, label) {
  label = label || "🔁 Повторить";
  btn.disabled = true;
  btn.textContent = "⏳...";
  try {
    const r = await fetch("/api/rerun/" + id, { method: "POST" });
    const data = await r.json();
    if (!r.ok) { alert(data.error); btn.disabled = false; btn.textContent = label; return; }
    // Switch to new job
    showJob(data.job_id);
  } catch (e) {
    alert("Ошибка: " + e.message);
    btn.disabled = false;
    btn.textContent = label;
  }
}

loadJobList();
setInterval(loadJobList, 30000);
</script>
</body>
</html>""".replace("__ACCEPT__", ACCEPT)


@app.get("/")
def index():
    resp = render_template_string(PAGE)
    resp = app.response_class(resp)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@socketio.on("join_job")
def on_join_job(data):
    pass  # Client joins room for this job


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)