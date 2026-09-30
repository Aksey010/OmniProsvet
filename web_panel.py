import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file
from loguru import logger

from src.collect import TRANSCRIPTS_DIR
from src.configuration import settings
from src.database import init_db
from src.processor import OUTPUT_DIR, run_pipeline

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

logger.add(BASE_DIR / "omniprosvet.log", rotation="1 week", encoding="utf-8")

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

jobs = {}


class _PhaseProgress:
    def __init__(self) -> None:
        self.current: str | None = None
        self.completed = 0.0
        self.weights: dict = {}
        self.fracs: dict = {}

    def feed(self, event: dict) -> None:
        if "phase" in event:
            name = event["phase"]
            if name != self.current:
                if self.current is not None:
                    self.completed += self.weights.get(self.current, 0)
                self.current = name
            self.weights[name] = event.get("weight", 0)
        elif "phase_progress" in event:
            self.fracs[event.get("p")] = max(0.0, min(1.0, event["phase_progress"]))

    def percent(self) -> int:
        frac = self.fracs.get(self.current, 0.0)
        weight = self.weights.get(self.current, 0)
        return min(100, round(self.completed + frac * weight))


def run_job(job_id: str, paths: list[str], mode: str, diarize: bool, model: str, language: str, platform: str) -> None:
    job = jobs[job_id]
    pp = _PhaseProgress()

    def progress(event: dict) -> None:
        if "log" in event:
            job["lines"].append(event["log"])
            return
        pp.feed(event)
        job["progress"] = pp.percent()
        if "file" in event:
            job["current"] = event["file"]
            if event.get("file_index"):
                job["file_index"] = event["file_index"]
            if event.get("file_total"):
                job["file_total"] = event["file_total"]

    try:
        init_db()
        job["state"] = "working"
        result = run_pipeline(
            paths,
            model=model,
            language=language,
            diarize=diarize,
            extract=True,
            process=(mode == "full"),
            images=(mode == "full"),
            out_dir=OUTPUT_DIR,
            progress=progress,
            platform=platform,
        )
        job["state"] = "done"
        job["progress"] = 100
        if result["report"]:
            job["report"] = str(result["report"])
        job["lines"].append(
            f"Готово. Транскриптов: {len(result['transcripts'])}, "
            f"отчётов по картинкам: {len(result['image_reports'])}"
        )
    except Exception as e:
        job["state"] = "error"
        job["error"] = str(e)
        logger.exception(f"Задача {job_id} упала с ошибкой")


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
    platform = request.form.get("platform", settings.analysis_platform)
    if platform not in ("deepseek", "nim"):
        platform = settings.analysis_platform
    diarize = request.form.get("diarize", "1") == "1"
    model = request.form.get("model", settings.whisper_model)
    if model not in MODEL_SIZES:
        model = settings.whisper_model
    language = request.form.get("language", settings.language)
    if language not in LANGUAGES:
        language = settings.language

    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "state": "queued", "progress": 0, "lines": [], "current": "",
        "file_index": 0, "file_total": 0, "error": "", "report": None,
    }
    threading.Thread(
        target=run_job, args=(job_id, paths, mode, diarize, model, language, platform), daemon=True
    ).start()
    return jsonify(job_id=job_id)


@app.get("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="Задача не найдена"), 404
    return jsonify(**job)


@app.get("/api/files/<job_id>")
def files_list(job_id):
    out = []
    for d in (UPLOADS_DIR / job_id, OUTPUT_DIR, TRANSCRIPTS_DIR, RESULTS_DIR):
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.txt")):
            out.append({"name": f.name, "path": str(f), "dir": str(f.parent.relative_to(BASE_DIR))})
    return jsonify(files=out)


@app.get("/api/result/<job_id>")
def result(job_id):
    job = jobs.get(job_id)
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
    <label>Платформа анализа</label>
    <div>
      <button type="button" class="srcbtn active" id="btnDeepseek">DeepSeek (браузер)</button>
      <button type="button" class="srcbtn" id="btnNim">NVIDIA NIM API</button>
    </div>
    <div class="hint" id="platformHint">Анализ через веб-чат DeepSeek: при запуске откроется окно браузера для входа</div>
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
    <button type="button" class="srcbtn" id="btnLogout">Сменить аккаунт DeepSeek</button>
  </div>

  <div class="card hidden" id="progress_card">
    <div id="status">Подготовка…</div>
    <div id="bar"><div id="fill"></div></div>
    <div id="log"></div>
    <div id="downloads"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
let jobId = null, timer = null, mode = "full", platform = "__PLATFORM__";

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

$("btnDeepseek").onclick = () => setPlatform("deepseek");
$("btnNim").onclick = () => setPlatform("nim");
function setPlatform(p) {
  platform = p;
  $("btnDeepseek").classList.toggle("active", p === "deepseek");
  $("btnNim").classList.toggle("active", p === "nim");
  const hint = $("platformHint");
  if (p === "nim") {
    hint.textContent = "Анализ через NVIDIA NIM API (ключ NIM_API_KEY в .env): браузер не нужен, параллельные запросы";
    $("btnLogout").style.display = "none";
  } else {
    hint.textContent = "Анализ через веб-чат DeepSeek: при запуске откроется окно браузера для входа";
    $("btnLogout").style.display = "";
  }
}
setPlatform(platform);

$("btnLogout").onclick = async () => {
  if (!confirm("Сбросить профиль браузера и выйти из аккаунта DeepSeek?")) return;
  await fetch("/api/logout", { method: "POST" });
  alert("Профиль сброшен — откроется окно входа DeepSeek");
};
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
  fd.append("platform", platform);
  fd.append("model", $("model").value);
  fd.append("language", $("language").value);
  fd.append("diarize", $("diarize").checked ? "1" : "0");
  const r = await fetch("/api/start", { method: "POST", body: fd });
  const data = await r.json();
  if (!r.ok) { alert(data.error); return; }
  jobId = data.job_id;
  $("progress_card").classList.remove("hidden");
  $("downloads").innerHTML = "";
  $("go").disabled = true;
  timer = setInterval(poll, 1000);
};

async function poll() {
  const r = await fetch("/api/status/" + jobId);
  if (!r.ok) {
    clearInterval(timer); timer = null;
    $("go").disabled = false;
    $("status").innerHTML = '<span class="err">Задача не найдена — перезагрузите страницу</span>';
    return;
  }
  const j = await r.json();
  $("fill").style.width = (j.progress || 0) + "%";
  if (j.state === "queued") $("status").textContent = "В очереди…";
  if (j.state === "working")
    $("status").textContent = (j.file_total > 1 ? "Файл " + j.file_index + " из " + j.file_total + ": " : "") + (j.current || "работаю…");
  if (j.error) $("status").innerHTML = '<span class="err">Ошибка: ' + j.error + "</span>";
  if (j.lines && j.lines.length) { $("log").textContent = j.lines.join("\\n"); $("log").scrollTop = 1e9; }
  if (j.state === "done" || j.state === "error") {
    clearInterval(timer);
    $("go").disabled = false;
    if (j.state === "done") $("status").textContent = "Готово ✔";
    loadFiles();
  }
}

async function loadFiles() {
  const r = await fetch("/api/files/" + jobId);
  if (!r.ok) return;
  const j = await r.json();
  const parts = j.files.map(f =>
    '<a class="dl" href="/api/download?path=' + encodeURIComponent(f.path) + '">⬇ ' + f.name + "</a>");
  if (parts.length) $("downloads").innerHTML = parts.join("<br>");
}
</script>
</body>
</html>""".replace("__ACCEPT__", ACCEPT).replace("__PLATFORM__", settings.analysis_platform)


@app.get("/")
def index():
    resp = render_template_string(PAGE)
    resp = app.response_class(resp)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/logout")
def logout():
    from src.deepseek_agent import agent

    agent.reset_profile()
    threading.Thread(target=_warmup_deepseek, daemon=True, name="deepseek-warmup").start()
    return jsonify(ok=True)


def _warmup_deepseek() -> None:
    if settings.analysis_platform == "nim":
        logger.info("Платформа анализа: NVIDIA NIM API — браузер для DeepSeek не запускается")
        return
    try:
        from src.deepseek_agent import agent

        with agent.lock:
            driver = agent._ensure_driver()
            agent._ensure_logged_in(driver, settings.deepseek_login, settings.deepseek_password)
        logger.info("DeepSeek: браузер открыт, вход выполнен — можно запускать обработку")
    except Exception as exc:
        logger.warning(f"Прогрев DeepSeek не удался: {exc} — вход будет запрошен при обработке")


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    threading.Thread(target=_warmup_deepseek, daemon=True, name="deepseek-warmup").start()
    app.run(host="127.0.0.1", port=5000, debug=False)
