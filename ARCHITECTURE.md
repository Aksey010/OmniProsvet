# OmniProsvet — Архитектура и принципы работы

## Обзор

OmniProsvet — это пайплайн для извлечения информации из аудио, видео, изображений и документов.
Использует **faster-whisper** для транскрибации и **NVIDIA NIM API** для LLM-анализа.

**Принцип работы с NIM**: бесплатный ключ build.nvidia.com ограничен **только числом запросов
в минуту (RPM)**. Стоимость запросов и расход токенов не учитывается — архитектура оптимизирует
параллелизм (запросы в секунду) и качество/читаемость выходных файлов, а не «экономию» на промптах.

**Формат выхода**: все итоговые файлы — обычный текст (.txt) **без markdown**: разделы — строки
заглавными буквами, темы — связной прозой, события и факты — перечислениями через дефис. Подстраховка —
санитайзер `src/outputs.py::plain_text()` применяется к каждому записываемому LLM-ответу.

---

## Структура исходников

| Файл | Ответственность |
|------|-----------------|
| `src/collect.py` | Сбор и классификация файлов |
| `src/whisper_extract.py` | Транскрибация faster-whisper + диаризация polyvoice |
| `src/transcription.py` | Параллельная транскрибация файлов, связка транскрибация → анализ |
| `src/utils.py` | Чтение транскриптов, чанкинг, хеш фрагмента |
| `src/analysis.py` | Map/Reduce анализ транскриптов (two-pass планирование) |
| `src/attachments.py` | Картинки, PDF (текст + Vision-fallback), документы |
| `src/nim_client.py` | Синхронный/асинхронный NIM-клиент, rate limit, retry |
| `src/prompts.py` | Загрузка промптов из `prompt.md` (fallback — `DEFAULT_PROMPT_MD`) |
| `src/outputs.py` | Сбор файлов задачи, ZIP, plain-text санитайзер |
| `src/progress.py` | Учёт прогресса параллельных фаз (PhaseProgress, FilesProgress) |
| `src/pipeline.py` | Оркестрация: веса фаз, параллельные вложения |
| `src/database.py` | SQLite-кэш фрагментов + записи задач |
| `web_panel.py` | Flask + SocketIO веб-панель |

---

## Этап 1: Сбор файлов (`src/collect.py`)

### `collect_media(path) -> (audio, video, images, documents)`
1. Рекурсивно обходит путь (файл или директория)
2. Пропускает служебные папки: `venv`, `.git`, `__pycache__`, `node_modules` и др.
3. Классифицирует по расширениям:
   - **Аудио**: wav, mp3, m4a, flac, ogg, opus, wma, aac
   - **Видео**: mp4, mkv, avi, mov, webm, m4v, wmv, flv, ts
   - **Изображения**: jpg, jpeg, png, webp, bmp, gif, tif, tiff
   - **Документы**: pdf, docx, doc, txt, rtf, csv, xlsx, xls, pptx, ppt, odt
4. Исключает уже обработанные файлы: `*.transcript.txt`, `report_*.txt`, `*.images.txt`, `*.documents.txt`

### `transcript_path_for(media, base)`
- Создаёт детерминированный путь для транскрипта на основе хеша абсолютного пути медиафайла
- Формат: `{base}/transcripts/{hash[:8]}/{stem}.transcript.txt`
- Позволяет находить существующие транскрипты без повторной обработки

---

## Этап 2: Транскрибация (`src/whisper_extract.py` + `src/transcription.py`)

### Параллельная транскрибация (`run_extract`, `run_extract_and_process`)

```
media_files (N файлов)
    │
    ├─► ThreadPoolExecutor(max_workers=min(4, N))
    │       ├─► Worker 1: _transcribe_single(file1)
    │       ├─► Worker 2: _transcribe_single(file2)
    │       ├─► Worker 3: _transcribe_single(file3)
    │       └─► Worker 4: _transcribe_single(file4)
    └─► Собирает результаты по мере завершения (as_completed)
```

### Прогресс: `FilesProgress` (`src/progress.py`)
Каждый воркер рапортует процент (0–100) своего файла в общий потокобезопасный аккумулятор.
Доля фазы = **среднее по всем файлам**, поэтому параллельные файлы, завершающиеся в произвольном
порядке, не дают «пилы» (старая индексная формула `(idx-1+pct)/total` двигала бар назад).

### `transcribe_file()` — детально

```python
def transcribe_file(media, model, language, diarize, on_progress, group_dir):
    1. Проверка кэша: find_existing_transcript() — если есть, возвращает сразу

    2. Загрузка модели Whisper (лениво, синглтон):
       - get_transcriber(size, fast) → WhisperModel или BatchedInferencePipeline
       - fast=True: BatchedInferencePipeline + beam_size=1 (ускорение ~3x)

    3. Декодирование аудио:
       - decode_audio(media, sampling_rate=16000) → float32 samples

    4. Параллельная диаризация (отдельный поток):
       - _diarize_samples(samples) → список сегментов с спикерами
       - Для аудио > 1 часа: разбивка на чанки с перекрытием 30 сек
       - polyvoice (balanced/mobile) — модели скачиваются автоматически

    5. Транскрибация:
       - transcriber.transcribe(samples, vad_filter=True, ...)
       - beam_size=1 (fast) или 5 (качественно)
       - batch_size из WHISPER_BATCH_SIZE (fast mode)
       - on_progress(pct) callback для UI (0–100, монотонно по файлу)

    6. Назначение спикеров сегментам:
       - _assign_speakers(segments, diarization_turns)
       - Пересечение по времени → лучший спикер

    7. Форматирование выхода:
       [timestamp] Спикер N: текст
       + заголовок (файл, дата, длительность, спикеры, кол-во сегментов)
       + футер

    8. Сохранение в {group_dir}/transcripts/{hash}/{stem}.transcript.txt
```

### Ключевые оптимизации
- **BatchedInferencePipeline** + `WHISPER_FAST=1` + `WHISPER_BATCH_SIZE` → ускорение в 3-5 раз
- **Параллельные воркеры** (до 4) → утилизация всех CPU ядер
- **Кэширование** по хешу файла → пропуск уже обработанного

---

## Этап 3: Планирование чанков — two-pass (`src/analysis.py::_plan_fragments`)

Перед отправкой первого NIM-запроса выполняется дешёвый проход по всем транскриптам:

1. `_prepare_transcript` (utils.py): читает строки, начинающиеся с `[` (таймстампы),
   обрезает каждую до 500 символов, префиксирует `[{stem}]`, режет на чанки
   (бюджет `nim_chunk_char_budget` = 8000 симв., `chunk_max_lines` = 300 строк).
2. Транскрипт из одного **малого** чанка (≤ половины бюджета и строк) копится в общий
   батч «мелкие файлы» — как в прежнем `_SmallBuffer`, но **до** старта запросов:
   сброс батча при превышении `chunk_char_budget` (65k) / `chunk_max_lines`.
3. Итог: упорядоченный список единиц `(источник, чанки)` и **суммарное число фрагментов
   известно заранее** — знаменатель прогресса фиксирован до первого «тика».

---

## Этап 4: Map/Reduce анализ через NIM (`src/analysis.py`)

```
Транскрипты (M файлов)
    │
    ├─► _plan_fragments: two-pass, мелкие склеены в батчи
    │
    ▼
Фрагменты (K суммарно), tracker.add_total(K) — один раз
    │
    ▼ MAP параллельно (ThreadPoolExecutor, max_workers=NIM_WORKERS)
    │   _process_fragment(idx, total, chunk, source)
    │       ├─► Кэш (SHA256(source + chunk)) → ответ из кэша
    │       ├─► NIM: build_map_prompt(idx, total, source, body)
    │       ├─► plain_text() санитайзер
    │       └─► Сохранение в кэш
    │
    ▼ REDUCE (один запрос)
    _reduce_and_save()
    │   ├─► parts_block = "### Фрагмент 1\n{конспект}..." (для промпта)
    │   ├─► NIM: build_reduce_prompt(files, chunks, parts)
    │   ├─► plain_text() санитайзер
    │   └─► Сохранение: output/report_{timestamp}.txt
```

- `_FragProgress` — тикает по завершении фрагмента; `frac = done/total × 0.9` (REDUCE_FRACTION).
- Фаза `analysis` закрывается явным событием `{"phase_done": "analysis"}` — на всех путях выхода.

---

## Этап 5: Обработка вложений (`src/attachments.py`)

### Параллельная обработка по типам (`src/pipeline.py::_attachments_worker`)

```
att_plan = [("images", [...], gdir), ("pdf", [...], gdir), ("documents", [...], gdir)]

ThreadPoolExecutor(max_workers=3):
    ├─► run_attachments(images, ...)     # Vision NIM
    ├─► run_pdfs(pdfs, ...)              # Текст + LLM / Vision fallback
    └─► run_attachments(documents, ...)  # Текст + LLM
```

Воркер анонсирует фазы с весами (`{"phase": kind, "weight": w}`) и закрывает их
по завершении (`{"phase_done": kind}`) — даже если тип упал с ошибкой.

### `run_attachments` (images/documents)
1. Батчинг: `min(images_per_batch, NIM_IMAGES_PER_BATCH)` файлов на запрос
2. Для каждого батча:
   - Хеш от имён+размеров → проверка кэша
   - `build_images_prompt(count)` / `build_documents_prompt(count)`
   - `nim_client.send_prompt(prompt, attachments=[paths...])` → авто-выбор Vision-модели
   - `plain_text()` санитайзер (и при чтении старого кэша)
   - Ошибка NIM при батче > 1 → рекурсивное деление батча пополам
   - Сохранение: `{stem}_{hash}.{kind}.txt` в групповой каталог

### `run_pdfs` — пер-файл, параллельно (max_workers=min(4, NIM_WORKERS))
```
Для каждого PDF:
    1. Извлечение текста через pypdf (по страницам)
    2. Есть текст → чанкинг (nim_chunk_char_budget=8000) → MAP (как этап 4)
       → конспект PDF → {stem}_{hash}.pdf.txt
    3. Нет текстового слоя (скан) → Vision fallback:
       - pdf2image (dpi 200) → PNG-страницы в tmp_pdfs/
       - батчи по nim_images_per_batch → Vision NIM (кэш по хешу страниц)
       - {stem}_scan_{hash}.txt
```
Склейка PDF (`_pdf_groups`/`_merge_pdf_group`) удалена из кода как мёртвая —
обработка строго пер-файл.

---

## Этап 6: NIM Client (`src/nim_client.py`)

### Синхронный `send_prompt()`
- Rate limiting: `_pace()` — min_interval = 60/RPM (единственный реальный лимит NIM)
- Retry logic: до 5 попыток, экспоненциальная задержка
- Обработка 429/5xx с `retry-after` заголовком
- Любой запрос с `attachments` автоматически идёт Vision-моделью (`NIM_VISION_MODEL`)

### Асинхронный `send_prompt_async()` + Connection Pooling
```python
limits = httpx.Limits(
    max_connections=NIM_WORKERS,      # 16
    max_keepalive_connections=NIM_WORKERS,
    keepalive_expiry=30.0,
)
```
- **Keep-alive** соединения → нет overhead на TLS handshake
- **Пул соединений** = NIM_WORKERS → полная утилизация rate limit

### Подготовка вложений (`_build_messages`)
- Изображения → base64 data URL → `image_url` в сообщении
- Документы (PDF, DOCX, XLSX, TXT, CSV, RTF) → извлечение текста → вставка в промпт
- Макс. 400k символов из документов на запрос

---

## Этап 7: Прогресс-бар (`src/progress.py` + `web_panel.py`)

### `PhaseProgress` — проценты по всем фазам сразу

Фазы идут **параллельно** (транскрибация + вложения + затем анализ), поэтому процент —
это не «текущая фаза», а сумма вкладов всех фаз:

```
percent() = Σ (вес завершённых фаз) + Σ (frac × вес) по активным фазам
```

События `feed()`:
- `{"phase": name, "weight": w}` — анонс фазы (идемпотентен, вес регистрируется один раз)
- `{"p": name, "phase_progress": 0..1}` — продвижение; **монотонно** (откаты игнорируются)
- `{"phase_done": name}` — фаза завершена: полный вес засчитывается один раз

Это устраняет «пинг-понг» и двойной подсчёт прошлой реализации.

### `FilesProgress` — аккумулятор параллельных файлов
Доля фазы = среднее процентов файлов; репортеры выдаются по индексу (`reporter(i, emit)`).

### Веса фаз (`src/pipeline.py::_weights_for`)
| Режим | Веса |
|-------|------|
| транскрибация + анализ + вложения | transcribe 45, analysis 30, вложения делят остаток 25 |
| транскрибация + анализ (без вложений) | transcribe 60, analysis 40 |
| только транскрибация | transcribe 100 (или 80 + вложения) |
| только анализ | analysis 100 (или 75 + вложения) |

### Монотонность в UI
`web_panel.run_job` дополнительно зажимает: `job["progress"] = max(старое, новое)` — бар не двигается назад.

---

## Этап 8: Веб-панель (`web_panel.py`)

### Архитектура
```
Flask + SocketIO (WebSocket)
    ├─► GET /                    → HTML страница
    ├─► POST /api/start          → Создание job, сохранение в БД, запуск потока
    ├─► POST /api/rerun/<id>     → Повторный запуск с теми же параметрами
    ├─► WebSocket /socket.io     → job_progress / job_log / job_done / job_error
    ├─► GET /api/status/<id>     → Статус job (REST fallback)
    ├─► GET /api/files/<id>      → Файлы ТОЛЬКО этой задачи (из job-записи)
    ├─► GET /api/zip/<id>        → Все файлы задачи одним ZIP-архивом
    ├─► GET /api/result/<id>     → Скачивание единого отчёта
    ├─► GET /api/download        → Скачивание любого файла (проверка path traversal)
    ├─► GET /api/jobs            → История задач (50 шт.)
    └─► DELETE /api/jobs/<id>    → Удаление job + upload папки
```

### Файлы задачи и ZIP
- После успеха `run_pipeline` результат (`report`, `transcripts`, `image_reports`)
  сохраняется в job-запись: `job["files"]` — дедуплицированный список `{name, path}`
  (`src/outputs.py::collect_job_files`).
- `/api/files/<id>` отдаёт **только** этот список — глобальные папки не сканируются.
  Задачи, созданные до этого механизма, честно показывают «файлы не записаны».
- `/api/zip/<id>` собирает архив на лету (`build_zip_bytes`, ZIP_DEFLATED, в памяти):

```
omniprosvet_{job_id[:8]}.zip
├── report_{timestamp}.txt     ← единый отчёт (корень)
├── transcripts/               ← полные транскрипты
└── summaries/                 ← конспекты PDF/сканов/картинок/документов
```

  Коллизии имён решаются суффиксом (`x.txt`, `x_2.txt`).

### Хранение задач (SQLite)
- Таблица `jobs`: `id`, `data(JSON)`, `created_at`, `updated_at`
- `save_job()` / `get_job()` / `list_jobs()` / `delete_job()` / `delete_old_jobs(days=7)`
- WAL режим для конкурентного доступа

---

## Этап 9: Промпты и формат выхода (`prompt.md` + `src/prompts.py`)

- Промпты живут в редактируемом файле `prompt.md`, секции разделены маркерами-комментариями
  (`ФРАГМЕНТ`, `ИТОГОВЫЙ ОТЧЁТ`, `КАРТИНКИ`, `ДОКУМЕНТЫ`); при отсутствии файла
  создаётся из `DEFAULT_PROMPT_MD`. Список файлов промптов — fallback на встроенные секции.
- **Формат вывода — строгий plain text** (выход читается в txt-ридере):
  - Никакого markdown: без решёток, звёздочек, тегов, жирного/курсива
  - Раздел — строка-заголовок заглавными буквами
  - Темы — монолитной прозой; события и факты — перечислениями через дефис
  - Структура единого отчёта: О ЧЁМ МАТЕРИАЛЫ → КЛЮЧЕВЫЕ СОБЫТИЯ И ОБЪЯВЛЕНИЯ →
    ТЕМЫ И ОБСУЖДЕНИЯ → ИЗВЛЕЧЁННЫЕ ФАКТЫ: ДАТЫ, ИМЕНА, СУММЫ, ССЫЛКИ
- Подстраховка — `plain_text()` (`src/outputs.py`): убирает `*`, `` ` ``, `#`-заголовки,
  `---`-линии, html-теги; `[текст](url)` → «текст (url)»; подчёркивания в именах файлов
  сохраняются. Применяется к каждому LLM-ответу **и к старым записям кэша** при чтении.

---

## Этап 10: Кэширование (`src/database.py`)

### Таблица `fragments`
```sql
CREATE TABLE fragments (
    hash TEXT PRIMARY KEY,    -- SHA256(source + chunk) или хеш файлов
    source TEXT,              -- Имя источника для дебага
    text TEXT,                -- Результат LLM
    created_at INTEGER        -- Unix timestamp
);
```
- Используется для: MAP фрагменты, REDUCE, изображения, PDF, документы
- Ключ: детерминированный хеш от контента → гарантированная идемпотентность

### Таблица `jobs`
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,      -- UUID hex
    data TEXT,                -- JSON с полным состоянием (+ files)
    created_at INTEGER,
    updated_at INTEGER
);
```
- Полное состояние job для восстановления после рестарта (`mark_interrupted_jobs`)
- TTL 7 дней (`delete_old_jobs`)

---

## Конфигурация (`.env`, дефолты в `src/configuration.py`)

```env
# Whisper — скорость
WHISPER_MODEL=small               # tiny/base/small/medium/large-v3
WHISPER_FAST=1                    # BatchedInferencePipeline
WHISPER_BATCH_SIZE=8              # Размер батча для fast mode
DIARIZATION=1                     # Спикеры (polyvoice); 0 = отключить для ускорения

# NIM — ограничение только RPM; стоимость запросов не учитывается
NIM_RPM=120                       # Requests per minute (квота build.nvidia.com)
NIM_WORKERS=16                    # Параллельные воркеры/соединения
NIM_MAX_TOKENS=8192               # Меньше токенов = быстрее ответ
NIM_CHUNK_CHAR_BUDGET=8000        # Меньшие чанки = больше параллелизма
NIM_IMAGES_PER_BATCH=3            # Меньше картинок на запрос = быстрее

# Чанкинг транскриптов
CHUNK_CHAR_BUDGET=65000           # Бюджет склейки мелких источников
CHUNK_MAX_LINES=300               # Максимум строк в чанке
```

---

## Поток данных (Data Flow)

```
ВХОД: /path/to/media/
    │
    ▼
collect_media() → (audio[], video[], images[], documents[])
    │
    ├─► audio+video → run_extract_and_process()
    │       ├─ ThreadPool(4) → transcribe_file() → transcripts[]
    │       │    (FilesProgress: средний % по файлам)
    │       └─ _plan_fragments → MAP NIM (workers=16) → REDUCE → report_*.txt
    │
    ├─► images[]  → run_attachments() → батчи по 3 → Vision NIM → *.images.txt
    ├─► pdfs[]    → run_pdfs() → пер-файл → pypdf → MAP NIM (или Vision fallback) → *.pdf.txt
    └─► documents[] → run_attachments() → батчи → NIM → *.documents.txt
    │                (все типы — параллельными потоками, фазы анонсируются и закрываются)
    ▼
run_pipeline → {"transcripts", "report", "image_reports"}
    │
    ▼
job["files"] = collect_job_files(result)   → UI: список файлов + кнопка ZIP
    │
    ▼
ВЫХОД:
    output/{group}/transcripts/{hash}/*.transcript.txt
    output/{group}/*.{images,pdf,documents}.txt  (+ *_scan_*.txt для сканов)
    output/report_{timestamp}.txt
    /api/zip/<job_id> → omniprosvet_{job_id8}.zip
```

---

## Масштабирование

| Компонент | Текущий лимит | Как масштабировать |
|-----------|---------------|-------------------|
| Транскрибация | 4 воркера | Увеличить `max_workers` в `run_extract*` (CPU-bound) |
| NIM MAP | NIM_WORKERS (16) | Увеличить квоту RPM, workers |
| NIM REDUCE | 1 запрос | Не параллелится (сериализация) |
| PDF | min(4, NIM_WORKERS) | Увеличить workers |
| Images/Docs | NIM_WORKERS | Увеличить квоту |

---

## Отладка и диагностика

### Логи
- `omniprosvet.log` — ротация 1 неделя, UTF-8
- Уровни: INFO (прогресс), WARNING (ретраи), ERROR (провалы)

### Кэш
- `data/cache.db` — SQLite с таблицами `fragments` и `jobs`
- Очистка: `DELETE FROM fragments WHERE created_at < ...`

### Тесты
- `py -m pytest -q` — весь suite (unit: чанкинг, планировщик, прогресс, санитайзер,
  сбор файлов, ZIP; endpoint-тесты веб-панели с подменённой БД)

### Типичные проблемы
1. **NIM 429** → увеличить `NIM_RPM` (квота) или снизить `NIM_WORKERS`
2. **Скан PDF без текста** → автоматически Vision-fallback; нужен установленный `pdf2image`
3. **Медленная транскрибация** → `WHISPER_MODEL=small`/`base`, `WHISPER_FAST=1`
4. **Нет спикеров** → `DIARIZATION=1` или проверить polyvoice загрузку
5. **Старая задача без файлов** → список файлов стал записываться недавно; задачи старше
   7 дней удаляются автоматически
