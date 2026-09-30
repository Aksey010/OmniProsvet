# OmniProsvet — Архитектура и принципы работы

## Обзор

OmniProsvet — это пайплайн для извлечения информации из аудио, видео, изображений и документов.
Использует **faster-whisper** для транскрибации и **NVIDIA NIM API** для LLM-анализа.

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

## Этап 2: Транскрибация (`src/transcription.py` + `src/whisper_extract.py`)

### Параллельная транскрибация (`run_extract`, `run_extract_and_process`)

```
media_files (N файлов)
    │
    ├─► ThreadPoolExecutor(max_workers=min(4, N))
    │       │
    │       ├─► Worker 1: transcribe_file(file1)
    │       ├─► Worker 2: transcribe_file(file2)
    │       ├─► Worker 3: transcribe_file(file3)
    │       └─► Worker 4: transcribe_file(file4)
    │
    └─► Собирает результаты по мере завершения (as_completed)
```

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
       - batch_size=8 (fast mode)
       - on_progress callback для UI
    
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
- **BatchedInferencePipeline** + `WHISPER_FAST=1` + `WHISPER_BATCH_SIZE=16` → ускорение в 3-5 раз
- **Параллельные воркеры** (до 4) → утилизация всех CPU ядер
- **Кэширование** по хешу файла → пропуск уже обработанного

---

## Этап 3: Подготовка чанков для LLM (`src/utils.py`)

### `_prepare_transcript(transcript_path, budget)`
1. Читает строки, начинающиеся с `[` (таймстампы)
2. Обрезает каждую строку до 500 символов
3. Префиксирует: `[{stem}] {line}`
4. Разбивает на чанки:
   - Бюджет: `settings.chunk_char_budget` (65k) или `nim_chunk_char_budget` (8k)
   - Макс. строк: `chunk_max_lines` (300)
   - Наполняет пока не превысит бюджет или макс. строк

### `_is_small_chunk(chunk, budget)`
- Малый чанк: размер ≤ budget/2 И строк ≤ max_lines/2
- Малые чанки копируются в общий буфер `_SmallBuffer` → батчятся вместе

---

## Этап 4: Map/Reduce анализ через NIM (`src/analysis.py`)

### Архитектура Map/Reduce

```
Транскрипты (M файлов)
    │
    ├─► Чанки транскрипта 1 → K1 чанков
    ├─► Чанки транскрипта 2 → K2 чанков
    │       │
    │       ▼ MAP параллельно (ThreadPoolExecutor, max_workers=NIM_WORKERS)
    │       _process_fragment(idx, total, chunk, source) для каждого чанка
    │       │
    │       ├─► Проверка кэша (SHA256(source + chunk))
    │       ├─► NIM: build_map_prompt(idx, total, source, body)
    │       ├─► nim_client.send_prompt(prompt, deepthink=False, expert=False)
    │       ├─► Сохранение в кэш
    │       └─► Возврат конспекта фрагмента
    │
    ▼
Все конспекты фрагментов (K1+K2+... суммарно)
    │
    ▼ REDUCE (один запрос)
    _reduce_and_save()
    │
    ├─► Сборка parts_block = "### Фрагмент 1\n{concept1}\n\n### Фрагмент 2\n{concept2}..."
    ├─► NIM: build_reduce_prompt(files_count, chunks_count, parts_block)
    ├─► nim_client.send_prompt(prompt, deepthink=True, expert=True, model=nim_reduce_model)
    └─► Сохранение: output/report_{timestamp}.txt
```

### `_FragProgress` — отслеживание прогресса
- Весовые фазы: `transcribe` (50%), `analysis` (35%), `images/pdf/documents` (остальное)
- `phase_progress` 0.0-1.0 внутри фазы
- Эмитит события для WebSocket/UI

### `_SmallBuffer` — батчинг мелких чанков
- Накапливает малые чанки до достижения бюджета
- Отправляет как один фрагмент "мелкие файлы"

---

## Этап 5: Обработка вложений (`src/attachments.py`)

### Параллельная обработка по типам

```
att_plan = [
    ("images", [img1, img2, ...], gdir),
    ("pdf", [pdf1, pdf2, ...], gdir),
    ("documents", [doc1, doc2, ...], gdir)
]

ThreadPoolExecutor(max_workers=3):
    ├─► run_attachments(images, ...)   # Vision модель
    ├─► run_pdfs(pdfs, ...)            # Текст + LLM
    └─► run_attachments(documents, ...) # Текст + LLM
```

### `run_attachments` (images/documents)
1. Батчинг: `NIM_IMAGES_PER_BATCH` (3) файлов на запрос
2. Для каждого батча:
   - Хеш от имён+размеров → проверка кэша
   - `build_images_prompt(count)` или `build_documents_prompt(count)`
   - `nim_client.send_prompt(prompt, attachments=[paths...])`
   - Vision модель для изображений: `meta/llama-3.2-90b-vision-instruct`
   - Сохранение: `results/{stem}_{hash}.images.txt`

### `run_pdfs` — пер-файл (без склейки)
```
Для каждого PDF параллельно (max_workers=min(NIM_WORKERS, 4)):
    1. Извлечение текста через pypdf (по страницам)
    2. Если пусто → skip (скан без текстового слоя)
    3. Чанкинг текста (nim_chunk_char_budget=8000)
    4. MAP: параллельные запросы к NIM для каждого чанка
    5. Сборка конспекта PDF
    6. Сохранение: results/{stem}_{hash}.pdf.txt
```
**Убрано**: склейка PDF в огромные файлы → не нужно, работает нативнее, меньше памяти.

---

## Этап 6: NIM Client (`src/nim_client.py`)

### Синхронный `send_prompt()`
- Rate limiting: `_pace()` — min_interval = 60/RPM
- Retry logic: до 5 попыток, экспоненциальная задержка
- Обработка 429/5xx с `retry-after` заголовком
- Кэширование rate limit заголовков (`x-ratelimit-remaining-requests`)

### Асинхронный `send_prompt_async()` + Connection Pooling
```python
async def _get_async_client() -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=NIM_WORKERS,      # 16
        max_keepalive_connections=NIM_WORKERS,
        keepalive_expiry=30.0,
    )
    return httpx.AsyncClient(
        base_url=NIM_BASE_URL,
        headers={"Authorization": "Bearer ...", "Accept": "application/json"},
        timeout=300.0,
        limits=limits,
    )
```
- **Keep-alive** соединения → нет overhead на TLS handshake
- **Пул соединений** = NIM_WORKERS (16) → полная утилизация rate limit
- Переиспользование между запросами

### Подготовка вложений (`_build_messages`)
- Изображения → base64 data URL → `image_url` в сообщении
- Документы (PDF, DOCX, XLSX, TXT, CSV, RTF) → извлечение текста → вставка в промпт
- Макс. 400k символов из документов на запрос

---

## Этап 7: Веб-панель (`web_panel.py`)

### Архитектура
```
Flask + SocketIO (WebSocket)
    │
    ├─► GET /                    → HTML страница
    ├─► POST /api/start          → Создание job, сохранение в БД, запуск потока
    ├─► WebSocket /socket.io     → Real-time прогресс
    │       ├─► job_progress    → {progress, state, current_file, lines...}
    │       ├─► job_log         → Логи
    │       ├─► job_done        → Результаты
    │       └─► job_error       → Ошибки
    ├─► GET /api/status/<id>     → Статус job (REST fallback)
    ├─► GET /api/files/<id>      → Список файлов результата
    ├─► GET /api/result/<id>     → Скачивание отчёта
    ├─► GET /api/download        → Скачивание любого файла (с проверкой path traversal)
    ├─► GET /api/jobs            → История задач (из БД, 50 шт.)
    └─► DELETE /api/jobs/<id>    → Удаление job + upload папки
```

### Хранение задач (SQLite)
- Таблица `jobs`: `id`, `data(JSON)`, `created_at`, `updated_at`
- `save_job()` / `get_job()` / `list_jobs()` / `delete_job()` / `delete_old_jobs(days=7)`
- WAL режим для конкурентного доступа

### Прогресс-бар (`_PhaseProgress`)
- Фазы с весами: `transcribe=50`, `analysis=35`, `images/pdf/documents=остальное`
- `feed(event)` → обновляет `completed` + `fracs[phase]`
- `percent()` → общий % для UI

---

## Этап 8: Кэширование (`src/database.py`)

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
    data TEXT,                -- JSON с полным состоянием
    created_at INTEGER,
    updated_at INTEGER
);
```
- Полное состояние job для восстановления после рестарта
- TTL 7 дней (`delete_old_jobs`)

---

## Конфигурация производительности (`.env`)

```env
# Whisper — скорость
WHISPER_MODEL=base              # tiny/base/small/medium/large-v3
WHISPER_FAST=1                  # BatchedInferencePipeline
WHISPER_BATCH_SIZE=16           # Размер батча для fast mode
DIARIZATION=0                   # Отключить для ускорения (если спикеры не нужны)

# NIM — пропускная способность
NIM_RPM=120                     # Requests per minute (зависит от квоты)
NIM_WORKERS=16                  # Параллельные воркеры/соединения
NIM_MAX_TOKENS=8192             # Меньше токенов = быстрее ответ
NIM_CHUNK_CHAR_BUDGET=8000      # Меньшие чанки = больше параллелизма
NIM_IMAGES_PER_BATCH=3          # Меньше картинок на запрос = быстрее
```

---

## Поток данных (Data Flow)

```
ВХОД: /path/to/media/
    │
    ▼
collect_media() → (audio[], video[], images[], documents[])
    │
    ├─► audio+video → run_extract() → ThreadPool(4) → transcribe_file() → transcripts[]
    │       │
    │       └─► faster-whisper (+ polyvoice опционально)
    │
    ├─► images[] → run_attachments() → батчи по 3 → Vision NIM → results/*.images.txt
    │
    ├─► pdfs[] → run_pdfs() → пер-файл параллельно → извлечение текста → MAP NIM → results/*.pdf.txt
    │
    └─► documents[] → run_attachments() → батчи → NIM → results/*.documents.txt
    │
    ▼
transcripts[] + in-memory анализ (MAP/REDUCE)
    │
    ├─► MAP: каждый чанк → NIM (deepthink=False) → конспект фрагмента (кэш)
    │
    └─► REDUCE: все конспекты → NIM (deepthink=True, expert=True) → report.txt
    │
    ▼
ВЫХОД:
    output/transcripts/*.transcript.txt
    output/report_{timestamp}.txt
    results/*.{images,pdf,documents}.txt
```

---

## Масштабирование

| Компонент | Текущий лимит | Как масштабировать |
|-----------|---------------|-------------------|
| Транскрибация | 4 воркера | Увеличить `max_workers` в `run_extract` (CPU-bound) |
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

### Типичные проблемы
1. **NIM 429** → увеличить `NIM_RPM` или снизить `NIM_WORKERS`
2. **OOM на PDF** → убрана склейка, пер-файл обработка
3. **Медленная транскрибация** → `WHISPER_MODEL=base`, `WHISPER_FAST=1`
4. **Нет спикеров** → `DIARIZATION=0` или проверить polyvoice загрузку