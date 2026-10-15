# OmniProsvet

Экстрактор информации из аудио, видео, картинок и документов. Использует Whisper для транскрибации и NVIDIA NIM API для анализа контента.

## Возможности

- **Транскрибация** аудио/видео с помощью faster-whisper (поддержка диаризации спикеров через polyvoice)
- **Анализ текста** через NVIDIA NIM API (Nemotron, GPT-OSS и другие модели)
- **Распознавание картинок** (OCR, описание изображений) через Vision модели NIM
- **Обработка PDF и документов** (извлечение текста, конспектирование)
- **Веб-интерфейс** с drag-and-drop, WebSocket прогрессом и историей задач
- **Кэширование** результатов LLM в SQLite для экономии запросов
- **Параллельная обработка** нескольких файлов

## Установка

### 1. Клонирование и зависимости

```bash
git clone <repo>
cd OmniProsvet
pip install -r requirements.txt
```

### 2. Настройка NVIDIA NIM API

1. Получите бесплатный API ключ на [build.nvidia.com](https://build.nvidia.com)
2. Скопируйте `.env.example` в `.env`:
   ```bash
   cp .env.example .env
   ```
3. Отредактируйте `.env` и добавьте ваш ключ:
   ```
   NIM_API_KEY=ваш_ключ_здесь
   ```

### 3. Рекомендуемые модели (в .env)

```env
# Текстовый анализ — быстрый и качественный
NIM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NIM_REDUCE_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b

# Визуальный анализ картинок
NIM_VISION_MODEL=meta/llama-3.2-90b-vision-instruct

# Лимиты запросов (бесплатный тариф: 40 RPM)
NIM_RPM=40
NIM_MAX_TOKENS=16384
NIM_WORKERS=8
```

Другие доступные модели на NIM:
- `nvidia/nemotron-3-ultra-550b-a55b` — лучшее качество, 1M контекст
- `deepseek-ai/deepseek-v4-pro-0813` — сильная альтернатива
- `openai/gpt-oss-20b` — открытая модель от OpenAI

## Ускорение на GPU (другая машина)

Транскрибация — единственная локально-CPU-тяжёлая фаза; NIM работает в облаке
и от видеокарты не зависит. Проект определяет GPU автоматически.

### Требования

1. NVIDIA GPU с compute capability ≥ 5.0 (GTX 1050+; реальный эффект — от RTX 3060
   с 8+ ГБ VRAM) и свежим драйвером (CUDA 12: драйвер ≥ 527.41).
2. CUDA-стек для CTranslate2 (faster-whisper):
   ```bash
   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   ```
3. Проверка: `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`
   — должно быть ≥ 1.

### Настройки (.env)

```env
WHISPER_DEVICE=auto        # auto (по умолчанию: cuda если есть GPU, иначе cpu) | cuda | cpu
WHISPER_COMPUTE_TYPE=int8  # на GPU int8 автоматически повышается до float16;
                           # задайте int8_float16 явно для экономии VRAM
WHISPER_WORKERS=1          # на одной карте оптимально 1..2 (не 4!)
WHISPER_BATCH_SIZE=16      # на GPU можно 16..32
WHISPER_MODEL=large-v3     # на GPU large-v3 быстрее, чем small на CPU, и точнее
```

При `WHISPER_DEVICE=cuda` без рабочего драйвера проект не падает: в логе
появится предупреждение и транскрибация уйдёт на CPU с обычным int8.

Ожидаемый эффект: транскрибация с ~3–5× realtime (CPU) до ~20–60× realtime
(batched + float16). Диаризация polyvoice работает параллельно и от GPU
не зависит.

## Использование

### Командная строка

```bash
# Полная обработка (транскрибация + анализ)
python main.py all /путь/к/файлам

# Только транскрибация
python main.py extract /путь/к/аудио

# Только анализ существующих транскриптов
python main.py process /путь/к/транскриптам

# Опции
python main.py all /путь -m small --language ru --no-diarize --no-images
```

### Веб-интерфейс

```bash
python web_panel.py
```

Откроется браузер на `http://127.0.0.1:5000`

Функции веб-панели:
- Drag-and-drop загрузка файлов (до 2 ГБ)
- Указание пути к папке на диске
- Реальное время прогресса через WebSocket
- История задач с сохранением между перезапусками
- Скачивание результатов

## Структура проекта

```
OmniProsvet/
├── main.py              # CLI точка входа
├── web_panel.py         # Flask + SocketIO веб-интерфейс
├── prompt.md            # Промпты для LLM (редактируются без кода)
├── requirements.txt     # Зависимости
├── .env                 # Настройки (не в git)
├── src/
│   ├── __init__.py
│   ├── configuration.py # Pydantic Settings с валидацией
│   ├── database.py      # SQLite: кэш фрагментов + истории задач
│   ├── collect.py       # Сбор медиа файлов
│   ├── whisper_extract.py   # Транскрибация (faster-whisper + polyvoice)
│   ├── nim_client.py    # NVIDIA NIM API клиент
│   ├── prompts.py       # Загрузка промптов из prompt.md
│   ├── utils.py         # Общие утилиты (чанкинг, хеширование)
│   ├── transcription.py # Модуль транскрибации
│   ├── analysis.py      # Map/Reduce анализ текста через NIM
│   ├── attachments.py   # Обработка картинок, PDF, документов
│   └── pipeline.py      # Оркестрация всего пайплайна
├── data/                # Кэш, БД, профиль браузера
├── output/              # Транскрипты и отчёты
├── results/             # Результаты обработки картинок/документов
├── transcripts/         # Кэш транскриптов
└── tmp_pdfs/            # Временные объединённые PDF
```

## Конфигурация (.env)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `WHISPER_MODEL` | `small` | Модель Whisper: tiny/base/small/medium/large-v3 |
| `LANGUAGE` | `auto` | Код языка: auto/ru/en/zh/ja/ko/de/fr/es |
| `WHISPER_FAST` | `1` | BatchedInferencePipeline (ускорение) |
| `WHISPER_BATCH_SIZE` | `8` | Размер батча для fast режима |
| `DIARIZATION` | `1` | Включить диаризацию спикеров |
| `POLYVOICE_VARIANT` | `balanced` | balanced/mobile |
| `NIM_API_KEY` | *обязательно* | Ключ с build.nvidia.com |
| `NIM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Основная модель текста |
| `NIM_REDUCE_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Модель для финального отчёта |
| `NIM_VISION_MODEL` | `meta/llama-3.2-90b-vision-instruct` | Vision модель |
| `NIM_RPM` | `40` | Requests per minute (лимит NIM) |
| `NIM_MAX_TOKENS` | `16384` | Макс. токенов в ответе |
| `NIM_TEMPERATURE` | `0.7` | Температура генерации |
| `NIM_WORKERS` | `8` | Параллельные воркеры для NIM |
| `NIM_IMAGES_PER_BATCH` | `5` | Картинок в одном запросе |
| `NIM_CHUNK_CHAR_BUDGET` | `12000` | Бюджет символов на чанк для NIM |
| `CHUNK_CHAR_BUDGET` | `65000` | Бюджет символов на чанк (общий) |
| `CHUNK_MAX_LINES` | `300` | Макс. строк в чанке |
| `IMAGES_PER_BATCH` | `25` | Картинок в батче (общий) |
| `PDF_MAX_SIZE_MB` | `40` | Макс. размер объединённого PDF |

## Промпты

Промпты хранятся в `prompt.md` и загружаются динамически. Можно редактировать без перезапуска кода.

Секции (разделены маркерами `<!-- ==== NAME ==== -->`):
- **ФРАГМЕНТ** — промпт для map-этапа (конспектирование чанка)
- **ИТОГОВЫЙ ОТЧЁТ** — промпт для reduce-этапа (объединение в единый отчёт)
- **КАРТИНКИ** — промпт для анализа изображений
- **ДОКУМЕНТЫ** — промпт для анализа документов

Плейсхолдеры: `{i}`, `{n}`, `{source}`, `{body}`, `{files}`, `{chunks}`, `{parts}`, `{count}`

## Архитектура пайплайна

```
Входные файлы
     │
     ├── Аудио/Видео ───► Whisper (транскрибация) ───► Транскрипты
     │                                                │
     ├── Картинки ──────────────────────────────────►│
     │                                                ▼
     ├── PDF ───────────────────────────────────────► NIM (Map: конспекты фрагментов)
     │                                                │
     └── Документы ─────────────────────────────────►│
                                                      ▼
                                               NIM (Reduce: единый отчёт)
                                                      │
                                                      ▼
                                              Результаты (output/, results/)
```

## Кэширование

- **Фрагменты текста**: хеш от `(source + chunk)` → SQLite `fragments` таблица
- **Картинки/PDF/документы**: хеш от имён файлов и размеров
- При повторном запуске используются кэшированные результаты

## Лицензия

MIT