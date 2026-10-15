import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _as_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _as_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _as_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


@dataclass(frozen=True)
class Settings:
    # Whisper
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    language: str = os.getenv("LANGUAGE", "auto")
    whisper_fast: bool = _as_bool("WHISPER_FAST", "1")
    whisper_batch_size: int = _as_int("WHISPER_BATCH_SIZE", 8)
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    # Устройство Whisper: auto (cuda при наличии GPU, иначе cpu) | cuda | cpu
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    # Воркеры транскрибации (CPU-bound; > числа физических ядер смысла нет)
    whisper_workers: int = _as_int("WHISPER_WORKERS", 4)

    # Диаризация (polyvoice)
    diarization: bool = _as_bool("DIARIZATION", "1")
    polyvoice_variant: str = os.getenv("POLYVOICE_VARIANT", "balanced")

    # Обработка
    chunk_char_budget: int = _as_int("CHUNK_CHAR_BUDGET", 65000)
    chunk_max_lines: int = _as_int("CHUNK_MAX_LINES", 300)
    images_per_batch: int = _as_int("IMAGES_PER_BATCH", 25)
    # Пропуск файлов по подстрокам имени (через запятую, регистр не важен).
    # Стикеры Telegram (123_sticker.webm/.webp, 123_animation.gif.mp4) не несут
    # речи и смыслового контента: транскрибация ломалась на них (388 ошибок
    # «tuple index out of range» в сессии 13.10), а .webp-стикеры зря гонялись
    # через дорогую vision-модель. Документы (pdf/docx/…) фильтром не трогаются.
    skip_file_patterns: str = os.getenv("SKIP_FILE_PATTERNS", "sticker,animation.gif")

    # NVIDIA NIM API (бесплатный ключ с build.nvidia.com)
    nim_api_key: str = os.getenv("NIM_API_KEY", "")
    nim_model: str = os.getenv("NIM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    nim_reduce_model: str = os.getenv("NIM_REDUCE_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    nim_vision_model: str = os.getenv("NIM_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct")
    nim_rpm: int = _as_int("NIM_RPM", 120)  # Увеличено для параллелизма
    nim_max_tokens: int = _as_int("NIM_MAX_TOKENS", 8192)  # Уменьшено для скорости
    nim_temperature: float = _as_float("NIM_TEMPERATURE", 0.7)
    nim_frequency_penalty: float = _as_float("NIM_FREQUENCY_PENALTY", 0.4)
    nim_presence_penalty: float = _as_float("NIM_PRESENCE_PENALTY", 0.3)
    nim_workers: int = _as_int("NIM_WORKERS", 16)  # Увеличено для параллелизма
    nim_images_per_batch: int = _as_int("NIM_IMAGES_PER_BATCH", 3)  # Меньше батч = больше параллелизма
    nim_chunk_char_budget: int = _as_int("NIM_CHUNK_CHAR_BUDGET", 8000)  # Меньшие чанки = больше параллелизма
    # Автопродолжение ответа при обрыве по max_tokens (0 = off)
    nim_max_continuations: int = _as_int("NIM_MAX_CONTINUATIONS", 3)
    # Фрагментов на промежуточную сводку в иерархическом reduce (0 = off)
    nim_reduce_group_size: int = _as_int("NIM_REDUCE_GROUP_SIZE", 24)
    # Максимум символов извлечённого текста документов НА ОДИН запрос.
    # Прежние 400 000 симв. ≈ 100к+ токенов гарантированно превышали контекст
    # моделей NIM (32k токенов) — запросы падали с HTTP 400.
    nim_doc_text_chars: int = _as_int("NIM_DOC_TEXT_CHARS", 60_000)
    # Read-таймаут HTTP-запроса к NIM (секунды). На урезанных картинках
    # (даунскейл до 2048px) и компактных промптах ответы приходят за секунды;
    # зависший запрос дешевле перезапустить (5 ретраев), чем ждать минуты.
    nim_read_timeout: int = _as_int("NIM_READ_TIMEOUT", 180)
    # Размер пула соединений httpx. Должен вмещать ОДНОВРЕМЕННЫЕ запросы всех
    # параллельных фаз (PDF/таблицы/презентации/картинки), иначе потоки ждут
    # свободный слот и падают с PoolTimeout (сессия 15.10: 62 PoolTimeout,
    # потерянные фото при живом сервере).
    nim_max_connections: int = _as_int("NIM_MAX_CONNECTIONS", 64)
    # Картинки тяжелее этого (байт) ужимаются перед отправкой (2048px, JPEG
    # q85): большие фото приводили к ReadTimeout на vision-модели.
    nim_image_max_bytes: int = _as_int("NIM_IMAGE_MAX_BYTES", 3_000_000)

    # Пути
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    output_dir: Path = base_dir / "output"
    transcripts_dir: Path = base_dir / "transcripts"
    results_dir: Path = base_dir / "results"
    tmp_pdfs_dir: Path = base_dir / "tmp_pdfs"
    cache_db_path: Path = data_dir / "cache.db"

    def validate(self) -> list[str]:
        """Валидация настроек. Возвращает список ошибок."""
        errors = []
        if not self.nim_api_key:
            errors.append("NIM_API_KEY не задан в .env — получите бесплатный ключ на https://build.nvidia.com")
        if self.nim_rpm <= 0:
            errors.append("NIM_RPM должен быть > 0")
        if self.nim_max_tokens <= 0:
            errors.append("NIM_MAX_TOKENS должен быть > 0")
        if not 0 <= self.nim_temperature <= 2:
            errors.append("NIM_TEMPERATURE должен быть в диапазоне 0-2")
        if self.nim_workers <= 0:
            errors.append("NIM_WORKERS должен быть > 0")
        return errors


settings = Settings()

# Создаём необходимые директории при импорте
for d in (settings.data_dir, settings.output_dir, settings.transcripts_dir, settings.results_dir, settings.tmp_pdfs_dir):
    d.mkdir(parents=True, exist_ok=True)
