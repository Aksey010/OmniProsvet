import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _as_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


@dataclass(frozen=True)
class Settings:
    # DeepSeek (веб-чат через Selenium)
    deepseek_login: str = os.getenv("DEEPSEEK_LOGIN", "")
    deepseek_password: str = os.getenv("DEEPSEEK_PASSWORD", "")
    deepseek_deepthink: bool = _as_bool("DEEPSEEK_DEEPTHINK", "1")
    deepseek_expert: bool = _as_bool("DEEPSEEK_EXPERT", "1")
    deepseek_workers: int = int(os.getenv("DEEPSEEK_WORKERS", "1") or "1")
    deepseek_headless: bool = _as_bool("DEEPSEEK_HEADLESS", "0")
    deepseek_hidden_window: bool = _as_bool("DEEPSEEK_HIDDEN_WINDOW", "0")
    deepseek_pause_min: int = int(os.getenv("DEEPSEEK_PAUSE_MIN", "15") or "15")
    deepseek_pause_max: int = int(os.getenv("DEEPSEEK_PAUSE_MAX", "60") or "60")

    # Whisper
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    language: str = os.getenv("LANGUAGE", "auto")
    whisper_fast: bool = _as_bool("WHISPER_FAST", "1")
    whisper_batch_size: int = int(os.getenv("WHISPER_BATCH_SIZE", "8") or "8")

    # Диаризация (polyvoice)
    diarization: bool = _as_bool("DIARIZATION", "1")
    polyvoice_variant: str = os.getenv("POLYVOICE_VARIANT", "balanced")

    # Обработка
    chunk_char_budget: int = int(os.getenv("CHUNK_CHAR_BUDGET", "65000") or "65000")
    chunk_max_lines: int = int(os.getenv("CHUNK_MAX_LINES", "300") or "300")
    images_per_batch: int = int(os.getenv("IMAGES_PER_BATCH", "25") or "25")
    pdf_max_size_mb: int = int(os.getenv("PDF_MAX_SIZE_MB", "40") or "40")

    # Платформа анализа: deepseek (браузер) или nim (NVIDIA NIM API)
    analysis_platform: str = os.getenv("ANALYSIS_PLATFORM", "deepseek")

    # NVIDIA NIM API (бесплатный ключ с build.nvidia.com)
    nim_api_key: str = os.getenv("NIM_API_KEY", "")
    nim_model: str = os.getenv("NIM_MODEL", "openai/gpt-oss-20b")
    nim_reduce_model: str = os.getenv("NIM_REDUCE_MODEL", "openai/gpt-oss-20b")
    nim_vision_model: str = os.getenv("NIM_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct")
    nim_rpm: int = int(os.getenv("NIM_RPM", "40") or "40")
    nim_max_tokens: int = int(os.getenv("NIM_MAX_TOKENS", "16384") or "16384")
    nim_temperature: float = float(os.getenv("NIM_TEMPERATURE", "0.7") or "0.7")
    nim_frequency_penalty: float = float(os.getenv("NIM_FREQUENCY_PENALTY", "0.4") or "0.4")
    nim_presence_penalty: float = float(os.getenv("NIM_PRESENCE_PENALTY", "0.3") or "0.3")
    nim_workers: int = int(os.getenv("NIM_WORKERS", "8") or "8")
    nim_images_per_batch: int = int(os.getenv("NIM_IMAGES_PER_BATCH", "5") or "5")
    nim_chunk_char_budget: int = int(os.getenv("NIM_CHUNK_CHAR_BUDGET", "12000") or "12000")


settings = Settings()
