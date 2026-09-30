import asyncio
import base64
import hashlib
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from src.configuration import settings

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


class NimError(RuntimeError):
    pass


class NimClient:
    def __init__(self, model: str | None = None, vision_model: str | None = None) -> None:
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._model = model or settings.nim_model
        self._vision_model = vision_model or settings.nim_vision_model
        self._requests_left: int | None = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._client_lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model

    def _pace(self) -> None:
        rpm = max(1, settings.nim_rpm)
        min_interval = 60.0 / rpm
        with self._lock:
            since = time.time() - self._last_call
            if self._requests_left is not None and self._requests_left <= 2:
                min_interval = max(min_interval, 5.0)
            if since < min_interval:
                time.sleep(min_interval - since)
            self._last_call = time.time()

    def _note_headers(self, headers) -> None:
        try:
            remaining = headers.get("x-ratelimit-remaining-requests")
            if remaining is not None:
                self._requests_left = int(remaining)
        except (ValueError, TypeError, AttributeError):
            pass

    @staticmethod
    def _image_data_url(path: Path) -> str:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        ext = path.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif", "bmp": "bmp", "tif": "tiff", "tiff": "tiff"}.get(ext, "png")
        return f"data:image/{mime};base64,{data}"

    @staticmethod
    def _extract_document_text(paths: list[str], max_chars: int) -> str:
        chunks: list[str] = []
        total = 0
        for p in paths:
            path = Path(p)
            ext = path.suffix.lower()
            text = ""
            try:
                if ext == ".pdf":
                    from pypdf import PdfReader

                    reader = PdfReader(str(path))
                    pages = []
                    for page in reader.pages:
                        t = (page.extract_text() or "").strip()
                        if t:
                            pages.append(t)
                    text = "\n\n".join(pages)
                elif ext in (".txt", ".csv"):
                    text = path.read_text(encoding="utf-8", errors="ignore")
                elif ext == ".docx":
                    from docx import Document

                    doc = Document(str(path))
                    parts = [par.text for par in doc.paragraphs if par.text.strip()]
                    for table in doc.tables:
                        for row in table.rows:
                            cells = [c.text.strip() for c in row.cells]
                            if any(cells):
                                parts.append(" | ".join(cells))
                    text = "\n".join(parts)
                elif ext in (".xlsx", ".xlsm"):
                    from openpyxl import load_workbook

                    wb = load_workbook(str(path), read_only=True, data_only=True)
                    sheets = []
                    for ws in wb.worksheets:
                        rows = []
                        for row in ws.iter_rows(values_only=True):
                            cells = ["" if c is None else str(c) for c in row]
                            if any(cells):
                                rows.append(" | ".join(cells))
                        if rows:
                            sheets.append(f"## Лист: {ws.title}\n" + "\n".join(rows))
                    text = "\n\n".join(sheets)
                elif ext == ".rtf":
                    import re

                    raw = path.read_text(encoding="utf-8", errors="ignore")
                    raw = re.sub(r"\\par[d]?\s?", "\n", raw)
                    raw = re.sub(r"\\[a-zA-Z]+-?\d*\s?", "", raw)
                    raw = raw.replace("{", "").replace("}", "")
                    text = raw
                else:
                    logger.warning(f"Документ {path.name}: формат {ext} не поддерживается для извлечения текста")
                    continue
            except Exception as exc:
                logger.warning(f"Документ {path.name} не читается: {type(exc).__name__}: {exc}")
                continue
            if not text.strip():
                continue
            chunks.append(text.strip())
            total += len(text)
            if total >= max_chars:
                break
        return "\n\n".join(chunks)[:max_chars]

    def _build_messages(self, prompt: str, attachments: list[str] | None) -> list:
        if not attachments:
            return [{"role": "user", "content": prompt}]

        images = [p for p in attachments if Path(p).suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
        }]
        pdfs = [p for p in attachments if p not in images]

        content: list = []
        pdf_text = ""
        if pdfs:
            pdf_text = self._extract_document_text(pdfs, max_chars=400_000)
        if pdf_text:
            content.append({"type": "text", "text": prompt + "\n\n# Извлечённый текст документов\n" + pdf_text})
        else:
            content.append({"type": "text", "text": prompt})
        for img in images:
            try:
                content.append({"type": "image_url", "image_url": {"url": self._image_data_url(Path(img))}})
            except OSError as exc:
                logger.warning(f"Картинка {Path(img).name} не читается: {exc}")
        return [{"role": "user", "content": content}]

    def send_prompt(
        self,
        prompt: str,
        deepthink: bool = True,
        expert: bool = True,
        attachments: list[str] | None = None,
        stable_rounds: int = 12,
        model: str | None = None,
    ) -> str:
        if not settings.nim_api_key:
            raise NimError("NIM_API_KEY не задан в .env — добавьте ключ с build.nvidia.com")
        messages = self._build_messages(prompt, attachments)
        if attachments:
            chosen = self._vision_model
        elif model:
            chosen = model
        else:
            chosen = self._model
        payload = {
            "model": chosen,
            "messages": messages,
            "max_tokens": settings.nim_max_tokens,
            "temperature": settings.nim_temperature,
            "frequency_penalty": settings.nim_frequency_penalty,
            "presence_penalty": settings.nim_presence_penalty,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {settings.nim_api_key}",
            "Accept": "application/json",
        }

        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            self._pace()
            logger.info(f"NIM: запрос к {chosen} ({len(prompt)} симв., попытка {attempt}/{MAX_RETRIES})…")
            try:
                response = httpx.post(
                    f"{NIM_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300.0,
                )
            except httpx.HTTPError as exc:
                last_error = f"Сеть: {type(exc).__name__}: {exc}"
                logger.warning(f"{last_error} — повтор {attempt}/{MAX_RETRIES}")
                time.sleep(min(30, 5 * attempt))
                continue

            self._note_headers(response.headers)

            if response.status_code == 200:
                try:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    finish = data["choices"][0].get("finish_reason", "") or ""
                except (KeyError, IndexError, ValueError) as exc:
                    raise NimError(f"NIM: неожиданный формат ответа: {exc}")
                if finish == "length":
                    logger.warning(
                        "NIM: ответ ОБРЕЗАН по NIM_MAX_TOKENS — фрагмент неполный, увеличьте NIM_MAX_TOKENS в .env"
                    )
                    text = text + "\n[ОТВЕТ ОБРЕЗАН ПО ЛИМИТУ — фрагмент неполный]"
                logger.info(f"NIM: ответ получен ({len(text)} симв., finish={finish or 'stop'})")
                return text.strip()

            if response.status_code in RETRYABLE_STATUS:
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else min(60, 10 * attempt)
                last_error = f"NIM: HTTP {response.status_code} (лимит запросов?)"
                logger.warning(f"{last_error} — жду {wait:.0f}с, повтор {attempt}/{MAX_RETRIES}")
                time.sleep(wait)
                continue

            detail = ""
            try:
                detail = str(response.json())
            except ValueError:
                detail = response.text[:300]
            raise NimError(f"NIM: HTTP {response.status_code}: {detail[:500]}")

        raise NimError(f"NIM: не удалось получить ответ после {MAX_RETRIES} попыток. {last_error}")

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling."""
        if self._async_client is None or self._async_client.is_closed:
            with self._client_lock:
                if self._async_client is None or self._async_client.is_closed:
                    limits = httpx.Limits(
                        max_connections=settings.nim_workers,
                        max_keepalive_connections=settings.nim_workers,
                        keepalive_expiry=30.0,
                    )
                    self._async_client = httpx.AsyncClient(
                        base_url=NIM_BASE_URL,
                        headers={
                            "Authorization": f"Bearer {settings.nim_api_key}",
                            "Accept": "application/json",
                        },
                        timeout=httpx.Timeout(300.0),
                        limits=limits,
                    )
        return self._async_client

    async def send_prompt_async(
        self,
        prompt: str,
        deepthink: bool = True,
        expert: bool = True,
        attachments: list[str] | None = None,
        stable_rounds: int = 12,
        model: str | None = None,
    ) -> str:
        """Async version of send_prompt with connection pooling."""
        if not settings.nim_api_key:
            raise NimError("NIM_API_KEY не задан в .env — добавьте ключ с build.nvidia.com")
        
        messages = self._build_messages(prompt, attachments)
        if attachments:
            chosen = self._vision_model
        elif model:
            chosen = model
        else:
            chosen = self._model
        
        payload = {
            "model": chosen,
            "messages": messages,
            "max_tokens": settings.nim_max_tokens,
            "temperature": settings.nim_temperature,
            "frequency_penalty": settings.nim_frequency_penalty,
            "presence_penalty": settings.nim_presence_penalty,
            "stream": False,
        }

        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            self._pace()
            logger.info(f"NIM async: запрос к {chosen} ({len(prompt)} симв., попытка {attempt}/{MAX_RETRIES})…")
            try:
                client = self._get_async_client()
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                    timeout=300.0,
                )
            except httpx.HTTPError as exc:
                last_error = f"Сеть: {type(exc).__name__}: {exc}"
                logger.warning(f"{last_error} — повтор {attempt}/{MAX_RETRIES}")
                await asyncio.sleep(min(30, 5 * attempt))
                continue

            self._note_headers(response.headers)

            if response.status_code == 200:
                try:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    finish = data["choices"][0].get("finish_reason", "") or ""
                except (KeyError, IndexError, ValueError) as exc:
                    raise NimError(f"NIM: неожиданный формат ответа: {exc}")
                if finish == "length":
                    logger.warning(
                        "NIM: ответ ОБРЕЗАН по NIM_MAX_TOKENS — фрагмент неполный, увеличьте NIM_MAX_TOKENS в .env"
                    )
                    text = text + "\n[ОТВЕТ ОБРЕЗАН ПО ЛИМИТУ — фрагмент неполный]"
                logger.info(f"NIM async: ответ получен ({len(text)} симв., finish={finish or 'stop'})")
                return text.strip()

            if response.status_code in RETRYABLE_STATUS:
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else min(60, 10 * attempt)
                last_error = f"NIM: HTTP {response.status_code} (лимит запросов?)"
                logger.warning(f"{last_error} — жду {wait:.0f}с, повтор {attempt}/{MAX_RETRIES}")
                await asyncio.sleep(wait)
                continue

            detail = ""
            try:
                detail = str(response.json())
            except ValueError:
                detail = response.text[:300]
            raise NimError(f"NIM: HTTP {response.status_code}: {detail[:500]}")

        raise NimError(f"NIM: не удалось получить ответ после {MAX_RETRIES} попыток. {last_error}")

    async def close(self) -> None:
        """Close async client."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None


client = NimClient()
