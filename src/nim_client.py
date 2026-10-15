import asyncio
import base64
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

# Расширения файлов-картинок: только их присутствие во вложениях переключает
# запрос на vision-модель. Документы (pdf/docx/xlsx/…) вкладываются как
# извлечённый текст — текстовой модели они не требуют vision.
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
}

# Просьба продолжить оборванный ответ (автопродолжение при finish_reason == "length")
CONTINUATION_PROMPT = (
    "Твой предыдущий ответ оборвался на лимите длины. Продолжай ровно с того места, "
    "где текст оборвался. Не повторяй уже написанное, не пиши вступление — только "
    "непосредственное продолжение текста в том же формате."
)

TRUNCATED_MARKER = "\n[ОТВЕТ ОБРЕЗАН ПО ЛИМИТУ — фрагмент неполный]"


def _strip_overlap(prev: str, nxt: str, window: int = 60, min_k: int = 10) -> str:
    """Убрать повтор хвоста prev в начале nxt (модель часто повторяет обрыв).

    Сравниваем только достаточно длинные совпадения (>= min_k символов),
    чтобы не отрезать случайные короткие совпадения.
    """
    nxt = (nxt or "").lstrip()
    if not nxt:
        return ""
    max_k = min(window, len(prev), len(nxt))
    for k in range(max_k, min_k - 1, -1):
        if prev.endswith(nxt[:k]):
            return nxt[k:]
    return nxt


class NimError(RuntimeError):
    pass


class NimClient:
    def __init__(self, model: str | None = None, vision_model: str | None = None) -> None:
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._model = model or settings.nim_model
        self._vision_model = vision_model or settings.nim_vision_model
        self._requests_left: int | None = None
        self._reset_after: float | None = None
        self._sync_client: Optional[httpx.Client] = None
        self._client_lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_reset(value) -> float | None:
        """Заголовок x-ratelimit-reset-requests: '1s', '2500ms', '0.5' → секунды."""
        if not value:
            return None
        s = str(value).strip().lower()
        try:
            if s.endswith("ms"):
                return float(s[:-2]) / 1000.0
            if s.endswith("s"):
                return float(s[:-1])
            return float(s)
        except ValueError:
            return None

    def _pace(self) -> None:
        rpm = max(1, settings.nim_rpm)
        min_interval = 60.0 / rpm
        with self._lock:
            since = time.time() - self._last_call
            if self._requests_left is not None and self._requests_left <= 2:
                # Окно почти исчерпано: ждём его сброса, а не фиксированные 5 с —
                # сброс часто через секунду, и прежние 5 с занижали темп до 12 RPM.
                reset = self._reset_after if self._reset_after is not None else 5.0
                min_interval = max(min_interval, min(5.0, reset + 0.05))
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
        try:
            self._reset_after = self._parse_reset(headers.get("x-ratelimit-reset-requests"))
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # HTTP-клиенты (keep-alive пул: без TLS-рукопожатия на каждый запрос)
    # ------------------------------------------------------------------

    @staticmethod
    def _auth_headers() -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.nim_api_key}",
            "Accept": "application/json",
        }

    @staticmethod
    def _pool_limits() -> httpx.Limits:
        """Пул должен вмещать одновременные запросы ВСЕХ параллельных фаз.

        Прежний размер = nim_workers (16) не учитывал, что фрагментные пулы
        у PDF/таблиц/презентаций суммируются (4 файла × 16 воркеров и т.д.):
        слоты заканчивались, ожидающие потоки получали PoolTimeout уже через
        30 с — файлы терялись при живом сервере (сессия 15.10, 62 случая).
        """
        pool = max(2, settings.nim_max_connections)
        return httpx.Limits(
            max_connections=pool,
            # Открытыми держим не больше сокетов, чем ходовых воркеров:
            # keepalive дешевле рукопожатий, но сотни висящих соединений не нужны
            max_keepalive_connections=min(pool, max(2, settings.nim_workers)),
            keepalive_expiry=30.0,
        )

    @staticmethod
    def _http_timeout() -> httpx.Timeout:
        # ReadTimeout гибли vision-запросы на тяжёлых картинках
        # (сессии 13–14.10) — read настраивается (NIM_READ_TIMEOUT, 180 с);
        # тяжёлые картинки дополнительно ужимаются перед отправкой.
        # pool=120: если слоты всё же заняты — поток стоит в очереди, а не
        # роняет файл (дождаться слота дешевле перезапроса с нуля).
        return httpx.Timeout(
            connect=15.0,
            read=float(settings.nim_read_timeout),
            write=120.0,
            pool=120.0,
        )

    def _get_sync_client(self) -> httpx.Client:
        """Единый httpx.Client с keep-alive: TCP/TLS ставится один раз, дальше
        соединения переиспользуются (раньше httpx.post() на каждый запрос —
        +0.2-0.4 с на рукопожатие)."""
        if self._sync_client is None or self._sync_client.is_closed:
            with self._client_lock:
                if self._sync_client is None or self._sync_client.is_closed:
                    self._sync_client = httpx.Client(
                        base_url=NIM_BASE_URL,
                        headers=self._auth_headers(),
                        timeout=self._http_timeout(),
                        limits=self._pool_limits(),
                    )
        return self._sync_client

    # ------------------------------------------------------------------
    # Подготовка сообщений
    # ------------------------------------------------------------------

    @staticmethod
    def _downscale_image(data: bytes, path: Path, max_side: int = 2048) -> bytes:
        """Уменьшить тяжёлую картинку (2048px, JPEG q85) для Vision-запроса.

        Фото в несколько мегабайт после base64 раздували запрос так, что
        vision-модель не успевала за 300 с — файл гибли после 5 ретраев
        (сессии 13–14.10: одиночные фото терялись по ReadTimeout). Даунскейл
        в разы ускоряет запрос, качество распознавания не страдает.
        """
        try:
            from io import BytesIO

            from PIL import Image

            img = Image.open(BytesIO(data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            out = buf.getvalue()
            if len(out) < len(data):
                logger.info(
                    f"Картинка {path.name}: {len(data) // 1024} КБ → {len(out) // 1024} КБ "
                    f"(даунскейл для Vision)"
                )
                return out
        except Exception as exc:
            logger.warning(f"Картинка {path.name}: даунскейл не удался ({type(exc).__name__}: {exc})")
        return data

    @classmethod
    def _image_data_url(cls, path: Path) -> str:
        data = path.read_bytes()
        if len(data) > settings.nim_image_max_bytes:
            data = cls._downscale_image(data, path)
        base64_data = base64.b64encode(data).decode("ascii")
        ext = path.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif", "bmp": "bmp", "tif": "tiff", "tiff": "tiff"}.get(ext, "png")
        return f"data:image/{mime};base64,{base64_data}"

    @staticmethod
    def _extract_pptx_text(path: Path) -> str:
        from pptx import Presentation

        prs = Presentation(str(path))
        parts: list[str] = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    t = (shape.text_frame.text or "").strip()
                    if t:
                        parts.append(t)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        cells = [c.text.strip() for c in row.cells]
                        if any(cells):
                            parts.append(" | ".join(cells))
            if getattr(slide, "has_notes_slide", False):
                nt = (slide.notes_slide.notes_text_frame.text or "").strip()
                if nt:
                    parts.append(nt)
        return "\n".join(parts)

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
                            # Фантомные колонки (форматирование до 16k+ колонок)
                            # режем: строка обрезается по последнюю непустую
                            # ячейку в разумном пределе ширины, иначе джойн
                            # раздувается пустыми « | » и артефактами копирования.
                            row = row[:128]
                            last = 0
                            for j, v in enumerate(row, 1):
                                if v is not None and str(v).strip():
                                    last = j
                            if not last:
                                continue
                            cells = ["" if c is None else str(c) for c in row[:last]]
                            while cells and not cells[-1].strip():
                                cells.pop()
                            if not any(c.strip() for c in cells):
                                continue
                            rows.append(" | ".join(cells))
                        if rows:
                            sheets.append(f"## Лист: {ws.title}\n" + "\n".join(rows))
                    text = "\n\n".join(sheets)
                elif ext == ".pptx":
                    text = NimClient._extract_pptx_text(path)
                elif ext == ".rtf":
                    # RTF — 7-bit ASCII с escape-последовательностями: \'hh (байты
                    # cp1251), \uN, служебные группы. Наивный регэксп оставлял
                    # \'d0\'cf литералами (кириллица умирала) — используем декодер.
                    from src.rtf_text import rtf_to_text

                    text = rtf_to_text(
                        path.read_bytes().decode("latin-1", errors="ignore")
                    )
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

        images = [p for p in attachments if Path(p).suffix.lower() in IMAGE_SUFFIXES]
        docs = [p for p in attachments if p not in images]

        content: list = []
        doc_text = ""
        if docs:
            doc_text = self._extract_document_text(docs, max_chars=settings.nim_doc_text_chars)
        if doc_text:
            content.append({"type": "text", "text": prompt + "\n\n# Извлечённый текст документов\n" + doc_text})
        else:
            content.append({"type": "text", "text": prompt})
        for img in images:
            try:
                content.append({"type": "image_url", "image_url": {"url": self._image_data_url(Path(img))}})
            except OSError as exc:
                logger.warning(f"Картинка {Path(img).name} не читается: {exc}")
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Один логический запрос (с ретраями) -> (text, finish_reason)
    # ------------------------------------------------------------------

    def _post_chat(self, messages: list, chosen: str, log_prefix: str = "NIM") -> tuple[str, str]:
        if not settings.nim_api_key:
            raise NimError("NIM_API_KEY не задан в .env — добавьте ключ с build.nvidia.com")
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
            logger.info(f"{log_prefix}: запрос к {chosen} (сообщений: {len(messages)}, попытка {attempt}/{MAX_RETRIES})…")
            try:
                response = self._get_sync_client().post("/chat/completions", json=payload)
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
                logger.info(f"{log_prefix}: ответ получен ({len(text)} симв., finish={finish or 'stop'})")
                return text, finish

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

    def _complete_with_continuation(self, messages: list, chosen: str, log_prefix: str = "NIM") -> tuple[str, str]:
        """Запрос + автопродолжение, если ответ оборван по max_tokens.

        Возвращает (полный_текст, finish_reason_последнего_запроса).
        """
        text, finish = self._post_chat(messages, chosen, log_prefix)
        if finish != "length" or settings.nim_max_continuations <= 0:
            if finish == "length":
                logger.warning("NIM: ответ обрезан по NIM_MAX_TOKENS — увеличьте NIM_MAX_TOKENS в .env")
            return text, finish

        parts = [text]
        for i in range(1, settings.nim_max_continuations + 1):
            logger.warning(f"NIM: ответ обрезан по max_tokens — автопродолжение {i}/{settings.nim_max_continuations}")
            cont_messages = messages + [
                {"role": "assistant", "content": "".join(parts)},
                {"role": "user", "content": CONTINUATION_PROMPT},
            ]
            tail, finish = self._post_chat(cont_messages, chosen, log_prefix)
            parts.append(_strip_overlap("".join(parts), tail))
            if finish != "length":
                break
        if finish == "length":
            logger.warning(
                f"NIM: ответ остаётся обрезанным после {settings.nim_max_continuations} продолжений — "
                "увеличьте NIM_MAX_TOKENS или NIM_MAX_CONTINUATIONS в .env"
            )
        return "".join(parts), finish

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def send_prompt(
        self,
        prompt: str,
        deepthink: bool = True,
        expert: bool = True,
        attachments: list[str] | None = None,
        stable_rounds: int = 12,
        model: str | None = None,
    ) -> str:
        messages = self._build_messages(prompt, attachments)
        # Vision-модель нужна только когда в запросе ЕСТЬ картинки. Батчи
        # документов (docx/xlsx/pdf-текст) вкладываются извлечённым текстом —
        # гнать их в llama-3.2-90b-vision было расточительно и ловило 400
        # по контексту.
        has_images = any(
            Path(p).suffix.lower() in IMAGE_SUFFIXES for p in (attachments or [])
        )
        chosen = self._vision_model if has_images else (model or self._model)
        text, finish = self._complete_with_continuation(messages, chosen, "NIM")
        if finish == "length":
            text = text + TRUNCATED_MARKER
        return text.strip()

    async def send_prompt_async(
        self,
        prompt: str,
        deepthink: bool = True,
        expert: bool = True,
        attachments: list[str] | None = None,
        stable_rounds: int = 12,
        model: str | None = None,
    ) -> str:
        """Асинхронная обёртка над синхронным клиентом.

        Синхронный путь теперь единственный: общий keep-alive пул + rate limit
        + автопродолжение. Дублировать логику в AsyncClient не нужно —
        поток берётся из пула asyncio.to_thread, блокируя только его.
        """
        return await asyncio.to_thread(
            self.send_prompt,
            prompt,
            deepthink,
            expert,
            attachments,
            stable_rounds,
            model,
        )


client = NimClient()
