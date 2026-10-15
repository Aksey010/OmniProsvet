"""Учёт прогресса для пайплайна с параллельными фазами.

Прогресс-бар собирается не по «текущей» фазе, а по всем фазам сразу:
    percent() = Σ весов завершённых фаз + Σ (продвижение × вес) по активным.

Это устраняет «пинг-понг» процента, когда параллельные потоки
(транскрибация, картинки, PDF, анализ) по очереди перетягивают «current phase».
"""

import threading


# ---------------------------------------------------------------------------
# Разграничение фаз — общие метки для веб-панели и CLI
# ---------------------------------------------------------------------------

# Человекочитаемые названия фаз. Фазы идут параллельно и раньше перебивали
# друг друга в едином поле «current»; теперь у каждой фазы своя надпись.
PHASE_LABELS = {
    "transcribe": "Транскрибация",
    "pdf": "PDF",
    "ppt": "Презентации",
    "spreadsheets": "Таблицы",
    "images": "Картинки",
    "documents": "Документы",
    "analysis": "Анализ",
}

# Порядок отображения надписей: основная фаза, анализ, затем параллельные вложения
PHASE_ORDER = ["transcribe", "analysis", "pdf", "ppt", "spreadsheets", "images", "documents"]


def phase_label(phase: str) -> str:
    return PHASE_LABELS.get(phase, phase or "")


def format_file_event(event: dict) -> str | None:
    """Строка прогресса для CLI из события пайплайна (None, если это не файл).

    Формат: «Транскрибация: [36/4000] имя.mp3» — метка фазы разграничивает
    параллельные процессы друг от друга.
    """
    if "file" not in event:
        return None
    label = phase_label(event.get("phase", ""))
    prefix = f"{label}: " if label else ""
    idx = event.get("file_index")
    total = event.get("file_total")
    counter = f"[{idx}/{total}] " if idx and total else ""
    return f"{prefix}{counter}{event['file']}".strip()


class PhaseProgress:
    """Проценты по всем фазам сразу; поддерживает параллельные фазы.

    События feed():
      {"phase": name, "weight": w}      — анонс фазы (регистрирует вес, идемпотентно)
      {"phase_progress": x, "p": name} — продвижение фазы, 0..1
      {"phase_done": name}              — фаза завершена: вес засчитывается целиком, один раз

    Продвижение фазы монотонно (откаты игнорируются) — устойчиво к гонкам
    параллельных воркеров, рапортующих свои доли в произвольном порядке.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._weights: dict[str, float] = {}
        self._fracs: dict[str, float] = {}
        self._done: set[str] = set()

    def feed(self, event: dict) -> None:
        with self._lock:
            if "phase" in event:
                name = event["phase"]
                self._weights.setdefault(name, float(event.get("weight", 0) or 0))
                self._fracs.setdefault(name, 0.0)
            elif "phase_done" in event:
                self._done.add(event["phase_done"])
            elif "phase_progress" in event:
                name = event.get("p")
                if name is None or name in self._done:
                    return
                val = max(0.0, min(1.0, float(event["phase_progress"])))
                if val > self._fracs.get(name, 0.0):
                    self._fracs[name] = val

    def percent(self) -> int:
        with self._lock:
            total = 0.0
            for name, weight in self._weights.items():
                if name in self._done:
                    total += weight
                else:
                    total += self._fracs.get(name, 0.0) * weight
            return min(100, round(total))


# Обратная совместимость с прежним именем в web_panel
_PhaseProgress = PhaseProgress


class FilesProgress:
    """Агрегированный прогресс N параллельных файлов одной фазы.

    Каждый воркер рапортует процент (0..100) своего файла; продвижение фазы =
    среднее по всем файлам. Файлы могут завершаться в любом порядке —
    суммарный прогресс монотонен (пила «по индексу» исчезает).
    """

    def __init__(self, total: int, phase: str = "transcribe") -> None:
        self._lock = threading.Lock()
        self._total = max(1, int(total))
        self._pcts: list[float] = [0.0] * self._total
        self._phase = phase

    def reporter(self, idx: int, emit):
        """Callback(pct) для файла с порядковым номером idx (1-based).

        emit принимает событие прогресса ({"p": phase, "phase_progress": frac}).
        """
        if idx < 1 or idx > self._total:
            raise IndexError(f"idx {idx} вне диапазона 1..{self._total}")

        def _on_progress(pct) -> None:
            val = max(0.0, min(1.0, float(pct) / 100.0))
            with self._lock:
                if val > self._pcts[idx - 1]:
                    self._pcts[idx - 1] = val
                frac = sum(self._pcts) / self._total
            if emit:
                emit({"p": self._phase, "phase_progress": frac})

        return _on_progress


_FilesProgress = FilesProgress
