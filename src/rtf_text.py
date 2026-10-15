r"""RTF → plain text: собственный декодер без сторонних зависимостей.

Наивная замена регэкспом не работает для русских документов: кириллица в RTF
хранится как escape-последовательности \'hh (байты кодовой страницы, обычно
cp1251), а юникод — как \uN. Декодер группами ({}) с учётом вложенности:

- декодирует \'hh в кодировке из \ansicpgN (по умолчанию 1251);
- декодирует \uN (signed 16-bit) и пропускает ANSI-fallback после него
  (количество fallback-символов — \ucN, по умолчанию 1; fallback может быть
  и \'hh-последовательностью);
- пропускает служебные destination-группы (fonttbl, colortbl, stylesheet,
  info, pict, object, header/footer, listtable и др.) и любые группы \*;
- пропускает бинарные данные после \bin N;
- декодирует \par/\line → перевод строки, \tab → табуляция, спецкавычки и
  тире; литеральные \\, \{, \};
- неизвестные control words игнорируются (вместе с параметром и разделителем).
"""

import codecs

# Кодовая страница по умолчанию (ANSI; русские документы почти всегда 1251)
DEFAULT_CODEPAGE = 1251

# Destination-группы, содержимое которых в текст не попадает
SKIP_DESTINATIONS = {
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "object",
    "header", "headerl", "headerr", "headerf",
    "footer", "footerl", "footerr", "footerf",
    "listtable", "listoverridetable", "rsidtbl", "generator",
    "themedata", "colorschememapping", "datastore", "xmlnstbl",
    "latentstyles", "filetbl", "revtbl", "pgptbl", "faltcontent",
}

_HEX = set("0123456789abcdefABCDEF")

_SPECIAL_SYMBOLS = {
    "emdash": "—",
    "endash": "–",
    "bullet": "•",
    "lquote": "‘",
    "rquote": "’",
    "ldblquote": "«",
    "rdblquote": "»",
}


def _codepage_encoding(n: int | None) -> str:
    if not n:
        return f"cp{DEFAULT_CODEPAGE}"
    try:
        codecs.lookup(f"cp{n}")
        return f"cp{n}"
    except LookupError:
        return f"cp{DEFAULT_CODEPAGE}"


def _is_alpha(ch: str) -> bool:
    # Control words RTF — только ASCII-буквы (Cyrillic .isalpha() не годится)
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def rtf_to_text(raw: str, default_codepage: int = DEFAULT_CODEPAGE) -> str:
    """Декодировать RTF-разметку в plain text."""
    out: list[str] = []
    byte_buf = bytearray()
    codepage = default_codepage
    uc = 1  # уровень ANSI-fallback для \uN (меняется словом \ucN)
    depth = 0
    skip_depth: int | None = None  # None — не пропускаем; иначе глубина выхода
    i = 0
    n = len(raw)

    def flush_bytes() -> None:
        if byte_buf:
            out.append(byte_buf.decode(_codepage_encoding(codepage), errors="ignore"))
            byte_buf.clear()

    while i < n:
        ch = raw[i]

        # --- режим пропуска группы: считаем скобки, экранирование уважаем ---
        if skip_depth is not None:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < skip_depth:
                    skip_depth = None
            elif ch == "\\":
                nxt = raw[i + 1] if i + 1 < n else ""
                if nxt in "{}\\":
                    i += 2
                    continue
                j = i + 1
                if nxt == "'" and i + 3 < n:
                    i += 4  # \'hh
                    continue
                if _is_alpha(nxt):
                    while j < n and _is_alpha(raw[j]):
                        j += 1
                    if j < n and raw[j] == "-":
                        j += 1
                    while j < n and raw[j].isdigit():
                        j += 1
                    if j < n and raw[j] == " ":
                        j += 1
                else:
                    j += 1  # одиночный символ (\~, \- и т.п.)
                i = j
                continue
            i += 1
            continue

        # --- обычный режим ---
        if ch == "{":
            flush_bytes()
            depth += 1
            i += 1
            continue
        if ch == "}":
            flush_bytes()
            depth -= 1
            i += 1
            continue
        if ch == "\\":
            flush_bytes()
            nxt = raw[i + 1] if i + 1 < n else ""
            if nxt in "{}\\":
                out.append(nxt)
                i += 2
                continue
            if nxt == "~":
                out.append(" ")  # неразрывный пробел → обычный
                i += 2
                continue
            if nxt in "-_":
                out.append("-")
                i += 2
                continue
            if nxt == "'":
                hh = raw[i + 2:i + 4]
                if len(hh) == 2 and hh[0] in _HEX and hh[1] in _HEX:
                    byte_buf.append(int(hh, 16))
                    i += 4
                    continue
                out.append(nxt)
                i += 2
                continue
            if nxt == "*":
                # \* — служебная/неизвестная destination-группа: пропустить всю.
                # Глубина выхода = ТЕКУЩЕЙ глубине: слово уже внутри своей группы.
                skip_depth = depth
                i += 2
                continue
            if nxt == "u" and (raw[i + 2:i + 3].isdigit() or raw[i + 2:i + 3] == "-"):
                # \uN — юникод (signed 16-bit), затем uc fallback-символов.
                # (Без цифр после \u это другое слово — \ucN/\ulongauto: общий путь.)
                j = i + 2
                sign = 1
                if j < n and raw[j] == "-":
                    sign = -1
                    j += 1
                k = j
                while k < n and raw[k].isdigit():
                    k += 1
                val = sign * int(raw[j:k])
                if val < 0:
                    val += 65536
                if 0 <= val <= 0x10FFFF:
                    out.append(chr(val))
                m = k
                for _ in range(max(0, uc)):
                    if m >= n:
                        break
                    if raw[m:m + 2] == "\\'":
                        m += 4  # fallback \'hh
                    elif raw[m] not in "{}":  # скобки — структура, не fallback
                        m += 1
                    else:
                        break
                i = m
                continue
            if raw[i + 1:i + 4] == "bin":
                # \bin N — пропустить N байт бинарных данных
                j = i + 4
                while j < n and raw[j] == " ":
                    j += 1
                k = j
                while k < n and raw[k].isdigit():
                    k += 1
                if k > j:
                    count = int(raw[j:k])
                    if k < n and raw[k] == " ":
                        k += 1
                    i = min(n, k + count)
                    continue
            if _is_alpha(nxt):
                # управляющее слово: буквы + параметр + разделительный пробел
                k = i + 1
                while k < n and _is_alpha(raw[k]):
                    k += 1
                word = raw[i + 1:k]
                p = k
                sign = 1
                if p < n and raw[p] == "-":
                    sign = -1
                    p += 1
                while p < n and raw[p].isdigit():
                    p += 1
                param = sign * int(raw[k:p]) if p > k else None
                end = p
                if p < n and raw[p] == " ":
                    end = p + 1  # разделитель — часть control word

                if word == "ansicpg" and param:
                    codepage = param
                elif word == "uc" and param is not None:
                    uc = param
                elif word in ("par", "line"):
                    out.append("\n")
                elif word == "tab":
                    out.append("\t")
                elif word in _SPECIAL_SYMBOLS:
                    out.append(_SPECIAL_SYMBOLS[word])
                elif word in SKIP_DESTINATIONS:
                    # Глубина выхода = текущей: слово уже внутри группы
                    skip_depth = depth

                i = end
                continue
            # одиночный неизвестный символ после \
            i += 2
            continue

        out.append(ch)
        i += 1

    flush_bytes()

    # нормализация: убрать \r, схлопнуть пустые строки, обрезать пробелы строк
    text = "".join(out).replace("\r", "")
    lines: list[str] = []
    blank = 0
    for ln in text.split("\n"):
        ln = ln.rstrip()
        if not ln:
            blank += 1
            if blank == 1:
                lines.append("")
        else:
            blank = 0
            lines.append(ln)
    return "\n".join(lines).strip()
