import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.rtf_text import rtf_to_text


def _cp1251(text: str) -> str:
    """Текст → RTF-последовательность \'hh (байты cp1251)."""
    return "".join(f"\\'{b:02x}" for b in text.encode("cp1251"))


HEADER = r"{\rtf1\ansi\ansicpg1251"


class TestCp1251:
    def test_cyrillic_hex_escapes(self):
        rtf = HEADER + r"{\fonttbl{\f0 Times New Roman;}}\f0\fs24 " + _cp1251("Привет, мир!") + "}"
        assert rtf_to_text(rtf) == "Привет, мир!"

    def test_ansicpg_1251_explicit(self):
        rtf = r"{\rtf1\ansi\ansicpg1251 " + _cp1251("Да") + "}"
        assert rtf_to_text(rtf) == "Да"

    def test_ansicpg_1252_western(self):
        # é в cp1252 = 0xE9 (в cp1251 это «й»)
        rtf = r"{\rtf1\ansi\ansicpg1252 caf\'e9}"
        assert rtf_to_text(rtf) == "café"

    def test_default_codepage_1251_without_ansicpg(self):
        rtf = r"{\rtf1\ansi " + _cp1251("Тест") + "}"
        assert rtf_to_text(rtf) == "Тест"

    def test_cyrillic_multigroup(self):
        rtf = HEADER + "{\\b " + _cp1251("Жирный") + "} и {\\i " + _cp1251("курсив") + "}"
        assert rtf_to_text(rtf) == "Жирный и курсив"


class TestUnicode:
    def test_unicode_escapes_with_fallback(self):
        # М = 1052, о = 1086; «?» — ANSI-fallback, пропускается
        assert rtf_to_text(r"\u1052?\u1086?р") == "Мор"

    def test_unicode_cyrillic_word(self):
        rtf = "".join(f"\\u{ord(c)}?" for c in "Привет")
        assert rtf_to_text(rtf) == "Привет"

    def test_negative_unicode(self):
        # -123 + 65536 = 65413; «?» — fallback (съедается), «x» — обычный текст
        assert rtf_to_text(r"\u-123?x") == chr(65413) + "x"

    def test_uc0_no_fallback(self):
        assert rtf_to_text(r"{\uc0\u1052}") == "М"

    def test_hh_fallback_skipped(self):
        # fallback после \uN — \'hh-последовательность
        rtf = r"\u1044\'cf"
        assert rtf_to_text(rtf) == "Д"


class TestSkipGroups:
    def test_fonttbl_colortbl_skipped(self):
        rtf = (
            HEADER
            + r"{\fonttbl{\f0\froman Times New Roman;}{\f1\fswiss Arial;}}"
            + r"{\colortbl;\red0\green0\blue0;\red255\green0\blue0;}"
            + r"{\stylesheet{\s1 Стиль;}}"
            + r"{\info{\title Внутренний заголовок}}"
            + r"\f0\fs24 " + _cp1251("Текст документа")
            + "}"
        )
        out = rtf_to_text(rtf)
        assert out == "Текст документа"
        assert "Times" not in out and "Arial" not in out
        assert "red" not in out and "Стиль" not in out
        assert "заголовок" not in out

    def test_pict_binary_skipped(self):
        rtf = (
            HEADER
            + r"{\pict{\*\picprop\plblayout\blishp}\pngblip"
            + "89504e470d0a1a0a0000000d49484452"
            + "}"
            + r"\par " + _cp1251("После картинки")
            + "}"
        )
        out = rtf_to_text(rtf)
        assert "89504e" not in out
        assert "picprop" not in out
        assert "После картинки" in out

    def test_object_skipped(self):
        rtf = HEADER + r"{\object\objemb{\*\objdata 00ff00ff}}\par " + _cp1251("Готово") + "}"
        out = rtf_to_text(rtf)
        assert "objdata" not in out and "00ff00ff" not in out
        assert "Готово" in out

    def test_header_footer_skipped(self):
        rtf = (
            HEADER
            + r"{\header\pard\qr " + _cp1251("Колонтитул") + r"\par}"
            + r"{\footer\pard\qc " + _cp1251("Страница") + r"\par}"
            + r"\pard " + _cp1251("Основной текст")
            + "}"
        )
        out = rtf_to_text(rtf)
        assert out == "Основной текст"


class TestStarDestinationAndBin:
    def test_star_group_skipped(self):
        rtf = HEADER + r"{\*\generator Riched20 10.0.19041}\par " + _cp1251("Слово") + "}"
        out = rtf_to_text(rtf)
        assert "Riched20" not in out
        assert "Слово" in out

    def test_star_nested_braces_counted(self):
        rtf = HEADER + r"{\*\company{Вложенные {двойные} скобки}} " + _cp1251("Ядро") + "}"
        assert rtf_to_text(rtf) == "Ядро"

    def test_bin_data_skipped(self):
        rtf = HEADER + r"{\*\datastore\bin 8 01234567}" + _cp1251("Финал") + "}"
        out = rtf_to_text(rtf)
        assert "01234567" not in out
        assert "Финал" in out


class TestEscaping:
    def test_escaped_braces(self):
        # В RTF литеральные скобки экранируются одним слэшем: \{ \}
        assert rtf_to_text(r"code \{x\}") == r"code {x}"

    def test_escaped_backslash(self):
        assert rtf_to_text(r"C:\\temp") == r"C:\temp"

    def test_tilde_nbsp(self):
        assert rtf_to_text(r"раз\~два") == "раз два"

    def test_minus_underscore(self):
        assert rtf_to_text(r"a\-b\_c") == "a-b-c"

    def test_quotes_and_dashes(self):
        # пробел после control word — разделитель (часть слова), не текст
        assert rtf_to_text(r"\lquoteв\rquote \endash\emdash") == "‘в’–—"


class TestParLineTab:
    def test_par_newline(self):
        out = rtf_to_text(HEADER + _cp1251("Строка раз") + r"\par " + _cp1251("Строка два") + "}")
        assert out == "Строка раз\nСтрока два"

    def test_line_newline(self):
        out = rtf_to_text(r"{\rtf1 a\line b}")
        assert out == "a\nb"

    def test_tab(self):
        assert rtf_to_text(r"{\rtf1 имя\tab значение}") == "имя\tзначение"

    def test_blank_lines_collapsed(self):
        out = rtf_to_text(r"{\rtf1 a\par\par\par\par b}")
        assert out == "a\n\nb"

    def test_trailing_whitespace_stripped(self):
        assert rtf_to_text(r"{\rtf1 \par " + _cp1251("Текст") + r"\par\par}") == "Текст"


class TestEdgeCases:
    def test_empty(self):
        assert rtf_to_text("") == ""
        assert rtf_to_text(r"{\rtf1}") == ""

    def test_unknown_control_words_ignored(self):
        out = rtf_to_text(HEADER + r"\pard\plain\fs24\lang1049 " + _cp1251("Чисто") + "}")
        assert out == "Чисто"

    def test_text_without_escapes(self):
        assert rtf_to_text(r"{\rtf1 plain text}") == "plain text"

    def test_no_cr_in_output(self):
        out = rtf_to_text(HEADER + r"\par " + _cp1251("а") + r"\par " + _cp1251("б") + "}")
        assert "\r" not in out


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
