import io
import sys
import zipfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from src.outputs import build_zip_bytes, collect_job_files, plain_text


def _make(p: Path, content: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestCollectJobFiles:
    def test_order_report_first_then_transcripts_then_attachments(self, tmp_path):
        report = _make(tmp_path / "output" / "report_1.txt")
        t1 = _make(tmp_path / "t" / "a.transcript.txt")
        t2 = _make(tmp_path / "t" / "b.transcript.txt")
        img = _make(tmp_path / "r" / "c.images.txt")
        result = {
            "report": report,
            "transcripts": [t1, t2],
            "image_reports": [img],
        }
        files = collect_job_files(result)
        assert [f["name"] for f in files] == [
            "report_1.txt", "a.transcript.txt", "b.transcript.txt", "c.images.txt",
        ]
        assert all(f["path"].endswith(f["name"]) for f in files)

    def test_dedupe_same_file(self, tmp_path):
        t = _make(tmp_path / "a.transcript.txt")
        result = {"report": None, "transcripts": [t, t], "image_reports": [t]}
        assert [f["name"] for f in collect_job_files(result)] == ["a.transcript.txt"]

    def test_empty_result(self):
        assert collect_job_files({}) == []
        assert collect_job_files({"report": None, "transcripts": [], "image_reports": []}) == []

    def test_accepts_str_paths(self, tmp_path):
        t = _make(tmp_path / "a.transcript.txt")
        result = {"report": str(t), "transcripts": [], "image_reports": []}
        files = collect_job_files(result)
        assert files == [{"name": "a.transcript.txt", "path": str(t)}]


class TestBuildZip:
    def test_zip_contains_exactly_job_files(self, tmp_path):
        report = _make(tmp_path / "report_20260101.txt", "ОТЧЁТ")
        tr = _make(tmp_path / "h" / "video.transcript.txt", "[00:00] Спикер 1: привет")
        sm = _make(tmp_path / "book_abc.pdf.txt", "конспект")
        data = build_zip_bytes([
            {"name": report.name, "path": str(report)},
            {"name": tr.name, "path": str(tr)},
            {"name": sm.name, "path": str(sm)},
            {"name": "ghost.txt", "path": str(tmp_path / "missing.txt")},  # не существует
        ])
        zf = zipfile.ZipFile(io.BytesIO(data))
        assert sorted(zf.namelist()) == [
            "report_20260101.txt",
            "summaries/book_abc.pdf.txt",
            "transcripts/video.transcript.txt",
        ]
        assert zf.read("report_20260101.txt").decode("utf-8") == "ОТЧЁТ"

    def test_scan_fallback_classifications(self, tmp_path):
        doc = _make(tmp_path / "doc_123.documents.txt", "д")
        scan = _make(tmp_path / "book_scan_abc.txt", "с")
        deck = _make(tmp_path / "deck_abc.ppt.txt", "п")
        data = build_zip_bytes([
            {"name": doc.name, "path": str(doc)},
            {"name": scan.name, "path": str(scan)},
            {"name": deck.name, "path": str(deck)},
        ])
        zf = zipfile.ZipFile(io.BytesIO(data))
        assert sorted(zf.namelist()) == [
            "summaries/book_scan_abc.txt",
            "summaries/deck_abc.ppt.txt",
            "summaries/doc_123.documents.txt",
        ]

    def test_name_collision_gets_suffix(self, tmp_path):
        a = _make(tmp_path / "g1" / "x.transcript.txt", "A")
        b = _make(tmp_path / "g2" / "x.transcript.txt", "B")
        data = build_zip_bytes([
            {"name": a.name, "path": str(a)},
            {"name": b.name, "path": str(b)},
        ])
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = sorted(zf.namelist())
        assert names == ["transcripts/x.transcript.txt", "transcripts/x.transcript_2.txt"]
        contents = {zf.read(n).decode("utf-8") for n in names}
        assert contents == {"A", "B"}  # содержимое не перепуталось

    def test_empty_files_list(self):
        assert build_zip_bytes([]) != b""
        zf = zipfile.ZipFile(io.BytesIO(build_zip_bytes([])))
        assert zf.namelist() == []


class TestPlainText:
    def test_strips_markdown(self):
        md = "# Заголовок\n\n**жирный** и *курсив* и `код`\n\n---\n\n- пункт"
        out = plain_text(md)
        assert "*" not in out
        assert "`" not in out
        assert "#" not in out
        assert "Заголовок" in out
        assert "жирный" in out and "курсив" in out and "код" in out
        assert "пункт" in out
        assert "---" not in out

    def test_html_tags_removed(self):
        assert plain_text("текст <b>жирный</b> конец") == "текст жирный конец"

    def test_links_converted(self):
        out = plain_text("сайт [example](https://example.com) тут")
        assert "example (https://example.com)" in out
        assert "[" not in out and "](" not in out

    def test_images_dropped(self):
        out = plain_text("до ![alt](pic.png) после")
        assert "до  после" == out or "до после" == out

    def test_underscore_in_filenames_preserved(self):
        s = "файл Chumakov_E_M_100_urokov.pdf и 05-5-01-008.pdf сохранён"
        assert plain_text(s) == s

    def test_emphasis_underscores_removed(self):
        assert plain_text("_выделено_ текст") == "выделено текст"

    def test_heading_hashes_stripped_but_text_kept(self):
        assert plain_text("### КЛЮЧЕВЫЕ СОБЫТИЯ\nтекст") == "КЛЮЧЕВЫЕ СОБЫТИЯ\nтекст"

    def test_collapses_blank_lines(self):
        assert plain_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_empty_and_none(self):
        assert plain_text("") == ""
        assert plain_text(None) == ""

    def test_no_changes_for_clean_text(self):
        s = "О ЧЁМ МАТЕРИАЛЫ\n\nОбычный абзац текста.\n\n- пункт списка"
        assert plain_text(s) == s


class TestPlainTextTables:
    def test_markdown_table_flattened(self):
        md = (
            "Подробные данные\n"
            "| Код | Наименование | Цена |\n"
            "|-----|---------------|------|\n"
            "| 47252 | ПОМАДА | 549 |\n"
            "Конец."
        )
        out = plain_text(md)
        assert "|" not in out, out
        assert "-----" not in out
        # разделитель удалён, данные сохранены и читаемы
        assert "Код — Наименование — Цена" in out
        assert "47252 — ПОМАДА — 549" in out

    def test_table_without_edge_spaces(self):
        out = plain_text("|47252|ПОМАДА|549|45,10|")
        assert "|" not in out
        assert "47252 — ПОМАДА — 549 — 45,10" in out

    def test_table_separator_row_dropped(self):
        out = plain_text("a | b\n| :---: | --- |\nпроект")
        assert "|" not in out
        assert "---" not in out
        assert "a — b" in out
        assert "проект" in out

    def test_plain_pipe_free_text_untouched(self):
        s = "обычный текст без труб"
        assert plain_text(s) == s


class TestPlainTextEmoji:
    def test_keycap_digit_kept(self):
        assert plain_text("1️⃣ Новинки каталога") == "1 Новинки каталога"

    def test_emoji_removed(self):
        out = plain_text("Прайс ✅ готов ⭐➿")
        assert "✅" not in out and "⭐" not in out and "➿" not in out
        assert "Прайс" in out and "готов" in out

    def test_flag_removed(self):
        out = plain_text("страна 🇷🇺 производитель")
        assert "🇷🇺" not in out
        assert "страна" in out and "производитель" in out

    def test_legit_symbols_kept(self):
        s = "Цена 549 ₽, № 12, ≈45%, диапазон 100—200, «кавычки»"
        out = plain_text(s)
        for sym in ("₽", "№", "≈", "—", "«"):
            assert sym in out


class TestPlainTextBullets:
    def test_bullet_replaced_with_dash(self):
        assert plain_text("• пункт первый") == "- пункт первый"

    def test_bullet_variants(self):
        for ch in ("•", "‣", "●", "▪", "○", "◦", "·"):
            out = plain_text(f"{ch} пункт")
            assert out == "- пункт", f"буллет {ch!r} не заменён: {out}"

    def test_indented_bullet(self):
        # отступ не первой строки сохраняется (общий .strip() трогает только края)
        assert plain_text("Заголовок\n  • вложенный пункт") == "Заголовок\n  - вложенный пункт"

    def test_mid_line_bullet_char_kept(self):
        # «·» внутри строки (десятичный разделитель) не трогается
        s = "скидка ≈ 0·34"
        assert plain_text(s) == s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
