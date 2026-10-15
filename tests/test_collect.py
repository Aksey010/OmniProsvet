import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import src.collect as collect_mod
from src.collect import collect_media, is_document, is_skipped, _iter_files, _skip_patterns


class TestCollectMedia:
    def test_collect_media_with_temp_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Create test files
            (tmp / "audio.mp3").write_text("dummy")
            (tmp / "video.mp4").write_text("dummy")
            (tmp / "image.jpg").write_text("dummy")
            (tmp / "doc.pdf").write_text("dummy")
            (tmp / "text.txt").write_text("dummy")
            (tmp / "ignore.transcript.txt").write_text("dummy")
            (tmp / "report_123.txt").write_text("dummy")

            audio, video, images, documents = collect_media(str(tmp))

            assert len(audio) == 1
            assert audio[0].name == "audio.mp3"
            assert len(video) == 1
            assert video[0].name == "video.mp4"
            assert len(images) == 1
            assert images[0].name == "image.jpg"
            # PDF and txt are documents, but transcript and report files are excluded
            assert len(documents) == 2
            doc_names = {d.name for d in documents}
            assert "doc.pdf" in doc_names
            assert "text.txt" in doc_names
            assert "ignore.transcript.txt" not in doc_names
            assert "report_123.txt" not in doc_names

    def test_collect_media_nonexistent_path(self):
        audio, video, images, documents = collect_media("/nonexistent/path")
        assert audio == []
        assert video == []
        assert images == []
        assert documents == []

    def test_collect_media_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            f = tmp / "test.mp3"
            f.write_text("dummy")
            audio, video, images, documents = collect_media(str(f))
            assert len(audio) == 1
            assert audio[0] == f


class TestSkipStickers:
    """Стикеры исключаются на этапе collect_media — до транскрибации и vision.

    В сессии 13.10: 274 .webm-стикера падали в транскрибации («tuple index
    out of range»), 383 .webp-стикера занимали запросы vision-модели.
    """

    def _make_tree(self, tmp: Path) -> None:
        for name in (
            "1785_sticker.webm",           # видео-стикер
            "29033_sticker.webm",
            "123_sticker.webp",            # картинка-стикер
            "456_sticker.png",
            "2769_animation.gif.mp4",      # анимированный стикер
            "audio.mp3",                   # обычные файлы не тронуты
            "video.mp4",
            "photo.jpg",
        ):
            (tmp / name).write_text("dummy")

    def test_stickers_not_collected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_tree(tmp)
            audio, video, images, documents = collect_media(str(tmp))
            assert audio == [tmp / "audio.mp3"]
            assert video == [tmp / "video.mp4"]
            assert images == [tmp / "photo.jpg"]
            assert documents == []

    def test_sticker_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "1_Sticker.webm").write_text("x")
            (tmp / "2_STICKER.webp").write_text("x")
            _, video, images, _ = collect_media(str(tmp))
            assert video == []
            assert images == []

    def test_documents_with_sticker_word_kept(self):
        """Фильтр не прячет содержательные документы со 'sticker' в имени."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "sticker_pack_brief.pdf").write_text("x")
            (tmp / "notes.txt").write_text("x")
            _, _, _, documents = collect_media(str(tmp))
            assert sorted(d.name for d in documents) == ["notes.txt", "sticker_pack_brief.pdf"]

    def test_custom_patterns(self, monkeypatch):
        monkeypatch.setattr(
            collect_mod, "settings", SimpleNamespace(skip_file_patterns="sticker")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "1_sticker.webm").write_text("x")
            (tmp / "2_animation.gif.mp4").write_text("x")
            _, video, _, _ = collect_media(str(tmp))
            # шаблон 'sticker' один: анимация НЕ отфильтрована
            assert [v.name for v in video] == ["2_animation.gif.mp4"]

    def test_empty_patterns_disable_filter(self, monkeypatch):
        monkeypatch.setattr(
            collect_mod, "settings", SimpleNamespace(skip_file_patterns="")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "1_sticker.webm").write_text("x")
            _, video, _, _ = collect_media(str(tmp))
            assert len(video) == 1

    def test_single_sticker_file_path_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            f = tmp / "9_sticker.webm"
            f.write_text("x")
            audio, video, images, documents = collect_media(str(f))
            assert audio == [] and video == [] and images == [] and documents == []

    def test_notify_reports_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_tree(tmp)
            seen: list[list[Path]] = []
            collect_media(str(tmp), notify=seen.append)
            assert len(seen) == 1
            names = {f.name for f in seen[0]}
            assert names == {
                "1785_sticker.webm", "29033_sticker.webm",
                "123_sticker.webp", "456_sticker.png",
                "2769_animation.gif.mp4",
            }

    def test_is_skipped_direct(self):
        assert is_skipped(Path("a/b/1_sticker.webm")) is True
        assert is_skipped(Path("a/b/1_animation.gif.mp4")) is True
        assert is_skipped(Path("a/b/holiday.mp4")) is False
        # регистр не мешает
        assert is_skipped(Path("1_Sticker.webm"), ("sticker",)) is True
        assert is_skipped(Path("x.mp4"), ()) is False

    def test_patterns_parsed(self, monkeypatch):
        monkeypatch.setattr(
            collect_mod, "settings", SimpleNamespace(skip_file_patterns=" Sticker , animation.gif ,")
        )
        assert _skip_patterns() == ("sticker", "animation.gif")


class TestIsDocument:
    def test_excludes_transcript_files(self):
        assert is_document(Path("test.transcript.txt")) is False
        assert is_document(Path("test.transcript.md")) is False

    def test_excludes_report_files(self):
        assert is_document(Path("report_123.txt")) is False

    def test_excludes_image_report_files(self):
        assert is_document(Path("test.images.txt")) is False
        assert is_document(Path("test.documents.txt")) is False

    def test_excludes_summary_outputs(self):
        # Выходные конспекты per-file обработки не являются входными документами
        assert is_document(Path("book_abc123.pdf.txt")) is False
        assert is_document(Path("deck_abc123.ppt.txt")) is False
        assert is_document(Path("deck_abc123.pdf.txt")) is False

    def test_includes_valid_documents(self):
        assert is_document(Path("test.pdf")) is True
        assert is_document(Path("test.docx")) is True
        assert is_document(Path("test.txt")) is True
        assert is_document(Path("test.csv")) is True


class TestIterFiles:
    def test_skips_hidden_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "visible.txt").write_text("a")
            hidden = tmp / ".git"
            hidden.mkdir()
            (hidden / "file.txt").write_text("b")
            venv = tmp / "venv"
            venv.mkdir()
            (venv / "file.txt").write_text("c")

            files = _iter_files(str(tmp))
            names = {f.name for f in files}
            assert "visible.txt" in names
            # .git and venv should be skipped
            assert len(files) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])