import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from src.collect import collect_media, is_document, _iter_files


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


class TestIsDocument:
    def test_excludes_transcript_files(self):
        assert is_document(Path("test.transcript.txt")) is False
        assert is_document(Path("test.transcript.md")) is False

    def test_excludes_report_files(self):
        assert is_document(Path("report_123.txt")) is False

    def test_excludes_image_report_files(self):
        assert is_document(Path("test.images.txt")) is False
        assert is_document(Path("test.documents.txt")) is False

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