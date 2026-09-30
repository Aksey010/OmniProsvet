import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from src.pipeline import _weights_for, _group_names


class TestPipelineHelpers:
    def test_weights_for_extract_and_process(self):
        weights = _weights_for(extract=True, process=True, att_kinds=["images", "pdf"])
        assert "transcribe" in weights
        assert "deepseek" in weights
        assert "images" in weights
        assert "pdf" in weights
        assert sum(weights.values()) == 100

    def test_weights_for_extract_only(self):
        weights = _weights_for(extract=True, process=False, att_kinds=[])
        assert weights == {"transcribe": 100}

    def test_weights_for_process_only(self):
        weights = _weights_for(extract=False, process=True, att_kinds=[])
        assert weights == {"deepseek": 100}

    def test_weights_for_attachments_only(self):
        weights = _weights_for(extract=False, process=False, att_kinds=["images", "documents"])
        assert "images" in weights
        assert "documents" in weights
        assert sum(weights.values()) == 100

    def test_group_names_unique(self):
        paths = [
            Path("/tmp/test"),
            Path("/tmp/test"),  # duplicate
            Path("/other/test"),
        ]
        names = _group_names(paths)
        assert len(names) == 2
        # Second should get hash suffix
        assert "test" in names.values() or "test_" in "".join(names.values())

    def test_group_names_sanitizes(self):
        paths = [Path("/tmp/my file (1).mp3")]
        names = _group_names(paths)
        name = list(names.values())[0]
        # Should only contain alphanumeric, space, underscore, dash
        for c in name:
            assert c.isalnum() or c in " _-"


class TestPipelineIntegration:
    @patch("src.pipeline.collect_media")
    @patch("src.pipeline.run_extract_and_process")
    @patch("src.pipeline.run_attachments")
    @patch("src.pipeline.run_pdfs")
    def test_run_pipeline_extract_and_process(
        self, mock_run_pdfs, mock_run_attachments, mock_run_extract, mock_collect
    ):
        from src.pipeline import run_pipeline

        mock_collect.return_value = ([], [], [], [])
        mock_run_extract.return_value = ([Path("transcript.txt")], Path("report.txt"))
        mock_run_attachments.return_value = []
        mock_run_pdfs.return_value = []

        result = run_pipeline(
            ["/fake/path"],
            extract=True,
            process=True,
            images=True,
            progress=lambda x: None,
        )

        assert "transcripts" in result
        assert "report" in result
        assert "image_reports" in result

    @patch("src.pipeline.collect_media")
    @patch("src.pipeline.run_extract")
    def test_run_pipeline_extract_only(self, mock_run_extract, mock_collect):
        from src.pipeline import run_pipeline

        mock_collect.return_value = ([Path("audio.mp3")], [], [], [])
        mock_run_extract.return_value = [Path("transcript.txt")]

        result = run_pipeline(
            ["/fake/path"],
            extract=True,
            process=False,
            images=False,
            progress=lambda x: None,
        )

        assert len(result["transcripts"]) == 1
        assert result["report"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])