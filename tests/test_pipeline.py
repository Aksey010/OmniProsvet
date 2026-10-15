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
        assert "analysis" in weights
        assert "images" in weights
        assert "pdf" in weights
        assert sum(weights.values()) == 100

    def test_weights_for_extract_only(self):
        weights = _weights_for(extract=True, process=False, att_kinds=[])
        assert weights == {"transcribe": 100}

    def test_weights_for_process_only(self):
        weights = _weights_for(extract=False, process=True, att_kinds=[])
        assert weights == {"analysis": 100}

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


class TestPresentationRouting:
    @patch("src.pipeline.run_presentations")
    @patch("src.pipeline.run_attachments")
    @patch("src.pipeline.run_pdfs")
    def test_pptx_and_ppt_routed_to_presentations(
        self, mock_run_pdfs, mock_run_att, mock_run_pres, tmp_path, monkeypatch
    ):
        """ppt/pptx идут в run_presentations (как pdf -> run_pdfs); docx — в documents."""
        import src.collect as collect_mod
        from src.pipeline import run_pipeline

        pptx = tmp_path / "deck.pptx"
        ppt = tmp_path / "legacy.ppt"
        docx = tmp_path / "doc.docx"
        for f in (pptx, ppt, docx):
            f.write_text("x")

        monkeypatch.setattr(
            collect_mod, "collect_media", lambda p, notify=None: ([], [], [], [pptx, ppt, docx])
        )
        mock_run_pres.return_value = [tmp_path / "deck.ppt.txt"]
        mock_run_att.return_value = []
        mock_run_pdfs.return_value = []

        result = run_pipeline(
            [str(tmp_path)], extract=False, process=False, images=False, progress=lambda e: None
        )

        assert mock_run_pres.called, "презентации должны обрабатываться run_presentations"
        pres_files = mock_run_pres.call_args[0][0]
        assert set(pres_files) == {pptx, ppt}
        assert mock_run_att.called
        assert mock_run_att.call_args[0][0] == [docx]
        assert mock_run_att.call_args[0][1] == "documents"
        assert not mock_run_pdfs.called
        assert result["image_reports"] == [tmp_path / "deck.ppt.txt"]


class TestSpreadsheetRouting:
    @patch("src.pipeline.run_spreadsheets")
    @patch("src.pipeline.run_attachments")
    @patch("src.pipeline.run_presentations")
    @patch("src.pipeline.run_pdfs")
    def test_xlsx_xls_xlsm_routed_to_spreadsheets(
        self, mock_run_pdfs, mock_run_pres, mock_run_att, mock_run_sheets, tmp_path, monkeypatch
    ):
        """xlsx/xlsm/xls идут в run_spreadsheets; docx остаётся в documents."""
        import src.collect as collect_mod
        from src.pipeline import run_pipeline

        xlsx = tmp_path / "price.xlsx"
        xlsm = tmp_path / "book.xlsm"
        xls = tmp_path / "legacy.xls"
        docx = tmp_path / "doc.docx"
        for f in (xlsx, xlsm, xls, docx):
            f.write_text("x")

        monkeypatch.setattr(
            collect_mod, "collect_media", lambda p, notify=None: ([], [], [], [xlsx, xlsm, xls, docx])
        )
        mock_run_sheets.return_value = [tmp_path / "price_abc.xlsx.txt"]
        mock_run_att.return_value = []
        mock_run_pres.return_value = []
        mock_run_pdfs.return_value = []

        result = run_pipeline(
            [str(tmp_path)], extract=False, process=False, images=False, progress=lambda e: None
        )

        assert mock_run_sheets.called, "таблицы должны обрабатываться run_spreadsheets"
        sheet_files = mock_run_sheets.call_args[0][0]
        assert set(sheet_files) == {xlsx, xlsm, xls}
        assert mock_run_att.called
        assert mock_run_att.call_args[0][0] == [docx]
        assert mock_run_att.call_args[0][1] == "documents"
        assert not mock_run_pres.called
        assert not mock_run_pdfs.called
        assert result["image_reports"] == [tmp_path / "price_abc.xlsx.txt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])