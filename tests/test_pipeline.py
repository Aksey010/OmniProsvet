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


class TestRawPipeline:
    @patch("src.raw_extract.run_raw_extract")
    @patch("src.pipeline.run_attachments")
    @patch("src.pipeline.run_pdfs")
    @patch("src.pipeline.run_presentations")
    @patch("src.pipeline.run_spreadsheets")
    @patch("src.pipeline.run_extract")
    @patch("src.collect.collect_media")  # run_pipeline импортирует его локально из src.collect
    def test_raw_mode_routes_to_local_extraction(
        self, mock_collect, mock_run_extract, mock_spreadsheets,
        mock_presentations, mock_pdfs, mock_att, mock_raw, tmp_path,
    ):
        """raw-режим: все вложения идут в локальное извлечение; NIM-фазы и
        анализ не вызываются вовсе."""
        from src.pipeline import run_pipeline

        doc = tmp_path / "doc.docx"
        doc.write_text("x")
        img = tmp_path / "p.jpg"
        img.write_bytes(b"jpg")
        mock_collect.return_value = ([], [], [img], [doc])
        raw_out = tmp_path / "doc_abc.docx.raw.txt"
        raw_out.write_text("извлечено", encoding="utf-8")
        mock_raw.return_value = [raw_out]
        mock_run_extract.return_value = []

        events: list[dict] = []
        result = run_pipeline(
            [str(tmp_path)], extract=True, process=False, images=True, raw=True,
            progress=events.append,
        )

        mock_raw.assert_called_once()
        # все файлы (документы + картинки) ушли в raw-извлечение
        raw_files_arg = mock_raw.call_args.args[0]
        assert sorted(p.name for p in raw_files_arg) == ["doc.docx", "p.jpg"]
        # NIM-фазы не вызывались
        mock_att.assert_not_called()
        mock_pdfs.assert_not_called()
        mock_presentations.assert_not_called()
        mock_spreadsheets.assert_not_called()
        # транскрибация выполнена, анализ отсутствует
        mock_run_extract.assert_called_once()
        assert result["report"] is None
        assert result["raw_files"] == [raw_out]
        # фаза raw закрыта
        assert {"phase_done": "raw"} in events

    @patch("src.raw_extract.run_raw_extract")
    @patch("src.pipeline.run_extract")
    @patch("src.collect.collect_media")
    def test_raw_respects_images_false(
        self, mock_collect, mock_run_extract, mock_raw, tmp_path,
    ):
        """--no-images в raw-режиме: картинки исключены из извлечения."""
        from src.pipeline import run_pipeline

        doc = tmp_path / "doc.docx"
        doc.write_text("x")
        img = tmp_path / "p.jpg"
        img.write_bytes(b"jpg")
        mock_collect.return_value = ([], [], [img], [doc])
        mock_raw.return_value = []
        mock_run_extract.return_value = []

        run_pipeline([str(tmp_path)], extract=True, process=False, images=False, raw=True)

        raw_files_arg = mock_raw.call_args.args[0]
        assert [p.name for p in raw_files_arg] == ["doc.docx"], "картинки должны быть исключены"


class TestProgressCompleteness:
    """Шкала обязана добирать до 100% в ЛЮБОМ режиме: фаза без данных
    анонсирует вес и закрывается сразу (раньше extract без аудио полз
    к 20% и скакал на 100 в конце)."""

    @patch("src.collect.collect_media")
    def test_raw_no_files_bar_reaches_100(self, mock_collect):
        from src.pipeline import run_pipeline
        from src.progress import PhaseProgress

        mock_collect.return_value = ([], [], [], [])
        events: list[dict] = []
        pp = PhaseProgress()

        def progress(e):
            pp.feed(e)
            events.append(e)

        run_pipeline(["/fake"], extract=True, process=False, raw=True, progress=progress)

        assert pp.percent() == 100, "шкала не добрала до 100%"
        assert {"phase": "raw", "weight": 20} in events
        assert {"phase_done": "raw"} in events
        assert {"phase": "transcribe", "weight": 80} in events
        assert {"phase_done": "transcribe"} in events

    @patch("src.pipeline.run_pdfs")
    @patch("src.pipeline.run_attachments")
    @patch("src.collect.collect_media")
    def test_extract_no_media_bar_reaches_100(self, mock_collect, mock_att, mock_pdfs, tmp_path):
        """Extract-режим, папка только с вложениями (без аудио): раньше
        transcribe (80%) терялся и шкала показывала максимум 20%."""
        from src.pipeline import run_pipeline
        from src.progress import PhaseProgress

        img = tmp_path / "p.jpg"
        img.write_bytes(b"jpg")
        mock_collect.return_value = ([], [], [img], [])
        mock_att.return_value = []
        mock_pdfs.return_value = []
        events: list[dict] = []
        pp = PhaseProgress()

        def progress(e):
            pp.feed(e)
            events.append(e)

        run_pipeline([str(tmp_path)], extract=True, process=False, progress=progress)

        assert pp.percent() == 100, "шкала не добрала до 100%"
        assert {"phase": "transcribe", "weight": 80} in events
        assert {"phase_done": "transcribe"} in events
        # вложения (20%) анонсированы и закрыты воркером
        assert any(e.get("phase") == "images" and "weight" in e for e in events)
        assert {"phase_done": "images"} in events


class TestEstimateWiring:
    @patch("src.pipeline.emit_estimate")
    @patch("src.pipeline.collect_media")
    @patch("src.pipeline.run_extract_and_process")
    def test_estimate_emitted_before_phases(
        self, mock_run_extract, mock_collect, mock_emit, tmp_path
    ):
        """Оценка времени выводится ДО запуска фаз (после сбора плана)."""
        from src.pipeline import run_pipeline

        order: list[str] = []
        mock_emit.side_effect = lambda *a, **k: order.append("estimate")
        mock_collect.return_value = ([], [], [], [])
        mock_run_extract.return_value = ([], None)

        events: list[dict] = []

        def progress(event: dict) -> None:
            if "phase" in event and "estimate" not in order:
                order.append("phase-first")
            events.append(event)

        run_pipeline([str(tmp_path)], extract=True, process=True, images=True, progress=progress)

        assert order[0] == "estimate", "оценка должна идти до первой фазы"
        mock_emit.assert_called_once()
        # в emit переданы план вложений и флаги режима
        kwargs = mock_emit.call_args.kwargs
        assert kwargs["extract"] is True and kwargs["process"] is True
        assert "att_plan" in kwargs and "media_files" in kwargs


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