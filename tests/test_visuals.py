"""Tests for src/visuals.py — image modality (infographics, flowcharts, gap cards)."""
from pathlib import Path

import pytest

from visuals import (
    generate_article_infographic, generate_flowchart,
    generate_gap_report_card, generate_coverage_chart,
    FLOWCHARTS, list_flowcharts,
)
from generator import GroundedAnswer
from templates_module import DraftResult


def _dummy_answer(n_citations: int = 2) -> GroundedAnswer:
    return GroundedAnswer(
        answer=(
            "Based on the PDPL and its regulations:\n"
            "- The Controller shall appoint a Data Protection Officer. [#1]\n"
            "- The DPO monitors compliance with the law. [#2]\n"
            "- The DPO is the contact point for the Competent Authority. [#1]"
        ),
        citations=[
            {"label": "IR-EN, Article 32, p.22", "doc": "IR-EN",
             "article": "Article 32", "page": 22,
             "snippet": "The Controller shall appoint...", "marker": "#1"},
            {"label": "PDPL-EN, Article 35, p.14", "doc": "PDPL-EN",
             "article": "Article 35", "page": 14,
             "snippet": "Penalties for violations...", "marker": "#2"},
        ][:n_citations],
        backend="extractive",
        query="What are the responsibilities of a DPO?",
        lang="en",
    )


def _dummy_draft(n_missing: int = 3) -> DraftResult:
    missing = [
        {"key": "purpose", "label": "Purpose", "citation": "PDPL Art. 12",
         "snippet": "The Controller shall specify..."},
        {"key": "legal_basis", "label": "Legal basis", "citation": "PDPL Art. 6",
         "snippet": "Processing requires a legal basis..."},
        {"key": "retention_period", "label": "Retention period",
         "citation": "IR Art. 19", "snippet": "Data shall not be retained..."},
    ][:n_missing]
    return DraftResult(
        template_id="privacy_notice",
        draft="Privacy Notice for Acme...",
        missing_fields=missing,
        citations=[{"label": "PDPL Art. 12", "snippet": "x"}],
        lang="en",
    )


class TestArticleInfographic:
    def test_returns_pil_image(self):
        img = generate_article_infographic(_dummy_answer())
        assert img is not None
        assert img.size[0] > 100 and img.size[1] > 100

    def test_works_with_no_citations(self):
        img = generate_article_infographic(_dummy_answer(n_citations=0))
        assert img is not None

    def test_saves_to_path(self, tmp_path):
        out = tmp_path / "test_infographic.png"
        img = generate_article_infographic(_dummy_answer(), output_path=out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_custom_title(self):
        img = generate_article_infographic(
            _dummy_answer(), title="Custom Title Here"
        )
        assert img is not None


class TestFlowchart:
    def test_all_flowcharts_render(self):
        for fid in FLOWCHARTS:
            img = generate_flowchart(fid, lang="en")
            assert img is not None, f"Flowchart {fid} failed"
            assert img.size[0] > 200

    def test_arabic_language(self):
        img = generate_flowchart("breach_notification", lang="ar")
        assert img is not None

    def test_unknown_flowchart_raises(self):
        with pytest.raises(ValueError):
            generate_flowchart("nonexistent_flowchart")

    def test_saves_to_path(self, tmp_path):
        out = tmp_path / "test_flowchart.png"
        generate_flowchart("breach_notification", output_path=out)
        assert out.exists()


class TestGapReportCard:
    def test_renders_with_missing_fields(self):
        img = generate_gap_report_card(_dummy_draft(n_missing=3))
        assert img is not None
        assert img.size[0] > 200

    def test_renders_with_no_missing_fields(self):
        img = generate_gap_report_card(_dummy_draft(n_missing=0))
        assert img is not None

    def test_saves_to_path(self, tmp_path):
        out = tmp_path / "test_gap.png"
        generate_gap_report_card(_dummy_draft(), output_path=out)
        assert out.exists()


class TestCoverageChart:
    def test_renders(self):
        stats = {
            "total_chunks": 238,
            "by_language": {"en": 140, "ar": 98},
            "by_document": {
                "PDPL-EN": 51, "PDPL-AR": 30,
                "IR-EN": 72, "IR-AR": 53,
                "TR-EN": 17, "TR-AR": 15,
            },
        }
        img = generate_coverage_chart(stats)
        assert img is not None

    def test_empty_stats_returns_none(self):
        assert generate_coverage_chart({"by_document": {}}) is None

    def test_saves_to_path(self, tmp_path):
        stats = {
            "by_document": {"DOC-A": 10, "DOC-B": 20},
            "total_chunks": 30,
        }
        out = tmp_path / "test_cov.png"
        generate_coverage_chart(stats, output_path=out)
        assert out.exists()


class TestListFlowcharts:
    def test_returns_all(self):
        fcs = list_flowcharts("en")
        assert len(fcs) == len(FLOWCHARTS)
        for fc in fcs:
            assert "id" in fc
            assert "title" in fc
            assert "steps" in fc

    def test_arabic_titles_differ(self):
        en_titles = {fc["title"] for fc in list_flowcharts("en")}
        ar_titles = {fc["title"] for fc in list_flowcharts("ar")}
        assert en_titles != ar_titles


class TestPipelineVisualIntegration:
    """Test visual methods on the live pipeline (skipped if index missing)."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from pipeline import GovRAGPipeline
        root = Path(__file__).resolve().parents[1]
        if not (root / "data" / "index" / "hybrid.pkl").exists():
            pytest.skip("Index not built.")
        return GovRAGPipeline(root, backend="extractive")

    def test_generate_infographic(self, pipeline):
        img, ans = pipeline.generate_infographic(
            "What are the responsibilities of a DPO?"
        )
        assert img is not None
        assert not ans.refused

    def test_generate_flowchart(self, pipeline):
        img = pipeline.generate_flowchart("breach_notification")
        assert img is not None

    def test_generate_gap_card(self, pipeline):
        img, draft = pipeline.generate_gap_card(
            "privacy_notice", {"controller_name": "Acme"}, lang="en"
        )
        assert img is not None
        assert len(draft.missing_fields) > 0

    def test_generate_coverage_chart(self, pipeline):
        img = pipeline.generate_coverage_chart()
        assert img is not None

    def test_list_flowcharts(self, pipeline):
        fcs = pipeline.list_flowcharts()
        assert len(fcs) == 5
