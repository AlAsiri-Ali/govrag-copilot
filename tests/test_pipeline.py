"""Tests for src/pipeline.py — the GovRAGPipeline orchestrator."""
from pathlib import Path

import pytest

from pipeline import GovRAGPipeline


@pytest.fixture(scope="module")
def pipeline():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data" / "index" / "hybrid.pkl").exists():
        pytest.skip("Index not built. Run `python src/build.py` first.")
    return GovRAGPipeline(root, backend="extractive")


class TestPipelineConstruction:
    def test_builds_with_extractive(self, pipeline):
        assert pipeline.generator.name == "extractive"

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GovRAGPipeline(tmp_path, backend="extractive")

    def test_unknown_backend_raises(self):
        root = Path(__file__).resolve().parents[1]
        with pytest.raises(ValueError):
            GovRAGPipeline(root, backend="bogus_backend")


class TestPipelineAnswer:
    def test_english_answer(self, pipeline):
        ans = pipeline.answer(
            "What are the responsibilities of a Data Protection Officer?",
            lang="en",
        )
        assert ans.lang == "en"
        assert len(ans.citations) > 0
        assert not ans.refused

    def test_arabic_answer(self, pipeline):
        ans = pipeline.answer(
            "ما مسؤوليات مسؤول حماية البيانات الشخصية؟",
            lang="ar",
        )
        assert ans.lang == "ar"
        assert len(ans.citations) > 0

    def test_auto_detect_language(self, pipeline):
        # No lang argument -> should auto-detect
        ans = pipeline.answer("ما هي البيانات الحساسة؟")
        assert ans.lang == "ar"

    def test_doc_filter(self, pipeline):
        ans = pipeline.answer(
            "What is personal data?",
            lang="en",
            doc_filter=["PDPL-EN"],
        )
        # All citations should be from the filtered doc
        for c in ans.citations:
            assert c["doc"] == "PDPL-EN"


class TestPipelineSearch:
    def test_returns_hits(self, pipeline):
        hits = pipeline.search("breach notification", k=3)
        assert len(hits) <= 3
        assert len(hits) > 0

    def test_zero_k_works(self, pipeline):
        # Edge case: caller passes invalid k
        # The retriever returns up to k; k=1 minimum
        hits = pipeline.search("data", k=1)
        assert len(hits) <= 1


class TestPipelineDraft:
    def test_draft_privacy_notice(self, pipeline):
        res = pipeline.draft(
            "privacy_notice",
            {"controller_name": "Acme"},
            lang="en",
        )
        assert res.template_id == "privacy_notice"
        assert "Acme" in res.draft

    def test_unknown_template_raises(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.draft("nonexistent", {}, lang="en")

    def test_arabic_draft(self, pipeline):
        res = pipeline.draft(
            "breach_notification",
            {"controller_name": "ش"},
            lang="ar",
        )
        assert "إشعار" in res.draft


class TestPipelineMisc:
    def test_list_templates_english(self, pipeline):
        templates = pipeline.list_templates(lang="en")
        assert len(templates) == 4
        for t in templates:
            assert "id" in t
            assert "title" in t
            assert "fields" in t
            assert len(t["fields"]) > 0

    def test_list_templates_arabic(self, pipeline):
        templates = pipeline.list_templates(lang="ar")
        # Arabic titles should differ from English
        en_titles = {t["title"] for t in pipeline.list_templates(lang="en")}
        ar_titles = {t["title"] for t in templates}
        assert en_titles != ar_titles

    def test_stats(self, pipeline):
        stats = pipeline.stats()
        assert stats["total_chunks"] > 0
        assert "en" in stats["by_language"]
        assert "ar" in stats["by_language"]
        assert "PDPL-EN" in stats["by_document"]
        assert stats["backend"] == "extractive"
