"""Tests for src/templates_module.py — template drafting and gap detection."""
from pathlib import Path

import pytest

from templates_module import (
    TEMPLATES, draft_with_gaps, DraftResult,
    PRIVACY_NOTICE, ROPA_ENTRY, BREACH_NOTIFICATION, TRANSFER_ASSESSMENT,
)
from index import HybridRetriever


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------
class TestTemplateRegistry:
    def test_four_templates_registered(self):
        assert "privacy_notice" in TEMPLATES
        assert "ropa_entry" in TEMPLATES
        assert "breach_notification" in TEMPLATES
        assert "transfer_assessment" in TEMPLATES

    def test_each_template_has_fields(self):
        for tid, t in TEMPLATES.items():
            assert len(t.fields) > 0, f"{tid} has no fields"

    def test_each_template_has_bilingual_titles(self):
        for tid, t in TEMPLATES.items():
            assert t.title_en, f"{tid} missing English title"
            assert t.title_ar, f"{tid} missing Arabic title"

    def test_field_keys_are_unique(self):
        for tid, t in TEMPLATES.items():
            keys = [f.key for f in t.fields]
            assert len(keys) == len(set(keys)), f"{tid} has duplicate field keys"


# ---------------------------------------------------------------------------
# Standalone render functions (no retriever needed)
# ---------------------------------------------------------------------------
class TestRenderPrivacyNotice:
    def test_renders_with_inputs(self):
        inputs = {
            "controller_name": "Acme Inc.",
            "purpose": "delivery",
            "legal_basis": "consent",
            "data_categories": "name, phone",
            "retention_period": "2 years",
            "disclosure_parties": "couriers",
            "contact_channel": "privacy@acme.sa",
        }
        out = PRIVACY_NOTICE.render(inputs, "en")
        assert "Acme Inc." in out
        assert "delivery" in out
        assert "consent" in out

    def test_arabic_render(self):
        inputs = {
            "controller_name": "شركة أكمي",
            "purpose": "تقديم خدمات",
        }
        out = PRIVACY_NOTICE.render(inputs, "ar")
        assert "شركة أكمي" in out
        assert "إشعار الخصوصية" in out

    def test_missing_inputs_show_placeholder(self):
        out = PRIVACY_NOTICE.render({}, "en")
        assert "[__]" in out


class TestRenderBreach:
    def test_includes_72h_reminder(self):
        out = BREACH_NOTIFICATION.render({"controller_name": "Co."}, "en")
        assert "72 hours" in out
        assert "Article 24" in out

    def test_arabic_version(self):
        out = BREACH_NOTIFICATION.render({"controller_name": "ش"}, "ar")
        assert "72" in out
        # The Arabic template references Article 24 with the لـ prefix
        assert "المادة 24" in out or "للمادة 24" in out


class TestRenderRopa:
    def test_uses_activity_name_in_title(self):
        out = ROPA_ENTRY.render({"activity_name": "Customer onboarding"}, "en")
        assert "Customer onboarding" in out


class TestRenderTransferAssessment:
    def test_mentions_scc_and_bcr(self):
        out = TRANSFER_ASSESSMENT.render({}, "en")
        assert "Standard Contractual Clauses" in out
        assert "BCR" in out


# ---------------------------------------------------------------------------
# Gap detection (needs the live retriever)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def retriever():
    root = Path(__file__).resolve().parents[1]
    idx_path = root / "data" / "index" / "hybrid.pkl"
    if not idx_path.exists():
        pytest.skip("Index not built. Run `python src/build.py` first.")
    return HybridRetriever.load(idx_path)


class TestGapDetection:
    def test_returns_draft_result(self, retriever):
        res = draft_with_gaps("privacy_notice", {}, retriever, lang="en")
        assert isinstance(res, DraftResult)
        assert res.template_id == "privacy_notice"

    def test_empty_inputs_flag_all_required(self, retriever):
        res = draft_with_gaps("privacy_notice", {}, retriever, lang="en")
        # All required fields should be flagged as missing
        required_count = sum(1 for f in TEMPLATES["privacy_notice"].fields if f.required)
        assert len(res.missing_fields) == required_count

    def test_full_inputs_flag_nothing(self, retriever):
        # Provide every required field
        complete = {
            f.key: "x" for f in TEMPLATES["privacy_notice"].fields if f.required
        }
        res = draft_with_gaps("privacy_notice", complete, retriever, lang="en")
        assert len(res.missing_fields) == 0

    def test_partial_inputs_flag_only_missing(self, retriever):
        # Only fill controller name
        res = draft_with_gaps(
            "privacy_notice",
            {"controller_name": "Acme"},
            retriever, lang="en",
        )
        keys_missing = {m["key"] for m in res.missing_fields}
        assert "controller_name" not in keys_missing
        assert "purpose" in keys_missing  # required, was empty

    def test_missing_fields_include_citation(self, retriever):
        res = draft_with_gaps("privacy_notice", {}, retriever, lang="en")
        for m in res.missing_fields:
            assert "citation" in m
            assert "snippet" in m
            assert m["citation"]  # non-empty

    def test_grounding_citations_attached(self, retriever):
        res = draft_with_gaps("privacy_notice",
                              {f.key: "x" for f in TEMPLATES["privacy_notice"].fields},
                              retriever, lang="en")
        # Even with all fields filled, anchor citations should still attach
        assert len(res.citations) > 0

    def test_arabic_drafting(self, retriever):
        res = draft_with_gaps("breach_notification", {"controller_name": "ش"},
                              retriever, lang="ar")
        assert "إشعار" in res.draft

    def test_unknown_template_raises(self, retriever):
        with pytest.raises(KeyError):
            draft_with_gaps("nonexistent_template", {}, retriever, lang="en")


class TestDraftResultMarkdown:
    def test_markdown_includes_missing_fields(self, retriever):
        res = draft_with_gaps("privacy_notice", {}, retriever, lang="en")
        md = res.to_markdown()
        assert "Missing required fields" in md or "missing" in md.lower()

    def test_markdown_includes_citations(self, retriever):
        res = draft_with_gaps("privacy_notice", {}, retriever, lang="en")
        md = res.to_markdown()
        assert "citations" in md.lower() or "References" in md
