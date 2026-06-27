"""Tests for src/evaluate.py — metrics and Arabic ordinal handling."""
from pathlib import Path

import pytest

from evaluate import (
    arabic_ordinal_to_number, _extract_article_number,
    citation_relevance, faithfulness, completeness, bilingual_consistency,
    TESTSET, run_eval,
)
from generator import GroundedAnswer
from pipeline import GovRAGPipeline


# ---------------------------------------------------------------------------
# Arabic ordinal mapper
# ---------------------------------------------------------------------------
class TestArabicOrdinals:
    @pytest.mark.parametrize("label,expected", [
        ("المادة الأولى", 1),
        ("المادة الاولى", 1),     # alef without hamza
        ("المادة الثانية", 2),
        ("المادة الثالثة", 3),
        ("المادة الرابعة", 4),
        ("المادة الخامسة", 5),
        ("المادة السادسة", 6),
        ("المادة السابعة", 7),
        ("المادة الثامنة", 8),
        ("المادة التاسعة", 9),
        ("المادة العاشرة", 10),
    ])
    def test_units_one_through_ten(self, label, expected):
        assert arabic_ordinal_to_number(label) == expected

    @pytest.mark.parametrize("label,expected", [
        ("المادة الحادية عشرة", 11),
        ("المادة الثانية عشرة", 12),
        ("المادة الثالثة عشرة", 13),
        ("المادة الرابعة عشرة", 14),
        ("المادة التاسعة عشرة", 19),
    ])
    def test_teens(self, label, expected):
        assert arabic_ordinal_to_number(label) == expected

    @pytest.mark.parametrize("label,expected", [
        ("المادة العشرون", 20),
        ("المادة الثلاثون", 30),
        ("المادة الأربعون", 40),
        ("المادة الخمسون", 50),
    ])
    def test_pure_tens(self, label, expected):
        assert arabic_ordinal_to_number(label) == expected

    @pytest.mark.parametrize("label,expected", [
        ("المادة الثانية والعشرون", 22),
        ("المادة الرابعة والعشرون", 24),
        ("المادة الثانية والثلاثون", 32),
        ("المادة الخامسة والثلاثون", 35),
    ])
    def test_compound_ordinals(self, label, expected):
        assert arabic_ordinal_to_number(label) == expected

    def test_with_trailing_colon(self):
        assert arabic_ordinal_to_number("المادة الرابعة والعشرون:") == 24

    def test_parenthesized_digit(self):
        assert arabic_ordinal_to_number("المادة (12)") == 12

    def test_unknown_returns_none(self):
        assert arabic_ordinal_to_number("nonsense text") is None
        assert arabic_ordinal_to_number("") is None


class TestExtractArticleNumber:
    def test_english_article(self):
        assert _extract_article_number("Article 24") == 24

    def test_english_with_parens(self):
        assert _extract_article_number("Article (12)") == 12

    def test_arabic_ordinal(self):
        assert _extract_article_number("المادة الثانية والثلاثون") == 32

    def test_empty_returns_none(self):
        assert _extract_article_number("") is None
        assert _extract_article_number(None) is None  # type: ignore


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------
def _ans(citations: list[dict], text: str = "answer body") -> GroundedAnswer:
    return GroundedAnswer(
        answer=text, citations=citations, backend="test",
        query="q", lang="en",
    )


class TestCitationRelevance:
    def test_perfect_match(self):
        ans = _ans([{"article": "Article 24", "label": "PDPL, Article 24"}])
        assert citation_relevance(ans, ["Article 24"]) == 1.0

    def test_no_match(self):
        ans = _ans([{"article": "Article 5", "label": "x"}])
        assert citation_relevance(ans, ["Article 24"]) == 0.0

    def test_arabic_citation_matches_english_gold(self):
        ans = _ans([{"article": "المادة الرابعة والعشرون", "label": "x"}])
        assert citation_relevance(ans, ["Article 24"]) == 1.0

    def test_partial_match(self):
        ans = _ans([
            {"article": "Article 24", "label": "x"},
            {"article": "Article 99", "label": "y"},
        ])
        assert citation_relevance(ans, ["Article 24"]) == 0.5

    def test_no_citations(self):
        assert citation_relevance(_ans([]), ["Article 24"]) == 0.0


class TestFaithfulness:
    def test_exact_copy_is_perfectly_faithful(self):
        text = "the controller shall notify the competent authority"
        ans = _ans([], text=text)
        assert faithfulness(ans, text) == 1.0

    def test_unrelated_answer_is_unfaithful(self):
        ans = _ans([], text="this is completely different content here")
        score = faithfulness(ans, "the controller shall notify within hours")
        assert score < 0.5

    def test_empty_source_returns_zero(self):
        ans = _ans([], text="some claim with multiple words present")
        assert faithfulness(ans, "x y") == 0.0


class TestCompleteness:
    def test_all_keywords_present(self):
        ans = _ans([], text="The Controller must notify within 72 hours.")
        score = completeness(ans, ["controller", "notify", "72", "hours"])
        assert score == 1.0

    def test_no_keywords_present(self):
        ans = _ans([], text="unrelated text")
        score = completeness(ans, ["controller", "notify", "72"])
        assert score == 0.0

    def test_partial(self):
        ans = _ans([], text="The Controller must act.")
        score = completeness(ans, ["controller", "notify", "72"])
        assert 0 < score < 1

    def test_arabic_keyword_matching(self):
        ans = _ans([], text="جهة التحكم تشعر الجهة المختصة خلال 72 ساعة")
        score = completeness(ans, ["72", "الجهة المختصة"])
        assert score == 1.0

    def test_empty_keywords(self):
        ans = _ans([], text="anything")
        assert completeness(ans, []) == 1.0


class TestBilingualConsistency:
    def test_identical_articles_score_one(self):
        en = _ans([{"article": "Article 24", "label": "x"}])
        ar = _ans([{"article": "المادة الرابعة والعشرون", "label": "y"}])
        assert bilingual_consistency(en, ar) == 1.0

    def test_disjoint_articles_score_zero(self):
        en = _ans([{"article": "Article 24", "label": "x"}])
        ar = _ans([{"article": "المادة الأولى", "label": "y"}])
        assert bilingual_consistency(en, ar) == 0.0

    def test_partial_overlap(self):
        en = _ans([
            {"article": "Article 24", "label": "x"},
            {"article": "Article 1", "label": "y"},
        ])
        ar = _ans([{"article": "المادة الأولى", "label": "z"}])
        # Jaccard = |{1}| / |{1, 24}| = 0.5
        assert bilingual_consistency(en, ar) == 0.5

    def test_both_empty_returns_one(self):
        assert bilingual_consistency(_ans([]), _ans([])) == 1.0

    def test_one_empty_returns_zero(self):
        en = _ans([{"article": "Article 1", "label": "x"}])
        assert bilingual_consistency(en, _ans([])) == 0.0


# ---------------------------------------------------------------------------
# Test set sanity
# ---------------------------------------------------------------------------
class TestTestSet:
    def test_each_question_has_required_fields(self):
        for case in TESTSET:
            assert case.get("id")
            assert case.get("question_en")
            assert case.get("question_ar")
            assert case.get("expected_articles")
            assert case.get("keywords_en")
            assert case.get("keywords_ar")

    def test_ids_unique(self):
        ids = [c["id"] for c in TESTSET]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# End-to-end eval (slow; skipped if index missing)
# ---------------------------------------------------------------------------
class TestRunEval:
    @pytest.fixture(scope="class")
    def report(self):
        root = Path(__file__).resolve().parents[1]
        idx = root / "data" / "index" / "hybrid.pkl"
        if not idx.exists():
            pytest.skip("Index not built. Run `python src/build.py` first.")
        pipeline = GovRAGPipeline(root)
        return run_eval(pipeline)

    def test_report_has_all_metrics(self, report):
        m = report["metrics"]
        for key in ["citation_relevance_avg", "faithfulness_avg",
                    "completeness_avg", "bilingual_consistency_avg"]:
            assert key in m
            assert 0.0 <= m[key] <= 1.0

    def test_per_lang_breakdown(self, report):
        m = report["metrics"]
        assert "citation_relevance_en" in m
        assert "citation_relevance_ar" in m
        assert "faithfulness_en" in m
        assert "faithfulness_ar" in m

    def test_per_question_rows(self, report):
        # Each question yields one EN row + one AR row
        assert len(report["per_question"]) == 2 * len(TESTSET)

    def test_minimum_quality_thresholds(self, report):
        """Smoke threshold: faithfulness must stay above 0.5, citation
        relevance above 0.4. These are conservative bounds; the real
        target reported in the README is much higher."""
        m = report["metrics"]
        assert m["faithfulness_avg"] >= 0.5
        assert m["citation_relevance_avg"] >= 0.4
