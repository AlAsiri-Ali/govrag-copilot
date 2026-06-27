"""Tests for src/index.py — tokenization, BM25, hybrid retrieval."""
from pathlib import Path

import numpy as np
import pytest

from index import (
    tokenize, BM25, HybridRetriever, expand_query,
    QUERY_EXPANSIONS_EN, QUERY_EXPANSIONS_AR,
)
from ingest import Chunk


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
class TestTokenize:
    def test_drops_english_stopwords(self):
        toks = tokenize("the controller and the processor")
        assert "the" not in toks
        assert "and" not in toks
        assert "controller" in toks
        assert "processor" in toks

    def test_drops_arabic_stopwords(self):
        toks = tokenize("في هذا النظام تكون الكلمات")
        assert "في" not in toks
        assert "هذا" not in toks

    def test_lowercases(self):
        assert tokenize("Controller") == ["controller"]

    def test_normalises_arabic(self):
        # alef variants should collapse
        assert tokenize("إدارة") == tokenize("ادارة")

    def test_drops_punctuation(self):
        toks = tokenize("Article 12: privacy.")
        assert "12" in toks or "article" in toks
        assert ":" not in toks
        assert "." not in toks

    def test_min_length_filter(self):
        # 1-character tokens dropped
        assert "a" not in tokenize("a quick brown fox")


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class TestBM25:
    @pytest.fixture
    def bm25(self):
        corpus = [
            ["controller", "shall", "appoint", "officer", "data", "protection"],
            ["personal", "data", "breach", "notification", "competent", "authority"],
            ["transfer", "outside", "kingdom", "personal", "data", "consent"],
            ["data", "subject", "rights", "access", "correction", "destruction"],
        ]
        return BM25(corpus)

    def test_returns_array_of_correct_length(self, bm25):
        scores = bm25.scores(["data", "protection"])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 4

    def test_relevant_document_scores_higher(self, bm25):
        scores = bm25.scores(["officer", "appoint"])
        # First doc mentions both, others don't
        assert scores[0] == max(scores)

    def test_unknown_terms_yield_zero(self, bm25):
        scores = bm25.scores(["xyznonexistent"])
        assert all(s == 0 for s in scores)

    def test_no_negative_scores(self, bm25):
        scores = bm25.scores(["data", "subject"])
        assert all(s >= 0 for s in scores)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
class TestQueryExpansion:
    def test_en_dpo_expansion(self):
        expanded = expand_query("Who is the DPO?", "en")
        assert "data protection officer" in expanded.lower()

    def test_en_breach_expansion(self):
        expanded = expand_query("breach question", "en")
        assert "notification" in expanded.lower()

    def test_ar_rights_expansion(self):
        expanded = expand_query("ما هي حقوق صاحب البيانات؟", "ar")
        # Should add canonical synonyms
        assert "حقوق" in expanded
        assert len(expanded) > len("ما هي حقوق صاحب البيانات؟")

    def test_no_match_returns_original(self):
        original = "completely unrelated query about apples"
        assert expand_query(original, "en") == original

    def test_ar_uses_arabic_table(self):
        # English triggers shouldn't fire on Arabic
        ar_query = "حقوق"
        en_expansion = expand_query(ar_query, "en")
        ar_expansion = expand_query(ar_query, "ar")
        # AR expansion should be longer because the AR table fires
        assert len(ar_expansion) > len(en_expansion)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------
class TestHybridRetriever:
    @pytest.fixture
    def small_retriever(self):
        chunks = [
            Chunk(
                chunk_id=f"DOC::000::0{i}",
                text=text,
                doc_filename="test.pdf",
                doc_title="Test Doc",
                doc_short_id="TEST",
                doc_type="law",
                lang=lang,
                article=f"Article {i+1}",
                page_start=i+1,
                page_end=i+1,
                char_count=len(text),
            )
            for i, (text, lang) in enumerate([
                ("The Controller shall appoint a Data Protection Officer", "en"),
                ("The Controller shall notify the Competent Authority of "
                 "personal data breach incidents within 72 hours", "en"),
                ("Personal data may be transferred outside the Kingdom only "
                 "with appropriate safeguards", "en"),
                ("جهة التحكم تقوم بتعيين مسؤول حماية البيانات الشخصية", "ar"),
                ("جهة التحكم تشعر الجهة المختصة بحوادث تسرب البيانات "
                 "الشخصية خلال 72 ساعة", "ar"),
            ])
        ]
        return HybridRetriever(chunks)

    def test_retrieves_relevant_chunk(self, small_retriever):
        hits = small_retriever.retrieve("data protection officer", k=2, lang="en")
        assert len(hits) >= 1
        assert hits[0].chunk.article == "Article 1"

    def test_breach_query_finds_breach_chunk(self, small_retriever):
        hits = small_retriever.retrieve("breach notification 72 hours", k=2, lang="en")
        assert any(h.chunk.article == "Article 2" for h in hits)

    def test_arabic_query_finds_arabic_chunks(self, small_retriever):
        hits = small_retriever.retrieve("مسؤول حماية البيانات", k=3, lang="ar")
        assert len(hits) > 0
        # Top hit should be Arabic-language
        assert hits[0].chunk.lang == "ar"

    def test_doc_filter(self, small_retriever):
        # Filter to a non-existent doc -> no hits
        hits = small_retriever.retrieve(
            "data protection officer", k=3, lang="en",
            doc_filter=["NONEXISTENT"],
        )
        assert hits == []

    def test_returns_at_most_k(self, small_retriever):
        hits = small_retriever.retrieve("data", k=2)
        assert len(hits) <= 2

    def test_empty_query_returns_empty(self, small_retriever):
        # All-stopwords query should produce no tokens
        hits = small_retriever.retrieve("the the the and and", k=3)
        assert hits == []

    def test_hits_have_score_breakdown(self, small_retriever):
        hits = small_retriever.retrieve("data protection officer", k=1, lang="en")
        if hits:
            h = hits[0]
            assert h.bm25_score >= 0
            assert h.tfidf_score >= 0
            assert h.score >= 0


# ---------------------------------------------------------------------------
# Round-trip persistence
# ---------------------------------------------------------------------------
class TestRetrieverPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        chunks = [
            Chunk(
                chunk_id="A::000::00", text="The Controller appoints a DPO.",
                doc_filename="x.pdf", doc_title="X", doc_short_id="X",
                doc_type="law", lang="en", article="Article 1",
                page_start=1, page_end=1, char_count=33,
            ),
            Chunk(
                chunk_id="A::001::00", text="Personal data breach notification rules.",
                doc_filename="x.pdf", doc_title="X", doc_short_id="X",
                doc_type="law", lang="en", article="Article 2",
                page_start=2, page_end=2, char_count=42,
            ),
        ]
        retriever = HybridRetriever(chunks)
        path = tmp_path / "test_idx.pkl"
        retriever.save(path)
        loaded = HybridRetriever.load(path)
        assert len(loaded.chunks) == len(retriever.chunks)
        assert loaded.chunks[0].chunk_id == retriever.chunks[0].chunk_id

        # Round-tripped retriever should still retrieve correctly
        hits = loaded.retrieve("breach notification", k=1, lang="en")
        assert len(hits) >= 1
        assert hits[0].chunk.article == "Article 2"


# ---------------------------------------------------------------------------
# Live index smoke test (skipped if index not built)
# ---------------------------------------------------------------------------
class TestLiveIndex:
    @pytest.fixture(scope="class")
    def live_retriever(self):
        root = Path(__file__).resolve().parents[1]
        idx_path = root / "data" / "index" / "hybrid.pkl"
        if not idx_path.exists():
            pytest.skip("Index not built. Run `python src/build.py` first.")
        return HybridRetriever.load(idx_path)

    def test_dpo_question_finds_article_32(self, live_retriever):
        hits = live_retriever.retrieve(
            "What are the responsibilities of a Data Protection Officer?",
            k=3, lang="en",
        )
        assert any("32" in (h.chunk.article or "") for h in hits)

    def test_breach_question_finds_article_24(self, live_retriever):
        hits = live_retriever.retrieve(
            "Within how many hours must a personal data breach be notified?",
            k=3, lang="en",
        )
        assert any("24" in (h.chunk.article or "") for h in hits)

    def test_arabic_dpo_finds_arabic_article(self, live_retriever):
        hits = live_retriever.retrieve(
            "ما مسؤوليات مسؤول حماية البيانات الشخصية؟",
            k=3, lang="ar",
        )
        # Should retrieve from an Arabic source
        assert any(h.chunk.lang == "ar" for h in hits)
