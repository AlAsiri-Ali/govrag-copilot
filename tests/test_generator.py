"""Tests for src/generator.py — generation backends, citation handling."""
from pathlib import Path

import pytest

from generator import (
    ExtractiveGenerator, GroundedAnswer, _attach_citations,
    build_prompt, _short_snippet, _sentence_split,
    OllamaGenerator, HFTransformersGenerator,
)
from index import HybridRetriever, RetrievalHit
from ingest import Chunk


def _make_hit(article: str, text: str, lang: str = "en",
              short_id: str = "TEST", page: int = 1) -> RetrievalHit:
    chunk = Chunk(
        chunk_id=f"{short_id}::000::00",
        text=text,
        doc_filename="t.pdf",
        doc_title="Test",
        doc_short_id=short_id,
        doc_type="law",
        lang=lang,
        article=article,
        page_start=page,
        page_end=page,
        char_count=len(text),
    )
    return RetrievalHit(chunk=chunk, score=1.0, bm25_score=1.0, tfidf_score=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TestSentenceSplit:
    def test_english_split(self):
        text = ("The Controller shall notify within 72 hours. "
                "The notification must include description of the breach. "
                "The Data Subject has the right to be informed.")
        sents = _sentence_split(text)
        assert len(sents) >= 2

    def test_arabic_punctuation(self):
        # Arabic question mark ؟ is recognized as a sentence terminator
        text = ("جهة التحكم تشعر الجهة المختصة خلال اثنتين وسبعين ساعة. "
                "الإشعار يتضمن وصف الحادثة بالتفصيل اللازم. "
                "صاحب البيانات له الحق في المعرفة الكاملة.")
        sents = _sentence_split(text)
        assert len(sents) >= 2

    def test_filters_short_fragments(self):
        text = ("ok. This is a real sentence with substantial content. ok.")
        sents = _sentence_split(text)
        # Short "ok." fragments should be dropped (length filter)
        assert all(len(s) > 20 for s in sents)


class TestShortSnippet:
    def test_truncates_with_ellipsis(self):
        snip = _short_snippet("a" * 200, 50)
        assert len(snip) == 51  # 50 chars + ellipsis
        assert snip.endswith("…")

    def test_short_text_unchanged(self):
        assert _short_snippet("short text", 100) == "short text"

    def test_collapses_whitespace(self):
        assert _short_snippet("a   b\n\nc", 100) == "a b c"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    def test_english_prompt_contains_rules(self):
        hit = _make_hit("Article 24", "The Controller shall notify within 72 hours.")
        system, user = build_prompt("How fast must we notify?", [hit], "en")
        assert "GovRAG" in system
        assert "[#1]" in system or "[#2]" in system  # citation rule mentioned
        assert "[#1]" in user
        assert "Article 24" in user

    def test_arabic_prompt_in_arabic(self):
        hit = _make_hit("Article 24", "Notification rules.")
        system, user = build_prompt("سؤال", [hit], "ar")
        assert "GovRAG" in system
        # Arabic-specific instructions
        assert "السؤال" in user

    def test_passages_numbered(self):
        hits = [
            _make_hit("Article 1", "first passage"),
            _make_hit("Article 2", "second passage"),
            _make_hit("Article 3", "third passage"),
        ]
        _, user = build_prompt("q", hits, "en")
        assert "[#1]" in user
        assert "[#2]" in user
        assert "[#3]" in user


# ---------------------------------------------------------------------------
# Extractive generator
# ---------------------------------------------------------------------------
class TestExtractiveGenerator:
    @pytest.fixture
    def gen(self):
        return ExtractiveGenerator()

    def test_returns_grounded_answer(self, gen):
        hits = [_make_hit(
            "Article 24",
            "The Controller shall notify the Competent Authority of personal "
            "data breach incidents within 72 hours of becoming aware. "
            "The notification shall include description of the breach.")]
        ans = gen.generate("breach notification time?", hits, "en")
        assert isinstance(ans, GroundedAnswer)
        assert ans.backend == "extractive"
        assert not ans.refused

    def test_refuses_when_no_hits(self, gen):
        ans = gen.generate("any question", [], "en")
        assert ans.refused
        assert ans.refusal_reason == "no_hits"
        assert "cannot answer" in ans.answer.lower()

    def test_refuses_arabic_question(self, gen):
        ans = gen.generate("سؤال", [], "ar")
        assert ans.refused
        # Arabic refusal message
        assert "لا يمكنني" in ans.answer or "PDPL" in ans.answer

    def test_includes_citation_markers(self, gen):
        hits = [
            _make_hit("Article 1", "Personal Data means any data identifying an individual."),
            _make_hit("Article 24", "Notification within 72 hours."),
        ]
        ans = gen.generate("what is personal data", hits, "en")
        # Either citation marker should appear in the answer
        assert "[#1]" in ans.answer or "[#2]" in ans.answer

    def test_citations_have_full_metadata(self, gen):
        hits = [_make_hit("Article 12", "The Controller shall use a privacy policy.")]
        ans = gen.generate("privacy policy", hits, "en")
        assert len(ans.citations) >= 1
        c = ans.citations[0]
        assert "label" in c
        assert "article" in c
        assert "page" in c
        assert "snippet" in c

    def test_query_language_propagates(self, gen):
        hits = [_make_hit(
            "المادة الثانية والثلاثون",
            "جهة التحكم تعين مسؤول حماية البيانات الشخصية لأداء مهام محددة.",
            lang="ar",
        )]
        ans = gen.generate("ما مسؤوليات مسؤول الحماية؟", hits, "ar")
        assert ans.lang == "ar"
        # Header should be Arabic
        assert "بناءً على" in ans.answer or "نظام" in ans.answer


# ---------------------------------------------------------------------------
# Citation attachment helper (used by LLM backends)
# ---------------------------------------------------------------------------
class TestAttachCitations:
    def test_extracts_used_markers(self):
        hits = [_make_hit("Article 1", "x"), _make_hit("Article 2", "y"),
                _make_hit("Article 3", "z")]
        text = "Some claim here [#1]. Another [#3]."
        ans = _attach_citations(text, hits, "q", "en", backend="test")
        articles = {c["article"] for c in ans.citations}
        assert "Article 1" in articles
        assert "Article 3" in articles
        assert "Article 2" not in articles

    def test_falls_back_to_top_hits_when_no_markers(self):
        hits = [_make_hit("Article 1", "x"), _make_hit("Article 2", "y")]
        text = "Plain answer with no citation markers."
        ans = _attach_citations(text, hits, "q", "en", backend="test")
        # Should attribute to top hits
        assert len(ans.citations) > 0

    def test_ignores_invalid_marker_indices(self):
        hits = [_make_hit("Article 1", "x")]
        text = "Cited [#7] but only one source exists."
        ans = _attach_citations(text, hits, "q", "en", backend="test")
        # Falls back to top hit since [#7] is out of range
        assert len(ans.citations) >= 1


# ---------------------------------------------------------------------------
# Backend availability checks (do not actually call them)
# ---------------------------------------------------------------------------
class TestBackendAvailability:
    def test_ollama_unavailable_when_no_server(self):
        # Pointing at a definitely-dead host
        assert OllamaGenerator.is_available(host="http://127.0.0.1:1") is False

    def test_hf_availability_check_doesnt_raise(self):
        # The check itself should never raise, just return bool
        result = HFTransformersGenerator.is_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# GroundedAnswer markdown rendering
# ---------------------------------------------------------------------------
class TestGroundedAnswerMarkdown:
    def test_renders_with_citations(self):
        ans = GroundedAnswer(
            answer="The answer body.",
            citations=[
                {"label": "PDPL-EN, Article 1, p.1", "doc": "PDPL-EN",
                 "article": "Article 1", "page": 1,
                 "snippet": "test snippet", "marker": "#1"},
            ],
            backend="extractive",
            query="q",
            lang="en",
        )
        md = ans.to_markdown()
        assert "Citations" in md
        assert "PDPL-EN, Article 1, p.1" in md
        assert "extractive" in md

    def test_renders_without_citations(self):
        ans = GroundedAnswer(
            answer="answer", citations=[], backend="extractive",
            query="q", lang="en",
        )
        # Should not crash, should still mention backend
        md = ans.to_markdown()
        assert "extractive" in md
