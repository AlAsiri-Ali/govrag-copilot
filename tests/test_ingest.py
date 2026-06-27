"""Tests for src/ingest.py — text utilities, archive loading, article splitting."""
from pathlib import Path

import pytest

from ingest import (
    detect_lang, normalize_arabic, clean_text, sub_chunk,
    split_into_articles, ARTICLE_EN_RE, ARTICLE_AR_RE,
    ingest_document,
)


class TestDetectLang:
    def test_english_text(self):
        assert detect_lang("The Controller shall notify the Competent Authority") == "en"

    def test_arabic_text(self):
        assert detect_lang("جهة التحكم تقوم بإشعار الجهة المختصة") == "ar"

    def test_mixed_majority_english(self):
        # Mostly English with a brief Arabic gloss; should still detect English
        assert detect_lang(
            "The Personal Data Protection Law applies to all controllers "
            "and processors operating in the Kingdom (المملكة)."
        ) == "en"

    def test_mixed_majority_arabic(self):
        assert detect_lang("جهة التحكم تقوم بإشعار الجهة المختصة during processing") == "ar"

    def test_empty_string(self):
        assert detect_lang("") == "unknown"
        assert detect_lang("   ") == "unknown"


class TestNormalizeArabic:
    def test_unifies_alef_forms(self):
        assert normalize_arabic("إأآا") == "اااا"

    def test_strips_diacritics(self):
        result = normalize_arabic("بَيَّنَ")
        # Tashkeel codepoints \u064B-\u065F should be gone
        assert all(not (0x064B <= ord(c) <= 0x065F) for c in result)

    def test_unifies_yaa(self):
        result = normalize_arabic("على")
        assert "ى" not in result and "ي" in result

    def test_unifies_taa_marbuta(self):
        assert "ة" not in normalize_arabic("شركة")

    def test_idempotent(self):
        text = "البيانات الشخصية"
        assert normalize_arabic(text) == normalize_arabic(normalize_arabic(text))


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("a    b\t\tc") == "a b c"

    def test_removes_null_bytes(self):
        assert "\x00" not in clean_text("hello\x00world")

    def test_strips_public_footer(self):
        assert clean_text("article body\nPublic") == "article body"

    def test_collapses_excess_blank_lines(self):
        assert "\n\n\n" not in clean_text("para1\n\n\n\n\npara2")


class TestArticleRegex:
    def test_en_matches_basic(self):
        m = ARTICLE_EN_RE.search("Article 12: privacy")
        assert m is not None and m.group(1) == "12"

    def test_en_matches_no_colon(self):
        m = ARTICLE_EN_RE.search("Article 4 The data subject")
        assert m is not None and m.group(1) == "4"

    def test_en_does_not_match_partial(self):
        # "particle" should not match (word boundary)
        assert ARTICLE_EN_RE.search("particle 12") is None

    def test_ar_matches_ordinal(self):
        assert ARTICLE_AR_RE.search("المادة الثانية والثلاثون: مسؤول") is not None

    def test_ar_does_not_match_inline_word(self):
        # "المادة" mid-sentence without a trailing colon should not match
        assert ARTICLE_AR_RE.search("وفقا لهذه المادة من النظام") is None


class TestSplitIntoArticles:
    def test_english_basic(self):
        pages = [
            (1, "Article 1\nDefinitions go here.\n\nArticle 2\nMore stuff."),
            (2, "Article 3: scope\nfinal text"),
        ]
        labels = [s[0] for s in split_into_articles(pages, "en")]
        assert "Article 1" in labels
        assert "Article 2" in labels
        assert "Article 3" in labels

    def test_no_article_markers_falls_back_to_pages(self):
        pages = [(1, "preamble"), (2, "more preamble")]
        sections = split_into_articles(pages, "en")
        assert len(sections) == 2
        assert all(s[0] is None for s in sections)

    def test_arabic_splits(self):
        pages = [
            (1, "المادة الأولى: التعريفات\nبيان."),
            (2, "المادة الثانية: النطاق\nنص."),
        ]
        labels = [s[0] for s in split_into_articles(pages, "ar")]
        assert any("الأولى" in (l or "") for l in labels)
        assert any("الثانية" in (l or "") for l in labels)

    def test_page_numbers_propagate(self):
        pages = [
            (1, "Article 1\ntext on page 1"),
            (2, "Article 2\ntext on page 2"),
        ]
        for label, ps, pe, _ in split_into_articles(pages, "en"):
            if label == "Article 2":
                assert ps == 2 and pe == 2


class TestSubChunk:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short article."
        assert sub_chunk(text, target_chars=1200) == [text]

    def test_long_text_splits(self):
        long = ". ".join(f"Sentence number {i} contains some words" for i in range(150))
        chunks = sub_chunk(long, target_chars=400, overlap=50)
        assert len(chunks) > 1
        assert all(len(c) <= 800 for c in chunks)

    def test_chunks_overlap(self):
        long = ". ".join(f"Sentence X{i} with content" for i in range(80))
        chunks = sub_chunk(long, target_chars=300, overlap=80)
        if len(chunks) >= 2:
            tail = chunks[0][-100:]
            head = chunks[1][:200]
            assert any(tail[i:i+15] in head for i in range(len(tail) - 15))


class TestIngestRealDoc:
    @pytest.fixture(scope="class")
    def pdpl_en_chunks(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "raw" / "PersonalDataProtectionLawEn.pdf"
        if not path.exists():
            pytest.skip("PDPL-EN raw file not present")
        return ingest_document(path)

    def test_produces_chunks(self, pdpl_en_chunks):
        assert len(pdpl_en_chunks) > 0

    def test_all_chunks_have_metadata(self, pdpl_en_chunks):
        for c in pdpl_en_chunks:
            assert c.doc_short_id == "PDPL-EN"
            assert c.lang == "en"
            assert c.page_start >= 1
            assert c.char_count > 0

    def test_finds_known_articles(self, pdpl_en_chunks):
        articles = {c.article for c in pdpl_en_chunks}
        assert "Article 1" in articles
        assert "Article 12" in articles
        assert "Article 35" in articles

    def test_citation_label_format(self, pdpl_en_chunks):
        label = pdpl_en_chunks[0].citation_label()
        assert "PDPL-EN" in label
        assert "p." in label
