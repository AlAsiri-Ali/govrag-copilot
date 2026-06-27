"""
GovRAG Copilot - Document Ingestion Module
============================================
Extracts text from PDPL/SDAIA PDFs, detects language, and segments
documents into Article-aware chunks with rich metadata for citation.

Pipeline: PDF -> page-level text -> Article splitter -> chunks with metadata
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import zipfile

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Document registry: maps filenames -> human-readable doc titles + language
# ---------------------------------------------------------------------------
DOC_REGISTRY: dict[str, dict] = {
    "PersonalDataProtectionLawEn.pdf": {
        "title": "Personal Data Protection Law (PDPL)",
        "title_ar": "نظام حماية البيانات الشخصية",
        "lang": "en",
        "doc_type": "law",
        "short_id": "PDPL-EN",
    },
    "PersonalDataProtectionLawAr.pdf": {
        "title": "نظام حماية البيانات الشخصية",
        "title_ar": "نظام حماية البيانات الشخصية",
        "lang": "ar",
        "doc_type": "law",
        "short_id": "PDPL-AR",
    },
    "ImplementingRegulationPersonalDataProtectionLawEn.pdf": {
        "title": "Implementing Regulation of the PDPL",
        "title_ar": "اللائحة التنفيذية لنظام حماية البيانات الشخصية",
        "lang": "en",
        "doc_type": "implementing_regulation",
        "short_id": "IR-EN",
    },
    "ImplementingRegulationPersonalDataProtectionLawAr.pdf": {
        "title": "اللائحة التنفيذية لنظام حماية البيانات الشخصية",
        "title_ar": "اللائحة التنفيذية لنظام حماية البيانات الشخصية",
        "lang": "ar",
        "doc_type": "implementing_regulation",
        "short_id": "IR-AR",
    },
    "RegulationonPersonalDataEn.pdf": {
        "title": "Regulation on Personal Data Transfer Outside the Kingdom",
        "title_ar": "لائحة نقل البيانات الشخصية خارج المملكة",
        "lang": "en",
        "doc_type": "transfer_regulation",
        "short_id": "TR-EN",
    },
    "RegulationonPersonalDataAr.pdf": {
        "title": "لائحة نقل البيانات الشخصية خارج المملكة",
        "title_ar": "لائحة نقل البيانات الشخصية خارج المملكة",
        "lang": "ar",
        "doc_type": "transfer_regulation",
        "short_id": "TR-AR",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A retrievable piece of regulatory text with full citation metadata."""
    chunk_id: str
    text: str
    doc_filename: str
    doc_title: str
    doc_short_id: str
    doc_type: str
    lang: str
    article: str | None
    page_start: int
    page_end: int
    char_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def citation_label(self) -> str:
        """Human-readable citation, e.g. 'PDPL-EN, Article 12, p.8'."""
        parts = [self.doc_short_id]
        if self.article:
            parts.append(self.article)
        parts.append(f"p.{self.page_start}" if self.page_start == self.page_end
                     else f"pp.{self.page_start}-{self.page_end}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def detect_lang(text: str) -> str:
    """Detect Arabic vs English by counting Arabic codepoints."""
    if not text.strip():
        return "unknown"
    arabic_chars = len(ARABIC_RE.findall(text))
    return "ar" if arabic_chars > len(text) * 0.15 else "en"


def normalize_arabic(text: str) -> str:
    """Light Arabic normalization: unify alef/yaa/taa-marbuta and strip diacritics."""
    text = unicodedata.normalize("NFKC", text)
    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    # Unify alef forms
    text = re.sub(r"[إأآا]", "ا", text)
    # Unify yaa
    text = text.replace("ى", "ي")
    # Unify taa marbuta
    text = text.replace("ة", "ه")
    return text


def clean_text(text: str) -> str:
    """Collapse whitespace, strip page-furniture artifacts."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip the trailing 'Public' classification footer common in SDAIA docs
    text = re.sub(r"\n?Public\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Return [(page_number, text), ...] (1-indexed pages).
    Supports two file formats transparently:
      1. Standard PDF (parsed with pypdf)
      2. ZIP archive containing per-page <N>.txt files + manifest.json
         (this is how the SDAIA documents in this project ship)
    """
    if zipfile.is_zipfile(pdf_path):
        return _extract_pages_from_archive(pdf_path)
    # Standard PDF path
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i, clean_text(txt)))
    return pages


def _extract_pages_from_archive(pdf_path: Path) -> list[tuple[int, str]]:
    """Read per-page text files from a ZIP bundle, ordered by page number."""
    pages: list[tuple[int, str]] = []
    with zipfile.ZipFile(pdf_path) as zf:
        # Prefer manifest order if available; otherwise sort N.txt by N.
        manifest_order: list[tuple[int, str]] = []
        if "manifest.json" in zf.namelist():
            try:
                manifest = json.loads(zf.read("manifest.json"))
                for entry in manifest.get("pages", []):
                    pn = entry["page_number"]
                    txt_path = entry.get("text", {}).get("path")
                    if txt_path:
                        manifest_order.append((pn, txt_path))
            except Exception:
                manifest_order = []
        if not manifest_order:
            # Fallback: any *.txt with numeric stem
            for name in zf.namelist():
                if name.endswith(".txt") and Path(name).stem.isdigit():
                    manifest_order.append((int(Path(name).stem), name))
            manifest_order.sort()

        for page_num, txt_path in manifest_order:
            try:
                raw = zf.read(txt_path).decode("utf-8", errors="replace")
            except KeyError:
                raw = ""
            pages.append((page_num, clean_text(raw)))
    return pages


# ---------------------------------------------------------------------------
# Article-aware splitting
# ---------------------------------------------------------------------------
# English article header, e.g. "Article 17:" or "Article 17"
ARTICLE_EN_RE = re.compile(r"\bArticle\s+(\d+)\b[:.\-\s]*", re.MULTILINE)
# Arabic article header. The SDAIA documents use ordinal forms
# ("المادة الأولى:", "المادة الثانية والثلاثون:") and occasionally a
# parenthesized number ("المادة (12):"). We anchor on the *colon* that
# always follows the article title to avoid catching the word
# "المادة" used inline elsewhere.
ARTICLE_AR_RE = re.compile(
    r"المادة\s+(?:\([^)]+\)|[\u0600-\u06FF\s]{1,40}?):"
)


def split_into_articles(pages: list[tuple[int, str]], lang: str
                        ) -> list[tuple[str | None, int, int, str]]:
    """
    Split joined-page text into (article_label, page_start, page_end, text)
    sections. Falls back to per-page chunks when no article markers found.
    """
    # Build a single string but track page offsets so we can recover page numbers
    segments: list[tuple[int, int, str]] = []  # (page, char_offset, text)
    full_text_parts: list[str] = []
    cursor = 0
    page_offsets: list[tuple[int, int]] = []  # (page, offset_start)
    for page_num, txt in pages:
        page_offsets.append((page_num, cursor))
        full_text_parts.append(txt)
        cursor += len(txt) + 2  # account for "\n\n"
    full_text = "\n\n".join(full_text_parts)

    def page_for_offset(offset: int) -> int:
        page = page_offsets[0][0]
        for p_num, p_off in page_offsets:
            if p_off <= offset:
                page = p_num
            else:
                break
        return page

    # Find all article matches with their offsets
    pattern = ARTICLE_EN_RE if lang == "en" else ARTICLE_AR_RE
    matches = list(pattern.finditer(full_text))

    if not matches:
        # Fallback: one chunk per page
        return [(None, p, p, t) for p, t in pages if t.strip()]

    sections: list[tuple[str | None, int, int, str]] = []

    # Capture preamble before first article (definitions, title page)
    if matches[0].start() > 200:
        preamble = full_text[: matches[0].start()].strip()
        if len(preamble) > 100:
            sections.append((
                "Preamble" if lang == "en" else "تمهيد",
                page_for_offset(0),
                page_for_offset(matches[0].start() - 1),
                preamble,
            ))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_text = full_text[start:end].strip()
        if not section_text:
            continue
        if lang == "en":
            label = f"Article {m.group(1)}"
        else:
            label = m.group(0).strip()
        sections.append((
            label,
            page_for_offset(start),
            page_for_offset(end - 1),
            section_text,
        ))
    return sections


# ---------------------------------------------------------------------------
# Chunking within articles (for very long articles)
# ---------------------------------------------------------------------------
def sub_chunk(text: str, target_chars: int = 1200, overlap: int = 150
              ) -> list[str]:
    """Split a long article into overlapping windows on sentence boundaries."""
    if len(text) <= target_chars:
        return [text]
    # Split on sentence boundaries (English period or Arabic full-stop)
    sentences = re.split(r"(?<=[\.\!\?؟])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > target_chars and current:
            chunks.append(" ".join(current))
            # Overlap: keep tail
            tail_len = 0
            tail: list[str] = []
            for s in reversed(current):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            current = tail
            current_len = tail_len
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Top-level ingest
# ---------------------------------------------------------------------------
def ingest_document(pdf_path: Path) -> list[Chunk]:
    meta = DOC_REGISTRY.get(pdf_path.name)
    if not meta:
        # Unknown doc -> autodetect language from first page
        pages_preview = extract_pages(pdf_path)[:2]
        sample = " ".join(t for _, t in pages_preview)
        meta = {
            "title": pdf_path.stem,
            "title_ar": pdf_path.stem,
            "lang": detect_lang(sample),
            "doc_type": "other",
            "short_id": pdf_path.stem[:12].upper(),
        }
    pages = extract_pages(pdf_path)
    sections = split_into_articles(pages, meta["lang"])

    chunks: list[Chunk] = []
    for sect_idx, (article, p_start, p_end, text) in enumerate(sections):
        for sub_idx, sub_text in enumerate(sub_chunk(text)):
            chunk_id = f"{meta['short_id']}::{sect_idx:03d}::{sub_idx:02d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=sub_text,
                doc_filename=pdf_path.name,
                doc_title=meta["title"],
                doc_short_id=meta["short_id"],
                doc_type=meta["doc_type"],
                lang=meta["lang"],
                article=article,
                page_start=p_start,
                page_end=p_end,
                char_count=len(sub_text),
            ))
    return chunks


def ingest_directory(raw_dir: Path, processed_dir: Path) -> list[Chunk]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: list[Chunk] = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        if "Proposal" in pdf_path.name:
            continue  # skip the proposal PDF itself
        print(f"  -> ingesting {pdf_path.name} ...")
        ch = ingest_document(pdf_path)
        all_chunks.extend(ch)
        print(f"     {len(ch)} chunks")

    # Persist chunks to JSONL for traceability
    out = processed_dir / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"  -> wrote {len(all_chunks)} chunks to {out}")
    return all_chunks


def load_chunks(processed_dir: Path) -> list[Chunk]:
    path = processed_dir / "chunks.jsonl"
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[1]
    chunks = ingest_directory(root / "data" / "raw", root / "data" / "processed")
    by_lang: dict[str, int] = {}
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_lang[c.lang] = by_lang.get(c.lang, 0) + 1
        by_doc[c.doc_short_id] = by_doc.get(c.doc_short_id, 0) + 1
    print("\nSummary:")
    print(f"  total chunks: {len(chunks)}")
    print(f"  by language:  {by_lang}")
    print(f"  by document:  {by_doc}")
