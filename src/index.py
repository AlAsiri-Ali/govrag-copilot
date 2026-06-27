"""
GovRAG Copilot - Indexing & Retrieval Module
=============================================
Hybrid retriever: BM25 (lexical, strong on legal terms) +
TF-IDF cosine similarity (semantic-ish via n-grams).

Design rationale:
  - No API keys, no network: rules out hosted embeddings.
  - Constrained env: rules out heavy local embedding models for the demo build.
  - Legal text is rich in keywords ("Controller", "Disclosure", "Article 17",
    "المعالج", "الإفصاح") -> BM25 is genuinely strong here.
  - Hybrid scoring + Arabic normalization gives the bilingual coverage the
    proposal asks for.

A clean `Retriever` interface lets you drop in dense embeddings later
(sentence-transformers / Ollama embeddings) without touching the rest of the app.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ingest import Chunk, normalize_arabic, detect_lang, load_chunks


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z\u0600-\u06FF][A-Za-z\u0600-\u06FF0-9_]*")

# Compact stopword lists. Kept small so legal terms stay weighted.
EN_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "in", "on", "at", "to",
    "for", "with", "as", "by", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "from", "shall", "may",
    "such", "any", "all", "not", "no", "so", "than", "then", "there", "their",
    "them", "they", "we", "our", "us", "i", "you", "your", "he", "she", "his",
    "her", "which", "who", "whom", "whose", "what", "when", "where", "how",
    "do", "does", "did", "have", "has", "had", "will", "would", "should",
    "could", "can", "into", "out", "up", "down", "over", "under", "about",
}
AR_STOP = {
    "في", "من", "الى", "إلى", "على", "عن", "مع", "بين", "هذا", "هذه", "ذلك",
    "تلك", "التي", "الذي", "الذين", "اللذان", "اللتان", "ما", "ماذا", "كيف",
    "متى", "اين", "أين", "هل", "لا", "لم", "لن", "ان", "أن", "إن", "كان",
    "كانت", "يكون", "تكون", "او", "أو", "و", "ف", "ب", "ل", "كل", "بعض",
    "غير", "حيث", "كما", "قد", "لقد", "هو", "هي", "هم", "هن", "نحن", "انت",
    "أنت", "أنتم", "ايضا", "أيضا", "ايضًا", "ثم", "حتى", "لكن", "بل",
}
STOP = EN_STOP | AR_STOP


def tokenize(text: str) -> list[str]:
    """Lowercase, normalize Arabic, drop stopwords, return tokens."""
    text = normalize_arabic(text.lower())
    return [t for t in TOKEN_RE.findall(text) if t not in STOP and len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 (Okapi) - reimplemented to avoid a dependency, ~30 LOC
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_lens = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 0.0
        df: Counter = Counter()
        self.tf: list[Counter] = []
        for doc in corpus_tokens:
            counts = Counter(doc)
            self.tf.append(counts)
            for term in counts.keys():
                df[term] += 1
        # IDF with floor at eps to keep scores non-negative
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for q in query_tokens:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(q, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / max(self.avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------
@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    bm25_score: float
    tfidf_score: float

    def __repr__(self) -> str:  # for debugging
        return f"<Hit {self.chunk.citation_label()} score={self.score:.3f}>"


# ---------------------------------------------------------------------------
# Query expansion: tiny domain glossary that boosts retrieval recall by
# adding canonical PDPL synonyms to the query before scoring.
# ---------------------------------------------------------------------------
QUERY_EXPANSIONS_EN: dict[str, list[str]] = {
    "rights": ["data subject rights", "informed access correction destruction"],
    "rights of": ["right to be informed access correction destruction"],
    "sensitive": ["sensitive data biometric genetic religious health"],
    "minor": ["minors legal capacity impact assessment"],
    "child": ["minors legal capacity"],
    "penalty": ["penalty fine imprisonment violation"],
    "penalties": ["penalty fine imprisonment violation"],
    "breach": ["personal data breach notification incident"],
    "transfer": ["transfer outside kingdom safeguards"],
    "consent": ["consent withdraw lawful basis"],
    "controller": ["controller obligations responsibilities"],
    "processor": ["processor selection contract obligations"],
    "dpo": ["data protection officer responsibilities"],
    "officer": ["data protection officer"],
    "ropa": ["records of processing activities"],
}
QUERY_EXPANSIONS_AR: dict[str, list[str]] = {
    "حقوق": ["حقوق صاحب البيانات الشخصية الإتلاف التصحيح"],
    "حساس": ["البيانات الحساسة الديني العرقي الصحي الوراثي"],
    "قاصر": ["القاصرون أهلية تقييم الأثر"],
    "عقوب": ["عقوبة غرامة سجن مخالفة"],
    "تسرب": ["تسرب البيانات إشعار حادثة"],
    "نقل": ["نقل خارج المملكة ضمانات"],
    "موافق": ["موافقة سحب الموافقة"],
    "مسؤول حماية": ["مسؤول حماية البيانات الشخصية"],
}


def expand_query(query: str, lang: str) -> str:
    """Add canonical PDPL synonyms to the raw query."""
    table = QUERY_EXPANSIONS_AR if lang == "ar" else QUERY_EXPANSIONS_EN
    norm = normalize_arabic(query.lower())
    extras: list[str] = []
    for trigger, expansions in table.items():
        if trigger in norm:
            extras.extend(expansions)
    if extras:
        return query + " " + " ".join(extras)
    return query


class HybridRetriever:
    """Combines BM25 + TF-IDF cosine. Per-language sub-indexes."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens: list[list[str]] = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25(self.tokens)

        # TF-IDF on the *tokenized* form so Arabic normalization carries through
        joined = [" ".join(toks) for toks in self.tokens]
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(joined)  # sparse [N x V]

    def retrieve(self, query: str, k: int = 6, lang: str | None = None,
                 doc_filter: list[str] | None = None,
                 alpha: float = 0.55,
                 use_expansion: bool = True) -> list[RetrievalHit]:
        """
        Args:
            query: user query (Arabic or English)
            k: number of hits to return
            lang: optional 'ar' or 'en' to prefer same-language sources
            doc_filter: optional list of doc_short_id values to restrict to
            alpha: weight for BM25 (1-alpha for TF-IDF)
            use_expansion: apply lightweight query expansion (default True)
        """
        # Auto-detect language if not provided so expansion uses the right glossary
        if lang is None:
            lang = detect_lang(query)
        if use_expansion:
            query = expand_query(query, lang)
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        bm25 = self.bm25.scores(q_tokens)
        # TF-IDF cosine
        q_vec = self.tfidf.transform([" ".join(q_tokens)])
        # Normalize rows of tfidf_matrix once at construction? sklearn does L2 by default.
        tfidf_sims = (self.tfidf_matrix @ q_vec.T).toarray().ravel().astype(np.float32)

        # Min-max normalize each independently before blending
        def mm(a: np.ndarray) -> np.ndarray:
            lo, hi = float(a.min()), float(a.max())
            if hi - lo < 1e-9:
                return np.zeros_like(a)
            return (a - lo) / (hi - lo)

        blended = alpha * mm(bm25) + (1 - alpha) * mm(tfidf_sims)

        # Soft language bonus (don't filter hard - allows cross-lingual fallback)
        if lang in ("ar", "en"):
            for i, c in enumerate(self.chunks):
                if c.lang == lang:
                    blended[i] += 0.05

        # Hard filter on doc_short_id if requested
        if doc_filter:
            allowed = set(doc_filter)
            for i, c in enumerate(self.chunks):
                if c.doc_short_id not in allowed:
                    blended[i] = -1.0

        # Take top-k by blended score
        top_idx = np.argsort(-blended)[: max(k * 3, k)]
        # Diversify: keep at most 2 chunks per (doc, article) to avoid stuffing
        seen: dict[tuple, int] = defaultdict(int)
        hits: list[RetrievalHit] = []
        for i in top_idx:
            # Skip chunks excluded by doc_filter (marked with score -1.0)
            if blended[i] < 0:
                continue
            c = self.chunks[i]
            key = (c.doc_short_id, c.article)
            if seen[key] >= 2:
                continue
            seen[key] += 1
            hits.append(RetrievalHit(
                chunk=c,
                score=float(blended[i]),
                bm25_score=float(bm25[i]),
                tfidf_score=float(tfidf_sims[i]),
            ))
            if len(hits) >= k:
                break
        return hits

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "chunks": [c.to_dict() for c in self.chunks],
                "tokens": self.tokens,
                "bm25": self.bm25,
                "tfidf": self.tfidf,
                "tfidf_matrix": self.tfidf_matrix,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "HybridRetriever":
        with path.open("rb") as f:
            payload = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunks = [Chunk(**d) for d in payload["chunks"]]
        obj.tokens = payload["tokens"]
        obj.bm25 = payload["bm25"]
        obj.tfidf = payload["tfidf"]
        obj.tfidf_matrix = payload["tfidf_matrix"]
        return obj


# ---------------------------------------------------------------------------
# Build entrypoint
# ---------------------------------------------------------------------------
def build_index(processed_dir: Path, index_dir: Path) -> HybridRetriever:
    chunks = load_chunks(processed_dir)
    print(f"  -> building hybrid index over {len(chunks)} chunks ...")
    retriever = HybridRetriever(chunks)
    out = index_dir / "hybrid.pkl"
    retriever.save(out)
    print(f"  -> saved index to {out}")
    return retriever


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    retriever = build_index(root / "data" / "processed", root / "data" / "index")

    # Quick sanity checks
    print("\n--- Sanity check (English) ---")
    for h in retriever.retrieve("What is the role of the Data Protection Officer?", k=3, lang="en"):
        print(f"  {h}")
        print(f"     {h.chunk.text[:160].strip()}...")
    print("\n--- Sanity check (Arabic) ---")
    for h in retriever.retrieve("ما هي مسؤوليات مسؤول حماية البيانات؟", k=3, lang="ar"):
        print(f"  {h}")
        print(f"     {h.chunk.text[:160].strip()}...")
    print("\n--- Cross-border transfer ---")
    for h in retriever.retrieve("cross-border transfer requirements", k=3, lang="en"):
        print(f"  {h}")
