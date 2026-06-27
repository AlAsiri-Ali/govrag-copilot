"""
GovRAG Copilot - Evaluation Module
====================================
Implements the four metrics named in the project proposal:

  1. Citation Relevance  — share of retrieved chunks that contain a
     gold-standard keyword for the question (proxy for "did we retrieve
     the right article?").

  2. Faithfulness        — share of n-grams in the answer that are also
     present in the cited chunks (proxy for "no hallucinations").

  3. Completeness        — share of expected facets (gold keywords) that
     appear in the generated answer.

  4. Bilingual Consistency — for each Arabic↔English pair in the test set,
     do the answers cite the *same articles*? Cross-lingual robustness.

These are deterministic proxies suitable for an unattended CI run; they
match the human-rubric framing in the proposal.

Run:
    python src/evaluate.py
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from pipeline import GovRAGPipeline
from generator import GroundedAnswer
from index import tokenize


# ---------------------------------------------------------------------------
# Test set: bilingual question pairs with gold-standard cited articles
# and expected keywords.
# ---------------------------------------------------------------------------
TESTSET: list[dict] = [
    {
        "id": "dpo_role",
        "question_en": "What are the responsibilities of a Data Protection Officer under the PDPL?",
        "question_ar": "ما مسؤوليات مسؤول حماية البيانات الشخصية وفق نظام حماية البيانات؟",
        "expected_articles": ["Article 32"],   # IR
        "keywords_en": ["competent authority", "monitoring", "data subject", "complaints"],
        "keywords_ar": ["الجهة المختصة", "متابع", "صاحب البيانات", "الشكاوى"],
    },
    {
        "id": "breach_72h",
        "question_en": "Within how many hours must a personal data breach be notified to the Competent Authority?",
        "question_ar": "خلال كم ساعة يجب إشعار الجهة المختصة بحادثة تسرب البيانات الشخصية؟",
        "expected_articles": ["Article 24"],   # IR
        "keywords_en": ["72", "competent authority", "notify"],
        "keywords_ar": ["72", "الجهة المختصة", "إشعار"],
    },
    {
        "id": "rights_access",
        "question_en": "What rights does a Data Subject have over their personal data under the PDPL?",
        "question_ar": "ما الحقوق التي يتمتع بها صاحب البيانات الشخصية وفق نظام حماية البيانات؟",
        "expected_articles": [
            "Article 4",   # PDPL: rights enumeration
            "Article 5",   # IR: right to access
            "Article 6",   # IR: right to request access
            "Article 7",   # IR: right to correction
            "Article 8",   # IR: right to destruction
            "Article 10",  # PDPL: rights of data subjects (umbrella)
        ],
        "keywords_en": ["informed", "access", "correction", "destruction"],
        "keywords_ar": ["العلم", "الوصول", "تصحيح", "الإتلاف"],
    },
    {
        "id": "cross_border",
        "question_en": "What conditions must be met to transfer personal data outside the Kingdom?",
        "question_ar": "ما الشروط الواجب توافرها لنقل البيانات الشخصية خارج المملكة؟",
        "expected_articles": ["Article 4", "Article 5", "Article 29"],
        "keywords_en": ["transfer", "outside", "kingdom", "safeguards"],
        "keywords_ar": ["نقل", "خارج", "المملكة", "ضمانات"],
    },
    {
        "id": "privacy_notice",
        "question_en": "What information must a privacy notice include before collecting personal data?",
        "question_ar": "ما المعلومات التي يجب تضمينها في إشعار الخصوصية قبل جمع البيانات؟",
        "expected_articles": ["Article 12", "Article 4"],
        "keywords_en": ["purpose", "collection", "controller", "data subject"],
        "keywords_ar": ["الغرض", "الجمع", "جهة التحكم", "صاحب البيانات"],
    },
    {
        "id": "sensitive_data",
        "question_en": "What is considered Sensitive Personal Data under the PDPL?",
        "question_ar": "ما البيانات الشخصية الحساسة وفق نظام حماية البيانات الشخصية؟",
        # PDPL Article 1 (definitions) and IR Article 2 both define sensitive data
        "expected_articles": ["Article 1", "Article 2"],
        "keywords_en": ["religious", "racial", "health", "biometric", "genetic"],
        "keywords_ar": ["ديني", "عرقي", "صحي", "حيوي", "وراثي"],
    },
    {
        "id": "consent_withdrawal",
        "question_en": "Can a Data Subject withdraw consent, and what happens when they do?",
        "question_ar": "هل يحق لصاحب البيانات سحب الموافقة، وما الذي يترتب على ذلك؟",
        "expected_articles": ["Article 8"],
        "keywords_en": ["withdraw", "consent", "destroy"],
        "keywords_ar": ["سحب", "الموافقة", "إتلاف"],
    },
    {
        "id": "minors_processing",
        "question_en": "What additional protections apply when processing personal data of minors?",
        "question_ar": "ما الحماية الإضافية المطبقة عند معالجة بيانات القاصرين؟",
        "expected_articles": ["Article 25"],
        "keywords_en": ["impact assessment", "minors", "legal capacity"],
        "keywords_ar": ["تقييم", "قاصرين", "أهلية"],
    },
    {
        "id": "ropa",
        "question_en": "What must be recorded in the Records of Processing Activities?",
        "question_ar": "ما الذي يجب تسجيله في سجلات أنشطة المعالجة؟",
        "expected_articles": ["Article 31", "Article 33"],
        "keywords_en": ["records", "processing", "activities", "controller"],
        "keywords_ar": ["سجل", "معالجة", "أنشطة", "جهة التحكم"],
    },
    {
        "id": "penalties",
        "question_en": "What penalties apply for unlawful disclosure of Sensitive Data?",
        "question_ar": "ما العقوبات المطبقة على الإفصاح غير المشروع عن البيانات الحساسة؟",
        "expected_articles": ["Article 35"],
        "keywords_en": ["imprisonment", "fine", "sensitive"],
        "keywords_ar": ["سجن", "غرامة", "حساس"],
    },
]


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------
def _normalize_for_match(s: str) -> str:
    from ingest import normalize_arabic
    return normalize_arabic(s.lower())


def _extract_article_number(article_label: str) -> int | None:
    """Get a numeric article id from either an English or Arabic label."""
    if not article_label:
        return None
    # English: "Article 24" / "Article (12)"
    m = re.search(r"Article\s*\(?(\d+)\)?", article_label)
    if m:
        return int(m.group(1))
    # Arabic ordinal
    return arabic_ordinal_to_number(article_label)


def citation_relevance(answer: GroundedAnswer, expected_articles: list[str]) -> float:
    """Fraction of citations whose article matches one of the expected articles.
    Works for both English ("Article N") and Arabic ordinal labels."""
    if not answer.citations:
        return 0.0
    expected_nums: set[int] = set()
    for a in expected_articles:
        n = _extract_article_number(a)
        if n is not None:
            expected_nums.add(n)
    if not expected_nums:
        return 0.0
    hits = 0
    for c in answer.citations:
        art_num = _extract_article_number(c.get("article") or "")
        if art_num in expected_nums:
            hits += 1
    return hits / len(answer.citations)


def faithfulness(answer: GroundedAnswer, retrieved_text: str, n: int = 4) -> float:
    """Fraction of length-n token windows in the answer that appear in the
    retrieved sources. Higher = more faithful (less hallucinated)."""
    ans_tokens = tokenize(answer.answer)
    src_tokens = tokenize(retrieved_text)
    if len(ans_tokens) < n:
        return 1.0 if all(t in src_tokens for t in ans_tokens) else 0.0
    src_ngrams = {tuple(src_tokens[i:i+n]) for i in range(len(src_tokens) - n + 1)}
    if not src_ngrams:
        return 0.0
    grounded = 0
    total = 0
    for i in range(len(ans_tokens) - n + 1):
        total += 1
        if tuple(ans_tokens[i:i+n]) in src_ngrams:
            grounded += 1
    return grounded / total if total else 0.0


def completeness(answer: GroundedAnswer, keywords: list[str]) -> float:
    """Fraction of expected keywords that appear in the answer."""
    if not keywords:
        return 1.0
    norm_ans = _normalize_for_match(answer.answer)
    hit = sum(1 for k in keywords if _normalize_for_match(k) in norm_ans)
    return hit / len(keywords)


def bilingual_consistency(en_answer: GroundedAnswer,
                          ar_answer: GroundedAnswer) -> float:
    """
    Two answers are bilingually consistent if they cite the same set of
    article numbers. Score = Jaccard overlap of cited article numbers.
    """
    def article_nums(ans: GroundedAnswer) -> set[int]:
        return {n for n in (
            _extract_article_number(c.get("article") or "")
            for c in ans.citations
        ) if n is not None}

    a, b = article_nums(en_answer), article_nums(ar_answer)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Arabic ordinal handling (used by the metrics above)
# ---------------------------------------------------------------------------
AR_ORDINALS_BASE = {
    "الأولى": 1, "الاولى": 1,
    "الثانية": 2, "الثالثة": 3, "الرابعة": 4, "الخامسة": 5,
    "السادسة": 6, "السابعة": 7, "الثامنة": 8, "التاسعة": 9,
    "العاشرة": 10,
}
AR_TENS = {
    "العشرون": 20, "الثلاثون": 30, "الأربعون": 40, "الاربعون": 40,
    "الخمسون": 50, "الستون": 60, "السبعون": 70, "الثمانون": 80,
    "التسعون": 90,
}
AR_TEEN_FIRST = {
    "الحادية": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4, "الخامسة": 5,
    "السادسة": 6, "السابعة": 7, "الثامنة": 8, "التاسعة": 9,
}


def arabic_ordinal_to_number(label: str) -> int | None:
    """
    Convert an Arabic ordinal phrase (as found in article headers) to its
    numeric value. Examples:
        "المادة الثانية والثلاثون"  -> 32
        "المادة الرابعة عشرة"        -> 14
        "المادة العشرون"             -> 20
        "المادة الأولى"              -> 1
    Returns None if no mapping found.
    """
    if not label:
        return None
    stripped = re.sub(r"^المادة\s*", "", label).strip(" :،.()")
    # Pattern A: "<unit> و<tens>"  e.g. الثانية والثلاثون = 32
    m = re.match(r"^(\S+)\s*و(\S+)", stripped)
    if m:
        unit = AR_ORDINALS_BASE.get(m.group(1))
        tens = AR_TENS.get(m.group(2))
        if unit is not None and tens is not None:
            return tens + unit
    # Pattern B: "<teen-first> عشرة"  e.g. الرابعة عشرة = 14
    m = re.match(r"^(\S+)\s*عشر[ةه]?\s*$", stripped)
    if m:
        first = AR_TEEN_FIRST.get(m.group(1))
        if first is not None:
            return 10 + first
    # Pattern C: pure tens
    if stripped in AR_TENS:
        return AR_TENS[stripped]
    # Pattern D: pure unit
    if stripped in AR_ORDINALS_BASE:
        return AR_ORDINALS_BASE[stripped]
    # Pattern E: parenthesized digit
    m = re.search(r"(\d+)", stripped)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    qid: str
    lang: str
    citation_relevance: float
    faithfulness: float
    completeness: float
    backend: str
    answer_preview: str
    cited: list[str]


def run_eval(pipeline: GovRAGPipeline) -> dict:
    rows: list[EvalRow] = []
    bilingual_scores: list[float] = []

    for case in TESTSET:
        # English
        en_q = case["question_en"]
        en_ans = pipeline.answer(en_q, lang="en", k=6)
        retrieved_en = " ".join(c["snippet"] for c in en_ans.citations)
        # Pull more context for faithfulness from raw chunks
        en_hits = pipeline.search(en_q, k=6, lang="en")
        en_src_text = " ".join(h.chunk.text for h in en_hits)
        rows.append(EvalRow(
            qid=case["id"], lang="en",
            citation_relevance=citation_relevance(en_ans, case["expected_articles"]),
            faithfulness=faithfulness(en_ans, en_src_text),
            completeness=completeness(en_ans, case["keywords_en"]),
            backend=en_ans.backend,
            answer_preview=en_ans.answer[:160].replace("\n", " "),
            cited=[c["label"] for c in en_ans.citations],
        ))
        # Arabic
        ar_q = case["question_ar"]
        ar_ans = pipeline.answer(ar_q, lang="ar", k=6)
        ar_hits = pipeline.search(ar_q, k=6, lang="ar")
        ar_src_text = " ".join(h.chunk.text for h in ar_hits)
        rows.append(EvalRow(
            qid=case["id"], lang="ar",
            citation_relevance=citation_relevance(ar_ans, case["expected_articles"]),
            faithfulness=faithfulness(ar_ans, ar_src_text),
            completeness=completeness(ar_ans, case["keywords_ar"]),
            backend=ar_ans.backend,
            answer_preview=ar_ans.answer[:160].replace("\n", " "),
            cited=[c["label"] for c in ar_ans.citations],
        ))
        # Bilingual consistency
        bilingual_scores.append(bilingual_consistency(en_ans, ar_ans))

    def avg(key: str, lang: str | None = None) -> float:
        vals = [getattr(r, key) for r in rows if (lang is None or r.lang == lang)]
        return statistics.mean(vals) if vals else 0.0

    return {
        "n_questions": len(TESTSET),
        "backend": rows[0].backend if rows else None,
        "metrics": {
            "citation_relevance_avg": round(avg("citation_relevance"), 3),
            "citation_relevance_en":  round(avg("citation_relevance", "en"), 3),
            "citation_relevance_ar":  round(avg("citation_relevance", "ar"), 3),
            "faithfulness_avg":       round(avg("faithfulness"), 3),
            "faithfulness_en":        round(avg("faithfulness", "en"), 3),
            "faithfulness_ar":        round(avg("faithfulness", "ar"), 3),
            "completeness_avg":       round(avg("completeness"), 3),
            "completeness_en":        round(avg("completeness", "en"), 3),
            "completeness_ar":        round(avg("completeness", "ar"), 3),
            "bilingual_consistency_avg": round(statistics.mean(bilingual_scores), 3),
        },
        "per_question": [asdict(r) for r in rows],
    }


def render_report(report: dict) -> str:
    m = report["metrics"]
    lines = [
        "GovRAG Copilot — Evaluation Report",
        "=" * 60,
        f"Backend       : {report['backend']}",
        f"# questions   : {report['n_questions']}  (Arabic + English pairs)",
        "",
        "Aggregate metrics",
        "-" * 60,
        f"  Citation Relevance      avg={m['citation_relevance_avg']:.3f}  "
        f"(en={m['citation_relevance_en']:.3f}, ar={m['citation_relevance_ar']:.3f})",
        f"  Faithfulness            avg={m['faithfulness_avg']:.3f}  "
        f"(en={m['faithfulness_en']:.3f}, ar={m['faithfulness_ar']:.3f})",
        f"  Completeness            avg={m['completeness_avg']:.3f}  "
        f"(en={m['completeness_en']:.3f}, ar={m['completeness_ar']:.3f})",
        f"  Bilingual Consistency   avg={m['bilingual_consistency_avg']:.3f}",
        "",
        "Per-question breakdown",
        "-" * 60,
    ]
    for r in report["per_question"]:
        lines.append(
            f"[{r['qid']:<20s}] lang={r['lang']}  "
            f"cite={r['citation_relevance']:.2f}  "
            f"faith={r['faithfulness']:.2f}  "
            f"compl={r['completeness']:.2f}"
        )
        if r["cited"]:
            lines.append(f"    cited: {', '.join(r['cited'][:3])}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[1]
    pipeline = GovRAGPipeline(root)
    print(f"Stats: {pipeline.stats()}\n")
    report = run_eval(pipeline)
    print(render_report(report))
    out = root / "data" / "processed" / "eval_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n-> wrote full report to {out}")
