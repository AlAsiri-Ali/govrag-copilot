"""
GovRAG Copilot - Generation Module
===================================
Produces evidence-grounded answers from retrieved chunks. Three backends
are supported, all without API keys:

1. ExtractiveGenerator  - Pure-Python deterministic answer assembled from the
                          retrieved passages. Fast, requires no LLM, faithful
                          by construction. Used as the default for the
                          sandbox demo and as a guaranteed fallback.

2. OllamaGenerator      - Talks to a local Ollama server (http://localhost:11434).
                          Recommended for the user's local machine. Just
                          `ollama pull qwen2.5:7b-instruct` and run.

3. HFTransformersGenerator - Loads an open-source model (e.g.
                          Qwen2.5-1.5B-Instruct) directly with HuggingFace
                          transformers. No API key, fully local.

Selection is automatic: prefers Ollama -> HF -> Extractive, but the user
can force a backend via constructor argument or env var.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from index import RetrievalHit
from ingest import detect_lang


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class GroundedAnswer:
    answer: str
    citations: list[dict]   # [{label, doc, article, page, snippet}]
    backend: str
    query: str
    lang: str
    refused: bool = False
    refusal_reason: str | None = None

    def to_markdown(self) -> str:
        out = [self.answer.strip(), ""]
        if self.citations:
            out.append("**Citations / المراجع:**")
            for i, c in enumerate(self.citations, 1):
                out.append(f"{i}. {c['label']} — _{c['snippet']}_")
        out.append("")
        out.append(f"<sub>Backend: `{self.backend}`</sub>")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Prompt builder (shared by LLM backends)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_EN = textwrap.dedent("""\
    You are GovRAG Copilot, a compliance assistant grounded in Saudi Arabia's
    Personal Data Protection Law (PDPL), its Implementing Regulation, and the
    Regulation on Personal Data Transfer Outside the Kingdom (SDAIA).

    Strict rules:
    1. Answer ONLY using facts present in the supplied CONTEXT passages.
    2. Every factual claim must include an inline citation in the form
       [#1], [#2], etc., matching the context passage numbers.
    3. If the answer is not supported by the context, reply exactly:
       "I cannot answer this from the provided PDPL/SDAIA sources."
    4. Be concise and structured. Use short paragraphs or bullet lists.
    5. Do NOT invent article numbers, page numbers, or quotations.
    6. Mirror the user's language (English or Arabic) in the response.
""")

SYSTEM_PROMPT_AR = textwrap.dedent("""\
    أنت GovRAG Copilot، مساعد امتثال يستند إلى نظام حماية البيانات الشخصية
    في المملكة العربية السعودية ولائحته التنفيذية ولائحة نقل البيانات الشخصية
    خارج المملكة (SDAIA).

    قواعد صارمة:
    1. استخدم فقط الحقائق الواردة في المقاطع المرجعية المرفقة (CONTEXT).
    2. كل ادعاء يجب أن يتبعه استشهاد على شكل [#1] أو [#2] إلخ، يطابق رقم المقطع.
    3. إذا لم تكن الإجابة مدعومة بالمقاطع، اكتب حرفياً:
       "لا يمكنني الإجابة على ذلك من المصادر المتاحة في PDPL/SDAIA."
    4. كن مختصراً ومنظماً، واستخدم فقرات قصيرة أو قوائم.
    5. لا تخترع أرقام مواد أو صفحات أو اقتباسات.
    6. أجب بلغة المستخدم (العربية أو الإنجليزية).
""")


def build_prompt(query: str, hits: list[RetrievalHit], lang: str) -> tuple[str, str]:
    """Return (system, user) prompt strings."""
    system = SYSTEM_PROMPT_AR if lang == "ar" else SYSTEM_PROMPT_EN

    ctx_parts: list[str] = []
    for i, h in enumerate(hits, start=1):
        ctx_parts.append(
            f"[#{i}] {h.chunk.citation_label()}\n{h.chunk.text.strip()}"
        )
    context_block = "\n\n---\n\n".join(ctx_parts)

    if lang == "ar":
        user = (
            f"السؤال: {query}\n\n"
            f"المقاطع المرجعية (CONTEXT):\n\n{context_block}\n\n"
            f"اكتب الإجابة الآن، مع الالتزام التام بالقواعد أعلاه."
        )
    else:
        user = (
            f"Question: {query}\n\n"
            f"CONTEXT passages:\n\n{context_block}\n\n"
            f"Write the answer now, strictly following the rules above."
        )
    return system, user


# ---------------------------------------------------------------------------
# Extractive generator (no LLM required)
# ---------------------------------------------------------------------------
class ExtractiveGenerator:
    """
    Builds a faithful answer by stitching together top-scoring sentences
    from the retrieved chunks. Cannot hallucinate because every output
    word came from the source text. Adds inline [#N] citations.
    """
    name = "extractive"

    def generate(self, query: str, hits: list[RetrievalHit], lang: str) -> GroundedAnswer:
        if not hits:
            msg = ("لا يمكنني الإجابة على ذلك من المصادر المتاحة في PDPL/SDAIA."
                   if lang == "ar"
                   else "I cannot answer this from the provided PDPL/SDAIA sources.")
            return GroundedAnswer(answer=msg, citations=[], backend=self.name,
                                  query=query, lang=lang, refused=True,
                                  refusal_reason="no_hits")

        from index import tokenize  # local import to avoid cycles
        q_tokens = set(tokenize(query))

        # Score candidate sentences by query-term overlap
        scored: list[tuple[float, int, str]] = []  # (score, hit_idx, sentence)
        for h_idx, hit in enumerate(hits):
            sentences = _sentence_split(hit.chunk.text)
            for sent in sentences:
                s_tokens = set(tokenize(sent))
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens)
                if overlap == 0:
                    continue
                # Prefer shorter sentences and higher-ranked hits
                rank_bonus = 1.0 / (1 + h_idx)
                length_pen = 1.0 / (1 + max(0, len(sent) - 250) / 100.0)
                scored.append((overlap * rank_bonus * length_pen, h_idx, sent))

        scored.sort(key=lambda x: -x[0])

        # Take top sentences, dedup by content, cap total length
        used_hits: set[int] = set()
        bullets: list[str] = []
        seen: set[str] = set()
        total_len = 0
        for score, h_idx, sent in scored:
            key = sent[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            sent_clean = re.sub(r"\s+", " ", sent.strip())
            sent_clean = sent_clean.rstrip(".،,") + "."
            bullets.append(f"- {sent_clean} [#{h_idx + 1}]")
            used_hits.add(h_idx)
            total_len += len(sent_clean)
            if len(bullets) >= 5 or total_len > 900:
                break

        if not bullets:
            # No sentence-level overlap; fall back to first sentences of top hits
            for h_idx, hit in enumerate(hits[:3]):
                first_sent = _sentence_split(hit.chunk.text)[:1]
                if first_sent:
                    s = re.sub(r"\s+", " ", first_sent[0].strip())
                    bullets.append(f"- {s} [#{h_idx + 1}]")
                    used_hits.add(h_idx)

        header = ("بناءً على نظام حماية البيانات الشخصية ولوائحه:"
                  if lang == "ar"
                  else "Based on the PDPL and its regulations:")
        answer = header + "\n" + "\n".join(bullets)

        citations = [
            {
                "label": hits[i].chunk.citation_label(),
                "doc": hits[i].chunk.doc_short_id,
                "article": hits[i].chunk.article,
                "page": hits[i].chunk.page_start,
                "snippet": _short_snippet(hits[i].chunk.text, 140),
                "marker": f"#{i + 1}",
            }
            for i in sorted(used_hits)
        ]
        return GroundedAnswer(answer=answer, citations=citations,
                              backend=self.name, query=query, lang=lang)


def _sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[\.\!\?؟])\s+|(?<=[\.\!\?؟])(?=[A-Z\u0600-\u06FF])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _short_snippet(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n] + ("…" if len(text) > n else "")


# ---------------------------------------------------------------------------
# Ollama generator (runs against http://localhost:11434)
# ---------------------------------------------------------------------------
class OllamaGenerator:
    """
    Uses a local Ollama server. To enable on your machine:
        $ curl -fsSL https://ollama.com/install.sh | sh
        $ ollama pull qwen2.5:7b-instruct       # bilingual, strong
        # or:  ollama pull llama3.1:8b-instruct
        # or:  ollama pull aya:8b               # tuned for 23 languages incl. Arabic

    No API key needed. The model runs entirely on your machine.
    """
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b-instruct",
                 host: str = "http://localhost:11434",
                 timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    @classmethod
    def is_available(cls, host: str = "http://localhost:11434") -> bool:
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def _chat(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "options": {"temperature": 0.1, "num_predict": 600},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read())
        return payload.get("message", {}).get("content", "").strip()

    def generate(self, query: str, hits: list[RetrievalHit], lang: str) -> GroundedAnswer:
        if not hits:
            msg = ("لا يمكنني الإجابة على ذلك من المصادر المتاحة في PDPL/SDAIA."
                   if lang == "ar"
                   else "I cannot answer this from the provided PDPL/SDAIA sources.")
            return GroundedAnswer(answer=msg, citations=[], backend=self.name,
                                  query=query, lang=lang, refused=True)
        system, user = build_prompt(query, hits, lang)
        text = self._chat(system, user)
        return _attach_citations(text, hits, query, lang, backend=self.name)


# ---------------------------------------------------------------------------
# HuggingFace transformers generator (purely local, no API key)
# ---------------------------------------------------------------------------
class HFTransformersGenerator:
    """
    Loads an open-source instruction model with HuggingFace transformers.
    Recommended models (all open-weight, no API key):
        - "Qwen/Qwen2.5-7B-Instruct"    (strong bilingual, needs GPU ~16GB)
        - "Qwen/Qwen2.5-3B-Instruct"    (good balance, GPU or high-RAM CPU)
        - "Qwen/Qwen2.5-1.5B-Instruct"  (smallest, runs on CPU ~5GB RAM,
                                           but weak on Arabic)
        - "CohereForAI/aya-expanse-8b"  (tuned for Arabic)

    First run downloads weights from HuggingFace Hub (no API key needed for
    public models). Subsequent runs use the local cache.

    NOTE: The 1.5B model is too small for reliable Arabic generation — it
    may produce mixed-language gibberish. Use 3B+ for bilingual queries.
    """
    name = "hf"

    def __init__(self, model_id: str | None = None):
        if model_id is None:
            model_id = self._auto_select_model()
        self.model_id = model_id
        self._tokenizer = None
        self._model = None

    @staticmethod
    def _auto_select_model() -> str:
        """Pick the best model that fits the available hardware."""
        try:
            import torch
            if torch.cuda.is_available():
                mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
                if mem_gb >= 20:
                    return "Qwen/Qwen2.5-7B-Instruct"
                else:
                    return "Qwen/Qwen2.5-3B-Instruct"
        except Exception:
            pass
        return "Qwen/Qwen2.5-1.5B-Instruct"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise RuntimeError(
                "transformers / torch not installed. Run:\n"
                "  pip install 'transformers>=4.45' 'torch>=2.2' accelerate\n"
            ) from e

        print(f"[hf] Loading {self.model_id} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"[hf] Model loaded on {self._model.device}.")

    @classmethod
    def is_available(cls) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, query: str, hits: list[RetrievalHit], lang: str) -> GroundedAnswer:
        if not hits:
            msg = ("لا يمكنني الإجابة على ذلك من المصادر المتاحة في PDPL/SDAIA."
                   if lang == "ar"
                   else "I cannot answer this from the provided PDPL/SDAIA sources.")
            return GroundedAnswer(answer=msg, citations=[], backend=self.name,
                                  query=query, lang=lang, refused=True)
        self._ensure_loaded()
        system, user = build_prompt(query, hits, lang)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        # Use the tokenizer's chat template to build a single prompt string,
        # then generate directly on the model.  This avoids the pipeline()
        # deprecation warnings about max_length vs max_new_tokens.
        import torch

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
            )

        # Decode only the new tokens (strip the input prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return _attach_citations(text, hits, query, lang, backend=self.name)


# ---------------------------------------------------------------------------
# Citation post-processing: turn [#N] markers into structured citation objects
# ---------------------------------------------------------------------------
CITE_MARKER_RE = re.compile(r"\[#(\d+)\]")


def _attach_citations(text: str, hits: list[RetrievalHit], query: str,
                      lang: str, backend: str) -> GroundedAnswer:
    used: set[int] = set()
    for m in CITE_MARKER_RE.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(hits):
            used.add(idx)
    # If model didn't cite, attribute to top-3 hits
    if not used:
        used = set(range(min(3, len(hits))))
    citations = [
        {
            "label": hits[i].chunk.citation_label(),
            "doc": hits[i].chunk.doc_short_id,
            "article": hits[i].chunk.article,
            "page": hits[i].chunk.page_start,
            "snippet": _short_snippet(hits[i].chunk.text, 140),
            "marker": f"#{i + 1}",
        }
        for i in sorted(used)
    ]
    return GroundedAnswer(answer=text, citations=citations,
                          backend=backend, query=query, lang=lang)


# ---------------------------------------------------------------------------
# Auto-select backend
# ---------------------------------------------------------------------------
def get_default_generator():
    """Pick the strongest backend that is actually available right now.

    Priority (auto-select): Ollama > HF Transformers > Extractive.
    Force a specific backend with GOVRAG_BACKEND env var.
    """
    forced = os.environ.get("GOVRAG_BACKEND", "").lower()
    if forced == "ollama":
        return OllamaGenerator(model=os.environ.get("GOVRAG_OLLAMA_MODEL",
                                                    "qwen2.5:7b-instruct"))
    if forced == "hf":
        return HFTransformersGenerator(
            model_id=os.environ.get("GOVRAG_HF_MODEL", None))
    if forced == "extractive":
        return ExtractiveGenerator()

    if OllamaGenerator.is_available():
        print("[generator] Using Ollama backend (local LLM, no API key).")
        return OllamaGenerator(model=os.environ.get("GOVRAG_OLLAMA_MODEL",
                                                    "qwen2.5:7b-instruct"))
    if HFTransformersGenerator.is_available():
        print("[generator] Using HuggingFace transformers backend (no API key).")
        return HFTransformersGenerator(
            model_id=os.environ.get("GOVRAG_HF_MODEL", None))
    print("[generator] Using extractive backend (no LLM required).")
    return ExtractiveGenerator()


if __name__ == "__main__":
    from index import HybridRetriever
    root = Path(__file__).resolve().parents[1]
    retriever = HybridRetriever.load(root / "data" / "index" / "hybrid.pkl")
    gen = get_default_generator()

    queries = [
        ("What are the responsibilities of a Data Protection Officer?", "en"),
        ("Within how many hours must a personal data breach be notified?", "en"),
        ("ما هي الحالات التي يحق فيها لصاحب البيانات طلب الإتلاف؟", "ar"),
        ("Can data be transferred outside the Kingdom without consent?", "en"),
    ]
    for q, lang in queries:
        print(f"\n{'='*72}\nQ ({lang}): {q}")
        hits = retriever.retrieve(q, k=5, lang=lang)
        ans = gen.generate(q, hits, lang)
        print(ans.to_markdown())
