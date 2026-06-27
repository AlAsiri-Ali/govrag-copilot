"""
GovRAG Copilot - Pipeline (top-level orchestrator)
====================================================
Wraps everything into a single GovRAGPipeline class:
  • answer(query, lang)            -> grounded Q&A
  • draft(template_id, inputs)     -> template draft + gap detection
  • search(query, k)               -> raw retrieval results

This is what the UI and tests call.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from index import HybridRetriever
from generator import (
    GroundedAnswer, get_default_generator, ExtractiveGenerator,
    OllamaGenerator, HFTransformersGenerator,
)
from templates_module import TEMPLATES, DraftResult, draft_with_gaps
from ingest import detect_lang


class GovRAGPipeline:
    def __init__(self, project_root: Path | str | None = None,
                 backend: str = "auto"):
        backend = backend.lower()
        valid_backends = {"auto", "extractive", "ollama", "hf"}
        if backend not in valid_backends:
            raise ValueError(f"unknown backend: {backend}")

        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        self.root = Path(project_root)
        self.index_path = self.root / "data" / "index" / "hybrid.pkl"
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {self.index_path}. "
                "Run `python src/build.py` first.")
        self.retriever = HybridRetriever.load(self.index_path)
        self.generator = self._select_backend(backend)

    @staticmethod
    def _select_backend(backend: str):
        backend = backend.lower()
        if backend == "auto":
            return get_default_generator()
        if backend == "extractive":
            return ExtractiveGenerator()
        if backend == "ollama":
            return OllamaGenerator()
        if backend == "hf":
            return HFTransformersGenerator()
        raise ValueError(f"unknown backend: {backend}")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def answer(self, query: str, lang: str | None = None,
               k: int = 6, doc_filter: list[str] | None = None
               ) -> GroundedAnswer:
        if not lang:
            lang = detect_lang(query)
            if lang == "unknown":
                lang = "en"
        hits = self.retriever.retrieve(query, k=k, lang=lang,
                                       doc_filter=doc_filter)
        return self.generator.generate(query, hits, lang)

    def search(self, query: str, k: int = 6, lang: str | None = None):
        if not lang:
            lang = detect_lang(query)
        return self.retriever.retrieve(query, k=k, lang=lang)

    def draft(self, template_id: str, inputs: dict,
              lang: str = "en") -> DraftResult:
        if template_id not in TEMPLATES:
            raise ValueError(f"unknown template: {template_id}. "
                             f"Available: {list(TEMPLATES)}")
        return draft_with_gaps(template_id, inputs, self.retriever, lang=lang)

    def list_templates(self, lang: str = "en") -> list[dict]:
        return [
            {
                "id": t.template_id,
                "title": t.title_ar if lang == "ar" else t.title_en,
                "description": t.description_ar if lang == "ar" else t.description_en,
                "fields": [
                    {
                        "key": f.key,
                        "label": f.label_ar if lang == "ar" else f.label_en,
                        "required": f.required,
                        "hint": f.hint_ar if lang == "ar" else f.hint_en,
                    }
                    for f in t.fields
                ],
            }
            for t in TEMPLATES.values()
        ]

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------
    def stats(self) -> dict:
        chunks = self.retriever.chunks
        by_lang: dict[str, int] = {}
        by_doc: dict[str, int] = {}
        for c in chunks:
            by_lang[c.lang] = by_lang.get(c.lang, 0) + 1
            by_doc[c.doc_short_id] = by_doc.get(c.doc_short_id, 0) + 1
        return {
            "total_chunks": len(chunks),
            "by_language": by_lang,
            "by_document": by_doc,
            "backend": self.generator.name,
        }

    # -----------------------------------------------------------------
    # Image modality (second modality for course requirement)
    # -----------------------------------------------------------------
    def generate_infographic(self, query: str, lang: str | None = None,
                             output_path: Path | None = None):
        """Generate a visual infographic card for a Q&A answer."""
        from visuals import generate_article_infographic
        ans = self.answer(query, lang=lang)
        img = generate_article_infographic(ans, output_path=output_path)
        return img, ans

    def generate_flowchart(self, flowchart_id: str, lang: str = "en",
                           output_path: Path | None = None):
        """Generate a PDPL compliance process flowchart."""
        from visuals import generate_flowchart
        return generate_flowchart(flowchart_id, lang=lang,
                                  output_path=output_path)

    def generate_gap_card(self, template_id: str, inputs: dict,
                          lang: str = "en",
                          output_path: Path | None = None):
        """Generate a visual gap-report card from template inputs."""
        from visuals import generate_gap_report_card
        draft = self.draft(template_id, inputs, lang=lang)
        img = generate_gap_report_card(draft, output_path=output_path)
        return img, draft

    def generate_coverage_chart(self, output_path: Path | None = None):
        """Generate a document coverage bar chart."""
        from visuals import generate_coverage_chart
        return generate_coverage_chart(self.stats(), output_path=output_path)

    def list_flowcharts(self, lang: str = "en") -> list[dict]:
        from visuals import list_flowcharts
        return list_flowcharts(lang)
