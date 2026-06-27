"""
GovRAG Copilot - Gradio Demo UI
================================
Implements the demo described in the project proposal:
  • Bilingual Q&A (Arabic / English toggle, with auto-detect)
  • Template-driven drafting with the four PDPL artifact templates
  • Missing-info checklist (gap detection)
  • Inline citations with article + page references
  • Export of drafts as Markdown

Run from the project root:
    python ui/app.py
Then open http://localhost:7860 in your browser.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

# Make `src/` importable when running this file directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from pipeline import GovRAGPipeline  # noqa: E402
from templates_module import TEMPLATES  # noqa: E402
from ingest import detect_lang  # noqa: E402
from visuals import (  # noqa: E402
    generate_article_infographic, generate_flowchart,
    generate_gap_report_card, generate_coverage_chart,
    FLOWCHARTS, list_flowcharts,
)


# ---------------------------------------------------------------------------
# Pipeline (built once on app start)
# ---------------------------------------------------------------------------
print("Loading GovRAG pipeline ...")
PIPELINE = GovRAGPipeline(ROOT, backend="auto")
STATS = PIPELINE.stats()
print(f"  -> ready. Backend: {STATS['backend']}, "
      f"chunks: {STATS['total_chunks']}")


# ---------------------------------------------------------------------------
# UI strings (bilingual)
# ---------------------------------------------------------------------------
UI = {
    "en": {
        "title": "GovRAG Copilot — PDPL & SDAIA Guidance Assistant",
        "subtitle": ("Evidence-grounded, bilingual answers and drafting for "
                     "Saudi Arabia's Personal Data Protection Law."),
        "tab_qa": "💬 Q&A",
        "tab_draft": "📝 Drafting",
        "tab_search": "🔍 Search",
        "tab_about": "ℹ️ About",
        "ask_label": "Your question",
        "ask_ph": "e.g. Within how many hours must a personal data breach be notified?",
        "lang_label": "Language",
        "lang_auto": "Auto-detect",
        "lang_en": "English",
        "lang_ar": "العربية",
        "doc_filter": "Limit to documents (optional)",
        "k_label": "Passages to retrieve",
        "ask_btn": "Ask",
        "answer": "Answer",
        "citations": "Citations",
        "draft_template": "Choose a template",
        "draft_btn": "Generate Draft",
        "draft_export": "Download Markdown",
        "missing": "⚠️ Missing required fields",
        "search_btn": "Search",
        "search_results": "Top retrieved passages",
        "lang_btn": "العربية",
    },
    "ar": {
        "title": "GovRAG Copilot — مساعد إرشادي لنظام حماية البيانات الشخصية",
        "subtitle": ("إجابات وصياغات ثنائية اللغة مستندة إلى نظام حماية "
                     "البيانات الشخصية في المملكة العربية السعودية."),
        "tab_qa": "💬 الأسئلة والأجوبة",
        "tab_draft": "📝 الصياغة",
        "tab_search": "🔍 البحث",
        "tab_about": "ℹ️ حول",
        "ask_label": "سؤالك",
        "ask_ph": "مثال: خلال كم ساعة يجب الإشعار بحادثة تسرب البيانات؟",
        "lang_label": "اللغة",
        "lang_auto": "اكتشاف تلقائي",
        "lang_en": "English",
        "lang_ar": "العربية",
        "doc_filter": "تقييد على وثائق (اختياري)",
        "k_label": "عدد المقاطع المسترجعة",
        "ask_btn": "اسأل",
        "answer": "الإجابة",
        "citations": "المراجع",
        "draft_template": "اختر قالباً",
        "draft_btn": "أنشئ المسودة",
        "draft_export": "تنزيل Markdown",
        "missing": "⚠️ حقول مطلوبة ناقصة",
        "search_btn": "بحث",
        "search_results": "أفضل المقاطع المسترجعة",
        "lang_btn": "English",
    },
}

DOC_CHOICES = [
    ("PDPL — English",                      "PDPL-EN"),
    ("Implementing Regulation — English",   "IR-EN"),
    ("Transfer Regulation — English",       "TR-EN"),
    ("نظام حماية البيانات — العربية",         "PDPL-AR"),
    ("اللائحة التنفيذية — العربية",            "IR-AR"),
    ("لائحة نقل البيانات — العربية",           "TR-AR"),
]


# ---------------------------------------------------------------------------
# Q&A handler
# ---------------------------------------------------------------------------
def handle_question(question: str, lang_choice: str, doc_filter: list[str],
                    k: int):
    if not question or not question.strip():
        return "_Please enter a question._", "", ""

    lang = None
    if lang_choice == "English":
        lang = "en"
    elif lang_choice in ("العربية", "Arabic"):
        lang = "ar"
    # else: auto

    df = doc_filter if doc_filter else None
    answer = PIPELINE.answer(question, lang=lang, k=int(k), doc_filter=df)

    # Format the answer
    answer_md = answer.answer

    # Format citations as a table
    if answer.citations:
        rows = ["| # | Reference | Snippet |", "|---|---|---|"]
        for i, c in enumerate(answer.citations, 1):
            snippet = c["snippet"].replace("|", "\\|").replace("\n", " ")
            rows.append(f"| {c.get('marker', f'#{i}')} | **{c['label']}** | "
                        f"{snippet} |")
        cite_md = "\n".join(rows)
    else:
        cite_md = "_No citations available._"

    info = (f"_Backend:_ `{answer.backend}` &nbsp;·&nbsp; "
            f"_Detected language:_ `{answer.lang}` &nbsp;·&nbsp; "
            f"_Hits considered:_ `{k}`")

    return answer_md, cite_md, info


# ---------------------------------------------------------------------------
# Drafting handler
# ---------------------------------------------------------------------------
def list_template_choices(lang: str) -> list[tuple[str, str]]:
    return [
        (t.title_ar if lang == "ar" else t.title_en, t.template_id)
        for t in TEMPLATES.values()
    ]


def get_template_field_layout(template_id: str, lang: str):
    """Return the metadata for building one Textbox per field dynamically."""
    if not template_id or template_id not in TEMPLATES:
        return []
    template = TEMPLATES[template_id]
    return [
        {
            "key": f.key,
            "label": (f.label_ar if lang == "ar" else f.label_en) +
                     (" *" if f.required else ""),
            "hint": f.hint_ar if lang == "ar" else f.hint_en,
            "required": f.required,
        }
        for f in template.fields
    ]


def handle_draft(template_id: str, lang_choice: str, *field_values):
    """All field Textboxes are passed as positional args."""
    if not template_id:
        return "_Please choose a template._", "", None

    lang = "ar" if lang_choice == "العربية" else "en"
    template = TEMPLATES[template_id]

    # Map positional values back to field keys
    inputs = {
        f.key: (v.strip() if isinstance(v, str) else "")
        for f, v in zip(template.fields, field_values)
    }

    res = PIPELINE.draft(template_id, inputs, lang=lang)

    # Build the draft markdown
    draft_md = "```\n" + res.draft + "\n```\n"

    # Missing fields block
    if res.missing_fields:
        miss = ["**⚠️ Missing required fields:**\n"]
        for m in res.missing_fields:
            miss.append(f"- **{m['label']}** — required by *{m['citation']}*: "
                        f"_{m['snippet']}_")
        miss_md = "\n".join(miss)
    else:
        miss_md = ("✅ **All required fields provided.**" if lang == "en"
                   else "✅ **جميع الحقول المطلوبة مكتملة.**")

    # Citations block
    if res.citations:
        cite_lines = ["\n**Grounding citations:**\n"]
        for c in res.citations:
            cite_lines.append(f"- **{c['label']}** — _{c['snippet']}_")
        miss_md += "\n" + "\n".join(cite_lines)

    # Export file
    export_path = ROOT / "data" / "processed" / f"draft_{template_id}_{lang}.md"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8") as f:
        f.write(f"# {template.title_ar if lang == 'ar' else template.title_en}\n\n")
        f.write(res.draft)
        f.write("\n\n---\n\n")
        if res.missing_fields:
            f.write("## Missing required fields\n\n")
            for m in res.missing_fields:
                f.write(f"- **{m['label']}** — {m['citation']}: {m['snippet']}\n")
            f.write("\n")
        f.write("## Citations\n\n")
        for c in res.citations:
            f.write(f"- {c['label']} — {c['snippet']}\n")

    return draft_md, miss_md, str(export_path)


# ---------------------------------------------------------------------------
# Search (raw retrieval) handler
# ---------------------------------------------------------------------------
def handle_search(query: str, lang_choice: str, k: int):
    if not query or not query.strip():
        return "_Please enter a search query._"

    lang = None
    if lang_choice == "English":
        lang = "en"
    elif lang_choice == "العربية":
        lang = "ar"

    hits = PIPELINE.search(query, k=int(k), lang=lang)
    if not hits:
        return "_No matches found._"

    rows = ["| Rank | Reference | Score | Snippet |",
            "|---|---|---|---|"]
    for i, h in enumerate(hits, 1):
        snippet = h.chunk.text[:220].replace("|", "\\|").replace("\n", " ")
        if len(h.chunk.text) > 220:
            snippet += "…"
        rows.append(f"| {i} | **{h.chunk.citation_label()}** | "
                    f"{h.score:.3f} | {snippet} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Visual handlers (image modality)
# ---------------------------------------------------------------------------
def handle_infographic(question: str, lang_choice: str):
    """Generate an article infographic from a Q&A answer."""
    if not question or not question.strip():
        return None, "_Please enter a question._"
    lang = None
    if lang_choice == "English":
        lang = "en"
    elif lang_choice == "العربية":
        lang = "ar"
    ans = PIPELINE.answer(question, lang=lang)
    img = generate_article_infographic(ans)
    return img, ans.to_markdown()


def handle_flowchart(flowchart_id: str, lang_choice: str):
    """Generate a compliance flowchart."""
    if not flowchart_id:
        return None
    lang = "ar" if lang_choice == "العربية" else "en"
    img = generate_flowchart(flowchart_id, lang=lang)
    return img


def handle_coverage_chart():
    """Generate the document coverage chart."""
    img = generate_coverage_chart(PIPELINE.stats())
    return img


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    ui = UI["en"]   # primary UI is bilingual labels but defaults to EN strings

    CUSTOM_CSS = """
    .gradio-container { max-width: 1200px !important; }
    .header-block h1 { margin-bottom: 0; }
    .header-block p  { color: #666; margin-top: 4px; }
    .stat-pill {
        display: inline-block; background: #eef2ff; color: #3b3b8c;
        padding: 2px 10px; border-radius: 12px; margin-right: 6px;
        font-size: 12px;
    }
    """

    with gr.Blocks(
        title="GovRAG Copilot",
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CUSTOM_CSS,
    ) as app:
        # Header
        gr.Markdown(
            f"""
            <div class="header-block">
              <h1>🛡️ GovRAG Copilot</h1>
              <p>Evidence-grounded PDPL & SDAIA guidance — bilingual (Arabic / English) — runs locally, no API keys.</p>
              <span class="stat-pill">backend: {STATS['backend']}</span>
              <span class="stat-pill">chunks: {STATS['total_chunks']}</span>
              <span class="stat-pill">EN: {STATS['by_language'].get('en', 0)}</span>
              <span class="stat-pill">AR: {STATS['by_language'].get('ar', 0)}</span>
            </div>
            """
        )

        with gr.Tabs():
            # =========================================================
            # TAB 1: Q & A
            # =========================================================
            with gr.Tab("💬 Q&A"):
                gr.Markdown(
                    "Ask any question about the PDPL, its Implementing "
                    "Regulation, or the Cross-Border Transfer Regulation. "
                    "The answer will cite the article and page it came from."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        question = gr.Textbox(
                            label="Your question / سؤالك",
                            placeholder="e.g. What are the responsibilities of "
                                        "a Data Protection Officer? / "
                                        "ما مسؤوليات مسؤول حماية البيانات؟",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        lang_dd = gr.Dropdown(
                            choices=["Auto-detect", "English", "العربية"],
                            value="Auto-detect", label="Language",
                        )
                        k_slider = gr.Slider(
                            minimum=3, maximum=10, value=6, step=1,
                            label="Passages to retrieve",
                        )
                with gr.Accordion("Filter by document (optional)", open=False):
                    doc_filter = gr.CheckboxGroup(
                        choices=DOC_CHOICES, label="",
                    )
                ask_btn = gr.Button("Ask / اسأل", variant="primary")

                gr.Markdown("### Answer / الإجابة")
                answer_box = gr.Markdown()
                info_box = gr.Markdown()
                gr.Markdown("### Citations / المراجع")
                cite_box = gr.Markdown()

                # Example questions
                gr.Examples(
                    examples=[
                        ["What are the responsibilities of a Data Protection Officer?",
                         "English", [], 6],
                        ["Within how many hours must a personal data breach be notified?",
                         "English", [], 6],
                        ["What conditions must be met to transfer personal data outside the Kingdom?",
                         "English", ["TR-EN", "PDPL-EN"], 6],
                        ["ما مسؤوليات مسؤول حماية البيانات الشخصية؟",
                         "العربية", [], 6],
                        ["خلال كم ساعة يجب الإشعار بحادثة تسرب البيانات؟",
                         "العربية", [], 6],
                        ["ما الحقوق التي يتمتع بها صاحب البيانات الشخصية؟",
                         "العربية", [], 6],
                    ],
                    inputs=[question, lang_dd, doc_filter, k_slider],
                    label="Example questions / أمثلة",
                )

                ask_btn.click(
                    handle_question,
                    inputs=[question, lang_dd, doc_filter, k_slider],
                    outputs=[answer_box, cite_box, info_box],
                )

            # =========================================================
            # TAB 2: Drafting
            # =========================================================
            with gr.Tab("📝 Drafting"):
                gr.Markdown(
                    "Generate PDPL artifacts (privacy notice, ROPA entry, breach "
                    "notification, transfer assessment) from your inputs. "
                    "Missing required fields are flagged with the article that "
                    "mandates them. Required fields are marked with `*`."
                )
                with gr.Row():
                    template_dd = gr.Dropdown(
                        choices=list_template_choices("en"),
                        label="Template / القالب", value="privacy_notice",
                    )
                    draft_lang = gr.Radio(
                        choices=["English", "العربية"], value="English",
                        label="Output language / لغة المخرجات",
                    )

                # We render up to N field boxes; only the active template's
                # fields are made visible.
                MAX_FIELDS = 12
                field_boxes: list[gr.Textbox] = []
                for i in range(MAX_FIELDS):
                    field_boxes.append(gr.Textbox(
                        label=f"field {i+1}", visible=False, lines=1,
                    ))

                draft_btn = gr.Button("Generate Draft / أنشئ المسودة",
                                      variant="primary")

                gr.Markdown("### Draft / المسودة")
                draft_out = gr.Markdown()
                gr.Markdown("### Compliance Check / فحص الامتثال")
                miss_out = gr.Markdown()
                draft_file = gr.File(label="Download as Markdown",
                                     visible=True, interactive=False)

                # Update fields when template changes
                def update_field_layout(template_id: str, lang_choice: str):
                    lang = "ar" if lang_choice == "العربية" else "en"
                    layout = get_template_field_layout(template_id, lang)
                    updates = []
                    for i in range(MAX_FIELDS):
                        if i < len(layout):
                            f = layout[i]
                            updates.append(gr.update(
                                label=f["label"],
                                placeholder=f.get("hint", "") or "",
                                visible=True,
                                value="",
                            ))
                        else:
                            updates.append(gr.update(visible=False, value=""))
                    return updates

                # Initial render on load: fill in fields for default template
                template_dd.change(
                    update_field_layout,
                    inputs=[template_dd, draft_lang],
                    outputs=field_boxes,
                )
                draft_lang.change(
                    update_field_layout,
                    inputs=[template_dd, draft_lang],
                    outputs=field_boxes,
                )

                draft_btn.click(
                    handle_draft,
                    inputs=[template_dd, draft_lang] + field_boxes,
                    outputs=[draft_out, miss_out, draft_file],
                )

            # =========================================================
            # TAB 3: Raw search
            # =========================================================
            with gr.Tab("🔍 Search"):
                gr.Markdown(
                    "Inspect the raw output of the hybrid retriever (BM25 + "
                    "TF-IDF). Useful for debugging or for finding the exact "
                    "article that defines a term."
                )
                with gr.Row():
                    s_query = gr.Textbox(
                        label="Search query / استعلام البحث",
                        placeholder="e.g. data minimisation",
                        lines=1,
                    )
                    s_lang = gr.Dropdown(
                        choices=["Auto-detect", "English", "العربية"],
                        value="Auto-detect", label="Language",
                    )
                    s_k = gr.Slider(minimum=3, maximum=15, value=8, step=1,
                                    label="# results")
                s_btn = gr.Button("Search / بحث", variant="primary")
                s_results = gr.Markdown()

                s_btn.click(
                    handle_search,
                    inputs=[s_query, s_lang, s_k],
                    outputs=s_results,
                )

            # =========================================================
            # TAB 4: Visuals (Image Modality)
            # =========================================================
            with gr.Tab("🖼️ Visuals"):
                gr.Markdown(
                    "**Image Modality** — generate compliance infographics, "
                    "PDPL process flowcharts, and document coverage charts. "
                    "These visuals can be downloaded and used in reports "
                    "or presentations."
                )

                with gr.Tabs():
                    # Sub-tab: Article Infographic
                    with gr.Tab("Article Infographic"):
                        gr.Markdown(
                            "Enter a PDPL question and get a visual "
                            "infographic card summarizing the answer."
                        )
                        with gr.Row():
                            info_query = gr.Textbox(
                                label="Question",
                                placeholder="e.g. What are the responsibilities of a DPO?",
                                lines=1, scale=3,
                            )
                            info_lang = gr.Dropdown(
                                choices=["Auto-detect", "English", "العربية"],
                                value="Auto-detect", label="Language",
                                scale=1,
                            )
                        info_btn = gr.Button(
                            "Generate Infographic", variant="primary")
                        info_img = gr.Image(label="Infographic", type="pil")
                        info_txt = gr.Markdown(label="Answer text")
                        info_btn.click(
                            handle_infographic,
                            inputs=[info_query, info_lang],
                            outputs=[info_img, info_txt],
                        )

                    # Sub-tab: Compliance Flowcharts
                    with gr.Tab("Compliance Flowcharts"):
                        gr.Markdown(
                            "Predefined PDPL process flowcharts — select a "
                            "workflow to visualize."
                        )
                        fc_choices = [
                            (fc["title_en"], fid)
                            for fid, fc in FLOWCHARTS.items()
                        ]
                        with gr.Row():
                            fc_dd = gr.Dropdown(
                                choices=fc_choices,
                                label="Workflow",
                                value="breach_notification",
                                scale=3,
                            )
                            fc_lang = gr.Dropdown(
                                choices=["English", "العربية"],
                                value="English", label="Language",
                                scale=1,
                            )
                        fc_btn = gr.Button(
                            "Generate Flowchart", variant="primary")
                        fc_img = gr.Image(label="Flowchart", type="pil")
                        fc_btn.click(
                            handle_flowchart,
                            inputs=[fc_dd, fc_lang],
                            outputs=[fc_img],
                        )

                    # Sub-tab: Coverage Chart
                    with gr.Tab("Coverage Chart"):
                        gr.Markdown(
                            "Visualize the document corpus — how many "
                            "article chunks per document and language."
                        )
                        cov_btn = gr.Button(
                            "Generate Coverage Chart", variant="primary")
                        cov_img = gr.Image(label="Coverage", type="pil")
                        cov_btn.click(
                            handle_coverage_chart,
                            inputs=[],
                            outputs=[cov_img],
                        )

            # =========================================================
            # TAB 5: About
            # =========================================================
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
### About GovRAG Copilot

**GovRAG Copilot** is an evidence-grounded compliance assistant for Saudi
Arabia's Personal Data Protection Law (PDPL), its Implementing Regulation,
and the Regulation on Personal Data Transfer Outside the Kingdom (issued
by SDAIA).

#### Why this matters
PDPL compliance work — privacy notices, ROPA entries, breach procedures,
DPO obligations, cross-border transfer safeguards — spans multiple
SDAIA-issued documents. A grounded assistant can speed up drafting,
improve consistency, and provide audit-ready traceability.

#### How it works
```
PDF/Archive → text + page-level metadata
            → Article-aware splitter (EN/AR regex)
            → Hybrid index (BM25 + TF-IDF, Arabic-normalised)
            → Retriever (top-k, language preference, document filter)
            → Generator (Extractive | Ollama | HF Transformers)
            → Grounded answer with [#N] citations
```

#### Backends (no API keys required)
- **Extractive** — pure-Python, faithful by construction. Default fallback.
- **Ollama** — local LLM via `http://localhost:11434`. Recommended:
  `qwen2.5:7b-instruct` (bilingual) or `aya:8b` (Arabic-tuned).
- **HuggingFace Transformers** — open-weight models loaded directly
  (e.g. `Qwen/Qwen2.5-1.5B-Instruct`).

Set `GOVRAG_BACKEND=ollama|hf|extractive` to force a backend.

#### Documents indexed
- Personal Data Protection Law (EN + AR)
- Implementing Regulation of the PDPL (EN + AR)
- Regulation on Personal Data Transfer Outside the Kingdom (EN + AR)

**Currently indexed:** {STATS['total_chunks']} article-level chunks
({STATS['by_language'].get('en', 0)} English, {STATS['by_language'].get('ar', 0)} Arabic).

#### Important
This is a research prototype. It does not provide legal advice and its
output should be reviewed by a qualified compliance officer or legal
counsel before being used in production.
                """)

        # Trigger initial field layout for the default template on load
        app.load(
            update_field_layout,
            inputs=[template_dd, draft_lang],
            outputs=field_boxes,
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
