"""
GovRAG Copilot - Visual / Image Modality Module
=================================================
Adds the second modality (IMAGE) required by the course guidelines.

Generates professional compliance infographics and flowcharts from the
RAG pipeline's grounded answers and templates. Three visual types:

  1. Article Infographic  — visual summary of a PDPL article with key
     provisions rendered as an infographic card.
  2. Compliance Flowchart — step-by-step process diagram for common PDPL
     workflows (breach notification, consent lifecycle, data transfer).
  3. Gap Report Card     — visual compliance scorecard from template
     gap detection (green = provided, red = missing, with article refs).

All outputs are PNG images saved to disk and returned as PIL Image
objects. The UI (Gradio) can display them inline.

Technology: matplotlib + Pillow (no external API, no network, no key).
"""
from __future__ import annotations

import io
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / Colab

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from generator import GroundedAnswer
from templates_module import DraftResult, TEMPLATES


# ---------------------------------------------------------------------------
# Theme / style constants
# ---------------------------------------------------------------------------
SDAIA_GREEN = "#1B7A4A"
SDAIA_DARK  = "#0D3B23"
ACCENT_BLUE = "#2563EB"
ACCENT_RED  = "#DC2626"
ACCENT_AMBER = "#D97706"
BG_LIGHT    = "#F8FAFC"
BG_CARD     = "#FFFFFF"
TEXT_DARK   = "#1E293B"
TEXT_MUTED  = "#64748B"
BORDER      = "#E2E8F0"
SUCCESS_GREEN = "#16A34A"
FAIL_RED    = "#DC2626"


def _fig_to_image(fig) -> "Image.Image":
    """Convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    if Image is not None:
        return Image.open(buf).copy()
    return None  # type: ignore


def _save_image(img: "Image.Image", path: Path) -> Path:
    """Save a PIL image and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return path


def _wrap(text: str, width: int = 50) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _arabic_safe(text: str) -> str:
    """For matplotlib which has limited native Arabic support, we
    use the text as-is.  matplotlib.rcParams are set per-call to use
    a font that handles Arabic glyphs (DejaVu Sans has partial coverage).
    For best results we keep labels short."""
    return text


# ---------------------------------------------------------------------------
# 1. Article Infographic
# ---------------------------------------------------------------------------
def generate_article_infographic(
    answer: GroundedAnswer,
    title: str | None = None,
    output_path: Path | None = None,
) -> "Image.Image":
    """
    Generate a professional infographic card summarizing a grounded answer.

    The card shows:
      • Title (the user's question)
      • Key answer points (extracted from the answer text)
      • Citation references with article + page
      • A color-coded header bar in SDAIA green
    """
    if title is None:
        title = answer.query

    # Parse answer into bullet points
    lines = answer.answer.strip().split("\n")
    bullets = []
    for line in lines:
        line = line.strip().lstrip("- •·")
        if len(line) > 15:
            bullets.append(line.strip())
    if not bullets:
        bullets = [answer.answer[:300]]

    # Limit to 6 points for visual clarity
    bullets = bullets[:6]

    n_bullets = len(bullets)
    fig_height = max(5, 2.5 + n_bullets * 0.9)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    fig.set_facecolor(BG_LIGHT)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # Background card
    card = FancyBboxPatch(
        (0.3, 0.3), 9.4, fig_height - 0.6,
        boxstyle="round,pad=0.15",
        facecolor=BG_CARD, edgecolor=BORDER, linewidth=1.5,
    )
    ax.add_patch(card)

    # Header bar
    header = FancyBboxPatch(
        (0.3, fig_height - 1.6), 9.4, 1.3,
        boxstyle="round,pad=0.1",
        facecolor=SDAIA_GREEN, edgecolor="none",
    )
    ax.add_patch(header)

    # Title
    wrapped_title = _wrap(title, 55)
    ax.text(5, fig_height - 0.95, wrapped_title,
            fontsize=12, fontweight="bold", color="white",
            ha="center", va="center", family="sans-serif")

    # Shield icon (simple text emoji)
    ax.text(0.8, fig_height - 0.95, "", fontsize=18, ha="center", va="center")

    # Bullet points
    y = fig_height - 2.2
    icons = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]"]
    for i, bullet in enumerate(bullets):
        icon = icons[i % len(icons)]
        ax.text(1.0, y, icon, fontsize=12, ha="center", va="top")
        wrapped = _wrap(bullet.rstrip(".") + ".", 65)
        ax.text(1.5, y + 0.05, wrapped, fontsize=8.5, color=TEXT_DARK,
                va="top", family="sans-serif",
                linespacing=1.4)
        y -= 0.85

    # Citations footer
    if answer.citations:
        y -= 0.15
        ax.plot([0.8, 9.2], [y + 0.3, y + 0.3], color=BORDER, linewidth=0.8)
        ax.text(0.8, y, "References:", fontsize=7.5, fontweight="bold",
                color=TEXT_MUTED, va="top")
        y -= 0.3
        for c in answer.citations[:4]:
            ref = f"• {c['label']}"
            ax.text(1.0, y, ref, fontsize=7, color=TEXT_MUTED, va="top",
                    family="sans-serif")
            y -= 0.25

    # Footer branding
    ax.text(5, 0.15, "GovRAG Copilot — PDPL & SDAIA Guidance",
            fontsize=6.5, color=TEXT_MUTED, ha="center", style="italic")

    img = _fig_to_image(fig)
    if output_path and img:
        _save_image(img, output_path)
    return img


# ---------------------------------------------------------------------------
# 2. Compliance Flowchart
# ---------------------------------------------------------------------------

# Predefined PDPL workflows
FLOWCHARTS = {
    "breach_notification": {
        "title_en": "Personal Data Breach Notification Process",
        "title_ar": "عملية الإشعار بحادثة تسرب البيانات الشخصية",
        "article": "IR Article 24",
        "steps": [
            ("Breach\nDetected", ACCENT_RED),
            ("Assess\nSeverity &\nScope", ACCENT_AMBER),
            ("Notify\nCompetent\nAuthority\n(≤72 hrs)", SDAIA_GREEN),
            ("Document\nIncident\nDetails", ACCENT_BLUE),
            ("Notify\nAffected Data\nSubjects", SDAIA_GREEN),
            ("Implement\nCorrective\nMeasures", ACCENT_BLUE),
            ("File Report\nwith SDAIA", SDAIA_DARK),
        ],
    },
    "consent_lifecycle": {
        "title_en": "Consent Lifecycle under PDPL",
        "title_ar": "دورة حياة الموافقة وفق نظام حماية البيانات",
        "article": "PDPL Articles 6-8 + IR Articles 10-12",
        "steps": [
            ("Determine\nLegal Basis", ACCENT_BLUE),
            ("Design\nConsent\nMechanism", ACCENT_BLUE),
            ("Obtain\nExplicit\nConsent", SDAIA_GREEN),
            ("Record &\nStore\nConsent", ACCENT_BLUE),
            ("Process\nData per\nPurpose", SDAIA_GREEN),
            ("Handle\nWithdrawal\nRequest", ACCENT_AMBER),
            ("Cease\nProcessing &\nNotify Third\nParties", ACCENT_RED),
        ],
    },
    "cross_border_transfer": {
        "title_en": "Cross-Border Data Transfer Assessment",
        "title_ar": "تقييم نقل البيانات خارج المملكة",
        "article": "Transfer Regulation Articles 4-5",
        "steps": [
            ("Identify\nTransfer\nNeed", ACCENT_BLUE),
            ("Check\nAdequacy\nDecision", SDAIA_GREEN),
            ("Select\nTransfer\nMechanism\n(SCC/BCR)", ACCENT_BLUE),
            ("Conduct\nRisk\nAssessment", ACCENT_AMBER),
            ("Apply\nSafeguards", SDAIA_GREEN),
            ("Obtain\nApproval\n(if needed)", ACCENT_AMBER),
            ("Document &\nMonitor", SDAIA_DARK),
        ],
    },
    "data_subject_rights": {
        "title_en": "Data Subject Rights Exercise Process",
        "title_ar": "عملية ممارسة حقوق صاحب البيانات",
        "article": "PDPL Article 4 + IR Articles 5-8",
        "steps": [
            ("Data Subject\nSubmits\nRequest", ACCENT_BLUE),
            ("Verify\nIdentity", ACCENT_AMBER),
            ("Assess\nRequest\nType", ACCENT_BLUE),
            ("Process\nwithin\nLegal\nTimeframe", SDAIA_GREEN),
            ("Respond to\nData Subject", SDAIA_GREEN),
            ("Log &\nDocument\nAction", SDAIA_DARK),
        ],
    },
    "privacy_notice": {
        "title_en": "Privacy Notice Preparation Workflow",
        "title_ar": "سير عمل إعداد إشعار الخصوصية",
        "article": "PDPL Article 12 + IR Article 4",
        "steps": [
            ("Map Data\nProcessing\nActivities", ACCENT_BLUE),
            ("Identify\nLegal Basis\n& Purpose", ACCENT_BLUE),
            ("Define Data\nCategories &\nRetention", ACCENT_AMBER),
            ("List\nDisclosure\nParties", ACCENT_AMBER),
            ("Draft\nPrivacy\nNotice", SDAIA_GREEN),
            ("Publish &\nMake\nAccessible", SDAIA_GREEN),
            ("Review\nPeriodically", SDAIA_DARK),
        ],
    },
}


def generate_flowchart(
    flowchart_id: str,
    lang: str = "en",
    output_path: Path | None = None,
) -> "Image.Image":
    """
    Generate a horizontal process-flow diagram for a predefined PDPL workflow.
    """
    if flowchart_id not in FLOWCHARTS:
        raise ValueError(f"Unknown flowchart: {flowchart_id}. "
                         f"Available: {list(FLOWCHARTS)}")

    fc = FLOWCHARTS[flowchart_id]
    steps = fc["steps"]
    title = fc["title_ar"] if lang == "ar" else fc["title_en"]
    n = len(steps)

    fig_width = max(12, n * 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    fig.set_facecolor(BG_LIGHT)
    ax.set_xlim(-0.5, n * 2.0 + 0.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    # Title bar
    title_bar = FancyBboxPatch(
        (-0.3, 3.3), n * 2.0 + 0.6, 1.0,
        boxstyle="round,pad=0.1",
        facecolor=SDAIA_GREEN, edgecolor="none",
    )
    ax.add_patch(title_bar)
    ax.text(n * 1.0, 3.8, f"🛡️  {title}",
            fontsize=13, fontweight="bold", color="white",
            ha="center", va="center", family="sans-serif")
    ax.text(n * 1.0, 3.4, f"[{fc['article']}]",
            fontsize=8, color="#D1FAE5", ha="center", va="center",
            style="italic")

    # Draw steps as rounded boxes with arrows
    box_w, box_h = 1.6, 1.8
    for i, (label, color) in enumerate(steps):
        cx = i * 2.0 + 1.0
        cy = 1.5

        # Step box
        box = FancyBboxPatch(
            (cx - box_w/2, cy - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white", linewidth=2,
            alpha=0.9,
        )
        ax.add_patch(box)

        # Step number circle
        circle = plt.Circle((cx - box_w/2 + 0.2, cy + box_h/2 - 0.2),
                           0.18, color="white", zorder=5)
        ax.add_patch(circle)
        ax.text(cx - box_w/2 + 0.2, cy + box_h/2 - 0.2,
                str(i + 1), fontsize=8, fontweight="bold",
                color=color, ha="center", va="center", zorder=6)

        # Step label
        ax.text(cx, cy - 0.05, label,
                fontsize=8, color="white", fontweight="bold",
                ha="center", va="center", family="sans-serif",
                linespacing=1.3)

        # Arrow to next step
        if i < n - 1:
            ax.annotate(
                "", xy=(cx + box_w/2 + 0.35, cy),
                xytext=(cx + box_w/2 + 0.05, cy),
                arrowprops=dict(arrowstyle="->", color=TEXT_MUTED,
                                lw=2, connectionstyle="arc3,rad=0"),
            )

    # Footer
    ax.text(n * 1.0, -0.2,
            "GovRAG Copilot — Compliance Flowchart",
            fontsize=7, color=TEXT_MUTED, ha="center", style="italic")

    img = _fig_to_image(fig)
    if output_path and img:
        _save_image(img, output_path)
    return img


# ---------------------------------------------------------------------------
# 3. Gap Report Card (from template gap detection)
# ---------------------------------------------------------------------------
def generate_gap_report_card(
    draft_result: DraftResult,
    output_path: Path | None = None,
) -> "Image.Image":
    """
    Generate a visual compliance scorecard from template gap detection.
    Each field is rendered as a row: green checkmark if provided,
    red X if missing, with the requiring article reference.
    """
    template = TEMPLATES.get(draft_result.template_id)
    if template is None:
        raise ValueError(f"Unknown template: {draft_result.template_id}")

    lang = draft_result.lang
    title = template.title_ar if lang == "ar" else template.title_en
    fields = template.fields
    missing_keys = {m["key"] for m in draft_result.missing_fields}
    missing_map = {m["key"]: m for m in draft_result.missing_fields}

    n_fields = len(fields)
    n_ok = sum(1 for f in fields if f.required and f.key not in missing_keys)
    n_missing = sum(1 for f in fields if f.required and f.key in missing_keys)
    n_optional = sum(1 for f in fields if not f.required)
    n_required = n_ok + n_missing
    score_pct = (n_ok / n_required * 100) if n_required > 0 else 100

    fig_height = max(5, 2.5 + n_fields * 0.55)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    fig.set_facecolor(BG_LIGHT)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # Card background
    card = FancyBboxPatch(
        (0.2, 0.2), 9.6, fig_height - 0.4,
        boxstyle="round,pad=0.15",
        facecolor=BG_CARD, edgecolor=BORDER, linewidth=1.5,
    )
    ax.add_patch(card)

    # Header
    header_color = SUCCESS_GREEN if n_missing == 0 else (
        ACCENT_AMBER if score_pct >= 60 else FAIL_RED)
    header = FancyBboxPatch(
        (0.2, fig_height - 1.8), 9.6, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=header_color, edgecolor="none",
    )
    ax.add_patch(header)

    status_text = (
        "✅ All Required Fields Complete" if n_missing == 0
        else f"⚠️ {n_missing} Required Field{'s' if n_missing > 1 else ''} Missing"
    )
    ax.text(5, fig_height - 0.75, f"Compliance Check: {title}",
            fontsize=11, fontweight="bold", color="white",
            ha="center", va="center", family="sans-serif")
    ax.text(5, fig_height - 1.2, status_text,
            fontsize=10, color="white",
            ha="center", va="center", family="sans-serif")
    ax.text(5, fig_height - 1.55,
            f"Score: {n_ok}/{n_required} required fields ({score_pct:.0f}%)",
            fontsize=8, color="#D1FAE5" if n_missing == 0 else "#FEF3C7",
            ha="center", va="center", style="italic")

    # Field rows
    y = fig_height - 2.3
    for f in fields:
        is_missing = f.required and f.key in missing_keys
        is_optional = not f.required
        is_ok = f.required and f.key not in missing_keys

        # Status icon
        if is_ok:
            icon, icon_color = "✓", SUCCESS_GREEN
        elif is_missing:
            icon, icon_color = "✗", FAIL_RED
        else:
            icon, icon_color = "○", TEXT_MUTED

        label = f.label_ar if lang == "ar" else f.label_en
        suffix = " (optional)" if is_optional else " *"

        # Row background for missing fields
        if is_missing:
            row_bg = FancyBboxPatch(
                (0.5, y - 0.18), 9.0, 0.42,
                boxstyle="round,pad=0.05",
                facecolor="#FEF2F2", edgecolor="#FECACA", linewidth=0.5,
            )
            ax.add_patch(row_bg)

        ax.text(1.0, y, icon, fontsize=12, fontweight="bold",
                color=icon_color, ha="center", va="center",
                family="sans-serif")
        ax.text(1.5, y, f"{label}{suffix}",
                fontsize=8.5, color=TEXT_DARK if not is_optional else TEXT_MUTED,
                va="center", fontweight="bold" if is_missing else "normal",
                family="sans-serif")

        # Show the requiring article for missing fields
        if is_missing and f.key in missing_map:
            citation = missing_map[f.key].get("citation", "")
            if citation:
                ax.text(8.5, y, citation,
                        fontsize=6.5, color=FAIL_RED,
                        va="center", ha="right", style="italic",
                        family="sans-serif")

        y -= 0.5

    # Legend
    y -= 0.3
    ax.plot([0.8, 9.2], [y + 0.25, y + 0.25], color=BORDER, linewidth=0.5)
    legend_items = [
        ("✓", SUCCESS_GREEN, "Provided"),
        ("✗", FAIL_RED, "Missing (required)"),
        ("○", TEXT_MUTED, "Optional"),
    ]
    lx = 1.5
    for icon, color, desc in legend_items:
        ax.text(lx, y, f"{icon} {desc}", fontsize=7, color=color,
                va="center", family="sans-serif")
        lx += 2.5

    # Footer
    ax.text(5, 0.1, "GovRAG Copilot — Gap Report Card",
            fontsize=6.5, color=TEXT_MUTED, ha="center", style="italic")

    img = _fig_to_image(fig)
    if output_path and img:
        _save_image(img, output_path)
    return img


# ---------------------------------------------------------------------------
# 4. Document Coverage Chart
# ---------------------------------------------------------------------------
def generate_coverage_chart(
    stats: dict,
    output_path: Path | None = None,
) -> "Image.Image":
    """
    Generate a horizontal bar chart showing document coverage in the
    corpus (chunks per document, by language).
    """
    by_doc = stats.get("by_document", {})
    if not by_doc:
        return None  # type: ignore

    docs = sorted(by_doc.keys())
    values = [by_doc[d] for d in docs]
    colors = [SDAIA_GREEN if d.endswith("-EN") else ACCENT_BLUE for d in docs]

    fig, ax = plt.subplots(figsize=(8, max(3, len(docs) * 0.7)))
    fig.set_facecolor(BG_LIGHT)

    bars = ax.barh(docs, values, color=colors, edgecolor="white", height=0.55)
    ax.set_xlabel("Number of Article Chunks", fontsize=9, color=TEXT_DARK)
    ax.set_title("Document Coverage in GovRAG Copilot",
                 fontsize=12, fontweight="bold", color=SDAIA_DARK, pad=15)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9, fontweight="bold",
                color=TEXT_DARK)

    # Legend
    en_patch = mpatches.Patch(color=SDAIA_GREEN, label="English")
    ar_patch = mpatches.Patch(color=ACCENT_BLUE, label="Arabic / العربية")
    ax.legend(handles=[en_patch, ar_patch], loc="lower right", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    total = stats.get("total_chunks", sum(values))
    ax.text(0.98, 0.02, f"Total: {total} chunks",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED,
            ha="right", va="bottom", style="italic")

    plt.tight_layout()
    img = _fig_to_image(fig)
    if output_path and img:
        _save_image(img, output_path)
    return img


# ---------------------------------------------------------------------------
# Convenience: list available flowcharts
# ---------------------------------------------------------------------------
def list_flowcharts(lang: str = "en") -> list[dict]:
    return [
        {
            "id": fid,
            "title": fc["title_ar"] if lang == "ar" else fc["title_en"],
            "article": fc["article"],
            "steps": len(fc["steps"]),
        }
        for fid, fc in FLOWCHARTS.items()
    ]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline import GovRAGPipeline

    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "processed" / "visuals"
    out.mkdir(parents=True, exist_ok=True)

    pipe = GovRAGPipeline(root)

    # 1. Article infographic
    print("Generating article infographic ...")
    ans = pipe.answer("What are the responsibilities of a Data Protection Officer?")
    img1 = generate_article_infographic(ans, output_path=out / "infographic_dpo.png")
    print(f"  -> saved {out / 'infographic_dpo.png'}")

    # 2. Flowcharts
    for fid in FLOWCHARTS:
        print(f"Generating flowchart: {fid} ...")
        img = generate_flowchart(fid, lang="en", output_path=out / f"flowchart_{fid}.png")
        print(f"  -> saved {out / f'flowchart_{fid}.png'}")

    # 3. Gap report card
    print("Generating gap report card ...")
    draft = pipe.draft("privacy_notice", {"controller_name": "Acme"}, lang="en")
    img3 = generate_gap_report_card(draft, output_path=out / "gap_report_card.png")
    print(f"  -> saved {out / 'gap_report_card.png'}")

    # 4. Coverage chart
    print("Generating coverage chart ...")
    img4 = generate_coverage_chart(pipe.stats(), output_path=out / "coverage_chart.png")
    print(f"  -> saved {out / 'coverage_chart.png'}")

    print(f"\n✅ All visuals saved to {out}")
