"""
GovRAG Copilot - Compliance Templates & Gap Detection
======================================================
Implements two of the proposal's "novel" features:

  • Template-driven drafting — fills PDPL artifacts (privacy notice clauses,
    ROPA entries, breach notification text, transfer SCC checklist) from
    user-supplied facts, with article citations grounded in the corpus.

  • Gap detection — given user inputs for a compliance task, flags missing
    mandatory fields per the PDPL/Implementing Regulation requirements and
    cites the article that mandates each field.

Each template defines its mandatory fields and the article that requires them
so the assistant can both draft *and* audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from index import HybridRetriever
from generator import GroundedAnswer, _short_snippet


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------
@dataclass
class FieldSpec:
    key: str
    label_en: str
    label_ar: str
    required: bool
    grounded_query: str   # query used to fetch the article that requires this field
    hint_en: str = ""
    hint_ar: str = ""


@dataclass
class TemplateSpec:
    template_id: str
    title_en: str
    title_ar: str
    description_en: str
    description_ar: str
    fields: list[FieldSpec]
    render: Callable[[dict, str], str]   # (inputs, lang) -> draft body


# --- 1. Privacy Notice (Article 12 of PDPL + Article 4 of Implementing Reg.) ---
def _render_privacy_notice(inputs: dict, lang: str) -> str:
    g = lambda k, fb="[__]": (inputs.get(k) or fb)
    if lang == "ar":
        return (
            f"إشعار الخصوصية لـ {g('controller_name')}\n\n"
            f"1. الغرض من جمع البيانات: {g('purpose')}.\n"
            f"2. الأساس النظامي للمعالجة: {g('legal_basis')}.\n"
            f"3. فئات البيانات الشخصية التي تُجمع: {g('data_categories')}.\n"
            f"4. مدة الاحتفاظ بالبيانات: {g('retention_period')}.\n"
            f"5. الجهات التي قد تُفصح إليها البيانات: {g('disclosure_parties')}.\n"
            f"6. هل ستُنقل البيانات خارج المملكة: {g('cross_border', 'لا')}.\n"
            f"7. حقوق صاحب البيانات: تشمل الحق في الاطلاع، التصحيح، الإتلاف، "
            f"وسحب الموافقة، وتُمارَس عبر التواصل مع: {g('contact_channel')}.\n"
            f"8. مسؤول حماية البيانات (إن وجد): {g('dpo_contact', 'غير معيّن')}.\n"
        )
    return (
        f"Privacy Notice for {g('controller_name')}\n\n"
        f"1. Purpose of collection: {g('purpose')}.\n"
        f"2. Legal basis for processing: {g('legal_basis')}.\n"
        f"3. Categories of personal data collected: {g('data_categories')}.\n"
        f"4. Retention period: {g('retention_period')}.\n"
        f"5. Parties to whom data may be disclosed: {g('disclosure_parties')}.\n"
        f"6. Whether data will be transferred outside the Kingdom: "
        f"{g('cross_border', 'No')}.\n"
        f"7. Data Subject rights: access, correction, destruction, withdrawal "
        f"of consent — exercised by contacting: {g('contact_channel')}.\n"
        f"8. Data Protection Officer (if any): {g('dpo_contact', 'not appointed')}.\n"
    )


PRIVACY_NOTICE = TemplateSpec(
    template_id="privacy_notice",
    title_en="Privacy Notice (PDPL Art. 12 / IR Art. 4)",
    title_ar="إشعار الخصوصية (المادة 12 من النظام / المادة 4 من اللائحة)",
    description_en="Notice that must be given to data subjects before collection.",
    description_ar="الإشعار الذي يجب تقديمه لصاحب البيانات قبل الجمع.",
    fields=[
        FieldSpec("controller_name", "Controller name", "اسم جهة التحكم", True,
                  "controller identification privacy notice"),
        FieldSpec("purpose", "Purpose of processing", "الغرض من المعالجة", True,
                  "purpose of collection privacy policy Article 12"),
        FieldSpec("legal_basis", "Legal basis", "الأساس النظامي", True,
                  "legal basis lawful processing Article 6"),
        FieldSpec("data_categories", "Data categories", "فئات البيانات", True,
                  "categories of personal data processed",
                  hint_en="e.g. name, contact info, identification numbers"),
        FieldSpec("retention_period", "Retention period", "مدة الاحتفاظ", True,
                  "retention period personal data minimum"),
        FieldSpec("disclosure_parties", "Disclosure parties", "جهات الإفصاح", True,
                  "disclosure of personal data to third parties Article 15"),
        FieldSpec("cross_border", "Cross-border transfer?", "النقل خارج المملكة؟",
                  False, "cross border transfer outside Kingdom"),
        FieldSpec("contact_channel", "Contact channel for rights",
                  "قناة ممارسة الحقوق", True,
                  "data subject rights exercise contact"),
        FieldSpec("dpo_contact", "DPO contact", "مسؤول حماية البيانات", False,
                  "data protection officer contact"),
    ],
    render=_render_privacy_notice,
)


# --- 2. ROPA entry (IR Art. 31) ---
def _render_ropa(inputs: dict, lang: str) -> str:
    g = lambda k, fb="[__]": (inputs.get(k) or fb)
    if lang == "ar":
        return (
            f"سجل أنشطة المعالجة — {g('activity_name')}\n\n"
            f"- جهة التحكم: {g('controller_name')}\n"
            f"- اسم النشاط: {g('activity_name')}\n"
            f"- الغرض: {g('purpose')}\n"
            f"- الأساس النظامي: {g('legal_basis')}\n"
            f"- فئات أصحاب البيانات: {g('data_subject_categories')}\n"
            f"- فئات البيانات الشخصية: {g('data_categories')}\n"
            f"- جهات الإفصاح / المعالجون: {g('processors_recipients')}\n"
            f"- النقل خارج المملكة: {g('cross_border', 'لا')}\n"
            f"- مدة الاحتفاظ: {g('retention_period')}\n"
            f"- الضوابط الأمنية: {g('security_controls')}\n"
        )
    return (
        f"ROPA Entry — {g('activity_name')}\n\n"
        f"- Controller: {g('controller_name')}\n"
        f"- Activity: {g('activity_name')}\n"
        f"- Purpose: {g('purpose')}\n"
        f"- Legal basis: {g('legal_basis')}\n"
        f"- Data subject categories: {g('data_subject_categories')}\n"
        f"- Personal data categories: {g('data_categories')}\n"
        f"- Recipients / processors: {g('processors_recipients')}\n"
        f"- Cross-border transfer: {g('cross_border', 'No')}\n"
        f"- Retention period: {g('retention_period')}\n"
        f"- Security controls: {g('security_controls')}\n"
    )


ROPA_ENTRY = TemplateSpec(
    template_id="ropa_entry",
    title_en="Record of Processing Activity (IR Art. 31)",
    title_ar="سجل أنشطة المعالجة (المادة 31 من اللائحة)",
    description_en="Per-activity ROPA entry as required by the Implementing Reg.",
    description_ar="مدخل ROPA لكل نشاط، كما تتطلبه اللائحة التنفيذية.",
    fields=[
        FieldSpec("controller_name", "Controller", "جهة التحكم", True,
                  "ROPA controller identification Article 31"),
        FieldSpec("activity_name", "Activity name", "اسم النشاط", True,
                  "processing activity record"),
        FieldSpec("purpose", "Purpose", "الغرض", True,
                  "purpose of processing ROPA"),
        FieldSpec("legal_basis", "Legal basis", "الأساس النظامي", True,
                  "legal basis processing Article 6"),
        FieldSpec("data_subject_categories", "Data subject categories",
                  "فئات أصحاب البيانات", True,
                  "categories of data subjects ROPA"),
        FieldSpec("data_categories", "Data categories", "فئات البيانات", True,
                  "categories of personal data ROPA"),
        FieldSpec("processors_recipients", "Recipients / processors",
                  "المعالجون والمستلمون", True,
                  "processors recipients disclosure ROPA"),
        FieldSpec("cross_border", "Cross-border transfer?", "نقل خارج المملكة؟",
                  False, "transfer outside Kingdom ROPA"),
        FieldSpec("retention_period", "Retention period", "مدة الاحتفاظ", True,
                  "retention period ROPA Article 19"),
        FieldSpec("security_controls", "Security controls", "الضوابط الأمنية",
                  True, "security controls protective measures Article 21"),
    ],
    render=_render_ropa,
)


# --- 3. Breach Notification (IR Art. 24) ---
def _render_breach(inputs: dict, lang: str) -> str:
    g = lambda k, fb="[__]": (inputs.get(k) or fb)
    if lang == "ar":
        return (
            f"إشعار حادثة تسرب بيانات شخصية\n\n"
            f"إلى: الجهة المختصة (SDAIA)\n"
            f"من: {g('controller_name')}\n\n"
            f"1. وصف الحادثة: {g('incident_description')}\n"
            f"   - تاريخ ووقت الحادثة: {g('incident_datetime')}\n"
            f"   - تاريخ ووقت العلم بالحادثة: {g('aware_datetime')}\n"
            f"2. فئات البيانات المتأثرة: {g('affected_categories')}\n"
            f"3. عدد أصحاب البيانات المتأثرين (تقريبي): {g('affected_count')}\n"
            f"4. الآثار المحتملة: {g('potential_impact')}\n"
            f"5. التدابير المتخذة أو المقترحة للتخفيف: {g('mitigation')}\n"
            f"6. هل تم إشعار أصحاب البيانات؟ {g('subjects_notified', 'لا')}\n"
            f"7. جهة الاتصال: {g('contact')}\n"
            f"\nتنبيه: يجب تقديم هذا الإشعار خلال 72 ساعة من العلم بالحادثة "
            f"وفقاً للمادة 24 من اللائحة التنفيذية."
        )
    return (
        f"Personal Data Breach Notification\n\n"
        f"To: Competent Authority (SDAIA)\n"
        f"From: {g('controller_name')}\n\n"
        f"1. Incident description: {g('incident_description')}\n"
        f"   - Date/time of incident: {g('incident_datetime')}\n"
        f"   - Date/time of awareness: {g('aware_datetime')}\n"
        f"2. Categories of data affected: {g('affected_categories')}\n"
        f"3. Approximate number of affected data subjects: {g('affected_count')}\n"
        f"4. Potential consequences: {g('potential_impact')}\n"
        f"5. Mitigation measures taken/proposed: {g('mitigation')}\n"
        f"6. Have data subjects been notified? {g('subjects_notified', 'No')}\n"
        f"7. Contact: {g('contact')}\n"
        f"\nReminder: this notification must be submitted within 72 hours of "
        f"becoming aware, per IR Article 24."
    )


BREACH_NOTIFICATION = TemplateSpec(
    template_id="breach_notification",
    title_en="Personal Data Breach Notification (IR Art. 24)",
    title_ar="إشعار حادثة تسرب بيانات (المادة 24 من اللائحة)",
    description_en="Notification to SDAIA within 72 hours of awareness.",
    description_ar="إشعار للهيئة خلال 72 ساعة من العلم بالحادثة.",
    fields=[
        FieldSpec("controller_name", "Controller", "جهة التحكم", True,
                  "controller breach notification"),
        FieldSpec("incident_description", "Incident description", "وصف الحادثة",
                  True, "description of personal data breach incident"),
        FieldSpec("incident_datetime", "Incident date/time",
                  "تاريخ ووقت الحادثة", True,
                  "time and date of breach"),
        FieldSpec("aware_datetime", "Awareness date/time",
                  "تاريخ العلم بالحادثة", True,
                  "time controller became aware breach"),
        FieldSpec("affected_categories", "Categories of data affected",
                  "فئات البيانات المتأثرة", True,
                  "categories of personal data breach"),
        FieldSpec("affected_count", "Number affected (approx.)",
                  "عدد المتأثرين", True,
                  "number of data subjects affected"),
        FieldSpec("potential_impact", "Potential impact",
                  "الآثار المحتملة", True,
                  "potential consequences of breach"),
        FieldSpec("mitigation", "Mitigation measures",
                  "تدابير التخفيف", True,
                  "measures to mitigate adverse effects breach"),
        FieldSpec("subjects_notified", "Subjects notified?",
                  "إشعار الأصحاب؟", False,
                  "notification of data subjects breach"),
        FieldSpec("contact", "Contact", "جهة الاتصال", True,
                  "contact details breach notification"),
    ],
    render=_render_breach,
)


# --- 4. Cross-border transfer assessment (Transfer Regulation Art. 4) ---
def _render_transfer_assessment(inputs: dict, lang: str) -> str:
    g = lambda k, fb="[__]": (inputs.get(k) or fb)
    if lang == "ar":
        return (
            f"ملخص تقييم مخاطر نقل البيانات خارج المملكة\n\n"
            f"- جهة التحكم: {g('controller_name')}\n"
            f"- الدولة المستقبلة: {g('destination_country')}\n"
            f"- الجهة المستقبلة: {g('recipient_entity')}\n"
            f"- الغرض من النقل: {g('transfer_purpose')}\n"
            f"- فئات البيانات المنقولة: {g('data_categories')}\n"
            f"- آلية النقل المستخدمة: {g('transfer_mechanism')}\n"
            f"  (مثل: قرار كفاية، شروط تعاقدية معيارية SCC، قواعد مؤسسية ملزمة BCR)\n"
            f"- الضمانات المطبقة: {g('safeguards')}\n"
            f"- المخاطر المحددة وتدابير التخفيف: {g('risks_and_mitigations')}\n"
            f"- نتيجة التقييم: {g('assessment_outcome')}\n"
        )
    return (
        f"Cross-Border Transfer Risk Assessment Summary\n\n"
        f"- Controller: {g('controller_name')}\n"
        f"- Destination country: {g('destination_country')}\n"
        f"- Recipient entity: {g('recipient_entity')}\n"
        f"- Transfer purpose: {g('transfer_purpose')}\n"
        f"- Data categories: {g('data_categories')}\n"
        f"- Transfer mechanism: {g('transfer_mechanism')}\n"
        f"  (e.g. adequacy decision, Standard Contractual Clauses, BCR)\n"
        f"- Safeguards in place: {g('safeguards')}\n"
        f"- Identified risks and mitigations: {g('risks_and_mitigations')}\n"
        f"- Assessment outcome: {g('assessment_outcome')}\n"
    )


TRANSFER_ASSESSMENT = TemplateSpec(
    template_id="transfer_assessment",
    title_en="Cross-Border Transfer Assessment (Transfer Reg. Art. 4)",
    title_ar="تقييم نقل البيانات خارج المملكة (المادة 4 من لائحة النقل)",
    description_en="Risk assessment summary required before transferring data outside KSA.",
    description_ar="ملخص تقييم المخاطر المطلوب قبل نقل البيانات خارج المملكة.",
    fields=[
        FieldSpec("controller_name", "Controller", "جهة التحكم", True,
                  "controller cross-border transfer"),
        FieldSpec("destination_country", "Destination country",
                  "الدولة المستقبلة", True, "destination country transfer"),
        FieldSpec("recipient_entity", "Recipient entity",
                  "الجهة المستقبلة", True, "recipient entity transfer"),
        FieldSpec("transfer_purpose", "Purpose of transfer",
                  "غرض النقل", True, "purpose of transfer Article 4"),
        FieldSpec("data_categories", "Data categories", "فئات البيانات",
                  True, "categories of data transferred"),
        FieldSpec("transfer_mechanism", "Transfer mechanism",
                  "آلية النقل", True,
                  "transfer mechanism standard contractual clauses BCR"),
        FieldSpec("safeguards", "Safeguards", "الضمانات", True,
                  "appropriate safeguards transfer"),
        FieldSpec("risks_and_mitigations", "Risks and mitigations",
                  "المخاطر والتخفيف", True,
                  "risk assessment transfer mitigation"),
        FieldSpec("assessment_outcome", "Assessment outcome",
                  "نتيجة التقييم", True,
                  "transfer risk assessment outcome conclusion"),
    ],
    render=_render_transfer_assessment,
)


TEMPLATES: dict[str, TemplateSpec] = {
    t.template_id: t for t in [
        PRIVACY_NOTICE, ROPA_ENTRY, BREACH_NOTIFICATION, TRANSFER_ASSESSMENT,
    ]
}


# ---------------------------------------------------------------------------
# Drafting + gap detection
# ---------------------------------------------------------------------------
@dataclass
class DraftResult:
    template_id: str
    draft: str
    missing_fields: list[dict]   # [{key, label, citation, snippet}]
    citations: list[dict]
    lang: str

    def to_markdown(self) -> str:
        out = [self.draft, ""]
        if self.missing_fields:
            out.append("**⚠️ Missing required fields / حقول مطلوبة ناقصة:**")
            for m in self.missing_fields:
                out.append(f"- **{m['label']}** — required by {m['citation']}: "
                           f"_{m['snippet']}_")
            out.append("")
        if self.citations:
            out.append("**Grounding citations / المراجع:**")
            for c in self.citations:
                out.append(f"- {c['label']} — _{c['snippet']}_")
        return "\n".join(out)


def draft_with_gaps(template_id: str, inputs: dict, retriever: HybridRetriever,
                    lang: str = "en") -> DraftResult:
    template = TEMPLATES[template_id]

    # Identify missing required fields and ground each one
    missing: list[dict] = []
    citations: dict[str, dict] = {}
    for fld in template.fields:
        if fld.required and not (inputs.get(fld.key) or "").strip():
            hits = retriever.retrieve(fld.grounded_query, k=1, lang=lang)
            if hits:
                hit = hits[0]
                cit = {
                    "label": hit.chunk.citation_label(),
                    "snippet": _short_snippet(hit.chunk.text, 160),
                }
                missing.append({
                    "key": fld.key,
                    "label": fld.label_ar if lang == "ar" else fld.label_en,
                    "citation": cit["label"],
                    "snippet": cit["snippet"],
                })
                citations[hit.chunk.chunk_id] = {
                    "label": cit["label"],
                    "snippet": cit["snippet"],
                }

    # Always ground the *template itself* with one or two anchor citations
    anchor_query = template.title_en  # use the article reference in the title
    for h in retriever.retrieve(anchor_query, k=2, lang=lang):
        citations[h.chunk.chunk_id] = {
            "label": h.chunk.citation_label(),
            "snippet": _short_snippet(h.chunk.text, 160),
        }

    draft = template.render(inputs, lang)
    return DraftResult(
        template_id=template_id,
        draft=draft,
        missing_fields=missing,
        citations=list(citations.values())[:6],
        lang=lang,
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from index import HybridRetriever

    root = Path(__file__).resolve().parents[1]
    retriever = HybridRetriever.load(root / "data" / "index" / "hybrid.pkl")

    # Demo: privacy notice with a deliberately incomplete input set
    inputs = {
        "controller_name": "Acme KSA Ltd.",
        "purpose": "delivering food orders to customers",
        "data_categories": "name, phone, address, payment data",
        "contact_channel": "privacy@acme.sa",
        # legal_basis, retention_period, disclosure_parties intentionally missing
    }
    print("="*72)
    print("DEMO 1: Privacy Notice draft + gap detection (English)")
    print("="*72)
    res = draft_with_gaps("privacy_notice", inputs, retriever, lang="en")
    print(res.to_markdown())

    print("\n" + "="*72)
    print("DEMO 2: Breach notification (Arabic, all fields present)")
    print("="*72)
    inputs_ar = {
        "controller_name": "شركة أكمي",
        "incident_description": "وصول غير مصرح به إلى قاعدة بيانات العملاء عبر بيانات اعتماد مسربة",
        "incident_datetime": "2026-04-30 02:15 AST",
        "aware_datetime": "2026-04-30 06:00 AST",
        "affected_categories": "اسم، بريد إلكتروني، رقم هاتف",
        "affected_count": "حوالي 4,200",
        "potential_impact": "محاولات تصيد، احتيال هوية",
        "mitigation": "إعادة تعيين كلمات المرور، تفعيل MFA، مراجعة السجلات",
        "subjects_notified": "نعم، عبر البريد الإلكتروني",
        "contact": "dpo@acme.sa",
    }
    res = draft_with_gaps("breach_notification", inputs_ar, retriever, lang="ar")
    print(res.to_markdown())
