"""
report_pdf.py — Compliance-Grade PDF Report Generator for TRACE
===============================================================
Generates high-fidelity, professional compliance PDF reports for AML investigations.
Adheres strictly to banking compliance documentation standards:
- Clean white background, restrained typography, subtle borders, professional navy accents
- Structured metadata: Report ID, Case ID, Subject Account, Timestamp
- Model assessment in compliance-safe language: "Model Risk Score", "Graph-based assessment"
- Zero prohibited legacy terminology
- Deterministic layout using ReportLab Platypus.

"""

import io
from datetime import datetime
from xml.sax.saxutils import escape

def escape_text(val) -> str:
    if val is None:
        return ''
    return escape(str(val))

def escape_multiline(val) -> str:
    if val is None:
        return ''
    return escape(str(val)).replace('\n', '<br/>')
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print 'Page X of Y' 
    and compliance confidentiality notices in the running footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running header on later pages
        if self._pageNumber > 1:
            self.drawString(54, 750, "TRACE AML Investigation Dossier")
            self.drawRightString(612 - 54, 750, f"Report ID: {getattr(self, 'report_id', 'REP')}")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        # Running footer
        footer_y = 36
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(54, footer_y + 12, 612 - 54, footer_y + 12)

        self.drawString(54, footer_y, "TRACE | Confidential & Proprietary — For Compliance & Investigation Purposes Only")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(612 - 54, footer_y, page_str)
        self.restoreState()


def format_currency(val):
    if not isinstance(val, (int, float)):
        return str(val)
    if val >= 1_000_000_000:
        return f"${val / 1_000_000_000:,.2f}B"
    if val >= 1_000_000:
        return f"${val / 1_000_000:,.2f}M"
    if val >= 1_000:
        return f"${val / 1_000:,.2f}K"
    return f"${val:,.2f}"


def generate_report_pdf(report: dict) -> bytes:
    """
    Renders an official compliance report dictionary into PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Color Palette: Professional Compliance Navy & Slate
    PRIMARY_NAVY = colors.HexColor("#0F172A")
    ACCENT_BLUE = colors.HexColor("#1D4ED8")
    SLATE_TEXT = colors.HexColor("#334155")
    MUTED_TEXT = colors.HexColor("#64748B")
    BORDER_COLOR = colors.HexColor("#E2E8F0")
    BG_LIGHT = colors.HexColor("#F8FAFC")

    # Risk colors
    risk_level = str(report.get("riskLevel", "MEDIUM")).upper()
    if risk_level == "HIGH":
        risk_badge_bg = colors.HexColor("#FEF2F2")
        risk_badge_fg = colors.HexColor("#DC2626")
        risk_border = colors.HexColor("#FECACA")
    elif risk_level == "LOW":
        risk_badge_bg = colors.HexColor("#F0FDF4")
        risk_badge_fg = colors.HexColor("#16A34A")
        risk_border = colors.HexColor("#BBF7D0")
    else:
        risk_badge_bg = colors.HexColor("#FFFBEB")
        risk_badge_fg = colors.HexColor("#D97706")
        risk_border = colors.HexColor("#FDE68A")

    # Typography styles
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PRIMARY_NAVY
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=MUTED_TEXT
    )
    style_h2 = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=PRIMARY_NAVY,
        spaceBefore=10,
        spaceAfter=4
    )
    style_body = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=SLATE_TEXT
    )
    style_mono = ParagraphStyle(
        "Mono_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=9,
        leading=12,
        textColor=PRIMARY_NAVY
    )
    style_meta_label = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=MUTED_TEXT
    )
    style_meta_val = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=9.5,
        leading=12,
        textColor=PRIMARY_NAVY
    )

    story = []

    # 1. Header Block: Organization & Classification
    header_table_data = [
        [
            Paragraph("TRACE &bull; FINANCIAL CRIME INTELLIGENCE<br/><font size='7' color='#64748B'>ANTI-MONEY LAUNDERING INVESTIGATION DIVISION</font>", style_subtitle),
            Paragraph(f"<b>REPORT ID:</b> {report.get('id', 'REP-2026-0001')}<br/><b>FILED DATE:</b> {report.get('filedDate', datetime.now().strftime('%Y-%m-%d'))}", ParagraphStyle("RightHdr", parent=style_body, alignment=2, fontSize=8, leading=10, textColor=MUTED_TEXT))
        ]
    ]
    header_table = Table(header_table_data, colWidths=[320, 184])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph("Investigation Adjudication Dossier", style_title))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_NAVY, spaceBefore=0, spaceAfter=12))

    # 2. Key Metadata Strip (Case ID, Account, Risk Level, Risk Score, Status)
    model_score = report.get("modelAssessment", {}).get("riskScore", 0.0)
    score_pct = f"{model_score * 100:.1f}%" if isinstance(model_score, (int, float)) else "N/A"
    model_assessment = report.get("modelAssessment", {})
    raw_thr = model_assessment.get("threshold", 0.3781)
    try:
        thr_str = f"{float(raw_thr):.4f}"
    except (ValueError, TypeError):
        thr_str = escape_text(str(raw_thr))

    meta_cells = [
        [
            Paragraph("CASE REFERENCE", style_meta_label),
            Paragraph("SUBJECT ACCOUNT", style_meta_label),
            Paragraph("MODEL RISK LEVEL", style_meta_label),
            Paragraph("MODEL RISK SCORE", style_meta_label),
        ],
        [
            Paragraph(escape_text(report.get("caseId", "CASE-2026-0000")), style_meta_val),
            Paragraph(escape_text(report.get("accountId", "UNKNOWN")), style_meta_val),
            Paragraph(f"<font color='{risk_badge_fg.hexval()}'><b>{escape_text(risk_level)}</b></font>", style_meta_val),
            Paragraph(f"<b>{score_pct}</b> (Threshold: {thr_str})", style_meta_val),
        ]
    ]
    meta_table = Table(meta_cells, colWidths=[126, 126, 126, 126])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
        ('TOPPADDING', (0, 1), (-1, 1), 2),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Section 1: Investigation Summary
    story.append(Paragraph("1. INVESTIGATION SUMMARY", style_h2))
    raw_summary = report.get("summary") or "Investigation opened on subject account following automated high-risk topological detection in the transaction graph."
    summary_text = escape_multiline(raw_summary)
    summary_table = Table([[Paragraph(summary_text, style_body)]], colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 4. Section 2: Observed Transaction Patterns
    story.append(Paragraph("2. OBSERVED TRANSACTION PATTERNS", style_h2))
    raw_patterns = report.get("evidence", {}).get("patterns") or report.get("patterns") or []
    if raw_patterns and isinstance(raw_patterns, list):
        rows = [
            [
                Paragraph("<b>Pattern Typology</b>", style_meta_label),
                Paragraph("<b>Observation Status</b>", style_meta_label),
                Paragraph("<b>Diagnostic Evidence & Analysis</b>", style_meta_label),
            ]
        ]
        for p in raw_patterns:
            if isinstance(p, str):
                p_name = escape_text(p)
                p_status = "Detected"
                p_desc = escape_text(f"Pattern '{p}' identified in case records.")
            elif isinstance(p, dict):
                p_name = escape_text(p.get("pattern", p.get("name", "Unknown Pattern")))
                p_status = escape_text(p.get("status", "Not detected"))
                p_desc = escape_text(p.get("evidence", p.get("description", p.get("notes", "Recorded during analysis."))))
            else:
                continue
            color = "#B45309" if p_status in ("Observed", "Detected", "Recorded") else "#64748B"
            rows.append([
                Paragraph(f"<b>{p_name}</b>", style_body),
                Paragraph(f"<font color='{color}'><b>{p_status}</b></font>", style_body),
                Paragraph(p_desc, style_body)
            ])
        pattern_data = rows
    else:
        raw_pattern = report.get("observedPattern")
        has_pattern = bool(raw_pattern and str(raw_pattern).strip() and str(raw_pattern).strip().lower() not in ("none", "none identified", "not available"))
        if has_pattern:
            pattern_name = escape_text(str(raw_pattern))
            pattern_status = "Recorded"
            pattern_status_color = "#B45309"
            pattern_desc = f"Pattern '{pattern_name}' registered from case file."
        else:
            pattern_name = "None identified"
            pattern_status = "Not detected"
            pattern_status_color = "#64748B"
            pattern_desc = "No anomalous structuring or laundering topology patterns detected."

        pattern_data = [
            [
                Paragraph("<b>Pattern Typology</b>", style_meta_label),
                Paragraph("<b>Observation Status</b>", style_meta_label),
                Paragraph("<b>Diagnostic Evidence & Analysis</b>", style_meta_label),
            ],
            [
                Paragraph(f"<b>{pattern_name}</b>", style_body),
                Paragraph(f"<font color='{pattern_status_color}'><b>{pattern_status}</b></font>", style_body),
                Paragraph(pattern_desc, style_body)
            ]
        ]
    pattern_table = Table(pattern_data, colWidths=[130, 94, 280])
    pattern_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(pattern_table)
    story.append(Spacer(1, 10))

    # 5. Section 3: Quantitative Evidence & Network Statistics
    story.append(Paragraph("3. QUANTITATIVE EVIDENCE & NETWORK STATISTICS", style_h2))
    ev = report.get("evidence", {})
    tot_vol = format_currency(ev.get("totalTransferred", 0))
    tx_count = str(ev.get("transactionCount", 0))
    conn_accs = str(ev.get("connectedAccountsCount", 0))

    ev_data = [
        [
            Paragraph("TOTAL TRANSACTIONS LOGGED", style_meta_label),
            Paragraph("CUMULATIVE TRANSACTION VOLUME", style_meta_label),
            Paragraph("DIRECT CONNECTED COUNTERPARTIES", style_meta_label),
        ],
        [
            Paragraph(f"<b>{tx_count}</b>", style_meta_val),
            Paragraph(f"<b>{tot_vol}</b>", style_meta_val),
            Paragraph(f"<b>{conn_accs}</b>", style_meta_val),
        ]
    ]
    ev_table = Table(ev_data, colWidths=[168, 168, 168])
    ev_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(ev_table)
    story.append(Spacer(1, 10))

    # 6. Section 4: Intelligence Engine Assessment (GraphSAGE GNN)
    story.append(Paragraph("4. GRAPH-BASED INTELLIGENCE ASSESSMENT", style_h2))
    gnn_note = (
        f"The relational Graph Neural Network (GraphSAGE inductive architecture) evaluated subject account "
        f"<b>{escape_text(report.get('accountId', ''))}</b> with a Model Risk Score of <b>{score_pct}</b> "
        f"relative to the calibrated operational threshold of <b>{thr_str}</b>. "
        f"This model-assessed risk serves as assistive intelligence to support the human investigator's adjudication."
    )
    gnn_table = Table([[Paragraph(gnn_note, style_body)]], colWidths=[504])
    gnn_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(gnn_table)
    story.append(Spacer(1, 10))

    # 7. Section 5: Investigator Findings & Compliance Decision
    story.append(Paragraph("5. COMPLIANCE ADJUDICATION & INVESTIGATOR FINDINGS", style_h2))
    decision_val = report.get("decision") or "Continue Surveillance"
    raw_notes = report.get("notes")
    if not raw_notes or not str(raw_notes).strip():
        notes_val = "None recorded"
    else:
        notes_val = str(raw_notes).strip()
    investigator_name = report.get("investigatorName") or "Harish GG (Lead AML Investigator)"

    decision_data = [
        [
            Paragraph("ADJUDICATED DECISION", style_meta_label),
            Paragraph(f"<b><font color='{ACCENT_BLUE.hexval()}'>{escape_text(decision_val)}</font></b>", style_body)
        ],
        [
            Paragraph("INVESTIGATOR NOTES & FINDINGS", style_meta_label),
            Paragraph(escape_multiline(notes_val), style_body)
        ],
        [
            Paragraph("ADJUDICATING INVESTIGATOR", style_meta_label),
            Paragraph(f"<b>{escape_text(investigator_name)}</b> &mdash; TRACE Compliance Division", style_body)
        ]
    ]
    dec_table = Table(decision_data, colWidths=[140, 364])
    dec_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0, 0), (0, -1), BG_LIGHT),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(dec_table)
    story.append(Spacer(1, 14))

    # Build PDF with dynamic NumberedCanvas for header/footer
    def canvas_maker(*args, **kwargs):
        c = NumberedCanvas(*args, **kwargs)
        c.report_id = report.get("id", "REP-2026-0001")
        return c

    doc.build(story, canvasmaker=canvas_maker)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
