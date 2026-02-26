"""PDF report generation for ATS resume analysis."""

from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from modules.scorer import ATSResult


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "<br/>".join(f"- {item}" for item in items)


def generate_pdf_report(candidate_name: str, ats_result: ATSResult, job_title: str = "") -> bytes:
    """Build and return a downloadable PDF report as bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    heading = styles["Heading1"]
    body = styles["BodyText"]
    body.leading = 14
    section = ParagraphStyle("Section", parent=styles["Heading3"], spaceBefore=10)

    story = [
        Paragraph("AI Resume Analyzer - ATS Report", heading),
        Spacer(1, 8),
        Paragraph(f"<b>Candidate:</b> {candidate_name}", body),
        Paragraph(f"<b>Job Title:</b> {job_title or 'N/A'}", body),
        Paragraph(f"<b>ATS Match Score:</b> {ats_result.score:.2f}%", body),
        Spacer(1, 12),
    ]

    summary_data = [
        ["Metric", "Value"],
        ["Matched Skills", str(len(ats_result.matched_skills))],
        ["Missing Skills", str(len(ats_result.missing_skills))],
        ["Overall Score", f"{ats_result.score:.2f}%"],
        ["SVM Confidence", f"{ats_result.confidence_score:.2f}%"],
    ]

    summary_table = Table(summary_data, colWidths=[180, 120])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.75, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )

    story.extend(
        [
            summary_table,
            Spacer(1, 14),
            Paragraph("Strength Analysis", section),
            Paragraph(ats_result.strength_analysis, body),
            Paragraph("Weakness Analysis", section),
            Paragraph(ats_result.weakness_analysis, body),
            Paragraph("Matched Skills", section),
            Paragraph(_bullet_list(ats_result.matched_skills), body),
            Paragraph("Missing Skills", section),
            Paragraph(_bullet_list(ats_result.missing_skills), body),
        ]
    )

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
