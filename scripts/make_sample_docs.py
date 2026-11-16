"""Generate fictional sample PDFs for demos and tests (requires reportlab)."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "sample_docs"

VENDOR = [
    ("Master Services Agreement - Acme Analytics Pvt Ltd", [
        ("1. Term", "This Agreement begins on 1 April 2026 and continues for an initial term of twenty-four (24) months. "
                    "It renews automatically for successive twelve-month periods unless either party gives notice of non-renewal."),
        ("2. Payment Terms", "The Client shall pay all undisputed invoices within thirty (30) days of receipt. "
                             "Late payments accrue interest at 1.5 percent per month. Fees are quoted in Indian Rupees and exclude GST."),
    ]),
    ("", [
        ("3. Termination", "Either party may terminate this Agreement for convenience by giving ninety (90) days written notice. "
                           "Either party may terminate immediately if the other commits a material breach and fails to cure it within thirty (30) days of notice."),
        ("4. Confidentiality", "Each party shall keep the other party's confidential information secret for five (5) years after termination "
                               "and shall use it only to perform this Agreement."),
        ("5. Liability", "Total liability of the Vendor under this Agreement is capped at the fees paid in the twelve months preceding the claim."),
    ]),
]
POLICY = [
    ("Employee Leave Policy", [
        ("Annual Leave", "Full-time employees receive 24 days of paid annual leave per calendar year. Unused leave up to 10 days may be carried forward."),
        ("Sick Leave", "Employees may take up to 12 days of paid sick leave per year. A medical certificate is required for absences longer than three consecutive days."),
        ("Parental Leave", "Primary caregivers receive 26 weeks of paid parental leave; secondary caregivers receive 4 weeks."),
    ]),
]


def build(name: str, pages):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT / name), pagesize=A4)
    story = []
    for i, (title, sections) in enumerate(pages):
        if i:
            story.append(PageBreak())
        if title:
            story += [Paragraph(title, styles["Title"]), Spacer(1, 12)]
        for head, body in sections:
            story += [Paragraph(head, styles["Heading2"]), Paragraph(body, styles["BodyText"]), Spacer(1, 8)]
    doc.build(story)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build("vendor_agreement.pdf", VENDOR)
    build("leave_policy.pdf", POLICY)
    print("Wrote sample PDFs to", OUT)
