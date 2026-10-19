from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

SAMPLES_DIR = Path(__file__).parent / "sample_docs"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

PAGE_1_TEXT = """NEXUS ENTERPRISE SERVICE LEVEL AGREEMENT (SLA)
Document Ref: NEX-SLA-2026-V4 | Page 1

1. UPTIME COMMITMENT & INFRASTRUCTURE GUARANTEE
Nexus Cloud guarantees a 99.95% monthly uptime percentage for all Production API instances.
If the uptime falls below 99.95% in any billing cycle, the customer is eligible for service credits:
- 99.0% to 99.95%: 10% credit of the monthly subscription fee.
- 95.0% to 99.0%: 25% credit of the monthly subscription fee.
- Below 95.0%: 50% credit of the monthly subscription fee.

2. SUPPORT SEVERITY LEVELS & INCIDENT RESPONSE TIMES
- Severity 1 (Critical Outage): Production system down, core APIs unresponsive. Initial response within 15 minutes, 24/7/365.
- Severity 2 (Major Impairment): High impact with workaround available. Initial response within 2 hours during business hours.
- Severity 3 (Minor Inquiry): General configuration or feature request. Initial response within 8 hours.
"""

PAGE_2_TEXT = """NEXUS ENTERPRISE SERVICE LEVEL AGREEMENT (SLA)
Document Ref: NEX-SLA-2026-V4 | Page 2

3. DATA ENCRYPTION & SECURITY PROTOCOLS
All customer payload data is encrypted both in transit and at rest:
- Data in transit: Mandatory TLS 1.3 transport encryption across all external API endpoints.
- Data at rest: AES-256 encryption across all primary databases, object stores, and snapshots.
- Cryptographic keys are rotated automatically every 90 days via AWS KMS Hardware Security Modules (HSM).

4. COMPLIANCE & AUDIT ASSURANCE
Nexus undergoes annual independent third-party audits and maintains active compliance certifications:
- SOC 2 Type II compliance report available under NDA.
- ISO/IEC 27001:2022 certified information security management system.
- Full compliance with EU GDPR and California CCPA regulations.
"""

PAGE_3_TEXT = """NEXUS ENTERPRISE SERVICE LEVEL AGREEMENT (SLA)
Document Ref: NEX-SLA-2026-V4 | Page 3

5. ACCOUNT TERMINATION & DATA RETENTION SCHEDULE
Upon official termination or expiration of the subscription agreement:
- Immediate Account Freeze: Read-only access is preserved for 30 days following termination to allow data export.
- Customer Data Deletion: Following the 30-day grace period, all active production tenant data is cryptographically purged.
- Disaster Recovery Purge: Encrypted secondary backups and replica snapshots are permanently expunged within 90 days.
- A signed Certificate of Data Destruction is provided to Enterprise tier clients upon request.
"""

def generate_pdf():
    pdf_path = SAMPLES_DIR / "enterprise_sla_policy.pdf"
    txt_path = SAMPLES_DIR / "enterprise_sla_policy.txt"

    # Save consolidated plain text
    full_text = f"{PAGE_1_TEXT}\n\n{PAGE_2_TEXT}\n\n{PAGE_3_TEXT}"
    txt_path.write_text(full_text, encoding="utf-8")

    # Generate 3-page PDF
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    pages = [PAGE_1_TEXT, PAGE_2_TEXT, PAGE_3_TEXT]
    for idx, page_content in enumerate(pages, start=1):
        # Header banner
        c.setFillColor(colors.HexColor("#0F172A"))
        c.rect(0, height - 60, width, 60, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 38, f"Nexus Cloud SLA & Security Policy - Section {idx}")

        c.setFillColor(colors.HexColor("#1E293B"))
        c.setFont("Helvetica", 10)

        y = height - 90
        for line in page_content.splitlines():
            c.drawString(40, y, line)
            y -= 15

        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(40, 30, f"Confidential & Proprietary - Page {idx} of {len(pages)}")
        c.showPage()

    c.save()
    print(f"Sample PDF created at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
