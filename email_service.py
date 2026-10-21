"""
email_service.py — Real Configurable SMTP Dispatch Boundary for TRACE
======================================================================
Provides genuine email dispatch for compliance investigation reports.
Configured strictly through environment variables:
- SMTP_HOST: Hostname of SMTP relay
- SMTP_PORT: Port (default: 587 for STARTTLS or 465 for SSL)
- SMTP_USERNAME: Username for authentication
- SMTP_PASSWORD: Password for authentication
- SMTP_FROM: Sender email address (default: compliance@trace.internal)
- SMTP_USE_TLS: Boolean flag ('true'/'false', default: true)

Guarantees honest failure handling with zero fake success states:
- Unconfigured relay -> explicit 'smtp_unconfigured' notification
- Invalid recipient -> 'invalid_recipient' 400 error
- SMTP connection or auth failure -> exact honest diagnosis
"""

import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Tuple, Dict, Any, Optional

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    if not email or not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def get_smtp_config() -> Dict[str, Any]:
    host = os.environ.get("SMTP_HOST", "").strip()
    port_str = os.environ.get("SMTP_PORT", "").strip()
    username = os.environ.get("SMTP_USERNAME", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr = os.environ.get("SMTP_FROM", "compliance@trace.internal").strip()
    use_tls_str = os.environ.get("SMTP_USE_TLS", "true").strip().lower()

    port = 587
    if port_str.isdigit():
        port = int(port_str)

    use_tls = use_tls_str not in ("0", "false", "no")

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "from_addr": from_addr,
        "use_tls": use_tls,
        "is_configured": bool(host),
    }


def send_report_email(
    report: Dict[str, Any],
    pdf_bytes: bytes,
    recipient_email: str
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Sends an official compliance report with the attached PDF to the recipient.
    Returns: (success: bool, status_code_or_reason: str, details: dict)
    """
    clean_recipient = (recipient_email or "").strip()

    # 1. Recipient Validation
    if not is_valid_email(clean_recipient):
        return False, "invalid_recipient", {
            "success": False,
            "status": "invalid_recipient",
            "message": f"Invalid recipient email address: '{clean_recipient}'."
        }

    # 2. Configuration Check
    cfg = get_smtp_config()
    if not cfg["is_configured"]:
        return False, "smtp_unconfigured", {
            "success": False,
            "status": "smtp_unconfigured",
            "message": "Email delivery is not configured. Set SMTP_HOST and credentials in the server environment."
        }

    # 3. PDF Byte Validation
    if not pdf_bytes or len(pdf_bytes) < 100:
        return False, "pdf_generation_failed", {
            "success": False,
            "status": "pdf_generation_failed",
            "message": "Email delivery could not proceed because the report PDF could not be generated."
        }

    rep_id = report.get("id", "REP-2026-UNKNOWN")
    case_id = report.get("caseId", "CASE-UNKNOWN")
    account_id = report.get("accountId", "UNKNOWN")
    risk_level = report.get("riskLevel", "MEDIUM")
    risk_score = report.get("modelAssessment", {}).get("riskScore", 0.0)
    score_pct = f"{risk_score * 100:.1f}%" if isinstance(risk_score, (int, float)) else "N/A"
    decision = report.get("decision", "Escalate for Further Review")

    # 4. Compose Email Message
    msg = MIMEMultipart()
    msg["Subject"] = f"TRACE Investigation Report — {rep_id}"
    msg["From"] = cfg["from_addr"]
    msg["To"] = clean_recipient

    body_text = f"""TRACE AML Investigation & Financial Crime Intelligence
Official Compliance Investigation Dossier Dispatch
=============================================================

Report ID:            {rep_id}
Case ID:              {case_id}
Subject Account:      {account_id}
Model Risk Level:     {risk_level}
Model Risk Score:     {score_pct} (Operating Threshold: 0.3781)
Investigator Decision:{decision}

Please find the attached compliance report PDF ({f"TRACE_REPORT_{rep_id}.pdf"}).
This document contains the official investigation summary, quantitative transaction evidence,
observed AML pattern diagnostics, and investigator findings.

CONFIDENTIALITY NOTICE:
This transmission and attached PDF are strictly confidential, legally protected, and intended
solely for compliance and regulatory review purposes. Unauthorized dissemination is prohibited.
"""
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    # Attach PDF
    pdf_attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
    pdf_attachment.add_header(
        "Content-Disposition",
        "attachment",
        filename=f"TRACE_REPORT_{rep_id}.pdf"
    )
    msg.attach(pdf_attachment)

    # 5. Dispatch via SMTP
    try:
        if cfg["port"] == 465:
            server = smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=10)
        else:
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=10)
            if cfg["use_tls"]:
                server.starttls()

        if cfg["username"] and cfg["password"]:
            server.login(cfg["username"], cfg["password"])

        server.send_message(msg)
        server.quit()

        return True, "sent", {
            "success": True,
            "status": "sent",
            "message": f"Investigation report {rep_id} successfully dispatched to {clean_recipient}.",
            "recipient": clean_recipient,
            "reportId": rep_id
        }

    except smtplib.SMTPAuthenticationError as e:
        return False, "smtp_auth_error", {
            "success": False,
            "status": "smtp_auth_error",
            "message": f"Email delivery failed: SMTP authentication error ({str(e)})."
        }
    except (smtplib.SMTPConnectError, ConnectionRefusedError, OSError) as e:
        return False, "smtp_connect_error", {
            "success": False,
            "status": "smtp_connect_error",
            "message": f"Email delivery failed: Unable to connect to SMTP host {cfg['host']}:{cfg['port']} ({str(e)})."
        }
    except Exception as e:
        return False, "smtp_send_error", {
            "success": False,
            "status": "smtp_send_error",
            "message": f"Email delivery failed: {str(e)}."
        }
