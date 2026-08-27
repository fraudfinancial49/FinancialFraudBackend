import logging
import smtplib
from email.mime.text import MIMEText

import resend

from app.core.config import settings

logger = logging.getLogger("email_service")

resend.api_key = settings.RESEND_API_KEY


def _send_via_resend(to_email: str, subject: str, body: str) -> bool:
    if not settings.RESEND_API_KEY:
        return False
    try:
        resend.Emails.send({
            "from": settings.RESEND_FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "text": body,
        })
        return True
    except Exception:
        logger.exception("Resend API send failed to %s — falling back to SMTP if configured.", to_email)
        return False


def _send_via_smtp(to_email: str, subject: str, body: str) -> bool:
    if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
        return False
    # Gmail (and most SMTP servers) require the From address to exactly match
    # the authenticated account. Fall back to SMTP_USERNAME when the config
    # hasn't been explicitly overridden from its placeholder default.
    from_addr = settings.SMTP_FROM_EMAIL or settings.SMTP_USERNAME
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_email
        raw = msg.as_string()
        if settings.SMTP_PORT == 465:
            # SSL (recommended for Render — port 587/STARTTLS is often blocked
            # on PaaS hosts). Set SMTP_PORT=465 in Render env vars to use this.
            with smtplib.SMTP_SSL(settings.SMTP_HOST, 465, timeout=10) as server:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.sendmail(from_addr, [to_email], raw)
        else:
            # STARTTLS (port 587) — works fine locally / on most VPS.
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                server.sendmail(from_addr, [to_email], raw)
        logger.info("SMTP email sent to %s via %s:%s", to_email, settings.SMTP_HOST, settings.SMTP_PORT)
        return True
    except Exception:
        logger.exception("SMTP send failed to %s", to_email)
        return False


def send_email(to_email: str, subject: str, body: str) -> bool:
    """Best-effort send. Returns True/False, never raises — a mail outage must
    not block the transaction pipeline. Tries Resend's HTTP API first (reliable
    from hosts that block raw SMTP), then falls back to SMTP if configured."""
    if _send_via_resend(to_email, subject, body):
        return True
    if _send_via_smtp(to_email, subject, body):
        return True
    logger.warning("No email delivery method configured/succeeded — OTP NOT sent to %s. Body: %s", to_email, body)
    return False


def send_otp_email(to_email: str, otp: str, transaction_id: str) -> bool:
    subject = "Your PaySim verification code"
    body = (
        f"Your one-time verification code is: {otp}\n"
        f"It expires in {settings.OTP_EXPIRE_MINUTES} minutes.\n"
        f"Transaction reference: {transaction_id}\n\n"
        "If you did not initiate this transaction, ignore this email — the transaction "
        "will not complete without this code."
    )
    # Always log the OTP prominently so it's visible in Render logs even if
    # email delivery fails. Remove this line before going to real production.
    logger.warning(
        "=== OTP DEBUG === transaction=%s recipient=%s OTP_CODE=%s ===",
        transaction_id, to_email, otp,
    )
    return send_email(to_email, subject, body)
