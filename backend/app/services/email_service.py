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
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = settings.SMTP_FROM_EMAIL
        msg["To"] = to_email
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM_EMAIL, [to_email], msg.as_string())
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
    return send_email(to_email, subject, body)
