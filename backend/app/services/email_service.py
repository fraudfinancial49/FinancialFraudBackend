import logging

import requests
import resend

from app.core.config import settings

logger = logging.getLogger("email_service")

resend.api_key = settings.RESEND_API_KEY


def _send_via_brevo(to_email: str, subject: str, body: str) -> bool:
    """Send via Brevo HTTP API — works on Render, no domain needed, sends to ANY email."""
    if not settings.BREVO_API_KEY or not settings.BREVO_SENDER_EMAIL:
        return False
    try:
        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "api-key": settings.BREVO_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "sender": {
                    "name": settings.BREVO_SENDER_NAME,
                    "email": settings.BREVO_SENDER_EMAIL,
                },
                "to": [{"email": to_email}],
                "subject": subject,
                "textContent": body,
            },
            timeout=15,
        )
        if response.status_code in (200, 201):
            logger.info("OTP email sent via Brevo to %s", to_email)
            return True
        logger.error(
            "Brevo API returned %s for %s: %s",
            response.status_code, to_email, response.text,
        )
        return False
    except Exception:
        logger.exception("Brevo send failed to %s", to_email)
        return False


def _send_via_resend(to_email: str, subject: str, body: str) -> bool:
    """Fallback: Resend HTTP API."""
    if not settings.RESEND_API_KEY:
        return False
    try:
        resend.Emails.send({
            "from": settings.RESEND_FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "text": body,
        })
        logger.info("OTP email sent via Resend to %s", to_email)
        return True
    except Exception:
        logger.exception("Resend send failed to %s", to_email)
        return False


def send_otp_email(to_email: str, otp: str, transaction_id: str) -> bool:
    """Send OTP email. Tries Brevo first, then Resend as fallback.
    Returns True on success, False on failure. Never raises."""
    subject = "Your FinWallet verification code"
    body = (
        f"Your one-time verification code is: {otp}\n"
        f"It expires in {settings.OTP_EXPIRE_MINUTES} minutes.\n"
        f"Transaction reference: {transaction_id}\n\n"
        "If you did not initiate this transaction, ignore this email — "
        "the transaction will not complete without this code."
    )
    if _send_via_brevo(to_email, subject, body):
        return True
    if _send_via_resend(to_email, subject, body):
        return True
    logger.error(
        "All email delivery methods failed — OTP NOT sent to %s. "
        "Check BREVO_API_KEY and BREVO_SENDER_EMAIL in environment variables.",
        to_email,
    )
    return False
