import logging

import resend

from app.core.config import settings

logger = logging.getLogger("email_service")

# Resend uses HTTP — works on Render and all PaaS hosts that block raw SMTP.
resend.api_key = settings.RESEND_API_KEY


def send_otp_email(to_email: str, otp: str, transaction_id: str) -> bool:
    """Send OTP via Resend HTTP API. Returns True on success, False on failure.
    Never raises — a mail outage must not block the transaction pipeline."""
    if not settings.RESEND_API_KEY:
        logger.error(
            "RESEND_API_KEY is not configured — OTP email NOT sent to %s. "
            "Set RESEND_API_KEY in your environment variables.",
            to_email,
        )
        return False

    subject = "Your PaySim verification code"
    body = (
        f"Your one-time verification code is: {otp}\n"
        f"It expires in {settings.OTP_EXPIRE_MINUTES} minutes.\n"
        f"Transaction reference: {transaction_id}\n\n"
        "If you did not initiate this transaction, ignore this email — "
        "the transaction will not complete without this code."
    )

    try:
        resend.Emails.send({
            "from": settings.RESEND_FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "text": body,
        })
        logger.info("OTP email sent via Resend to %s (tx=%s)", to_email, transaction_id)
        return True
    except Exception:
        logger.exception("Resend failed to send OTP email to %s (tx=%s)", to_email, transaction_id)
        return False
