import logging
import smtplib
from email.mime.text import MIMEText

from app.core.config import settings

logger = logging.getLogger("email_service")


def send_email(to_email: str, subject: str, body: str) -> bool:
    """Best-effort send. Returns True/False, never raises — a mail outage must
    not block the transaction pipeline."""
    if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
        logger.warning("SMTP not configured — OTP email NOT sent to %s. Body: %s", to_email, body)
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
        logger.exception("Failed to send email to %s", to_email)
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
