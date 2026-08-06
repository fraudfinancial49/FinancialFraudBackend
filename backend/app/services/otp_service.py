
import hashlib
import logging
import random
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import models

logger = logging.getLogger("otp_service")


def _hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def generate_and_store_otp(db: Session, transaction_id: str) -> str:
    """Creates a fresh numeric OTP, persists only its hash, returns the plaintext
    so the caller can email it immediately."""
    code = "".join(random.choices("0123456789", k=settings.OTP_LENGTH))
    row = models.OTPCode(
        transaction_id=transaction_id,
        code_hash=_hash(code),
        expires_at=datetime.utcnow() + timedelta(minutes=settings.OTP_EXPIRE_MINUTES),
    )
    db.add(row)
    db.commit()
    return code


def verify_otp(db: Session, transaction_id: str, submitted_code: str) -> tuple[bool, str]:
    """Returns (is_valid, reason). Consumes the OTP row on success or on final failed attempt."""
    otp_row = (
        db.query(models.OTPCode)
        .filter(models.OTPCode.transaction_id == transaction_id, models.OTPCode.consumed == False)  # noqa: E712
        .order_by(models.OTPCode.created_at.desc())
        .first()
    )
    if otp_row is None:
        return False, "No active OTP for this transaction — it may already be resolved."
    if datetime.utcnow() > otp_row.expires_at:
        otp_row.consumed = True
        db.commit()
        return False, "OTP expired."
    if otp_row.attempts >= otp_row.max_attempts:
        otp_row.consumed = True
        db.commit()
        return False, "Too many incorrect attempts — OTP locked."

    if _hash(submitted_code) != otp_row.code_hash:
        otp_row.attempts += 1
        db.commit()
        remaining = otp_row.max_attempts - otp_row.attempts
        return False, f"Incorrect code. {max(remaining, 0)} attempt(s) remaining."

    otp_row.consumed = True
    db.commit()
    return True, "OK"
