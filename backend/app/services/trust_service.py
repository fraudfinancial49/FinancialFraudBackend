"""
Trust Isolation Contract:
  - `read_trust_score` is the ONLY function callable from the transaction
    assessment path. It never writes anything.
  - `record_confirmed_outcome` may ONLY be called after a real-world
    confirmation (OTP success, admin vault override, manual review) — never
    from raw model output. This is enforced structurally: the transactions
    router never imports `record_confirmed_outcome`, only vault.py does.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.db import models

DEFAULT_TRUST_SCORE = 50.0  # neutral prior for accounts with no confirmed history


def read_trust_score(db: Session, account_id: str) -> float:
    latest: Optional[models.TrustHistory] = (
        db.query(models.TrustHistory)
        .filter(models.TrustHistory.account_id == account_id)
        .order_by(models.TrustHistory.created_at.desc())
        .first()
    )
    return float(latest.trust_score) if latest is not None else DEFAULT_TRUST_SCORE


def record_confirmed_outcome(
    db: Session, account_id: str, trust_score: float, outcome_source: str
) -> models.TrustHistory:
    if outcome_source not in ("otp_verified", "admin_override", "manual_review"):
        raise ValueError(f"Invalid outcome_source for trust write: {outcome_source}")
    entry = models.TrustHistory(
        account_id=account_id,
        trust_score=trust_score,
        outcome_source=outcome_source,
        created_at=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
