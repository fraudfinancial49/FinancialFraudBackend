
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.db import models
from app.core.config import settings

logger = logging.getLogger("honeypot_service")


def _get_or_create_profile(db: Session, browser_fingerprint: str, simulated_ip: str) -> models.AttackerProfile:
    profile = (
        db.query(models.AttackerProfile)
        .filter(models.AttackerProfile.browser_fingerprint == browser_fingerprint)
        .first()
    )
    if profile is None:
        profile = models.AttackerProfile(
            browser_fingerprint=browser_fingerprint,
            first_seen_ip=simulated_ip,
            threat_score=0.0,
            interaction_count=0,
            first_seen_at=datetime.utcnow(),
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def _auto_block_if_needed(db: Session, account_id: str, threat_score: float) -> None:
    """No admin click — the honeypot blocks the account itself once it's confident."""
    if threat_score < settings.HONEYPOT_AUTO_BLOCK_THREAT_SCORE or not account_id:
        return
    existing = (
        db.query(models.BlockedAccount)
        .filter(models.BlockedAccount.account_id == account_id, models.BlockedAccount.is_active.is_(True))
        .first()
    )
    if existing is not None:
        return
    block = models.BlockedAccount(
        account_id=account_id,
        is_active=True,
        reason=f"Auto-blocked by honeypot: threat_score={threat_score:.1f} "
               f">= HONEYPOT_AUTO_BLOCK_THREAT_SCORE={settings.HONEYPOT_AUTO_BLOCK_THREAT_SCORE}",
        blocked_by_user_id="system:honeypot",
        blocked_at=datetime.utcnow(),
    )
    db.add(block)
    db.commit()
    logger.warning("Auto-blocked account %s via honeypot (threat_score=%.1f)", account_id, threat_score)


def record_interaction(
    db: Session,
    session_id: str,
    stage: str,
    account_id: str = "",
    browser_fingerprint: str = "",
    simulated_ip: str = "",
    detail: str = "",
) -> models.AttackerProfile:
    """Called by every decoy endpoint (and by the initial trigger in transactions.py).
    Logs the event, escalates/creates the AttackerProfile, and auto-blocks once the
    threat score is high enough. Returns the (possibly updated) profile."""
    session = db.query(models.HoneypotSession).filter(models.HoneypotSession.id == session_id).first()
    if session is not None:
        session.stage = stage
        db.add(session)

    event = models.HoneypotEvent(
        session_id=session_id, stage=stage, detail=detail, occurred_at=datetime.utcnow(),
    )
    db.add(event)
    db.commit()

    fingerprint = browser_fingerprint or (session.browser_fingerprint if session else "") or "unknown"
    ip = simulated_ip or (session.simulated_ip if session else "")

    profile = _get_or_create_profile(db, fingerprint, ip)
    profile.interaction_count = (profile.interaction_count or 0) + 1
    profile.threat_score = min(100.0, (profile.threat_score or 0.0) + settings.HONEYPOT_THREAT_SCORE_INCREMENT)
    profile.last_seen_at = datetime.utcnow()
    db.add(profile)
    db.commit()
    db.refresh(profile)

    resolved_account = account_id or (session.transaction.name_orig if session and session.transaction else "")
    _auto_block_if_needed(db, resolved_account, profile.threat_score)

    return profile
