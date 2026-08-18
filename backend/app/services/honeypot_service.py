import ipaddress
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from fastapi import Request
from sqlalchemy.orm import Session

from app.db import models
from app.core.config import settings

logger = logging.getLogger("honeypot_service")


def _is_private_or_loopback(ip_address: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_address)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback or addr.is_link_local


def extract_client_ip(request: Request) -> str:
    """Resolves the real client IP behind any reverse proxy: the left-most hop of
    X-Forwarded-For, then X-Real-IP, then the raw socket peer as a last resort.
    X-Forwarded-For / X-Real-IP are attacker-controllable unless a trusted proxy
    strips/overwrites them upstream -- fine for this project's honeypot/telemetry
    use case, but not a substitute for a real trusted-proxy allowlist in production.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def get_geo_location(ip_address: str) -> str:
    """Fetches approximate location for the given IP."""
    if not ip_address or ip_address == "unknown" or _is_private_or_loopback(ip_address):
        return "Local Network"
    try:
        # Using a free GeoIP service for the project
        response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=2)
        data = response.json()
        if data.get("status") == "success":
            return f"{data.get('city')}, {data.get('country')}"
    except Exception:
        pass
    return "Unknown Location"


def _get_or_create_profile(db: Session, browser_fingerprint: str, simulated_ip: str) -> models.AttackerProfile:
    profile = (
        db.query(models.AttackerProfile)
        .filter(models.AttackerProfile.browser_fingerprint == browser_fingerprint)
        .first()
    )
    if profile is None:
        profile = models.AttackerProfile(
            browser_fingerprint=browser_fingerprint,
            simulated_ip=simulated_ip,
            threat_score=0.0,
            total_sessions=0,
            last_seen_at=datetime.utcnow(),
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
    event_type: Optional[str] = None,
    event_payload: Optional[Dict[str, Any]] = None,
) -> models.AttackerProfile:
    """Called by every decoy endpoint (and by the initial trigger in transactions.py).
    Logs the event, escalates/creates the AttackerProfile, and auto-blocks once the
    threat score is high enough. Returns the (possibly updated) profile."""
    session = db.query(models.HoneypotSession).filter(models.HoneypotSession.id == session_id).first()
    if session is not None:
        session.stage = stage
        db.add(session)

    event = models.HoneypotEvent(
        session_id=session_id,
        stage=stage,
        detail=detail,
        event_type=event_type or stage,
        payload=event_payload,
        occurred_at=datetime.utcnow(),
    )
    db.add(event)
    db.commit()

    fingerprint = browser_fingerprint or (session.browser_fingerprint if session else "") or "unknown"
    ip = simulated_ip or (session.simulated_ip if session else "")

    profile = _get_or_create_profile(db, fingerprint, ip)
    # Threat score is owned exclusively by this realtime path -- the offline K-Means
    # job (threat_intel.py) must never overwrite it, only set a baseline for brand-new
    # profiles it discovers before they've ever hit this function.
    profile.threat_score = min(100.0, (profile.threat_score or 0.0) + settings.HONEYPOT_THREAT_SCORE_INCREMENT)
    profile.last_seen_at = datetime.utcnow()

    # --- Bubble up the real network origin to the Attacker Profile ---
    if session and session.actual_ip:
        profile.actual_ip = session.actual_ip
        profile.location = session.location

    db.add(profile)
    db.commit()
    db.refresh(profile)

    resolved_account = account_id or (session.transaction.name_orig if session and session.transaction else "")
    _auto_block_if_needed(db, resolved_account, profile.threat_score)

    return profile
