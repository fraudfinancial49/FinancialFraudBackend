import random
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.services import honeypot_service
from app.schemas.schemas import (
    DecoyBalanceRequest, DecoyBalanceResponse, DecoyTransferRequest,
    DecoyTransferResponse, DecoyOtpRequest, DecoyOtpResponse,
    TelemetryEventRequest,
)

router = APIRouter(prefix="/api/v1/honeypot", tags=["honeypot"])


@router.post("/{session_id}/decoy/balance", response_model=DecoyBalanceResponse)
def decoy_balance(session_id: str, payload: DecoyBalanceRequest, db: Session = Depends(get_db)):
    """Fake balance an attacker sees after 'logging in' inside the honeypot."""
    honeypot_service.record_interaction(
        db, session_id, stage="viewed_decoy_balance",
        browser_fingerprint=payload.browser_fingerprint, simulated_ip=payload.simulated_ip,
        detail="Attacker viewed decoy balance.",
    )
    # Deterministic-looking but fake — never a real account's real balance.
    fake_balance = round(random.uniform(5000, 250000), 2)
    return DecoyBalanceResponse(account_id=payload.account_id, balance=fake_balance, updated_at=datetime.utcnow())


@router.post("/{session_id}/decoy/transfer", response_model=DecoyTransferResponse)
def decoy_transfer(session_id: str, payload: DecoyTransferRequest, db: Session = Depends(get_db)):
    """Fake 'transfer succeeded' response — no money moves, no ledger touched."""
    profile = honeypot_service.record_interaction(
        db, session_id, stage="attempted_decoy_transfer", account_id=payload.name_dest,
        browser_fingerprint=payload.browser_fingerprint, simulated_ip=payload.simulated_ip,
        detail=f"Attacker attempted decoy transfer of {payload.amount} to {payload.name_dest}.",
    )
    return DecoyTransferResponse(
        transaction_id=str(uuid.uuid4()), status="completed",
        message="Transfer completed successfully.", threat_score=profile.threat_score,
    )


@router.post("/{session_id}/decoy/otp", response_model=DecoyOtpResponse)
def decoy_otp(session_id: str, payload: DecoyOtpRequest, db: Session = Depends(get_db)):
    """Any code 'works' — the goal is engagement time and fingerprint data, not a real gate."""
    profile = honeypot_service.record_interaction(
        db, session_id, stage="submitted_decoy_otp",
        browser_fingerprint=payload.browser_fingerprint, simulated_ip=payload.simulated_ip,
        detail="Attacker submitted a decoy OTP.",
    )
    return DecoyOtpResponse(verified=True, threat_score=profile.threat_score)


@router.post("/{session_id}/telemetry")
def log_micro_interaction(
    session_id: str, request: Request, payload: TelemetryEventRequest, db: Session = Depends(get_db)
):
    """Silent endpoint to capture raw clicks, keystrokes, and mouse movements."""
    actual_ip = honeypot_service.extract_client_ip(request)

    # Update session with actual IP and Location if not already set
    session = db.query(models.HoneypotSession).filter(models.HoneypotSession.id == session_id).first()
    if session and not session.actual_ip:
        session.actual_ip = actual_ip
        session.location = honeypot_service.get_geo_location(actual_ip)
        db.add(session)

    # Log the exact click/action -- both a human-readable summary and the
    # structured payload (queryable, unlike the free-text `detail` string).
    event_detail = (
        f"Action: {payload.action_type} on '{payload.target_element}' "
        f"(X:{payload.x_coord}, Y:{payload.y_coord})"
    )

    honeypot_service.record_interaction(
        db, session_id, stage="micro_interaction",
        detail=event_detail,
        event_type=payload.action_type,
        event_payload=payload.model_dump(),
    )
    return {"status": "logged"}


@router.get("/{session_id}/events")
def get_session_events(session_id: str, db: Session = Depends(get_db)):
    """Fetches the detailed chronological timeline of an attacker's actions."""
    events = db.query(models.HoneypotEvent).filter(
        models.HoneypotEvent.session_id == session_id
    ).order_by(models.HoneypotEvent.occurred_at.asc()).all()

    return [
        {
            "event_id": str(e.id),
            "stage": e.stage,
            "detail": e.detail,
            "occurred_at": e.occurred_at.isoformat() if e.occurred_at else None
        }
        for e in events
    ]