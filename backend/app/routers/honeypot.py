from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import get_current_user
from app.schemas.schemas import (
    HoneypotStartRequest, HoneypotAdvanceRequest, HoneypotCloseRequest, GenericStatus
)

router = APIRouter(prefix="/honeypot", tags=["honeypot"])


@router.post("/start", response_model=GenericStatus)
def start_session(
    payload: HoneypotStartRequest, db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    session = models.HoneypotSession(
        transaction_id=payload.transaction_id, simulated_ip=payload.simulated_ip,
        user_agent=payload.user_agent, browser_fingerprint=payload.browser_fingerprint,
        stage="started", risk_score_at_entry=payload.risk_score_at_entry,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return GenericStatus(status="started", message="Fake banking session initialized.",
                          data={"session_id": session.id})


@router.post("/advance", response_model=GenericStatus)
def advance_session(
    payload: HoneypotAdvanceRequest, db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    session = db.query(models.HoneypotSession).filter(models.HoneypotSession.id == payload.session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Honeypot session not found.")
    if session.stage == "closed":
        raise HTTPException(status_code=400, detail="Session already closed.")

    next_index = db.query(models.HoneypotEvent).filter(
        models.HoneypotEvent.session_id == session.id
    ).count()
    event = models.HoneypotEvent(
        session_id=session.id, event_type=payload.event_type, sequence_index=next_index,
        headers=payload.headers, payload=payload.payload,
    )
    db.add(event)
    session.stage = "advancing"
    db.add(session)
    db.commit()
    return GenericStatus(status="advanced", message="Simulated banking flow step recorded.",
                          data={"sequence_index": next_index})


@router.post("/close", response_model=GenericStatus)
def close_session(
    payload: HoneypotCloseRequest, db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    session = db.query(models.HoneypotSession).filter(models.HoneypotSession.id == payload.session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Honeypot session not found.")
    session.stage = "closed"
    session.closed_at = datetime.utcnow()
    db.add(session)
    db.commit()
    n_events = db.query(models.HoneypotEvent).filter(models.HoneypotEvent.session_id == session.id).count()
    return GenericStatus(status="closed", message="Honeypot session closed.",
                          data={"total_events": n_events})
