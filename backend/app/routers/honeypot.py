
import random
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.services import honeypot_service
from app.schemas.schemas import (
    DecoyBalanceRequest, DecoyBalanceResponse, DecoyTransferRequest,
    DecoyTransferResponse, DecoyOtpRequest, DecoyOtpResponse,
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
