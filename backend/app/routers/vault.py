import random
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import get_current_user, require_admin
from app.services import trust_service
from app.schemas.schemas import VaultOTPVerifyRequest, VaultAdminReviewRequest, VaultMoveRequest, GenericStatus

router = APIRouter(prefix="/api/v1/vault", tags=["safe-vault"])


def _log_audit(db: Session, actor_id: str, action: str, target_id: str, details: dict):
    db.add(models.AuditLog(actor_user_id=actor_id, action=action, target_type="safevault_transaction",
                           target_id=target_id, details=details))
    db.commit()


@router.get("/cases")
def list_vault_cases(
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    """Fetches all Safe Vault transactions ordered chronologically (newest first)."""
    records = db.query(models.SafeVaultTransaction).order_by(models.SafeVaultTransaction.created_at.desc()).all()
    
    return [
        {
            # Explicit string casting ensures UUID objects serialize correctly to JSON
            "vault_id": str(r.id) if r.id else "",
            "transaction_id": str(r.transaction_id) if r.transaction_id else "",
            "status": r.status,
            "reason": r.admin_override_reason, 
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in records
    ]


@router.post("/otp", response_model=GenericStatus)
def generate_or_verify_otp(
    payload: VaultOTPVerifyRequest, db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    vault_record = db.query(models.SafeVaultTransaction).filter(
        models.SafeVaultTransaction.id == payload.vault_id
    ).first()
    
    if vault_record is None:
        raise HTTPException(status_code=404, detail="Vault record not found.")
    if vault_record.status != "frozen":
        raise HTTPException(status_code=400, detail=f"Vault record is not frozen (status={vault_record.status}).")

    # Check if the frontend intentionally requested a new OTP by sending an empty string
    if not payload.otp_code:
        code = f"{random.randint(0, 999999):06d}"
        vault_record.otp_code = code
        vault_record.otp_expires_at = datetime.utcnow() + timedelta(minutes=5)
        vault_record.otp_attempts = 0  # Reset attempts for the new code
        db.add(vault_record)
        db.commit()
        return GenericStatus(status="otp_issued", message="OTP generated.", data={"otp_code": code})

    # Verification Step
    vault_record.otp_attempts += 1
    if vault_record.otp_expires_at and datetime.utcnow() > vault_record.otp_expires_at:
        raise HTTPException(status_code=400, detail="OTP expired. Request a new one.")
    if payload.otp_code != vault_record.otp_code:
        db.add(vault_record)
        db.commit()
        raise HTTPException(status_code=401, detail="Incorrect OTP.")

    vault_record.status = "otp_verified"
    db.add(vault_record)
    db.commit()

    tx = db.query(models.Transaction).filter(models.Transaction.id == vault_record.transaction_id).first()
    trust_service.record_confirmed_outcome(db, tx.name_orig, trust_score=75.0, outcome_source="otp_verified")
    return GenericStatus(status="otp_verified", message="Transaction released from Safe Vault.")


@router.post("/review", response_model=GenericStatus)
def admin_review(
    payload: VaultAdminReviewRequest, db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    vault_record = db.query(models.SafeVaultTransaction).filter(
        models.SafeVaultTransaction.id == payload.vault_id
    ).first()
    
    if vault_record is None:
        raise HTTPException(status_code=404, detail="Vault record not found.")

    vault_record.status = "released" if payload.decision == "approve" else "rejected"
    vault_record.admin_override_by_user_id = current_admin.id
    vault_record.admin_override_decision = payload.decision
    vault_record.admin_override_reason = payload.reason
    vault_record.admin_override_at = datetime.utcnow()
    db.add(vault_record)
    db.commit()

    tx = db.query(models.Transaction).filter(models.Transaction.id == vault_record.transaction_id).first()
    trust_score = 80.0 if payload.decision == "approve" else 10.0
    trust_service.record_confirmed_outcome(db, tx.name_orig, trust_score=trust_score, outcome_source="admin_override")
    
    _log_audit(db, current_admin.id, f"vault_review_{payload.decision}", vault_record.id, {"reason": payload.reason})
    
    return GenericStatus(status="reviewed", message=f"Admin override applied: {payload.decision}.")


@router.post("/move-to-vault", response_model=GenericStatus)
def move_to_vault(
    payload: VaultMoveRequest, db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    tx = db.query(models.Transaction).filter(models.Transaction.id == payload.transaction_id).first()
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found.")
        
    existing = db.query(models.SafeVaultTransaction).filter(
        models.SafeVaultTransaction.transaction_id == tx.id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Transaction is already in the Safe Vault.")

    vault_record = models.SafeVaultTransaction(transaction_id=tx.id, status="frozen")
    db.add(vault_record)
    db.commit()
    db.refresh(vault_record)
    
    _log_audit(db, current_admin.id, "manual_escalation_to_vault", vault_record.id, {"reason": payload.reason})
    
    return GenericStatus(status="moved_to_vault", message="Transaction manually escalated to Safe Vault.",
                          data={"vault_id": str(vault_record.id)})
