from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import get_current_user

router = APIRouter(prefix="/api/v1/vault", tags=["safe-vault"])


@router.get("/cases")
def list_vault_cases(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Fetches all Safe Vault transactions ordered chronologically (newest first),
    enriched with the underlying transaction's details so the admin UI can show a
    focused, read-only case view (transaction details + Vault ID). Deliberately
    read-only: admins cannot manually generate/verify OTPs or override a case's
    outcome here -- resolution happens entirely through the account holder's own
    step-up OTP flow on the customer portal (see /api/v1/transactions/{id}/verify-otp)."""
    records = db.query(models.SafeVaultTransaction).order_by(models.SafeVaultTransaction.created_at.desc()).all()

    result = []
    for r in records:
        tx = db.query(models.Transaction).filter(models.Transaction.id == r.transaction_id).first()
        pred = (
            db.query(models.ModelPrediction)
            .filter(models.ModelPrediction.transaction_id == r.transaction_id)
            .order_by(models.ModelPrediction.created_at.desc())
            .first()
        )
        result.append({
            # Explicit string casting ensures UUID objects serialize correctly to JSON
            "vault_id": str(r.id) if r.id else "",
            "transaction_id": str(r.transaction_id) if r.transaction_id else "",
            "status": r.status,
            "reason": r.admin_override_reason,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "name_orig": tx.name_orig if tx else None,
            "name_dest": tx.name_dest if tx else None,
            "type": tx.type if tx else None,
            "amount": tx.amount if tx else None,
            "final_risk_score": pred.final_risk_score if pred else None,
            "timestamp": tx.timestamp.isoformat() if tx and tx.timestamp else None,
        })
    return result
