from fastapi import APIRouter, Depends, HTTPException
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
    """Minimal list view: only non-sensitive identifiers + status. Full transaction
    details are NOT sent here -- they're fetched on demand per case via
    GET /cases/{vault_id} when the admin clicks 'Look Up ID'."""
    records = db.query(models.SafeVaultTransaction).order_by(models.SafeVaultTransaction.created_at.desc()).all()
    return [
        {
            "vault_id": str(r.id) if r.id else "",
            "transaction_id": str(r.transaction_id) if r.transaction_id else "",
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


@router.get("/cases/{vault_id}")
def get_vault_case(
    vault_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Look-up: fetches one case's full details straight from the database."""
    r = db.query(models.SafeVaultTransaction).filter(models.SafeVaultTransaction.id == vault_id).first()
    if r is None:
        raise HTTPException(status_code=404, detail="Vault case not found.")
    tx = db.query(models.Transaction).filter(models.Transaction.id == r.transaction_id).first()
    pred = (
        db.query(models.ModelPrediction)
        .filter(models.ModelPrediction.transaction_id == r.transaction_id)
        .order_by(models.ModelPrediction.created_at.desc())
        .first()
    )
    return {
        "vault_id": str(r.id),
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
    }
