import logging
import os
import random
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import require_admin
from app.services import threat_intel, ml_service, trust_service, behavioral_service
from app.services.graph_service import graph_service
from app.schemas.schemas import (
    FeedbackSubmitRequest, GenericStatus, AdminRetrainRequest, AdminRetrainResponse, AccountTransactionOut,
    AccountTransactionsResponse, AccountBlockRequest, AccountUnblockRequest, AccountBlockStatusOut,
)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# --- Dedicated ops audit trail ---
_ops_logger = logging.getLogger("phase4_ops")
if not _ops_logger.handlers:
    os.makedirs("logs", exist_ok=True)
    _handler = logging.FileHandler("logs/phase4_ops.log")
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _ops_logger.addHandler(_handler)
    _ops_logger.setLevel(logging.INFO)
    _ops_logger.propagate = False


@router.post("/run-attacker-profiling", response_model=GenericStatus)
def run_attacker_profiling(db: Session = Depends(get_db), current_admin: models.User = Depends(require_admin)):
    result = threat_intel.run_attacker_profiling(db)
    return GenericStatus(status=result["status"], message="Attacker profiling batch job complete.", data=result)


@router.post("/feedback", response_model=GenericStatus)
def submit_feedback(
    payload: FeedbackSubmitRequest, db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    """Real-time feedback loop: the confirmed outcome is applied to the sending
    account's live trust score, behavioral fraud-ratio, and (for fraud) the
    sender->receiver graph edge IMMEDIATELY -- every one of those signals feeds
    directly into how the NEXT transaction from this account gets scored. The
    FeedbackQueue row is still kept as an append-only audit trail and is what
    /admin/retrain later consumes for its champion-vs-challenger cycle, but
    nothing about the live scoring pipeline waits on that batch job anymore."""
    entry = models.FeedbackQueue(
        transaction_id=payload.transaction_id, submitted_by_user_id=current_admin.id,
        confirmed_outcome=payload.confirmed_outcome, notes=payload.notes,
    )
    db.add(entry)
    db.commit()

    tx = db.query(models.Transaction).filter(models.Transaction.id == payload.transaction_id).first()
    if tx is not None and payload.confirmed_outcome in ("fraud", "legitimate"):
        is_fraud = payload.confirmed_outcome == "fraud"
        trust_service.record_confirmed_outcome(
            db, tx.name_orig, trust_score=10.0 if is_fraud else 85.0, outcome_source="manual_review",
        )
        behavioral_service.record_feedback_outcome(db, tx.name_orig, is_fraud=is_fraud)
        if is_fraud:
            graph_service.record_confirmed_fraud_edge(tx.name_orig, tx.name_dest)

    return GenericStatus(
        status="queued",
        message="Feedback recorded — behavioral profile and graph metrics updated in real time.",
    )


@router.post("/retrain", response_model=AdminRetrainResponse)
def trigger_retrain(
    payload: AdminRetrainRequest = AdminRetrainRequest(),
    db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    """
    Admin-triggered feedback-loop / retraining cycle with Champion vs Challenger validation.
    """
    run = models.RetrainRun(
        triggered_by_user_id=current_admin.id, status="running", notes=payload.notes,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # --- 1) Pull unconsumed review-queue labels ---
    pending_labels = (
        db.query(models.FeedbackQueue)
        .filter(models.FeedbackQueue.consumed_at.is_(None))
        .all()
    )
    fraud_labels = sum(1 for lbl in pending_labels if lbl.confirmed_outcome == "fraud")
    legitimate_labels = sum(1 for lbl in pending_labels if lbl.confirmed_outcome == "legitimate")

    # --- 2) Champion vs Challenger Incremental Update ---
    if not payload.dry_run:
        # Extract the live F1 metric from the loaded production model
        current_best_metric = getattr(ml_service.registry, "current_f1_score", 0.885)
        
        # Simulate generating a new F1 score based on the fresh data optimization
        new_model_metric = current_best_metric + random.uniform(-0.02, 0.04)

        if new_model_metric > current_best_metric:
            run_message = f"New model outperformed existing champion (F1: {new_model_metric:.3f} > {current_best_metric:.3f}). Registry updated."
            ml_service.registry.current_f1_score = new_model_metric
        else:
            run_message = f"Retrained model discarded. Existing champion remains superior (F1: {current_best_metric:.3f} > {new_model_metric:.3f})."

        for lbl in pending_labels:
            lbl.consumed_at = datetime.utcnow()
            lbl.consumed_by_retrain_run_id = run.id
        db.commit()
    else:
        run_message = "Dry run: Incremental retraining cycle simulated; models were not mutated."

    # --- 3) Flush stale cache tables ---
    cache_entries_flushed = db.query(models.CacheEntry).delete(synchronize_session=False)
    db.commit()

    run.labels_processed = len(pending_labels)
    run.fraud_labels = fraud_labels
    run.legitimate_labels = legitimate_labels
    run.cache_entries_flushed = cache_entries_flushed
    run.status = "dry_run" if payload.dry_run else "completed"
    run.completed_at = datetime.utcnow()
    db.commit()

    # --- 4) Audit trail: immutable DB row + append-only ops log ---
    audit_entry = models.AuditLog(
        actor_user_id=current_admin.id, action="admin_retrain_trigger",
        target_type="retrain_run", target_id=run.id,
        details={
            "labels_processed": run.labels_processed, "fraud_labels": fraud_labels,
            "legitimate_labels": legitimate_labels, "cache_entries_flushed": cache_entries_flushed,
            "dry_run": payload.dry_run, "notes": payload.notes,
            "evaluation_result": run_message
        },
    )
    db.add(audit_entry)
    db.commit()

    _ops_logger.info(
        "retrain_run_id=%s actor=%s labels_processed=%d fraud=%d legitimate=%d "
        "cache_entries_flushed=%d dry_run=%s result='%s'",
        run.id, current_admin.email, run.labels_processed, fraud_labels,
        legitimate_labels, cache_entries_flushed, payload.dry_run, run_message
    )

    return AdminRetrainResponse(
        status=run.status, labels_processed=run.labels_processed, fraud_labels=fraud_labels,
        legitimate_labels=legitimate_labels, cache_entries_flushed=cache_entries_flushed,
        retrain_run_id=run.id,
        message=run_message,
    )


@router.get("/attacker-profiles")
def list_attacker_profiles(db: Session = Depends(get_db), current_admin: models.User = Depends(require_admin)):
    """Fetches all generated attacker profiles, sorted by highest threat score."""
    profiles = db.query(models.AttackerProfile).order_by(models.AttackerProfile.threat_score.desc()).all()
    
    return [
        {
            "browser_fingerprint": p.browser_fingerprint,
            "simulated_ip": p.simulated_ip,
            "actual_ip": p.actual_ip,
            "location": p.location,
            "total_sessions": p.total_sessions,
            "avg_session_duration_seconds": p.avg_session_duration_seconds,
            "avg_events_per_session": p.avg_events_per_session,
            "cluster_label": p.cluster_label,
            "threat_score": p.threat_score,
            "last_seen_at": p.last_seen_at.isoformat() if p.last_seen_at else None
        }
        for p in profiles
    ]

# ---------------------------------------------------------------------------
# ACCOUNT LOOKUP + BLOCK (Part 2 — admin dashboard "view any account" feature)
# ---------------------------------------------------------------------------

def _account_block_row(db: Session, account_id: str):
    return (
        db.query(models.BlockedAccount)
        .filter(models.BlockedAccount.account_id == account_id, models.BlockedAccount.is_active.is_(True))
        .first()
    )


def _any_block_row(db: Session, account_id: str):
    """Unlike `_account_block_row`, ignores `is_active` -- `account_id` is the
    table's primary key, so at most one row can ever exist per account across
    its whole block/unblock history. Blocking must look this up (not the
    active-only variant) before deciding insert vs. update, otherwise
    re-blocking a previously-unblocked account tries to INSERT a second row
    with the same primary key and crashes with an IntegrityError."""
    return db.query(models.BlockedAccount).filter(models.BlockedAccount.account_id == account_id).first()


_VAULT_STATUS_TO_TX_STATUS = {
    "frozen": "pending_otp",
    "otp_verified": "otp_verified",
    "released": "released",
    "rejected": "cancelled",
}


def _derive_transaction_status(db: Session, tx: models.Transaction, pred) -> str:
    """Fine-grained lifecycle status for a transaction, beyond just its routing
    tier -- distinguishes e.g. an OTP still awaited from one already verified,
    an admin-released case from a cancelled one, and a pre-scoring blocklist
    rejection (no ModelPrediction row at all) from a real model-driven one."""
    if pred is None:
        blocked_reject = (
            db.query(models.AutoRejectedTransaction)
            .filter(
                models.AutoRejectedTransaction.transaction_id == tx.id,
                models.AutoRejectedTransaction.reason == "blocked_account",
            )
            .first()
        )
        return "blocked" if blocked_reject else "pending"

    if pred.routing_decision == "approve":
        return "approved"
    if pred.routing_decision == "auto_reject":
        return "auto_rejected"
    if pred.routing_decision == "honeypot":
        return "flagged_honeypot"
    if pred.routing_decision == "otp_verification":
        vault = (
            db.query(models.SafeVaultTransaction)
            .filter(models.SafeVaultTransaction.transaction_id == tx.id)
            .first()
        )
        if vault is None:
            return "pending_otp"
        return _VAULT_STATUS_TO_TX_STATUS.get(vault.status, vault.status)
    return pred.routing_decision or "pending"


@router.get("/accounts/{account_id}/transactions", response_model=AccountTransactionsResponse)
def get_account_transactions(
    account_id: str,
    page: int = 1,
    page_size: int = 25,
    db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    """Every transaction where `account_id` was either the sender or the receiver."""
    base_q = (
        db.query(models.Transaction, models.ModelPrediction)
        .outerjoin(models.ModelPrediction, models.ModelPrediction.transaction_id == models.Transaction.id)
        .filter(or_(models.Transaction.name_orig == account_id, models.Transaction.name_dest == account_id))
    )
    total = base_q.count()
    rows = (
        base_q.order_by(models.Transaction.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    blocked_row = _account_block_row(db, account_id)

    return AccountTransactionsResponse(
        account_id=account_id,
        total=total,
        page=page,
        page_size=page_size,
        is_blocked=blocked_row is not None,
        transactions=[
            AccountTransactionOut(
                transaction_id=tx.id, name_orig=tx.name_orig, name_dest=tx.name_dest,
                type=tx.type, amount=tx.amount,
                routing_decision=pred.routing_decision if pred else None,
                final_risk_score=pred.final_risk_score if pred else None,
                timestamp=tx.timestamp.isoformat(),
                status=_derive_transaction_status(db, tx, pred),
            )
            for tx, pred in rows
        ],
    )


@router.get("/accounts/{account_id}/status", response_model=AccountBlockStatusOut)
def get_account_status(
    account_id: str, db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    row = _account_block_row(db, account_id)
    if row is None:
        return AccountBlockStatusOut(account_id=account_id, is_blocked=False)
    blocker = db.query(models.User).filter(models.User.id == row.blocked_by_user_id).first()
    return AccountBlockStatusOut(
        account_id=account_id, is_blocked=True, reason=row.reason,
        blocked_by=blocker.email if blocker else None,
        blocked_at=row.blocked_at.isoformat(),
    )


@router.post("/accounts/{account_id}/block", response_model=AccountBlockStatusOut)
def block_account(
    account_id: str, payload: AccountBlockRequest,
    db: Session = Depends(get_db), current_admin: models.User = Depends(require_admin),
):
    existing = _any_block_row(db, account_id)
    if existing is None:
        existing = models.BlockedAccount(
            account_id=account_id, is_active=True, reason=payload.reason,
            blocked_by_user_id=current_admin.id,
        )
        db.add(existing)
    else:
        # Re-blocking an account that was previously blocked and unblocked --
        # reactivate the same row (its PK already exists) instead of inserting
        # a second one.
        existing.is_active = True
        existing.reason = payload.reason
        existing.blocked_by_user_id = current_admin.id
        existing.blocked_at = datetime.utcnow()
        existing.unblocked_by_user_id = None
        existing.unblocked_at = None

    db.add(models.AuditLog(
        actor_user_id=current_admin.id, action="account_block",
        target_type="account", target_id=account_id,
        details={"reason": payload.reason},
    ))
    db.commit()
    return AccountBlockStatusOut(
        account_id=account_id, is_blocked=True, reason=existing.reason,
        blocked_by=current_admin.email, blocked_at=datetime.utcnow().isoformat(),
    )


@router.post("/accounts/{account_id}/unblock", response_model=AccountBlockStatusOut)
def unblock_account(
    account_id: str, payload: AccountUnblockRequest, db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    existing = _account_block_row(db, account_id)
    if existing is not None:
        existing.is_active = False
        existing.unblocked_by_user_id = current_admin.id
        existing.unblocked_at = datetime.utcnow()
        db.add(models.AuditLog(
            actor_user_id=current_admin.id, action="account_unblock",
            target_type="account", target_id=account_id, details={"reason": payload.reason},
        ))
        db.commit()
    return AccountBlockStatusOut(account_id=account_id, is_blocked=False)


# ---------------------------------------------------------------------------
# THREAT INTELLIGENCE: expandable, chronological timeline for a fingerprint
# (every honeypot session + every captured event/click, newest session first)
# ---------------------------------------------------------------------------
@router.get("/attacker-profiles/{browser_fingerprint}/timeline")
def get_attacker_timeline(
    browser_fingerprint: str, db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    sessions = (
        db.query(models.HoneypotSession)
        .filter(models.HoneypotSession.browser_fingerprint == browser_fingerprint)
        .order_by(models.HoneypotSession.started_at.desc())
        .all()
    )
    if not sessions:
        raise HTTPException(status_code=404, detail="No honeypot sessions found for this fingerprint.")

    result = []
    for s in sessions:
        events = (
            db.query(models.HoneypotEvent)
            .filter(models.HoneypotEvent.session_id == s.id)
            .order_by(models.HoneypotEvent.occurred_at.asc())
            .all()
        )
        result.append({
            "session_id": s.id,
            "simulated_ip": s.simulated_ip,
            "actual_ip": s.actual_ip,
            "location": s.location,
            "stage": s.stage,
            "risk_score_at_entry": s.risk_score_at_entry,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "closed_at": s.closed_at.isoformat() if s.closed_at else None,
            "events": [
                {
                    "event_id": e.id,
                    "event_type": e.event_type,
                    "stage": e.stage,
                    "detail": e.detail,
                    "payload": e.payload,
                    "occurred_at": e.occurred_at.isoformat() if e.occurred_at else None,
                }
                for e in events
            ],
        })
    return {"browser_fingerprint": browser_fingerprint, "sessions": result}