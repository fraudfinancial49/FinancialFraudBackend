import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import require_admin
from app.services import threat_intel
from app.schemas.schemas import (
    FeedbackSubmitRequest, GenericStatus, AdminRetrainRequest, AdminRetrainResponse,
)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# --- Dedicated ops audit trail: every admin-triggered retrain cycle leaves a
# line here, independent of the `audit_logs` DB table, so an operator can
# `tail -f logs/phase4_ops.log` without a database connection. Handler
# registration is idempotent (guarded on `_ops_logger.handlers`) so re-importing
# this module -- e.g. under the smoke test's repeated app re-imports -- never
# duplicates log lines or re-opens the file handle.
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
    """Append-only feedback log for OFFLINE, asynchronous retraining. Never
    triggers inline retraining or mutates any live model."""
    entry = models.FeedbackQueue(
        transaction_id=payload.transaction_id, submitted_by_user_id=current_admin.id,
        confirmed_outcome=payload.confirmed_outcome, notes=payload.notes,
    )
    db.add(entry)
    db.commit()
    return GenericStatus(status="queued", message="Feedback appended to the offline retraining queue.")


@router.post("/retrain", response_model=AdminRetrainResponse)
def trigger_retrain(
    payload: AdminRetrainRequest = AdminRetrainRequest(),
    db: Session = Depends(get_db),
    current_admin: models.User = Depends(require_admin),
):
    """
    Admin-triggered feedback-loop / retraining cycle.

    This endpoint deliberately does NOT retrain, refit, or mutate any frozen
    Phase 3 model in-process — doing that synchronously inside a request handler
    would risk serving an unvalidated model to live traffic the instant this
    call returns. Instead it:

      1) Pulls every not-yet-consumed row off the append-only `feedback_queue`
         table — false-positive corrections and admin-verified fraud labels
         submitted via `POST /admin/feedback`.
      2) Simulates the lightweight incremental-update step a real offline
         retraining job would perform against those labels: tallies them by
         outcome and marks them consumed, so a second call never double-counts
         the same label.
      3) Flushes every cached live-SHAP explanation (`cache_entries` table) so
         no analyst is ever served an XAI explanation computed against a
         model/feature-schema version this cycle may have since superseded.
      4) Writes an immutable `audit_logs` row plus a line in
         `logs/phase4_ops.log`, so every retrain trigger is independently
         traceable from both the database and the filesystem.

    An actual model weight update remains an offline, asynchronous job (e.g. a
    scheduled batch process reading this same `feedback_queue` table) that swaps
    in a newly-validated `*.joblib` bundle — never something this
    request/response cycle performs synchronously.
    """
    run = models.RetrainRun(
        triggered_by_user_id=current_admin.id, status="running", notes=payload.notes,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # --- 1) Pull unconsumed review-queue labels -----------------------------
    pending_labels = (
        db.query(models.FeedbackQueue)
        .filter(models.FeedbackQueue.consumed_at.is_(None))
        .all()
    )
    fraud_labels = sum(1 for lbl in pending_labels if lbl.confirmed_outcome == "fraud")
    legitimate_labels = sum(1 for lbl in pending_labels if lbl.confirmed_outcome == "legitimate")

    # --- 2) Simulate the lightweight incremental update ---------------------
    if not payload.dry_run:
        for lbl in pending_labels:
            lbl.consumed_at = datetime.utcnow()
            lbl.consumed_by_retrain_run_id = run.id
        db.commit()

    # --- 3) Flush stale cache tables -----------------------------------------
    cache_entries_flushed = db.query(models.CacheEntry).delete(synchronize_session=False)
    db.commit()

    run.labels_processed = len(pending_labels)
    run.fraud_labels = fraud_labels
    run.legitimate_labels = legitimate_labels
    run.cache_entries_flushed = cache_entries_flushed
    run.status = "dry_run" if payload.dry_run else "completed"
    run.completed_at = datetime.utcnow()
    db.commit()

    # --- 4) Audit trail: immutable DB row + append-only ops log --------------
    audit_entry = models.AuditLog(
        actor_user_id=current_admin.id, action="admin_retrain_trigger",
        target_type="retrain_run", target_id=run.id,
        details={
            "labels_processed": run.labels_processed, "fraud_labels": fraud_labels,
            "legitimate_labels": legitimate_labels, "cache_entries_flushed": cache_entries_flushed,
            "dry_run": payload.dry_run, "notes": payload.notes,
        },
    )
    db.add(audit_entry)
    db.commit()

    _ops_logger.info(
        "retrain_run_id=%s actor=%s labels_processed=%d fraud=%d legitimate=%d "
        "cache_entries_flushed=%d dry_run=%s",
        run.id, current_admin.email, run.labels_processed, fraud_labels,
        legitimate_labels, cache_entries_flushed, payload.dry_run,
    )

    return AdminRetrainResponse(
        status=run.status, labels_processed=run.labels_processed, fraud_labels=fraud_labels,
        legitimate_labels=legitimate_labels, cache_entries_flushed=cache_entries_flushed,
        retrain_run_id=run.id,
        message="Incremental retraining cycle simulated; frozen models were not mutated in-process.",
    )
