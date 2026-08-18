import time
import logging
from datetime import datetime, date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import get_current_user, require_roles
from app.schemas.schemas import (
    OTPVerifyRequest,
    TransactionAssessRequest, TransactionAssessResponse, TransactionExplainResponse,
    TransactionListItem, TransactionListResponse,
)
from app.services import ml_service, feature_pipeline, graph_service as graph_svc_module
from app.services import honeypot_service
from app.services import behavioral_service, trust_service, risk_fusion
from app.services.ml_service import ShapExplainerError
from app.core.config import settings
from app.routers.analytics import _date_bounds
from app.services import otp_service, email_service, ledger_service

logger = logging.getLogger("transactions_router")
router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])

@router.post("/assess", response_model=TransactionAssessResponse)
def assess_transaction(
    payload: TransactionAssessRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    t0 = time.time()
    registry = ml_service.registry
    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Model registry not loaded yet — try again shortly.")

    # --- Customer-initiated transfers: derive identity + balances server-side ---
    # A real customer never supplies nameOrig / old-and-new balance fields --
    # those are ML feature-engineering plumbing meant for the admin Sandbox's
    # hand-built test transactions. For a customer, trusting client-submitted
    # values here would also let an authenticated customer impersonate any
    # account (nameOrig is what the ledger later debits/credits from), so we
    # overwrite them unconditionally from the authenticated session + live
    # Balance rows rather than merely defaulting when absent.
    if current_user.role == "customer":
        customer = db.query(models.Customer).filter(models.Customer.user_id == current_user.id).first()
        if customer is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a customer account.")

        recipient_balance = db.query(models.Balance).filter(models.Balance.account_id == payload.nameDest).first()
        if not recipient_balance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Recipient Account ID not found. Please check the ID and try again.",
            )
        sender_balance = db.query(models.Balance).filter(models.Balance.account_id == customer.account_id).first()
        if sender_balance is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No balance record found for your account.")
        if payload.amount > sender_balance.amount:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Insufficient balance.")

        payload.nameOrig = customer.account_id
        payload.oldbalanceOrg = sender_balance.amount
        payload.newbalanceOrig = sender_balance.amount - payload.amount
        payload.oldbalanceDest = recipient_balance.amount
        payload.newbalanceDest = recipient_balance.amount + payload.amount
        payload.step = int(time.time() // 3600) % 744
    else:
        # Sandbox / admin-analyst path: these fields are hand-built test data
        # and must be fully supplied, same strictness the schema used to enforce
        # on its own before these fields became Optional to support customers.
        missing = [
            f for f in ("nameOrig", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "step")
            if getattr(payload, f) is None
        ]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required field(s): {', '.join(missing)}",
            )

    # 1) Persist the raw transaction record.
    tx = models.Transaction(
        name_orig=payload.nameOrig, name_dest=payload.nameDest, type=payload.type,
        amount=payload.amount, old_balance_orig=payload.oldbalanceOrg,
        new_balance_orig=payload.newbalanceOrig, old_balance_dest=payload.oldbalanceDest,
        new_balance_dest=payload.newbalanceDest, step=payload.step,
        timestamp=datetime.utcnow(), created_by_user_id=current_user.id,
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)

    # --- Part 2: automatic block-list enforcement (no admin step) ---
    # Checks BOTH sides of the transaction -- a blocked receiver is just as much a
    # reason to stop the transfer before it's ever scored as a blocked sender.
    blocked = (
        db.query(models.BlockedAccount)
        .filter(
            models.BlockedAccount.account_id.in_([tx.name_orig, tx.name_dest]),
            models.BlockedAccount.is_active.is_(True),
        )
        .first()
    )
    if blocked is not None:
        rejected = models.AutoRejectedTransaction(
            transaction_id=tx.id, final_risk_score=100.0, reason="blocked_account",
        )
        db.add(rejected)
        db.add(models.AuditLog(
            actor_user_id=current_user.id, action="auto_reject_blocked_account",
            target_type="transaction", target_id=tx.id,
            details={"blocked_account_id": blocked.account_id, "final_risk_score": 100.0},
        ))
        db.commit()
        db.refresh(rejected)
        return TransactionAssessResponse(
            transaction_id=tx.id, final_risk_score=100.0, routing_decision="auto_reject",
            message=f"Account {blocked.account_id} is blocked by an administrator.",
            latency_ms=0.0, auto_reject_id=rejected.id,
        )

    # 2) Reconstruct live features.
    raw_tx = feature_pipeline.RawTransaction(
        nameOrig=payload.nameOrig, nameDest=payload.nameDest, type=payload.type,
        amount=payload.amount, oldbalanceOrg=payload.oldbalanceOrg,
        newbalanceOrig=payload.newbalanceOrig, oldbalanceDest=payload.oldbalanceDest,
        newbalanceDest=payload.newbalanceDest, step=payload.step, timestamp=tx.timestamp,
    )
    behavioral_snapshot = behavioral_service.snapshot(db, payload.nameOrig)
    trust_score = trust_service.read_trust_score(db, payload.nameOrig)  # READ-ONLY on this path

    # Frozen Phase 2 structural snapshot needs BOTH sides of the transaction
    # (community/bridge comparisons are inherently pairwise).
    graph_snapshot = graph_svc_module.graph_service.account_risk_snapshot(payload.nameOrig, payload.nameDest)

    feature_dict = feature_pipeline.build_feature_dict(
        raw_tx, behavioral_snapshot, trust_score, graph_snapshot
    )

    # 3) Hard schema enforcement + matrix construction for each engine family.
    try:
        feature_pipeline.validate_feature_schema(feature_dict, registry.tree_feature_cols)
        feature_pipeline.validate_feature_schema(feature_dict, registry.deep_feature_cols)
    except feature_pipeline.FeatureSchemaError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    X_tree = feature_pipeline.to_ordered_frame(feature_dict, registry.tree_feature_cols)
    X_deep_raw = feature_pipeline.to_ordered_frame(feature_dict, registry.deep_feature_cols)
    X_deep_scaled = X_deep_raw.copy()

    # --- DEEP SCALER FIX APPLIED HERE ---
    if registry.deep_scaler is not None:
        # Extract only the continuous columns the scaler was actually fit on
        scaler_cols = registry.deep_scaler.feature_names_in_
        X_deep_scaled[scaler_cols] = registry.deep_scaler.transform(X_deep_raw[scaler_cols])

    # 4) Run every frozen, calibrated supervised + Isolation Forest engine.
    calibrated_probabilities = registry.predict_proba_all(X_tree, X_deep_scaled)
    iso_anomaly = calibrated_probabilities.get("isolation_forest", 0.0)
    best_model_probability = calibrated_probabilities.get(
        registry.best_model_name, max(calibrated_probabilities.values())
    )

    # 5) Threat intelligence lookup (never hardcoded; defaults to 0.0).
    threat_score = risk_fusion.get_threat_score(db, payload.browser_fingerprint or "")

    # 6) Hybrid risk fusion, using the frozen Phase 3 `fusion_weights` parsed
    # out of phase3_metadata_registry.json (there is no hybrid_risk_engine.joblib).
    trust_risk = risk_fusion.normalize_trust_risk(trust_score, registry.trust_score_range)
    behavioral_risk = risk_fusion.normalize_behavioral_risk(
        feature_dict["behavioral_risk_score"], registry.behavioral_risk_score_range
    )
    final_risk_score = risk_fusion.fuse(
        calibrated_probabilities=calibrated_probabilities,
        trust_risk=trust_risk,
        behavioral_risk=behavioral_risk,
        graph_risk=graph_snapshot["graph_risk"],
        threat_score=threat_score,
        fusion_weights=registry.fusion_weights,
    )
    # --- 4-tier routing (Part 1): approve / otp_verification / auto_reject / honeypot ---
    routing_decision = risk_fusion.route(
        final_risk_score, settings.LOW_RISK_MAX, settings.MODERATE_RISK_MAX, settings.HIGH_RISK_MAX
    )

    latency_ms = (time.time() - t0) * 1000.0

    prediction = models.ModelPrediction(
        transaction_id=tx.id, ml_ensemble_score=best_model_probability * 100.0,
        behavioral_risk_score=feature_dict["behavioral_risk_score"], trust_score=trust_score,
        graph_risk_score=graph_snapshot["graph_risk"] * 100.0, threat_score=threat_score,
        final_risk_score=final_risk_score, best_model_name=registry.best_model_name,
        isolation_forest_anomaly=iso_anomaly, routing_decision=routing_decision,
        latency_ms=latency_ms,
        # Exact tree-feature vector, in the tree model's fitted column order --
        # the single source of truth POST /explain reads from for live SHAP.
        tree_feature_vector=X_tree.iloc[0].to_dict(),
    )
    db.add(prediction)
    db.commit()

    # 7) Update the sending account's behavioral profile (incremental, Welford)
    # and Phase 4's own live edge bookkeeping (the frozen Phase 2 snapshot is
    # never mutated). Auto-rejected transactions still count for behavioral
    # tracking -- the sender genuinely attempted the transfer.
    behavioral_service.update_profile(db, payload.nameOrig, payload.amount, payload.nameDest)
    graph_svc_module.graph_service.add_edge_incremental(payload.nameOrig, payload.nameDest, payload.amount)

    honeypot_session_id = None
    vault_id = None
    auto_reject_id = None
    message = "Transaction approved."

    if routing_decision == "approve":
        ledger_service.settle_transaction(db, payload.nameOrig, payload.nameDest, payload.amount)

    if routing_decision == "otp_verification":
        vault_record = models.SafeVaultTransaction(transaction_id=tx.id, status="frozen")
        db.add(vault_record)
        db.commit()
        db.refresh(vault_record)
        vault_id = vault_record.id

        otp_code = otp_service.generate_and_store_otp(db, tx.id)
        customer = db.query(models.Customer).filter(models.Customer.user_id == current_user.id).first()
        if customer is not None:
            email_service.send_otp_email(customer.email, otp_code, tx.id)
        message = "Transaction frozen pending step-up verification (OTP). A code has been emailed to you."

    elif routing_decision == "auto_reject":
        reason = (
            f"Auto-rejected by risk router: final_risk_score={final_risk_score:.2f} fell in the "
            f"[{settings.MODERATE_RISK_MAX}, {settings.HIGH_RISK_MAX}) auto-reject band. "
            f"best_model={registry.best_model_name} best_model_probability={best_model_probability:.4f} "
            f"threat_score={threat_score:.2f}."
        )
        reject_record = models.AutoRejectedTransaction(
            transaction_id=tx.id, final_risk_score=final_risk_score, reason=reason,
        )
        db.add(reject_record)
        db.add(models.AuditLog(
            actor_user_id=current_user.id, action="auto_reject",
            target_type="transaction", target_id=tx.id,
            details={"final_risk_score": final_risk_score, "reason": reason},
        ))
        db.commit()
        db.refresh(reject_record)
        auto_reject_id = reject_record.id
        # No human touches this path -- treat it as a confirmed-negative outcome
        # immediately, same convention the vault-reject / admin-override path uses.
        trust_service.record_confirmed_outcome(
            db, payload.nameOrig, trust_score=10.0, outcome_source="auto_reject"
        )
        message = "Transaction automatically rejected due to high fraud risk."

    elif routing_decision == "honeypot":
        actual_ip = honeypot_service.extract_client_ip(request)
        location = honeypot_service.get_geo_location(actual_ip)

        session = models.HoneypotSession(
            transaction_id=tx.id, simulated_ip=payload.simulated_ip,
            actual_ip=actual_ip, location=location,
            user_agent=payload.user_agent, browser_fingerprint=payload.browser_fingerprint,
            stage="started", risk_score_at_entry=final_risk_score,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        honeypot_session_id = session.id
        # Part 5: score + (if warranted) auto-block from the very first interaction — no admin step.
        honeypot_service.record_interaction(
            db, session.id, stage="started", account_id=payload.nameOrig,
            browser_fingerprint=payload.browser_fingerprint, simulated_ip=payload.simulated_ip,
            detail="Honeypot triggered by risk router.",
        )
        message = "Transaction completed successfully."  # simulated completion message shown to the attacker

    return TransactionAssessResponse(
        transaction_id=tx.id, final_risk_score=final_risk_score, routing_decision=routing_decision,
        message=message, latency_ms=latency_ms, honeypot_session_id=honeypot_session_id, vault_id=vault_id,
        auto_reject_id=auto_reject_id,
        individual_scores=calibrated_probabilities,
        fusion_weights=registry.fusion_weights,
    )


@router.post("/{transaction_id}/explain", response_model=TransactionExplainResponse)
def explain_transaction(
    transaction_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("analyst", "admin")),
):
    """
    Real-time, per-transaction Explainable AI: returns SHAP contributions for the
    positive (fraud) class, computed live against the frozen Phase 3 champion
    tree model. Restricted to 'analyst' and 'admin' roles via JWT + RBAC.
    """
    t0 = time.time()
    registry = ml_service.registry

    tx = db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first()
    if tx is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found.")

    prediction = (
        db.query(models.ModelPrediction)
        .filter(models.ModelPrediction.transaction_id == transaction_id)
        .order_by(models.ModelPrediction.created_at.desc())
        .first()
    )
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No stored model prediction for this transaction — run "
            "POST /api/v1/transactions/assess for it first.",
        )
    if not prediction.tree_feature_vector:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="This transaction's prediction predates live-XAI persistence — no "
            "stored engineered feature matrix is available to explain.",
        )

    # --- Serve from cache if a prior /explain call hasn't been invalidated yet ---
    cache_key = f"shap_explanation:{transaction_id}"
    cached_entry = db.query(models.CacheEntry).filter(models.CacheEntry.cache_key == cache_key).first()
    if cached_entry is not None:
        payload = dict(cached_entry.payload)
        payload["latency_ms"] = (time.time() - t0) * 1000.0
        payload["cached"] = True
        return TransactionExplainResponse(**payload)

    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Model registry not loaded yet — try again shortly.")

    try:
        contributions = registry.compute_live_shap(prediction.tree_feature_vector)
    except ShapExplainerError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    latency_ms = (time.time() - t0) * 1000.0
    response_payload = {
        "transaction_id": transaction_id,
        "model_name": registry.best_model_name,
        "final_risk_score": prediction.final_risk_score,
        "contributions": contributions,
        "latency_ms": latency_ms,
        "cached": False,
    }

    cache_row = models.CacheEntry(
        cache_key=cache_key, cache_type="shap_explanation",
        payload={k: v for k, v in response_payload.items() if k not in ("latency_ms", "cached")},
    )
    db.add(cache_row)
    db.commit()

    return TransactionExplainResponse(**response_payload)


@router.post("/{transaction_id}/verify-otp")
def verify_otp_endpoint(
    transaction_id: str,
    payload: OTPVerifyRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Fully automated step-up verification — no admin involved at any point."""
    vault_record = (
        db.query(models.SafeVaultTransaction)
        .filter(models.SafeVaultTransaction.transaction_id == transaction_id)
        .first()
    )
    if vault_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No vault record for this transaction.")
    if vault_record.status != "frozen":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Transaction already {vault_record.status}.")

    tx = db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first()
    is_valid, reason = otp_service.verify_otp(db, transaction_id, payload.otp)

    if is_valid:
        vault_record.status = "otp_verified"
        db.commit()
        ledger_service.settle_transaction(db, tx.name_orig, tx.name_dest, tx.amount)
        trust_service.record_confirmed_outcome(db, tx.name_orig, trust_score=70.0, outcome_source="otp_verified")
        return {"status": "verified", "message": "OTP verified — transaction completed."}

    if "expired" in reason.lower() or "locked" in reason.lower():
        vault_record.status = "rejected"
        db.commit()
        trust_service.record_confirmed_outcome(db, tx.name_orig, trust_score=15.0, outcome_source="otp_failed")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=reason)

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=reason)