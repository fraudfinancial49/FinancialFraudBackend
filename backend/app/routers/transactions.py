import time
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import get_current_user, require_roles
from app.schemas.schemas import (
    TransactionAssessRequest, TransactionAssessResponse, TransactionExplainResponse,
)
from app.services import ml_service, feature_pipeline, graph_service as graph_svc_module
from app.services import behavioral_service, trust_service, risk_fusion
from app.services.ml_service import ShapExplainerError
from app.core.config import settings
from app.routers.analytics import _date_bounds
logger = logging.getLogger("transactions_router")
router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])


@router.post("/assess", response_model=TransactionAssessResponse)
def assess_transaction(
    payload: TransactionAssessRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    t0 = time.time()
    registry = ml_service.registry
    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Model registry not loaded yet — try again shortly.")

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
    routing_decision = risk_fusion.route(final_risk_score, settings.LOW_RISK_MAX, settings.MODERATE_RISK_MAX)

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
    # never mutated).
    behavioral_service.update_profile(db, payload.nameOrig, payload.amount, payload.nameDest)
    graph_svc_module.graph_service.add_edge_incremental(payload.nameOrig, payload.nameDest, payload.amount)

    honeypot_session_id = None
    vault_id = None
    message = "Transaction approved."

    if routing_decision == "vault":
        vault_record = models.SafeVaultTransaction(transaction_id=tx.id, status="frozen")
        db.add(vault_record)
        db.commit()
        db.refresh(vault_record)
        vault_id = vault_record.id
        message = "Transaction frozen pending step-up verification (Safe Vault)."
    elif routing_decision == "honeypot":
        session = models.HoneypotSession(
            transaction_id=tx.id, simulated_ip=payload.simulated_ip,
            user_agent=payload.user_agent, browser_fingerprint=payload.browser_fingerprint,
            stage="started", risk_score_at_entry=final_risk_score,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        honeypot_session_id = session.id
        message = "Transaction completed successfully."  # simulated completion message shown to the attacker

    return TransactionAssessResponse(
        transaction_id=tx.id, final_risk_score=final_risk_score, routing_decision=routing_decision,
        message=message, latency_ms=latency_ms, honeypot_session_id=honeypot_session_id, vault_id=vault_id,
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


@router.get("", response_model=TransactionListResponse)
def list_transactions(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    routing_decision: Optional[str] = Query(None, pattern="^(approve|vault|honeypot)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles("analyst", "admin")),
):
    start_dt, end_dt = _date_bounds(start_date, end_date)

    base_query = (
        db.query(models.Transaction, models.ModelPrediction)
        .join(models.ModelPrediction, models.ModelPrediction.transaction_id == models.Transaction.id)
        .filter(models.Transaction.timestamp >= start_dt, models.Transaction.timestamp < end_dt)
    )
    if routing_decision:
        base_query = base_query.filter(models.ModelPrediction.routing_decision == routing_decision)

    total = base_query.count()
    rows = (
        base_query.order_by(models.Transaction.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = [
        TransactionListItem(
            transaction_id=tx.id, name_orig=tx.name_orig, name_dest=tx.name_dest,
            type=tx.type, amount=tx.amount, final_risk_score=pred.final_risk_score,
            routing_decision=pred.routing_decision, timestamp=tx.timestamp, source=tx.source,
        )
        for tx, pred in rows
    ]
    return TransactionListResponse(items=items, total=total, page=page, page_size=page_size)
