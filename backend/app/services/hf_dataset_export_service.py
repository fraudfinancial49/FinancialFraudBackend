"""Exports transaction history -- raw fields, risk scores, the full engineered
feature vector (behavioral/graph/trust metadata), and any confirmed human
outcome -- to a Hugging Face Dataset repo, so real production data accumulates
for future model retraining.

This is a WRITE to Hugging Face, unlike every other use of HF_TOKEN in this
app (ml_service.py / graph_service.py only download frozen model artifacts;
xai_narrative_service.py only calls Inference Providers). The token needs the
separate "Write access to contents of all repos under your personal
namespace" fine-grained permission or every call here fails on auth, even
though the same token already works for those read-only uses.

Runs only from the admin-triggered /admin/retrain cycle (never on the live
/assess request path) -- consistent with how every other batch/offline job in
this codebase (threat_intel's K-Means clustering, the retrain cycle itself)
is manually triggered rather than tied to per-request latency.
"""
import io
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from huggingface_hub import HfApi

from app.db import models
from app.core.config import settings

logger = logging.getLogger("hf_dataset_export_service")

EXPORT_BATCH_LIMIT = 2000  # keep a single upload's size/time bounded


class HFDatasetExportError(Exception):
    pass


def _latest_confirmed_outcome(db, transaction_id: str) -> Optional[str]:
    row = (
        db.query(models.FeedbackQueue)
        .filter(models.FeedbackQueue.transaction_id == transaction_id)
        .order_by(models.FeedbackQueue.created_at.desc())
        .first()
    )
    return row.confirmed_outcome if row else None


def _build_records(db, rows: List[models.Transaction]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for tx in rows:
        pred = (
            db.query(models.ModelPrediction)
            .filter(models.ModelPrediction.transaction_id == tx.id)
            .order_by(models.ModelPrediction.created_at.desc())
            .first()
        )
        record: Dict[str, object] = {
            "transaction_id": tx.id,
            "name_orig": tx.name_orig,
            "name_dest": tx.name_dest,
            "type": tx.type,
            "amount": tx.amount,
            "old_balance_orig": tx.old_balance_orig,
            "new_balance_orig": tx.new_balance_orig,
            "old_balance_dest": tx.old_balance_dest,
            "new_balance_dest": tx.new_balance_dest,
            "step": tx.step,
            "timestamp": tx.timestamp.isoformat() if tx.timestamp else None,
            "source": tx.source,
        }
        if pred is not None:
            record.update({
                "final_risk_score": pred.final_risk_score,
                "routing_decision": pred.routing_decision,
                "best_model_name": pred.best_model_name,
                "ml_ensemble_score": pred.ml_ensemble_score,
                "behavioral_risk_score": pred.behavioral_risk_score,
                "trust_score": pred.trust_score,
                "graph_risk_score": pred.graph_risk_score,
                "threat_score": pred.threat_score,
                "isolation_forest_anomaly": pred.isolation_forest_anomaly,
                "latency_ms": pred.latency_ms,
            })
            # The full 39-feature engineered vector (behavioral + graph + trust +
            # deterministic transaction features) this prediction was actually
            # scored against -- unpacked into individual columns so the dataset
            # is directly usable for retraining without a separate join/parse step.
            if pred.tree_feature_vector:
                for feature_name, value in pred.tree_feature_vector.items():
                    record[f"feature__{feature_name}"] = value
        record["confirmed_outcome"] = _latest_confirmed_outcome(db, tx.id)
        records.append(record)
    return records


def export_pending_transactions(db, limit: int = EXPORT_BATCH_LIMIT) -> Dict[str, object]:
    """Exports every Transaction not yet marked `exported_to_hf`, up to `limit`.
    Marks them exported only after a successful upload -- a failed upload leaves
    the watermark untouched so the next retrain cycle retries the same rows."""
    pending = (
        db.query(models.Transaction)
        .filter(models.Transaction.exported_to_hf.is_(False))
        .order_by(models.Transaction.timestamp.asc())
        .limit(limit)
        .all()
    )
    if not pending:
        return {"status": "no_new_data", "exported_count": 0, "repo_id": settings.HF_DATASET_REPO_ID, "file_path": None}

    token = os.getenv("HF_TOKEN")
    if not token:
        raise HFDatasetExportError("HF_TOKEN is not configured -- cannot upload to Hugging Face.")

    records = _build_records(db, pending)
    df = pd.DataFrame.from_records(records)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path_in_repo = f"transactions/export_{timestamp}.csv"

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=settings.HF_DATASET_REPO_ID, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=io.BytesIO(csv_bytes),
            path_in_repo=path_in_repo,
            repo_id=settings.HF_DATASET_REPO_ID,
            repo_type="dataset",
        )
    except Exception as exc:
        logger.exception("Hugging Face dataset export failed (repo=%s).", settings.HF_DATASET_REPO_ID)
        raise HFDatasetExportError(
            f"Upload to Hugging Face dataset repo '{settings.HF_DATASET_REPO_ID}' failed: {exc}. "
            "Confirm HF_TOKEN has WRITE access to repos (a separate permission from read/inference)."
        ) from exc

    for tx in pending:
        tx.exported_to_hf = True
    db.commit()

    logger.info(
        "Exported %d transaction(s) to Hugging Face dataset '%s' at %s.",
        len(pending), settings.HF_DATASET_REPO_ID, path_in_repo,
    )
    return {
        "status": "exported",
        "exported_count": len(pending),
        "repo_id": settings.HF_DATASET_REPO_ID,
        "file_path": path_in_repo,
    }
