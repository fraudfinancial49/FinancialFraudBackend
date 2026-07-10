from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.services import ml_service, graph_service as graph_svc_module

router = APIRouter(tags=["ops"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(db: Session = Depends(get_db)):
    checks = {"database": False, "models_loaded": False, "graph_loaded": False}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False
    checks["models_loaded"] = ml_service.registry.loaded
    checks["graph_loaded"] = graph_svc_module.graph_service.loaded
    overall = all(checks.values())
    return {"status": "ready" if overall else "not_ready", "checks": checks}


@router.get("/model-info")
def model_info():
    registry = ml_service.registry
    if not registry.loaded:
        return {"status": "not_loaded"}
    # `baseline_metric_panel.csv` is indexed by model name (no "model" column
    # header) -- reset_index so /model-info still returns a clean, named
    # per-model record list, matching the old model_metrics.csv shape.
    if registry.model_metrics is not None:
        metrics = registry.model_metrics.reset_index().rename(columns={"index": "model"}).to_dict(orient="records")
    else:
        metrics = []
    return {
        "best_model": registry.best_model_name,
        "champion_model": registry.champion_model,
        "engines_loaded": list(registry.models.keys()),
        "calibrators_loaded": list(registry.calibrators.keys()),
        "tree_feature_count": len(registry.tree_feature_cols),
        "deep_feature_count": len(registry.deep_feature_cols),
        "fusion_signal_names": registry.fusion_signal_names,
        "model_metrics": metrics,
    }
