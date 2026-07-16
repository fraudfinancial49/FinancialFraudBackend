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
        
    checks["models_loaded"] = getattr(ml_service.registry, "loaded", False)
    checks["graph_loaded"] = getattr(graph_svc_module.graph_service, "loaded", False)
    
    overall = all(checks.values())
    return {"status": "ready" if overall else "not_ready", "checks": checks}

@router.get("/model-info")
def model_info():
    registry = ml_service.registry
    
    if not getattr(registry, "loaded", False):
        return {"status": "not_loaded"}
        
    # Read the active calibrators from RAM instead of the purged models
    live_engines = list(getattr(registry, "calibrators", {}).keys())
    
    metrics = []
    if getattr(registry, "model_metrics", None) is not None:
        df = registry.model_metrics.reset_index().rename(columns={"index": "model"})
        if live_engines:
            df = df[df["model"].isin(live_engines)]
        metrics = df.to_dict(orient="records")
        
    return {
        "best_model": getattr(registry, "best_model_name", "Unknown"),
        "champion_model": getattr(registry, "champion_model", "Unknown"),
        "engines_loaded": live_engines,
        "calibrators_loaded": live_engines,
        "tree_feature_count": len(getattr(registry, "tree_feature_cols", [])),
        "deep_feature_count": len(getattr(registry, "deep_feature_cols", [])),
        "fusion_signal_names": getattr(registry, "fusion_signal_names", []),
        "model_metrics": metrics,
    }
    
