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
        
    metrics = []
    loaded_models = []
    
    if getattr(registry, "model_metrics", None) is not None:
        # 1. Create a safe copy and extract the model names
        df = registry.model_metrics.copy().reset_index().rename(columns={"index": "model"})
        
        # 2. Normalize the column headers to perfectly match what React expects
        # Converts "F1 Score" to "f1_score", "PR-AUC" to "pr_auc", etc.
        df.columns = [str(c).lower().replace(" ", "_").replace("-", "_") for c in df.columns]
        
        # 3. Catch standard variations and force them to React's exact internal keys
        rename_map = {
            "pr_auc": "average_precision",
            "f1": "f1_score"
        }
        df = df.rename(columns=rename_map)
        
        metrics = df.to_dict(orient="records")
        loaded_models = df["model"].tolist()
        
    return {
        "best_model": getattr(registry, "best_model_name", "Unknown"),
        "champion_model": getattr(registry, "champion_model", "Unknown"),
        # Use the actual model list pulled from the HF metrics CSV
        "engines_loaded": loaded_models,
        "calibrators_loaded": loaded_models,
        "tree_feature_count": len(getattr(registry, "tree_feature_cols", [])),
        "deep_feature_count": len(getattr(registry, "deep_feature_cols", [])),
        "fusion_signal_names": getattr(registry, "fusion_signal_names", []),
        "model_metrics": metrics,
    }
    
