"""
Centralized application settings, loaded from environment variables / .env file.

Rewritten to natively address the REAL Google Drive artifact tree (target
folder ID 1ZKZfZRv2xWU9lK9zYYsao16dQRKRCG9f), rooted at `PROJECT_ROOT`:

    PROJECT_ROOT/
      artifacts/phase1/...
      artifacts/phase2/{graph_weight_maps.joblib, schema_registry.json, ...}
      artifacts/phase3/
        models/{model}.joblib, {model}_isotonic_calibrator.joblib
        phase3_metadata_registry.json
        deep_matrix_scaler.joblib, label_encoder.joblib, shap_explainer.joblib
        baseline_metric_panel.csv

No file is renamed or moved. There is no `best_model.joblib` or
`hybrid_risk_engine.joblib` on disk -- those were never real Phase 3 exports --
so this settings module no longer references them. Every piece of "which model
is champion" / "what are the fusion weights" metadata Phase 4 needs is parsed
out of `phase3_metadata_registry.json` at runtime by `ml_service.py`.
"""
import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- App metadata ---
    APP_NAME: str = "PaySim Fraud Prevention API"
    APP_VERSION: str = "4.0.0"
    ENV: str = Field(default="development")

    # --- Database ---
    DATABASE_URL: str = Field(default="sqlite:///./paysim_backend.db")

    # --- JWT / Auth ---
    SECRET_KEY: str = Field(default="CHANGE_ME_super_secret_key_min_32_chars_long")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- Project root: wherever the Google Drive target folder tree is mounted
    # (e.g. a local rclone/gdown sync, a Colab `drive.mount`, or a Kaggle dataset
    # attachment). Every artifact path below is resolved relative to this ONE
    # setting -- override it in `.env` to point at wherever this tree actually
    # lives; nothing else needs to change. ---
    PROJECT_ROOT: str = Field(default=".")

    # --- Phase 1 artifacts (artifacts/phase1/) ---
    PHASE1_ARTIFACTS_SUBDIR: str = "artifacts/phase1"
    PHASE1_LABEL_ENCODER_FILE: str = "label_encoder.joblib"

    # --- Phase 2 artifacts (artifacts/phase2/) ---
    PHASE2_ARTIFACTS_SUBDIR: str = "artifacts/phase2"
    PHASE2_GRAPH_WEIGHT_MAPS_FILE: str = "graph_weight_maps.joblib"
    PHASE2_SCHEMA_REGISTRY_FILE: str = "schema_registry.json"

    # --- Phase 3 artifacts (artifacts/phase3/ + artifacts/phase3/models/) ---
    PHASE3_ARTIFACTS_SUBDIR: str = "artifacts/phase3"
    PHASE3_MODELS_SUBDIR: str = "artifacts/phase3/models"
    PHASE3_METADATA_REGISTRY_FILE: str = "phase3_metadata_registry.json"
    PHASE3_DEEP_SCALER_FILE: str = "deep_matrix_scaler.joblib"
    PHASE3_LABEL_ENCODER_FILE: str = "label_encoder.joblib"
    PHASE3_SHAP_EXPLAINER_FILE: str = "shap_explainer.joblib"
    PHASE3_BASELINE_METRIC_PANEL_FILE: str = "baseline_metric_panel.csv"
    ISOTONIC_CALIBRATOR_SUFFIX: str = "_isotonic_calibrator.joblib"

    # The 6 models Phase 3 actually trains + calibrates, exactly as named on disk
    # under artifacts/phase3/models/ (`{name}.joblib`, `{name}_isotonic_calibrator.joblib`).
    MODEL_NAMES: List[str] = Field(default_factory=lambda: [
        "logistic_regression", "random_forest", "xgboost",
        "lightgbm", "mlp", "isolation_forest",
    ])

    # --- Phase 4 own runtime state (NOT part of the Drive tree -- purely local,
    # in-process bookkeeping for incremental graph/behavioral updates between
    # restarts). Created automatically; safe to delete/reset at any time. ---
    RUNTIME_STATE_DIR: str = Field(default="./backend/runtime_state")
    LIVE_GRAPH_STATE_FILE: str = "live_graph_state.joblib"
    GRAPH_PERSIST_EVERY_N_EDGES: int = 25

    # --- Risk routing thresholds (percent, 0-100 scale, matches Phase 3's hybrid
    # risk score bins) ---
    LOW_RISK_MAX: float = 30.0
    MODERATE_RISK_MAX: float = 60.0

    # --- CORS ---
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    class Config:
        env_file = ".env"
        case_sensitive = True

    # --- Resolved absolute-ish directory helpers ---------------------------
    @property
    def PHASE1_ARTIFACTS_DIR(self) -> str:
        return os.path.join(self.PROJECT_ROOT, self.PHASE1_ARTIFACTS_SUBDIR)

    @property
    def PHASE2_ARTIFACTS_DIR(self) -> str:
        return os.path.join(self.PROJECT_ROOT, self.PHASE2_ARTIFACTS_SUBDIR)

    @property
    def PHASE3_ARTIFACTS_DIR(self) -> str:
        return os.path.join(self.PROJECT_ROOT, self.PHASE3_ARTIFACTS_SUBDIR)

    @property
    def PHASE3_MODELS_DIR(self) -> str:
        return os.path.join(self.PROJECT_ROOT, self.PHASE3_MODELS_SUBDIR)


settings = Settings()

# Only the Phase-4-owned runtime state directory is ever created by Phase 4 --
# every artifacts/ directory above is treated as read-only, frozen output from
# earlier phases and is never written to, moved, or renamed.
os.makedirs(settings.RUNTIME_STATE_DIR, exist_ok=True)
