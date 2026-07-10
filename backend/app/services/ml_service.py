"""
Loads Phase 3 artifacts from Hugging Face Hub using a transient, lazy-loading 
strategy to prevent Out-Of-Memory (OOM) crashes on 512MB RAM limits.

- Lightweight artifacts (Metadata, Scalers, Encoders) are kept in RAM.
- Heavy artifacts (Models, Calibrators, SHAP Explainers) are loaded strictly 
  on-demand per transaction, evaluated, and immediately garbage-collected.
"""
import os
import json
import logging
import gc
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from app.core.config import settings

logger = logging.getLogger("ml_service")

DEEP_MODELS = {"logistic_regression", "mlp"}
TREE_MODELS = {"random_forest", "xgboost", "lightgbm", "isolation_forest"}


class ShapExplainerError(Exception):
    """Raised when a live, single-transaction SHAP explanation cannot be computed."""


class ModelRegistry:
    """Container for transient Phase 3 artifact loading + persistent metadata."""

    def __init__(self) -> None:
        # Lightweight components kept in RAM
        self.deep_scaler = None
        self.label_encoder = None
        self.metadata: Dict[str, object] = {}
        self.phase2_schema: Dict[str, object] = {}
        self.model_metrics: Optional[pd.DataFrame] = None
        
        self.tree_feature_cols: List[str] = []
        self.deep_feature_cols: List[str] = []
        self.model_matrix_kind: Dict[str, str] = {}
        self.graph_metric_cols: List[str] = []
        self.fusion_weights: Dict[str, float] = {}
        self.fusion_signal_names: List[str] = []
        self.champion_model: str = "random_forest"
        self.trust_score_range: List[float] = [0.0, 100.0]
        self.behavioral_risk_score_range: List[float] = [0.0, 5.0]
        self.loaded: bool = False

        # Hugging Face Configuration
        self.repo_id = "ff49/financialfraudmodel"  # UPDATE if your repo name differs
        self.token = os.getenv("HF_TOKEN")

    def _fetch_artifact(self, repo_filepath: str) -> str:
        """Downloads/Locates a file from HF Hub and returns the local cache path."""
        return hf_hub_download(repo_id=self.repo_id, filename=repo_filepath, token=self.token)

    def _fetch_optional_artifact(self, repo_filepath: str) -> Optional[str]:
        """Safely attempts to fetch an artifact that might not exist (e.g., calibrators)."""
        try:
            return self._fetch_artifact(repo_filepath)
        except (EntryNotFoundError, Exception):
            return None

    def load(self) -> None:
        """Loads strictly the lightweight metadata and schemas at process startup."""
        
        # --- 1) Metadata registry ---
        try:
            metadata_path = self._fetch_artifact("artifacts/phase3/phase3_metadata_registry.json")
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Required Phase 3 metadata missing from HF Hub: {str(e)}")

        self.tree_feature_cols = list(self.metadata.get("tree_feature_cols", []))
        self.deep_feature_cols = list(self.metadata.get("deep_feature_cols", []))
        self.model_matrix_kind = dict(self.metadata.get("model_matrix_kind", {}))
        self.graph_metric_cols = list(self.metadata.get("graph_metric_cols", []))
        self.fusion_weights = dict(self.metadata.get("fusion_weights", {}))
        self.fusion_signal_names = list(self.metadata.get("fusion_signal_names", []))
        self.champion_model = str(self.metadata.get("champion_model", "random_forest"))

        # --- 2) Phase 2 schema registry ---
        phase2_schema_path = self._fetch_optional_artifact("artifacts/phase2/schema_registry.json")
        if phase2_schema_path:
            with open(phase2_schema_path) as f:
                self.phase2_schema = json.load(f)
            self.trust_score_range = list(self.phase2_schema.get("trust_score_range", self.trust_score_range))
            self.behavioral_risk_score_range = list(self.phase2_schema.get("behavioral_risk_score_range", self.behavioral_risk_score_range))
        else:
            logger.warning("Phase 2 schema_registry.json not found -- falling back to default ranges.")

        # --- 3) Deep Matrix scaler ---
        scaler_path = self._fetch_optional_artifact("artifacts/phase3/deep_matrix_scaler.joblib")
        if scaler_path:
            self.deep_scaler = joblib.load(scaler_path)

        # --- 4) Label encoder ---
        encoder_path = self._fetch_optional_artifact("artifacts/phase3/label_encoder.joblib")
        if not encoder_path:
            encoder_path = self._fetch_optional_artifact("artifacts/phase1/label_encoder.joblib")
        if encoder_path:
            self.label_encoder = joblib.load(encoder_path)

        # --- 5) Metric panel ---
        metrics_path = self._fetch_optional_artifact("artifacts/phase3/baseline_metric_panel.csv")
        if metrics_path:
            self.model_metrics = pd.read_csv(metrics_path, index_col=0)

        # --- 6) Fallback feature-column resolution ---
        if not self.tree_feature_cols or not self.deep_feature_cols:
            logger.info("Feature columns missing from metadata. Running transient fallback resolution...")
            self._resolve_feature_cols_fallback()

        self.loaded = True
        logger.info(
            "ModelRegistry Base loaded: tree_cols=%d, deep_cols=%d, champion_model=%s",
            len(self.tree_feature_cols), len(self.deep_feature_cols), self.champion_model,
        )

    def _resolve_feature_cols_fallback(self) -> None:
        """Loads one tree and one deep model briefly just to extract column names, then deletes them."""
        if not self.tree_feature_cols:
            path = self._fetch_optional_artifact("artifacts/phase3/models/random_forest.joblib")
            if path:
                model = joblib.load(path)
                self.tree_feature_cols = list(getattr(model, "feature_names_in_", []))
                del model
        
        if not self.deep_feature_cols:
            path = self._fetch_optional_artifact("artifacts/phase3/models/logistic_regression.joblib")
            if path:
                model = joblib.load(path)
                self.deep_feature_cols = list(getattr(model, "feature_names_in_", []))
                del model
        gc.collect()

    @staticmethod
    def _positive_class_shap_row(raw_shap) -> np.ndarray:
        if isinstance(raw_shap, list):
            arr = np.asarray(raw_shap[1])
        else:
            arr = np.asarray(raw_shap)
            if arr.ndim == 3:
                arr = arr[:, :, 1]
        return arr[0]

    def compute_live_shap(self, transaction_features: Dict[str, float]) -> Dict[str, float]:
        """Compute SHAP using a strictly transient Explainer to save RAM."""
        if not self.tree_feature_cols:
            raise ShapExplainerError("Tree feature column order is unresolved.")

        explainer_path = self._fetch_optional_artifact("artifacts/phase3/shap_explainer.joblib")
        if not explainer_path:
            raise ShapExplainerError("Live SHAP explainer missing from HF Hub.")

        # Transient Load
        explainer = joblib.load(explainer_path)

        row = {col: float(transaction_features.get(col, 0.0)) for col in self.tree_feature_cols}
        matrix = pd.DataFrame([row], columns=self.tree_feature_cols)

        raw_shap = explainer.shap_values(matrix)
        fraud_vector = self._positive_class_shap_row(raw_shap)
        contributions = {col: float(val) for col, val in zip(self.tree_feature_cols, fraud_vector)}
        
        # Explicit Memory Cleanup
        del explainer
        gc.collect()

        return dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def predict_proba_all(self, X_tree: pd.DataFrame, X_deep_scaled: pd.DataFrame) -> Dict[str, float]:
        """
        Runs all 6 Phase 3 engines sequentially. 
        Loads a model, predicts, and explicitly purges it from RAM to avoid OOM crashes.
        """
        out: Dict[str, float] = {}
        
        for name in settings.MODEL_NAMES:
            model_path = self._fetch_artifact(f"artifacts/phase3/models/{name}.joblib")
            calibrator_path = self._fetch_optional_artifact(f"artifacts/phase3/models/{name}{settings.ISOTONIC_CALIBRATOR_SUFFIX}")
            
            # Transient Load
            model = joblib.load(model_path)
            calibrator = joblib.load(calibrator_path) if calibrator_path else None

            # Prediction Logic
            if name == "isolation_forest":
                raw_score = -model.decision_function(X_tree)[0]
                unified_score = float(1.0 / (1.0 + np.exp(-raw_score)))
                out[name] = float(calibrator.predict([unified_score])[0]) if calibrator else unified_score
            else:
                input_data = X_deep_scaled if name in DEEP_MODELS else X_tree
                raw_prob = float(model.predict_proba(input_data)[:, 1][0])
                out[name] = float(calibrator.predict([raw_prob])[0]) if calibrator else raw_prob

            # Explicit Memory Cleanup
            del model
            if calibrator:
                del calibrator
            gc.collect()

        return out

    @property
    def best_model_name(self) -> str:
        return self.champion_model


registry = ModelRegistry()
higher = more anomalous). Phase 3's
        exact train-fit min/max normalization bounds are not persisted anywhere in
        `phase3_metadata_registry.json`, so this uses a documented, monotone
        sigmoid-squash approximation of the same raw anomaly signal instead --
        the ordering and the calibrator fit on top of it are unaffected by this
        substitution, only the pre-calibration scale is approximate."""
        model = self.models["isolation_forest"]
        raw = -model.decision_function(X_tree)[0]  # higher raw = more anomalous
        return float(1.0 / (1.0 + np.exp(-raw)))

    @staticmethod
    def _positive_class_shap_row(raw_shap) -> np.ndarray:
        """Normalize a single-row `TreeExplainer.shap_values()` result -- which may
        come back as a length-2 list of 2D arrays, a 3D `(n_samples, n_features,
        n_classes)` array, or already a plain 2D `(n_samples, n_features)` array
        depending on the installed shap/sklearn version -- down to one clean 1D
        positive-class (fraud) vector. Mirrors Phase 3 Block 8's
        `shap_positive_class_matrix` helper exactly, so the live serving path and
        the offline analysis notebook never disagree about output shape."""
        if isinstance(raw_shap, list):
            arr = np.asarray(raw_shap[1])
        else:
            arr = np.asarray(raw_shap)
            if arr.ndim == 3:
                arr = arr[:, :, 1]
        return arr[0]

    def compute_live_shap(self, transaction_features: Dict[str, float]) -> Dict[str, float]:
        """Compute real-time, per-transaction SHAP contributions for the positive
        (fraud) class using the frozen Phase 3 TreeExplainer.

        `transaction_features` is a flat {column_name: value} dict for ONE
        transaction -- typically the exact tree-feature vector already persisted
        on that transaction's `ModelPrediction.tree_feature_vector` row. This
        method never trusts the caller's key set or ordering: it defensively
        reindexes onto `self.tree_feature_cols` (from `phase3_metadata_registry.json`),
        filling any absent column with 0.0 and silently dropping any extra key,
        before building the 2D matrix SHAP requires.

        Returns a dict of {feature_name: shap_value}, sorted by absolute
        contribution magnitude descending -- the most decision-relevant features
        (whichever direction they push) surface first, matching the case-level
        "why was this flagged" view a fraud analyst needs.
        """
        if self.explainer is None:
            raise ShapExplainerError(
                "Live SHAP explainer is not loaded -- artifacts/phase3/shap_explainer.joblib "
                "is missing. Real-time XAI is unavailable until this artifact exists."
            )
        if not self.tree_feature_cols:
            raise ShapExplainerError(
                "Tree feature column order is unresolved -- neither phase3_metadata_registry.json "
                "nor any loaded tree model's feature_names_in_ produced a column list. Refusing to "
                "build an unaligned SHAP matrix."
            )

        row = {col: float(transaction_features.get(col, 0.0)) for col in self.tree_feature_cols}
        matrix = pd.DataFrame([row], columns=self.tree_feature_cols)

        raw_shap = self.explainer.shap_values(matrix)
        fraud_vector = self._positive_class_shap_row(raw_shap)

        contributions = {col: float(val) for col, val in zip(self.tree_feature_cols, fraud_vector)}
        return dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def _calibrated_score(self, name: str, unified_score: float) -> float:
        """Applies the frozen isotonic calibrator for `name` on top of its raw
        [0, 1] unified score -- exactly mirroring Phase 3 Block 6's
        `test_scores_calibrated`, the same calibrated scores `fusion_weights`
        were objectively fit against (inverse-Brier). Falls back to the raw
        score if a calibrator is missing for this engine."""
        calibrator = self.calibrators.get(name)
        if calibrator is None:
            return unified_score
        return float(calibrator.predict([unified_score])[0])

    def predict_proba_all(self, X_tree: pd.DataFrame, X_deep_scaled: pd.DataFrame) -> Dict[str, float]:
        """Run every one of the 6 frozen Phase 3 engines and return
        {model_name: calibrated_fraud_probability}, matching the exact
        `test_scores_calibrated` convention `fusion_weights` were fit on."""
        out: Dict[str, float] = {}
        for name in ("random_forest", "xgboost", "lightgbm"):
            raw = float(self.models[name].predict_proba(X_tree)[:, 1][0])
            out[name] = self._calibrated_score(name, raw)
        for name in ("logistic_regression", "mlp"):
            raw = float(self.models[name].predict_proba(X_deep_scaled)[:, 1][0])
            out[name] = self._calibrated_score(name, raw)
        out["isolation_forest"] = self._calibrated_score(
            "isolation_forest", self.isolation_forest_unified_score(X_tree)
        )
        return out

    @property
    def best_model_name(self) -> str:
        return self.champion_model


registry = ModelRegistry()
