"""
Loads every frozen Phase 3 artifact EXACTLY ONCE at process startup and never
retrains, refits, or mutates any of them.

Rewritten to natively match the real `artifacts/phase3/` tree:
  - Models + isotonic calibrators live under `artifacts/phase3/models/`.
  - There is NO `best_model.joblib` bundle and NO `hybrid_risk_engine.joblib`
    on disk -- those files never existed. Every piece of equivalent metadata
    (champion model name, tree/deep feature-column order, fusion weights,
    threshold strategy, etc.) is parsed directly out of the single
    `phase3_metadata_registry.json` file Phase 3 Block 9 actually writes.
  - `feature_names_in_` on the loaded estimators is kept ONLY as a fallback
    cross-check if the registry's column lists are ever empty -- the registry
    itself (ground truth straight from the Phase 3 run) is preferred.
"""
import os
import json
import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger("ml_service")

DEEP_MODELS = {"logistic_regression", "mlp"}
TREE_MODELS = {"random_forest", "xgboost", "lightgbm", "isolation_forest"}


class ShapExplainerError(Exception):
    """Raised when a live, single-transaction SHAP explanation cannot be computed:
    the `shap_explainer.joblib` artifact is missing, the tree-feature column order
    is unresolved, or the live feature vector cannot be safely aligned to it. Maps
    to HTTP 503 at the router layer -- this is a service-availability problem, not
    a client request-validation problem.
    """


class ModelRegistry:
    """Container for every loaded Phase 3 artifact + the metadata registry."""

    def __init__(self) -> None:
        self.models: Dict[str, object] = {}
        self.calibrators: Dict[str, object] = {}
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
        self.explainer = None  # frozen Phase 3 SHAP TreeExplainer, for live per-transaction XAI
        self.loaded: bool = False

    # --- path helpers --------------------------------------------------------
    def _phase3_path(self, filename: str) -> str:
        return os.path.join(settings.PHASE3_ARTIFACTS_DIR, filename)

    def _phase2_path(self, filename: str) -> str:
        return os.path.join(settings.PHASE2_ARTIFACTS_DIR, filename)

    def _phase1_path(self, filename: str) -> str:
        return os.path.join(settings.PHASE1_ARTIFACTS_DIR, filename)

    def _model_path(self, filename: str) -> str:
        return os.path.join(settings.PHASE3_MODELS_DIR, filename)

    def load(self) -> None:
        """Load all Phase 3 artifacts natively from the real directory tree.
        Raises if any required file is missing -- Phase 4 must never silently
        serve with a partial/corrupt model set."""

        # --- 1) Metadata registry: the single source of truth for champion
        # model name, feature-column order, and fusion weights. ---
        metadata_path = self._phase3_path(settings.PHASE3_METADATA_REGISTRY_FILE)
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Required Phase 3 artifact missing: {metadata_path}. Phase 4 reads the "
                f"champion model name, feature-column order, and fusion weights directly "
                f"out of this file -- it does not retrain or reconstruct them."
            )
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.tree_feature_cols = list(self.metadata.get("tree_feature_cols", []))
        self.deep_feature_cols = list(self.metadata.get("deep_feature_cols", []))
        self.model_matrix_kind = dict(self.metadata.get("model_matrix_kind", {}))
        self.graph_metric_cols = list(self.metadata.get("graph_metric_cols", []))
        self.fusion_weights = dict(self.metadata.get("fusion_weights", {}))
        self.fusion_signal_names = list(self.metadata.get("fusion_signal_names", []))
        self.champion_model = str(self.metadata.get("champion_model", "random_forest"))

        # --- 2) Phase 2 schema registry: gives the frozen trust_score /
        # behavioral_risk_score ranges needed to min-max scale those two
        # signals into fusion, exactly as Phase 3 Block 7 did. ---
        phase2_schema_path = self._phase2_path(settings.PHASE2_SCHEMA_REGISTRY_FILE)
        if os.path.isfile(phase2_schema_path):
            with open(phase2_schema_path) as f:
                self.phase2_schema = json.load(f)
            self.trust_score_range = list(
                self.phase2_schema.get("trust_score_range", self.trust_score_range)
            )
            self.behavioral_risk_score_range = list(
                self.phase2_schema.get("behavioral_risk_score_range", self.behavioral_risk_score_range)
            )
        else:
            logger.warning(
                "Phase 2 schema_registry.json not found at %s -- falling back to default "
                "trust/behavioral risk ranges for fusion normalization.", phase2_schema_path,
            )

        # --- 3) Models + isotonic calibrators, from artifacts/phase3/models/ ---
        for name in settings.MODEL_NAMES:
            model_path = self._model_path(f"{name}.joblib")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Required Phase 3 artifact missing: {model_path}. Phase 4 does not "
                    f"retrain models -- artifacts/phase3/models/ must be populated."
                )
            self.models[name] = joblib.load(model_path)

            calibrator_path = self._model_path(f"{name}{settings.ISOTONIC_CALIBRATOR_SUFFIX}")
            if os.path.isfile(calibrator_path):
                self.calibrators[name] = joblib.load(calibrator_path)
            else:
                logger.warning(
                    "Isotonic calibrator missing for '%s' at %s -- serving raw (uncalibrated) "
                    "scores for this engine.", name, calibrator_path,
                )

        # --- 4) Deep Matrix scaler (replaces the old, nonexistent mlp_scaler.joblib
        # -- Phase 3 fits ONE scaler shared by both deep-matrix models). ---
        self.deep_scaler = joblib.load(self._phase3_path(settings.PHASE3_DEEP_SCALER_FILE))

        # --- 5) Label encoder: prefer the Phase 3 re-persisted copy, fall back to
        # Phase 1's original. ---
        label_encoder_path = self._phase3_path(settings.PHASE3_LABEL_ENCODER_FILE)
        if not os.path.isfile(label_encoder_path):
            label_encoder_path = self._phase1_path(settings.PHASE1_LABEL_ENCODER_FILE)
        self.label_encoder = joblib.load(label_encoder_path) if os.path.isfile(label_encoder_path) else None

        # --- 6) Metric panel (replaces the old, nonexistent model_metrics.csv) ---
        metrics_path = self._phase3_path(settings.PHASE3_BASELINE_METRIC_PANEL_FILE)
        self.model_metrics = pd.read_csv(metrics_path, index_col=0) if os.path.isfile(metrics_path) else None

        # --- 7) Fallback feature-column resolution, only if the registry ever
        # ships an empty list (defensive; should not happen against a real
        # Phase 3 export). ---
        if not self.tree_feature_cols:
            self.tree_feature_cols = self._resolve_feature_cols_from_models(TREE_MODELS)
        if not self.deep_feature_cols:
            self.deep_feature_cols = self._resolve_feature_cols_from_models(DEEP_MODELS)

        # --- 8) Real-time XAI deployment hook: the champion model's fitted SHAP
        # TreeExplainer, loaded once so live single-transaction requests can be
        # explained without re-fitting anything. Missing this one artifact must
        # never block the rest of the API from starting. ---
        shap_explainer_path = self._phase3_path(settings.PHASE3_SHAP_EXPLAINER_FILE)
        if os.path.isfile(shap_explainer_path):
            self.explainer = joblib.load(shap_explainer_path)
            logger.info(
                "Loaded live SHAP TreeExplainer (champion=%s) from %s for real-time XAI serving.",
                self.champion_model, shap_explainer_path,
            )
        else:
            self.explainer = None
            logger.warning(
                "SHAP explainer artifact not found at %s -- POST /explain will return 503 "
                "until artifacts/phase3/shap_explainer.joblib is present.", shap_explainer_path,
            )

        self.loaded = True
        logger.info(
            "ModelRegistry loaded: %d engines, %d calibrators, tree_cols=%d, deep_cols=%d, "
            "champion_model=%s, fusion_signals=%s, shap_loaded=%s",
            len(self.models), len(self.calibrators), len(self.tree_feature_cols),
            len(self.deep_feature_cols), self.champion_model, self.fusion_signal_names,
            self.explainer is not None,
        )

    def _resolve_feature_cols_from_models(self, model_names: set) -> List[str]:
        for name in model_names:
            model = self.models.get(name)
            cols = getattr(model, "feature_names_in_", None)
            if cols is not None:
                return list(cols)
        return []

    def isolation_forest_unified_score(self, X_tree: pd.DataFrame) -> float:
        """[0, 1]-scaled pseudo-probability for Isolation Forest, matching Phase
        3 Block 5's `unified_score` convention (higher = more anomalous). Phase 3's
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
