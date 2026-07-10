"""
Aggregates every risk signal into the single `final_risk_score` (0-100).

There is no `hybrid_risk_engine.joblib` anywhere in the real artifact tree --
it never existed as a standalone file. The frozen fusion weights Phase 3
Block 7/8 actually computes (an objective, inverse-Brier blend across the 6
calibrated model scores + `trust_risk` + `behavioral_risk` + `graph_risk`) are
persisted as plain dict entries -- `fusion_weights` / `fusion_signal_names` --
inside `phase3_metadata_registry.json`, and are read from there (via
`ml_service.registry`) instead.

`threat_score` (browser-fingerprint threat intelligence) has no Phase 3
offline equivalent -- it is a Phase-4-only, request-time addition layered on
top at a small, fixed supplemental weight, with every weight (Phase 3's frozen
9 signals + this one) renormalized together so the final blend still sums to 1.0.
"""
from typing import Dict

from sqlalchemy.orm import Session

from app.db import models
import logging
from typing import Dict

from sqlalchemy.orm import Session

from app.db import models

# Initialize logger for this file
logger = logging.getLogger("risk_fusion")

# Phase-4-only signal, not present in Phase 3's frozen `fusion_weights`.
THREAT_SCORE_SUPPLEMENTAL_WEIGHT = 0.10

DEFAULT_FUSION_WEIGHTS = {
    "logistic_regression": 0.15, "random_forest": 0.15, "xgboost": 0.15,
    "lightgbm": 0.15, "mlp": 0.10, "isolation_forest": 0.10,
    "trust_risk": 0.08, "behavioral_risk": 0.06, "graph_risk": 0.06,
}


def get_threat_score(db: Session, browser_fingerprint: str) -> float:
    """Fetched directly from the threat-intelligence table. Defaults to 0.0 for
    unknown entities -- NEVER a hardcoded non-zero placeholder."""
    if not browser_fingerprint:
        return 0.0
    profile = (
        db.query(models.AttackerProfile)
        .filter(models.AttackerProfile.browser_fingerprint == browser_fingerprint)
        .first()
    )
    return float(profile.threat_score) if profile is not None else 0.0


def normalize_trust_risk(trust_score: float, trust_score_range: list) -> float:
    """1 - min-max(trust_score) -> higher output = riskier, exactly matching
    Phase 3 Block 7's `trust_risk` fusion signal convention. `trust_score_range`
    comes from Phase 2's frozen `schema_registry.json` (the closest available
    persisted bound; Phase 3's own train-fit-slice bound is not persisted)."""
    lo, hi = (trust_score_range + [0.0, 100.0])[:2]
    if hi <= lo:
        return 0.0
    scaled = max(0.0, min(1.0, (trust_score - lo) / (hi - lo)))
    return float(1.0 - scaled)


def normalize_behavioral_risk(behavioral_risk_score: float, behavioral_risk_score_range: list) -> float:
    """Min-max(behavioral_risk_score) into [0, 1], matching Phase 3 Block 7's
    `behavioral_risk` fusion signal convention."""
    lo, hi = (behavioral_risk_score_range + [0.0, 5.0])[:2]
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (behavioral_risk_score - lo) / (hi - lo))))


def fuse(
    calibrated_probabilities: Dict[str, float],
    trust_risk: float,
    behavioral_risk: float,
    graph_risk: float,
    threat_score: float,
    fusion_weights: Dict[str, float],
) -> float:
    """0-100 weighted fusion. `calibrated_probabilities` keys are the 6 Phase 3
    `MODEL_NAMES`; `trust_risk` / `behavioral_risk` / `graph_risk` are already
    scaled to [0, 1] (higher = riskier), matching Phase 3's `fusion_signals_*`
    convention exactly. Falls back to a documented default weight set if the
    registry's `fusion_weights` is ever empty (never crashes the request path
    on a partially-populated artifact)."""
    weights = fusion_weights or DEFAULT_FUSION_WEIGHTS

    components: Dict[str, float] = dict(calibrated_probabilities)
    components["trust_risk"] = trust_risk
    components["behavioral_risk"] = behavioral_risk
    components["graph_risk"] = graph_risk

    phase3_weight_sum = sum(weights.get(k, 0.0) for k in components) or 1.0
    # Renormalize Phase 3's frozen weights down to (1 - supplemental), then add
    # the Phase-4-only threat_score term at the supplemental weight, so the
    # full blend -- frozen + live -- still sums to 1.0.
    scale = (1.0 - THREAT_SCORE_SUPPLEMENTAL_WEIGHT) / phase3_weight_sum
    fused_0_1 = sum(components[k] * weights.get(k, 0.0) * scale for k in components)
    fused_0_1 += (threat_score / 100.0) * THREAT_SCORE_SUPPLEMENTAL_WEIGHT
    
    # FIXED: Replaced external variable scopes with local parameters
    logger.info(
        "calibrated_probabilities=%s trust_risk=%.4f behavioral_risk=%.4f graph_risk=%.4f threat_score=%.4f weights=%s", 
        calibrated_probabilities, trust_risk, behavioral_risk, graph_risk, threat_score, weights
    )
    
    return float(min(max(fused_0_1 * 100.0, 0.0), 100.0))


def route(final_risk_score: float, low_max: float, moderate_max: float) -> str:
    if final_risk_score < low_max:
        return "approve"
    if final_risk_score < moderate_max:
        return "vault"
    return "honeypot"
