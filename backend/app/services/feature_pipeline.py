"""
Reconstructs live features and maps them to the exact column layout 
expected by the Phase 3 engines. Updated to include Phase 2 rolling window mappings.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("feature_pipeline")

KNOWN_TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

KNOWN_ENGINEERABLE_FEATURES = {
    # Phase 1 -- deterministic
    "amount", "amount_log", "amount_scaled", "balance_before", "balance_after",
    "oldbalanceDest", "newbalanceDest", "sender_balance_change", "receiver_balance_change",
    "balance_difference", "sender_balance_ratio", "receiver_balance_ratio",
    "balance_conservation_error", "balance_conservation_abs_error", "balance_drain_ratio",
    "hour", "day", "day_of_week", "week", "month",
    "is_weekend", "is_business_hour", "is_night", "type_encoded",
    "step", "oldbalanceOrg", "newbalanceOrig", "sender_had_zero_balance",
    *[f"type_{t}" for t in KNOWN_TRANSACTION_TYPES],
    
    # Phase 2 -- behavioral (incrementally-approximated live history)
    "sender_transaction_count", "sender_unique_receivers", "sender_account_age_hours",
    "sender_average_amount", "sender_median_amount", "sender_amount_std", "sender_max_amount",
    "sender_min_amount", "sender_running_total_amount", "sender_historical_fraud_ratio",
    "trust_score", "behavioral_risk_score", "transaction_regularity_score",
    "velocity_ratio", "receiver_novelty_score", "hour_deviation",
    "account_amount_expanding_std", "account_rolling_1h_txn_count", "account_rolling_1h_amount_sum",
    "account_rolling_24h_txn_count", "account_rolling_24h_amount_sum", "account_rolling_7d_amount_sum",
    
    # Phase 2 -- causal topological graph forensics
    "sender_graph_out_degree", "sender_graph_pagerank", "sender_graph_hub_score",
    "sender_graph_authority_score", "sender_community_id", "sender_community_size",
    "receiver_graph_out_degree", "receiver_graph_pagerank", "receiver_graph_hub_score",
    "receiver_graph_authority_score", "receiver_community_id", "receiver_community_size",
    "is_bridge_transaction",
}

class FeatureSchemaError(Exception):
    pass

@dataclass
class RawTransaction:
    nameOrig: str
    nameDest: str
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    step: int
    timestamp: datetime

def _phase1_deterministic_features(tx: RawTransaction) -> Dict[str, float]:
    ts = tx.timestamp
    eps = 1e-9

    balance_before = tx.oldbalanceOrg
    balance_after = tx.newbalanceOrig
    debited = balance_before - balance_after
    credited = tx.newbalanceDest - tx.oldbalanceDest

    hour = ts.hour
    day_of_week = ts.weekday()
    is_weekend = int(day_of_week in (5, 6))
    is_business_hour = int(9 <= hour < 18 and not is_weekend)
    is_night = int(hour >= 22 or hour < 6)

    feats: Dict[str, float] = {
        "step": tx.step,
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "sender_had_zero_balance": float(tx.oldbalanceOrg == 0.0),
        "amount": tx.amount,
        "amount_log": float(np.log1p(tx.amount)),
        "amount_scaled": tx.amount / (balance_before + 1.0),
        "balance_before": balance_before,
        "balance_after": balance_after,
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "sender_balance_change": balance_after - balance_before,
        "receiver_balance_change": tx.newbalanceDest - tx.oldbalanceDest,
        "sender_balance_ratio": balance_after / (balance_before + 1.0),
        "receiver_balance_ratio": tx.oldbalanceDest / (tx.newbalanceDest + 1.0),
        "balance_difference": debited - tx.amount,
        "balance_conservation_error": debited - credited,
        "balance_conservation_abs_error": abs(debited - credited),
        "balance_drain_ratio": tx.amount / (balance_before + eps),
        "hour": hour,
        "day": ts.day,
        "day_of_week": day_of_week,
        "week": int(ts.isocalendar()[1]),
        "month": ts.month,
        "is_weekend": is_weekend,
        "is_business_hour": is_business_hour,
        "is_night": is_night,
    }
    type_index = {t: i for i, t in enumerate(sorted(KNOWN_TRANSACTION_TYPES))}
    feats["type_encoded"] = type_index.get(tx.type, -1)
    for t in KNOWN_TRANSACTION_TYPES:
        feats[f"type_{t}"] = int(tx.type == t)
    return feats

def build_feature_dict(
    tx: RawTransaction,
    behavioral_snapshot: Dict[str, float],
    trust_score: float,
    graph_snapshot: Dict[str, float],
) -> Dict[str, float]:
    feats = _phase1_deterministic_features(tx)
    feats.update(behavioral_snapshot)
    feats["trust_score"] = trust_score
    feats.update({k: v for k, v in graph_snapshot.items() if k != "graph_risk"})

    velocity_ratio = feats.get("velocity_ratio", 0.0)
    receiver_novelty = feats.get("receiver_novelty_score", 0.0)
    hour_deviation = min(abs(feats["hour"] - behavioral_snapshot.get("preferred_hour", feats["hour"])),
                         24 - abs(feats["hour"] - behavioral_snapshot.get("preferred_hour", feats["hour"])))
    feats["hour_deviation"] = float(hour_deviation)
    
    # Map real-time Welford stats to the Phase 2 rolling window names
    feats["account_amount_expanding_std"] = behavioral_snapshot.get("sender_amount_std", 0.0)
    feats["account_rolling_1h_txn_count"] = behavioral_snapshot.get("sender_transaction_count", 0.0)
    feats["account_rolling_24h_txn_count"] = behavioral_snapshot.get("sender_transaction_count", 0.0)
    feats["account_rolling_1h_amount_sum"] = behavioral_snapshot.get("sender_running_total_amount", 0.0)
    feats["account_rolling_24h_amount_sum"] = behavioral_snapshot.get("sender_running_total_amount", 0.0)
    feats["account_rolling_7d_amount_sum"] = behavioral_snapshot.get("sender_running_total_amount", 0.0)

    risk_weights = {"amount": 1.0, "velocity": 1.2, "time": 0.6, "receiver": 0.9}
    feats["behavioral_risk_score"] = float(np.sqrt(
        risk_weights["amount"] * max(feats["amount_log"], 0) ** 2 * 0.01
        + risk_weights["velocity"] * max(velocity_ratio, 0) ** 2
        + risk_weights["time"] * (hour_deviation / 12.0) ** 2
        + risk_weights["receiver"] * max(receiver_novelty, 0) ** 2
    ))
    return feats

def validate_feature_schema(feature_dict: Dict[str, float], expected_cols: List[str]) -> None:
    if not expected_cols:
        raise FeatureSchemaError("Expected feature column list is empty.")
    unknown = [c for c in expected_cols if c not in KNOWN_ENGINEERABLE_FEATURES]
    if unknown:
        raise FeatureSchemaError(f"{len(unknown)} column(s) required by the frozen Phase 3 model are not producible by this feature pipeline: {unknown}")

def to_ordered_frame(feature_dict: Dict[str, float], expected_cols: List[str]) -> pd.DataFrame:
    row = {col: float(feature_dict.get(col, 0.0)) for col in expected_cols}
    return pd.DataFrame([row], columns=expected_cols)