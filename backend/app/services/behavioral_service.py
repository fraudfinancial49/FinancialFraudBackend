"""
Incremental, per-account behavioral profiling using Welford's online mean/
variance algorithm — average and standard deviation stay numerically stable
without ever scanning that account's historical transaction rows.
"""
import math
from datetime import datetime
from typing import Dict

from sqlalchemy.orm import Session

from app.db import models


def welford_update(count: int, mean: float, m2: float, new_value: float) -> tuple:
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m2 += delta * delta2
    return count, mean, m2


def get_or_create_profile(db: Session, account_id: str) -> models.BehavioralProfile:
    profile = db.query(models.BehavioralProfile).filter(
        models.BehavioralProfile.account_id == account_id
    ).first()
    if profile is None:
        profile = models.BehavioralProfile(account_id=account_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def snapshot(db: Session, account_id: str) -> Dict[str, float]:
    """Read-only snapshot of an account's current behavioral profile, safe to
    call before the transaction has been recorded."""
    profile = db.query(models.BehavioralProfile).filter(
        models.BehavioralProfile.account_id == account_id
    ).first()
    if profile is None or profile.transaction_count == 0:
        return {
            "sender_transaction_count": 0.0, "sender_unique_receivers": 0.0,
            "sender_account_age_hours": 0.0, "sender_average_amount": 0.0,
            "sender_median_amount": 0.0, "sender_amount_std": 0.0,
            "sender_max_amount": 0.0, "sender_min_amount": 0.0,
            "sender_running_total_amount": 0.0, "sender_historical_fraud_ratio": 0.0,
            "transaction_regularity_score": 0.0, "velocity_ratio": 0.0,
            "receiver_novelty_score": 0.0,
        }
    variance = profile.amount_m2 / profile.transaction_count if profile.transaction_count > 1 else 0.0
    return {
        "sender_transaction_count": float(profile.transaction_count),
        "sender_unique_receivers": float(profile.unique_receivers),
        "sender_account_age_hours": float(
            (datetime.utcnow() - profile.updated_at).total_seconds() / 3600.0
        ) if profile.updated_at else 0.0,
        "sender_average_amount": float(profile.amount_mean),
        "sender_median_amount": float(profile.amount_mean),  # median approximated by mean (streaming-safe)
        "sender_amount_std": float(math.sqrt(max(variance, 0.0))),
        "sender_max_amount": float(profile.last_amount),
        "sender_min_amount": float(profile.last_amount),
        "sender_running_total_amount": float(profile.amount_mean * profile.transaction_count),
        "sender_historical_fraud_ratio": 0.0,  # only updated via confirmed feedback, never raw model output
        "transaction_regularity_score": 0.5,
        "velocity_ratio": 1.0,
        "receiver_novelty_score": 0.0 if profile.unique_receivers > 0 else 1.0,
    }


def update_profile(db: Session, account_id: str, amount: float, receiver_id: str) -> None:
    """Apply exactly ONE incremental Welford update for the sending account.
    Called only AFTER a transaction has been assessed and logged."""
    profile = get_or_create_profile(db, account_id)
    count, mean, m2 = welford_update(
        profile.transaction_count, profile.amount_mean, profile.amount_m2, amount
    )
    profile.transaction_count = count
    profile.amount_mean = mean
    profile.amount_m2 = m2
    profile.last_amount = amount
    profile.unique_receivers = profile.unique_receivers + 1  # simple upper-bound proxy
    profile.updated_at = datetime.utcnow()
    db.add(profile)
    db.commit()
