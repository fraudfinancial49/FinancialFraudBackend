"""
Incremental, per-account behavioral profiling using Welford's online mean/
variance algorithm — average and standard deviation stay numerically stable
without ever scanning that account's historical transaction rows.
"""
import math
import statistics
from collections import Counter
from datetime import datetime
from typing import Dict, List

from sqlalchemy import func
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
        # A profile can still carry confirmed feedback (fraud/legitimate) even
        # with zero Welford transaction history -- e.g. an admin confirming an
        # outcome before this account has any other recorded activity. That
        # signal must survive this early-return branch, not get zeroed out.
        confirmed_fraud = profile.confirmed_fraud_count if profile else 0
        confirmed_legit = profile.confirmed_legitimate_count if profile else 0
        confirmed_total = (confirmed_fraud or 0) + (confirmed_legit or 0)
        fraud_ratio = (confirmed_fraud or 0) / confirmed_total if confirmed_total > 0 else 0.0
        return {
            "sender_transaction_count": 0.0, "sender_unique_receivers": 0.0,
            "sender_account_age_hours": 0.0, "sender_average_amount": 0.0,
            "sender_median_amount": 0.0, "sender_amount_std": 0.0,
            "sender_max_amount": 0.0, "sender_min_amount": 0.0,
            "sender_running_total_amount": 0.0, "sender_historical_fraud_ratio": float(fraud_ratio),
            "transaction_regularity_score": 0.0, "velocity_ratio": 0.0,
            "receiver_novelty_score": 0.0,
        }
    variance = profile.amount_m2 / profile.transaction_count if profile.transaction_count > 1 else 0.0
    confirmed_total = (profile.confirmed_fraud_count or 0) + (profile.confirmed_legitimate_count or 0)
    fraud_ratio = (profile.confirmed_fraud_count or 0) / confirmed_total if confirmed_total > 0 else 0.0
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
        # Real-time feedback loop: updated immediately by record_feedback_outcome()
        # whenever an admin confirms fraud/legitimate -- never from raw model output.
        "sender_historical_fraud_ratio": float(fraud_ratio),
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


def record_feedback_outcome(db: Session, account_id: str, is_fraud: bool) -> models.BehavioralProfile:
    """Real-time feedback loop: called immediately when an admin confirms a
    transaction's outcome (POST /admin/feedback), NOT deferred to a batch job.
    Updates the sending account's confirmed fraud/legitimate counters, which
    directly feed `sender_historical_fraud_ratio` -- so every future transaction
    from this account is scored with this outcome already baked in."""
    profile = get_or_create_profile(db, account_id)
    if is_fraud:
        profile.confirmed_fraud_count = (profile.confirmed_fraud_count or 0) + 1
    else:
        profile.confirmed_legitimate_count = (profile.confirmed_legitimate_count or 0) + 1
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


# ---------------------------------------------------------------------------
# BEHAVIORAL ANOMALY DETECTION (Account Lookup dashboard panel)
# ---------------------------------------------------------------------------
# Not part of the live scoring pipeline -- purely a read-only analytical scan,
# admin-triggered (GET /admin/behavioral-anomalies), that flags accounts whose
# most recent transaction deviates sharply from their own historical baseline:
# a previously-safe account (mostly "approve") that just triggered a riskier
# tier, and/or a sudden multi-x spike in transaction amount versus their own
# average. Distinct from the live per-account risk score -- this looks at the
# TREND for one account across its own history, not the absolute risk of a
# single transaction.
_ROUTING_SEVERITY = {"otp_verification": 50.0, "auto_reject": 70.0, "honeypot": 90.0}
_MIN_BASELINE_APPROVE_RATIO = 0.6   # baseline must be "mostly approved" to call a shift meaningful
_MIN_SPIKE_RATIO = 2.5              # recent amount must be >= 2.5x the historical average
_MIN_TRANSACTIONS = 3               # need at least this many scored transactions for a baseline
_MAX_CANDIDATE_ACCOUNTS = 300       # bounds the scan on a demo-scale database


def detect_behavioral_anomalies(db: Session, limit: int = 15) -> List[Dict[str, object]]:
    """Scans accounts with enough scored history and flags ones whose most
    recent transaction broke sharply from their own baseline -- either a
    routing-tier shift (previously safe, now risky) or an amount spike (or
    both). Read-only, admin-triggered; never affects live scoring."""
    candidate_rows = (
        db.query(models.Transaction.name_orig)
        .join(models.ModelPrediction, models.ModelPrediction.transaction_id == models.Transaction.id)
        .group_by(models.Transaction.name_orig)
        .having(func.count(models.ModelPrediction.id) >= _MIN_TRANSACTIONS)
        .limit(_MAX_CANDIDATE_ACCOUNTS)
        .all()
    )

    results: List[Dict[str, object]] = []
    for (account_id,) in candidate_rows:
        rows = (
            db.query(models.Transaction, models.ModelPrediction)
            .join(models.ModelPrediction, models.ModelPrediction.transaction_id == models.Transaction.id)
            .filter(models.Transaction.name_orig == account_id)
            .order_by(models.Transaction.timestamp.asc())
            .all()
        )
        if len(rows) < _MIN_TRANSACTIONS:
            continue

        historical = rows[:-1]
        recent_tx, recent_pred = rows[-1]

        baseline_counts = Counter(pred.routing_decision for _, pred in historical)
        baseline_dominant = baseline_counts.most_common(1)[0][0]
        baseline_approve_ratio = baseline_counts.get("approve", 0) / len(historical)
        recent_routing = recent_pred.routing_decision

        routing_shift = (
            baseline_approve_ratio >= _MIN_BASELINE_APPROVE_RATIO
            and recent_routing in _ROUTING_SEVERITY
        )

        historical_amounts = [tx.amount for tx, _ in historical]
        hist_mean = statistics.fmean(historical_amounts) if historical_amounts else 0.0
        recent_amount = recent_tx.amount
        spike_ratio = (recent_amount / hist_mean) if hist_mean > 0 else 0.0
        amount_spike = hist_mean > 0 and spike_ratio >= _MIN_SPIKE_RATIO

        if not routing_shift and not amount_spike:
            continue

        severity = 0.0
        if routing_shift:
            severity += _ROUTING_SEVERITY.get(recent_routing, 40.0) * baseline_approve_ratio
        if amount_spike:
            severity += min(40.0, (spike_ratio - 1.0) * 8.0)
        severity = min(100.0, severity)

        if routing_shift and amount_spike:
            anomaly_type = "both"
        elif routing_shift:
            anomaly_type = "routing_shift"
        else:
            anomaly_type = "amount_spike"

        results.append({
            "account_id": account_id,
            "anomaly_type": anomaly_type,
            "baseline_routing": baseline_dominant,
            "baseline_approve_ratio": round(baseline_approve_ratio, 3),
            "recent_routing": recent_routing,
            "historical_avg_amount": round(hist_mean, 2),
            "recent_amount": recent_amount,
            "spike_ratio": round(spike_ratio, 2),
            "severity_score": round(severity, 1),
            "transaction_count": len(rows),
            "last_transaction_at": recent_tx.timestamp,
        })

    results.sort(key=lambda r: r["severity_score"], reverse=True)
    return results[:limit]
