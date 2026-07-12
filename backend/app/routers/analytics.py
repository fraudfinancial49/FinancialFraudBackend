from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, case
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import require_roles
from app.schemas.schemas import TransactionAnalyticsSummary, TransactionTimeseriesPoint

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

# Bank-wide analytics are restricted to analyst/admin, same as /explain --
# a plain "user" role has no business seeing aggregate fraud figures.
_ROLE_DEP = Depends(require_roles("analyst", "admin"))


def _date_bounds(start_date: Optional[date], end_date: Optional[date]) -> tuple[datetime, datetime]:
    """Inclusive [start_date, end_date] -> half-open [start_dt, end_dt) datetime bounds.
    Defaults to the last 7 days if neither is given."""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
    return start_dt, end_dt


@router.get("/summary", response_model=TransactionAnalyticsSummary)
def analytics_summary(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
    _current_user=_ROLE_DEP,
):
    start_dt, end_dt = _date_bounds(start_date, end_date)

    row = (
        db.query(
            func.count(models.Transaction.id),
            func.coalesce(func.sum(models.Transaction.amount), 0.0),
            func.coalesce(func.sum(case((models.ModelPrediction.routing_decision == "approve", 1), else_=0)), 0),
            func.coalesce(func.sum(case((models.ModelPrediction.routing_decision == "vault", 1), else_=0)), 0),
            func.coalesce(func.sum(case((models.ModelPrediction.routing_decision == "honeypot", 1), else_=0)), 0),
            func.coalesce(func.avg(models.ModelPrediction.final_risk_score), 0.0),
            func.coalesce(func.avg(models.ModelPrediction.latency_ms), 0.0),
        )
        .join(models.ModelPrediction, models.ModelPrediction.transaction_id == models.Transaction.id)
        .filter(models.Transaction.timestamp >= start_dt, models.Transaction.timestamp < end_dt)
        .one()
    )
    total, volume, approve_ct, vault_ct, honeypot_ct, avg_risk, avg_latency = row
    flagged_ct = vault_ct + honeypot_ct
    fraud_rate = (flagged_ct / total) if total > 0 else 0.0

    return TransactionAnalyticsSummary(
        start_date=start_dt.date().isoformat(),
        end_date=(end_dt - timedelta(days=1)).date().isoformat(),
        total_transactions=total,
        total_volume=float(volume),
        approve_count=approve_ct,
        vault_count=vault_ct,
        honeypot_count=honeypot_ct,
        flagged_count=flagged_ct,
        fraud_rate=float(fraud_rate),
        avg_risk_score=float(avg_risk),
        avg_latency_ms=float(avg_latency),
    )


@router.get("/timeseries", response_model=list[TransactionTimeseriesPoint])
def analytics_timeseries(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
    _current_user=_ROLE_DEP,
):
    start_dt, end_dt = _date_bounds(start_date, end_date)
    day_bucket = func.date(models.Transaction.timestamp)

    rows = (
        db.query(
            day_bucket.label("day"),
            func.count(models.Transaction.id),
            func.coalesce(func.sum(case((models.ModelPrediction.routing_decision == "approve", 1), else_=0)), 0),
            func.coalesce(func.sum(case((models.ModelPrediction.routing_decision == "vault", 1), else_=0)), 0),
            func.coalesce(func.sum(case((models.ModelPrediction.routing_decision == "honeypot", 1), else_=0)), 0),
        )
        .join(models.ModelPrediction, models.ModelPrediction.transaction_id == models.Transaction.id)
        .filter(models.Transaction.timestamp >= start_dt, models.Transaction.timestamp < end_dt)
        .group_by(day_bucket)
        .order_by(day_bucket)
        .all()
    )

    return [
        TransactionTimeseriesPoint(
            date=str(day),
            total=total,
            approve_count=approve_ct,
            vault_count=vault_ct,
            honeypot_count=honeypot_ct,
            flagged_count=vault_ct + honeypot_ct,
        )
        for day, total, approve_ct, vault_ct, honeypot_ct in rows
    ]
