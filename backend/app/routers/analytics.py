import logging
from datetime import date, datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy import func, case
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.db import models
from app.core.deps import require_roles
from app.schemas import schemas

logger = logging.getLogger("analytics_router")
router = APIRouter(prefix="/api/v1", tags=["analytics"])

# Bank-wide analytics are restricted to analyst/admin
_ROLE_DEP = Depends(require_roles("analyst", "admin"))


def _date_bounds(start_date: Optional[date], end_date: Optional[date]) -> tuple[datetime, datetime]:
    """Inclusive [start_date, end_date] -> half-open [start_dt, end_dt) datetime bounds."""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
    return start_dt, end_dt


@router.get("/analytics/summary", response_model=schemas.TransactionAnalyticsSummary)
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

    return schemas.TransactionAnalyticsSummary(
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


@router.get("/analytics/timeseries", response_model=List[schemas.TransactionTimeseriesPoint])
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
        schemas.TransactionTimeseriesPoint(
            date=str(day),
            total=total,
            approve_count=approve_ct,
            vault_count=vault_ct,
            honeypot_count=honeypot_ct,
            flagged_count=vault_ct + honeypot_ct,
        )
        for day, total, approve_ct, vault_ct, honeypot_ct in rows
    ]


@router.get("/transactions", response_model=schemas.TransactionListResponse)
def list_transactions(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    routing_decision: Optional[str] = Query(None, pattern="^(approve|vault|honeypot)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=200),
    db: Session = Depends(get_db),
    _current_user=_ROLE_DEP,
):
    start_dt, end_dt = _date_bounds(start_date, end_date)

    base_query = (
        db.query(models.Transaction, models.ModelPrediction)
        .join(models.ModelPrediction, models.ModelPrediction.transaction_id == models.Transaction.id)
        .filter(models.Transaction.timestamp >= start_dt, models.Transaction.timestamp < end_dt)
    )
    if routing_decision:
        base_query = base_query.filter(models.ModelPrediction.routing_decision == routing_decision)

    total = base_query.count()
    rows = (
        base_query.order_by(models.Transaction.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = [
        schemas.TransactionListItem(
            transaction_id=tx.id,
            name_orig=tx.name_orig,
            name_dest=tx.name_dest,
            type=tx.type,
            amount=tx.amount,
            final_risk_score=pred.final_risk_score,
            routing_decision=pred.routing_decision,
            timestamp=tx.timestamp,
            source=tx.source,
        )
        for tx, pred in rows
    ]
    return schemas.TransactionListResponse(items=items, total=total, page=page, page_size=page_size)
