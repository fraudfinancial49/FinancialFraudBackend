import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.db.base import get_db
from app.db import models
from app.core.deps import get_current_user
from app.core.security import hash_password, verify_password, create_access_token
from app.schemas.schemas import (
    CustomerRegisterRequest, CustomerLoginRequest, CustomerAuthResponse,
    BalanceOut, CustomerTransactionOut,
)

logger = logging.getLogger("customer_router")
router = APIRouter(prefix="/api/v1/customer", tags=["customer"])


@router.post("/register", response_model=CustomerAuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: CustomerRegisterRequest, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == payload.email).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered.")

    account_id = f"C{uuid.uuid4().hex[:15].upper()}"

    user = models.User(
        email=payload.email, hashed_password=hash_password(payload.password), role="customer",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    customer = models.Customer(
        user_id=user.id, account_id=account_id, full_name=payload.full_name, email=payload.email,
    )
    db.add(customer)

    balance = models.Balance(account_id=account_id, amount=payload.opening_balance)
    db.add(balance)
    db.commit()

    token = create_access_token(subject=user.id, role=user.role)
    return CustomerAuthResponse(access_token=token, account_id=account_id, full_name=payload.full_name)


@router.post("/login", response_model=CustomerAuthResponse)
def login(payload: CustomerLoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    customer = db.query(models.Customer).filter(models.Customer.user_id == user.id).first()
    if customer is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a customer account.")

    token = create_access_token(subject=user.id, role=user.role)
    return CustomerAuthResponse(access_token=token, account_id=customer.account_id, full_name=customer.full_name)


def _current_customer(db: Session, current_user: models.User) -> models.Customer:
    customer = db.query(models.Customer).filter(models.Customer.user_id == current_user.id).first()
    if customer is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a customer account.")
    return customer


@router.get("/me/balance", response_model=BalanceOut)
def get_my_balance(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    customer = _current_customer(db, current_user)
    balance = db.query(models.Balance).filter(models.Balance.account_id == customer.account_id).first()
    if balance is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No balance record found.")
    return balance


@router.get("/me/transactions", response_model=list[CustomerTransactionOut])
def get_my_transactions(
    page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user),
):
    customer = _current_customer(db, current_user)
    rows = (
        db.query(models.Transaction)
        .filter(or_(models.Transaction.name_orig == customer.account_id,
                     models.Transaction.name_dest == customer.account_id))
        .order_by(models.Transaction.timestamp.desc())
        .offset((page - 1) * page_size).limit(page_size)
        .all()
    )
    out = []
    for tx in rows:
        prediction = (
            db.query(models.ModelPrediction)
            .filter(models.ModelPrediction.transaction_id == tx.id)
            .order_by(models.ModelPrediction.created_at.desc()).first()
        )
        routing = prediction.routing_decision if prediction else None
        vault = db.query(models.SafeVaultTransaction).filter(models.SafeVaultTransaction.transaction_id == tx.id).first()
        tx_status = vault.status if (routing == "otp_verification" and vault) else (
            "rejected" if routing == "auto_reject" else "completed" if routing else "processing"
        )
        out.append(CustomerTransactionOut(
            id=tx.id, name_orig=tx.name_orig, name_dest=tx.name_dest, type=tx.type,
            amount=tx.amount, routing_decision=routing, status=tx_status, timestamp=tx.timestamp,
        ))
    return out
