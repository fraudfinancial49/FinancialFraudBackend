from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, EmailStr, Field


# --- Auth ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = Field(default="user", pattern="^(user|analyst|admin)$")


class UserOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- Transaction assessment ---
class TransactionAssessRequest(BaseModel):
    nameOrig: str
    nameDest: str
    type: str = Field(pattern="^(CASH_IN|CASH_OUT|DEBIT|PAYMENT|TRANSFER)$")
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    step: int = Field(ge=0)
    simulated_ip: Optional[str] = None
    user_agent: Optional[str] = None
    browser_fingerprint: Optional[str] = None


class TransactionAssessResponse(BaseModel):
    transaction_id: str
    final_risk_score: float
    routing_decision: str
    message: str
    latency_ms: float
    honeypot_session_id: Optional[str] = None
    vault_id: Optional[str] = None


# --- Safe Vault ---
class VaultOTPVerifyRequest(BaseModel):
    vault_id: str
    otp_code: str


class VaultAdminReviewRequest(BaseModel):
    vault_id: str
    decision: str = Field(pattern="^(approve|reject)$")
    reason: Optional[str] = None


class VaultMoveRequest(BaseModel):
    transaction_id: str
    reason: str


# --- Honeypot ---
class HoneypotStartRequest(BaseModel):
    transaction_id: Optional[str] = None
    simulated_ip: Optional[str] = None
    user_agent: Optional[str] = None
    browser_fingerprint: Optional[str] = None
    risk_score_at_entry: float = 0.0


class HoneypotAdvanceRequest(BaseModel):
    session_id: str
    event_type: str
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, Any]] = None


class HoneypotCloseRequest(BaseModel):
    session_id: str


# --- Feedback ---
class FeedbackSubmitRequest(BaseModel):
    transaction_id: str
    confirmed_outcome: str = Field(pattern="^(fraud|legitimate|unknown)$")
    notes: Optional[str] = None


class GenericStatus(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# --- Live XAI (real-time SHAP explanation) ---
class TransactionExplainResponse(BaseModel):
    transaction_id: str
    model_name: str
    final_risk_score: Optional[float] = None
    contributions: Dict[str, float]
    latency_ms: float
    cached: bool = False


# --- Admin retrain trigger ---
class AdminRetrainRequest(BaseModel):
    notes: Optional[str] = None
    dry_run: bool = False


class AdminRetrainResponse(BaseModel):
    status: str
    labels_processed: int
    fraud_labels: int
    legitimate_labels: int
    cache_entries_flushed: int
    retrain_run_id: str
    message: str
