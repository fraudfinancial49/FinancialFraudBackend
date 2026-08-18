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
    wallet_balance: float = 0.0  # Added for customer website

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- Transaction assessment ---
class TransactionAssessRequest(BaseModel):
    # nameOrig and the balance/step fields are Optional at the schema level
    # because a customer-initiated transfer never supplies them -- the router
    # derives and overwrites them server-side for that path. For the admin/
    # analyst Sandbox path they are still required; the router enforces that
    # explicitly since Pydantic can't conditionally require a field based on
    # the caller's role.
    nameOrig: Optional[str] = None
    nameDest: str
    type: str = Field(pattern="^(CASH_IN|CASH_OUT|DEBIT|PAYMENT|TRANSFER)$")
    amount: float = Field(gt=0)
    oldbalanceOrg: Optional[float] = Field(default=None, ge=0)
    newbalanceOrig: Optional[float] = Field(default=None, ge=0)
    oldbalanceDest: Optional[float] = Field(default=None, ge=0)
    newbalanceDest: Optional[float] = Field(default=None, ge=0)
    step: Optional[int] = Field(default=None, ge=0)
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
    auto_reject_id: Optional[str] = None
    # --- NEW EXPLAINABILITY FIELDS ---
    individual_scores: Optional[Dict[str, float]] = None
    fusion_weights: Optional[Dict[str, float]] = None


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


class AutoRejectedTransactionOut(BaseModel):
    id: str
    transaction_id: str
    final_risk_score: float
    reason: str
    rejected_at: datetime

    class Config:
        from_attributes = True


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


class TelemetryEventRequest(BaseModel):
    action_type: str
    target_element: str
    x_coord: float
    y_coord: float


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


# --- Analytics & transaction history ---
class TransactionAnalyticsSummary(BaseModel):
    start_date: str
    end_date: str
    total_transactions: int
    total_volume: float
    approve_count: int
    vault_count: int
    auto_reject_count: int
    block_count: int          # New Block Tier
    honeypot_count: int
    flagged_count: int
    fraud_rate: float
    avg_risk_score: float
    avg_latency_ms: float


class TransactionTimeseriesPoint(BaseModel):
    date: str
    total: int
    approve_count: int
    vault_count: int
    auto_reject_count: int
    block_count: int          # New Block Tier
    honeypot_count: int
    flagged_count: int


class TransactionListItem(BaseModel):
    transaction_id: str
    name_orig: str
    name_dest: str
    type: str
    amount: float
    final_risk_score: float
    routing_decision: str
    timestamp: datetime
    source: str


class TransactionListResponse(BaseModel):
    items: List[TransactionListItem]
    total: int
    page: int
    page_size: int


# --- NEW: Admin Controls & Customer Portal ---
class AccountStatusUpdate(BaseModel):
    is_active: bool
    reason: Optional[str] = None


class AccountProfileResponse(BaseModel):
    user_id: str
    email: EmailStr
    role: str
    is_active: bool
    wallet_balance: float
    recent_transactions: List[TransactionListItem]


class WalletBalanceResponse(BaseModel):
    user_id: str
    wallet_balance: float
    last_updated: datetime


class AccountTransactionOut(BaseModel):
    transaction_id: str
    name_orig: str
    name_dest: str
    type: str
    amount: float
    routing_decision: Optional[str] = None
    final_risk_score: Optional[float] = None
    timestamp: str


class AccountTransactionsResponse(BaseModel):
    account_id: str
    total: int
    page: int
    page_size: int
    is_blocked: bool
    transactions: List[AccountTransactionOut]


class AccountBlockRequest(BaseModel):
    reason: str = Field(..., min_length=3, description="Required justification, logged to the audit trail.")


class AccountUnblockRequest(BaseModel):
    reason: str = Field(..., min_length=3, description="Required justification, logged to the audit trail.")


class AccountBlockStatusOut(BaseModel):
    account_id: str
    is_blocked: bool
    reason: Optional[str] = None
    blocked_by: Optional[str] = None
    blocked_at: Optional[str] = None


# --- Customer-facing website (Part 4) ---
class CustomerRegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    opening_balance: float = Field(default=2000.0)


class CustomerLoginRequest(BaseModel):
    email: str
    password: str


class CustomerAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    account_id: str
    full_name: str


class BalanceOut(BaseModel):
    account_id: str
    amount: float
    updated_at: datetime

    class Config:
        from_attributes = True


class CustomerTransactionOut(BaseModel):
    id: str
    name_orig: str
    name_dest: str
    type: str
    amount: float
    routing_decision: Optional[str] = None
    status: str
    timestamp: datetime

    class Config:
        from_attributes = True


class OTPVerifyRequest(BaseModel):
    otp: str


# --- Honeypot decoy endpoints (Part 5 — full automation) ---
class DecoyBalanceRequest(BaseModel):
    account_id: str
    browser_fingerprint: str = ""
    simulated_ip: str = ""


class DecoyBalanceResponse(BaseModel):
    account_id: str
    balance: float
    updated_at: datetime


class DecoyTransferRequest(BaseModel):
    name_dest: str
    amount: float
    browser_fingerprint: str = ""
    simulated_ip: str = ""


class DecoyTransferResponse(BaseModel):
    transaction_id: str
    status: str
    message: str
    threat_score: float


class DecoyOtpRequest(BaseModel):
    otp: str
    browser_fingerprint: str = ""
    simulated_ip: str = ""


class DecoyOtpResponse(BaseModel):
    verified: bool
    threat_score: float