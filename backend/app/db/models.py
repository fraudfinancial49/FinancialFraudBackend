"""
PostgreSQL schema (SQLAlchemy ORM). UUID primary keys throughout; explicit
indexes on every column that the serving layer filters/sorts/joins on.

Design note: a single `users` table carries a `role` column
("user" | "analyst" | "admin") rather than several separate role-specific tables. This is the standard
RBAC pattern for a table with identical auth fields and avoids duplicating
the login/password-hash machinery across two tables — `role` is indexed so
admin-only queries are just as fast as a dedicated table would be.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


def gen_uuid():
    return str(uuid.uuid4())


# Portable UUID type: real UUID column on Postgres, CHAR(36) string on SQLite
# (used automatically by the smoke test / local dev), so no model code differs
# between environments.
class GUID(String):
    pass


def uuid_column(primary_key=False):
    return Column(String(36), primary_key=primary_key, default=gen_uuid, index=True)


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = uuid_column(primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="user", index=True)  # "user" | "analyst" | "admin"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ---------------------------------------------------------------------------
# CORE TRANSACTIONS
# ---------------------------------------------------------------------------
class Transaction(Base):
    __tablename__ = "transactions"

    id = uuid_column(primary_key=True)
    name_orig = Column(String(64), nullable=False, index=True)
    name_dest = Column(String(64), nullable=False, index=True)
    type = Column(String(20), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    old_balance_orig = Column(Float, nullable=False)
    new_balance_orig = Column(Float, nullable=False)
    old_balance_dest = Column(Float, nullable=False)
    new_balance_dest = Column(Float, nullable=False)
    step = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)

    predictions = relationship("ModelPrediction", back_populates="transaction", uselist=False)
    vault_record = relationship("SafeVaultTransaction", back_populates="transaction", uselist=False)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = uuid_column(primary_key=True)
    transaction_id = Column(String(36), ForeignKey("transactions.id"), nullable=False, index=True)

    ml_ensemble_score = Column(Float, nullable=False)
    behavioral_risk_score = Column(Float, nullable=False)
    trust_score = Column(Float, nullable=False)
    graph_risk_score = Column(Float, nullable=False)
    threat_score = Column(Float, nullable=False, default=0.0)
    final_risk_score = Column(Float, nullable=False, index=True)

    best_model_name = Column(String(50), nullable=False)
    isolation_forest_anomaly = Column(Float, nullable=False)
    routing_decision = Column(String(20), nullable=False, index=True)  # approve|vault|honeypot
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Exact {column_name: value} tree-feature vector this prediction was scored
    # against, frozen at assess-time and keyed to the model's exact fitted column
    # order (`ml_service.registry.tree_feature_cols`). This is the single source
    # of truth POST /explain reads from -- it is never re-derived from live
    # behavioral/graph state, which may have since drifted.
    tree_feature_vector = Column(JSON, nullable=True)

    transaction = relationship("Transaction", back_populates="predictions")


# ---------------------------------------------------------------------------
# SAFE VAULT (moderate-risk holding pen)
# ---------------------------------------------------------------------------
class SafeVaultTransaction(Base):
    __tablename__ = "safevault_transactions"

    id = uuid_column(primary_key=True)
    transaction_id = Column(String(36), ForeignKey("transactions.id"), nullable=False, index=True)

    status = Column(String(20), nullable=False, default="frozen", index=True)  # frozen|otp_verified|released|rejected
    otp_code = Column(String(10), nullable=True)
    otp_expires_at = Column(DateTime, nullable=True)
    otp_attempts = Column(Integer, default=0, nullable=False)

    admin_override_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    admin_override_decision = Column(String(20), nullable=True)  # approve|reject
    admin_override_reason = Column(Text, nullable=True)
    admin_override_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    transaction = relationship("Transaction", back_populates="vault_record")


# ---------------------------------------------------------------------------
# HONEYPOT SANDBOX (high/critical-risk deception layer)
# ---------------------------------------------------------------------------
class HoneypotSession(Base):
    __tablename__ = "honeypot_sessions"

    id = uuid_column(primary_key=True)
    transaction_id = Column(String(36), ForeignKey("transactions.id"), nullable=True, index=True)
    simulated_ip = Column(String(64), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    browser_fingerprint = Column(String(128), nullable=True, index=True)
    stage = Column(String(30), nullable=False, default="started", index=True)  # started|advancing|closed
    risk_score_at_entry = Column(Float, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    closed_at = Column(DateTime, nullable=True)

    events = relationship("HoneypotEvent", back_populates="session")


class HoneypotEvent(Base):
    __tablename__ = "honeypot_events"

    id = uuid_column(primary_key=True)
    session_id = Column(String(36), ForeignKey("honeypot_sessions.id"), nullable=False, index=True)
    event_type = Column(String(40), nullable=False, index=True)
    sequence_index = Column(Integer, nullable=False)
    headers = Column(JSON, nullable=True)
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    session = relationship("HoneypotSession", back_populates="events")


class AttackerProfile(Base):
    __tablename__ = "attacker_profiles"

    id = uuid_column(primary_key=True)
    browser_fingerprint = Column(String(128), nullable=False, unique=True, index=True)
    simulated_ip = Column(String(64), nullable=True, index=True)
    total_sessions = Column(Integer, default=0, nullable=False)
    avg_session_duration_seconds = Column(Float, default=0.0, nullable=False)
    avg_events_per_session = Column(Float, default=0.0, nullable=False)
    cluster_label = Column(String(40), nullable=True, index=True)  # e.g. "Automated Bot"
    threat_score = Column(Float, default=0.0, nullable=False, index=True)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# ---------------------------------------------------------------------------
# FEEDBACK / RETRAINING QUEUE (append-only, offline/async consumer only)
# ---------------------------------------------------------------------------
class FeedbackQueue(Base):
    __tablename__ = "feedback_queue"

    id = uuid_column(primary_key=True)
    transaction_id = Column(String(36), ForeignKey("transactions.id"), nullable=False, index=True)
    submitted_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    confirmed_outcome = Column(String(20), nullable=False, index=True)  # fraud|legitimate|unknown
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Set the moment an admin retrain cycle folds this label into its lightweight
    # incremental update -- NULL means "still pending review by the next cycle".
    consumed_at = Column(DateTime, nullable=True, index=True)
    consumed_by_retrain_run_id = Column(String(36), ForeignKey("retrain_runs.id"), nullable=True, index=True)


# ---------------------------------------------------------------------------
# AUDIT LOG (immutable trail of every state-altering admin action)
# ---------------------------------------------------------------------------
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = uuid_column(primary_key=True)
    actor_user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(60), nullable=False, index=True)
    target_type = Column(String(40), nullable=False, index=True)
    target_id = Column(String(36), nullable=False, index=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


# ---------------------------------------------------------------------------
# BEHAVIORAL PROFILE (incremental, Welford running stats — one row per account)
# ---------------------------------------------------------------------------
class BehavioralProfile(Base):
    __tablename__ = "behavioral_profiles"

    account_id = Column(String(64), primary_key=True)
    transaction_count = Column(Integer, default=0, nullable=False)
    amount_mean = Column(Float, default=0.0, nullable=False)
    amount_m2 = Column(Float, default=0.0, nullable=False)  # Welford's running sum of squared diffs
    unique_receivers = Column(Integer, default=0, nullable=False)
    last_amount = Column(Float, default=0.0, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# ---------------------------------------------------------------------------
# TRUST HISTORY (write-only after a CONFIRMED real-world outcome)
# ---------------------------------------------------------------------------
class TrustHistory(Base):
    __tablename__ = "trust_history"

    id = uuid_column(primary_key=True)
    account_id = Column(String(64), nullable=False, index=True)
    trust_score = Column(Float, nullable=False)
    outcome_source = Column(String(30), nullable=False)  # otp_verified|admin_override|manual_review
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


# ---------------------------------------------------------------------------
# LIVE SERVING CACHE (flushed wholesale by every admin-triggered retrain cycle)
# ---------------------------------------------------------------------------
class CacheEntry(Base):
    __tablename__ = "cache_entries"

    id = uuid_column(primary_key=True)
    cache_key = Column(String(128), nullable=False, unique=True, index=True)
    cache_type = Column(String(40), nullable=False, index=True)  # e.g. "shap_explanation"
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


# ---------------------------------------------------------------------------
# RETRAIN AUDIT (one row per admin-triggered feedback-loop / retrain cycle)
# ---------------------------------------------------------------------------
class RetrainRun(Base):
    __tablename__ = "retrain_runs"

    id = uuid_column(primary_key=True)
    triggered_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    labels_processed = Column(Integer, default=0, nullable=False)
    fraud_labels = Column(Integer, default=0, nullable=False)
    legitimate_labels = Column(Integer, default=0, nullable=False)
    cache_entries_flushed = Column(Integer, default=0, nullable=False)
    status = Column(String(20), nullable=False, default="running", index=True)  # running|completed|dry_run
    notes = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
