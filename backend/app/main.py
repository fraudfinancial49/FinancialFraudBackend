import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db.base import Base, engine
from app.services.feature_pipeline import FeatureSchemaError
from app.services.ml_service import ShapExplainerError, registry as ml_registry
from app.routers import auth, transactions, vault, honeypot, admin, ops, analytics, customer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Attach Routers
app.include_router(auth.router)
app.include_router(transactions.router)
app.include_router(customer.router)
app.include_router(vault.router)
app.include_router(honeypot.router)
app.include_router(admin.router)
# FIXED: Prefix attached to operations paths to meet frontend endpoint alignment requirements
app.include_router(ops.router, prefix="/api/v1")
app.include_router(analytics.router)


def _ensure_transactions_source_column():
    """Lightweight, idempotent schema patch."""
    inspector = inspect(engine)
    existing_cols = {col["name"] for col in inspector.get_columns("transactions")}
    if "source" in existing_cols:
        logger.info("'transactions.source' column already present — skipping migration.")
        return

    logger.info("Patching missing 'transactions.source' column onto existing table...")
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE transactions ADD COLUMN source VARCHAR(20) NOT NULL DEFAULT 'manual_sandbox'"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_transactions_source ON transactions (source)"
        ))
    logger.info("'transactions.source' column added successfully.")


def _ensure_incrementally_added_columns():
    """Lightweight, idempotent schema patch. `Base.metadata.create_all()` only
    creates tables that don't exist yet -- it never ALTERs an existing table, so
    an older deployed database that already had these tables before a given
    column was added to the ORM model is permanently missing it without this
    patch. Add a (table, column, sql_type) tuple here whenever a new column is
    added to an existing model. 4th element (default True) controls whether an
    index is also created -- must be False for JSON columns, since Postgres'
    default btree access method doesn't support the native `json` type."""
    inspector = inspect(engine)
    patches = [
        ("attacker_profiles", "actual_ip", "VARCHAR(64)", True),
        ("attacker_profiles", "location", "VARCHAR(128)", True),
        ("honeypot_sessions", "actual_ip", "VARCHAR(64)", True),
        ("honeypot_sessions", "location", "VARCHAR(128)", True),
        ("behavioral_profiles", "confirmed_fraud_count", "INTEGER NOT NULL DEFAULT 0", True),
        ("behavioral_profiles", "confirmed_legitimate_count", "INTEGER NOT NULL DEFAULT 0", True),
        # honeypot_events -- table predates these columns being added to the model.
        ("honeypot_events", "event_type", "VARCHAR(40)", True),
        ("honeypot_events", "sequence_index", "INTEGER", False),
        ("honeypot_events", "headers", "JSON", False),
        ("honeypot_events", "payload", "JSON", False),
        ("honeypot_events", "stage", "VARCHAR(40)", True),
        ("honeypot_events", "detail", "TEXT", False),
    ]
    for table, column, col_type, should_index in patches:
        if table not in inspector.get_table_names():
            continue  # table itself doesn't exist yet -- create_all() will create it with the column already present
        existing_cols = {col["name"] for col in inspector.get_columns(table)}
        if column in existing_cols:
            continue
        logger.info("Patching missing '%s.%s' column onto existing table...", table, column)
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
            if should_index:
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_{column} ON {table} ({column})"))
        logger.info("'%s.%s' column added successfully.", table, column)


def _ensure_honeypot_fingerprint_backfill():
    """One-time, idempotent data repair. Sessions created before honeypot_service's
    "unknown" fallback was applied consistently were left with a NULL/blank
    browser_fingerprint, while the AttackerProfile they roll up into was already
    keyed by the string "unknown" -- an exact-match timeline lookup for that
    profile could never find them. Runs on every boot but only ever touches rows
    once; a no-op once every row has a non-empty fingerprint."""
    inspector = inspect(engine)
    if "honeypot_sessions" not in inspector.get_table_names():
        return
    with engine.begin() as conn:
        result = conn.execute(text(
            "UPDATE honeypot_sessions SET browser_fingerprint = 'unknown' "
            "WHERE browser_fingerprint IS NULL OR browser_fingerprint = ''"
        ))
        if result.rowcount:
            logger.info("Backfilled browser_fingerprint on %d honeypot session(s).", result.rowcount)


@app.on_event("startup")
def on_startup():
    """Executes core framework initializations on boot."""
    logger.info("Syncing relational database schemas...")
    Base.metadata.create_all(bind=engine)
    _ensure_transactions_source_column()
    _ensure_incrementally_added_columns()
    _ensure_honeypot_fingerprint_backfill()

    logger.info("Loading metadata registry configurations from Hugging Face Hub...")
    # Flags the model registry as loaded so endpoints can begin processing requests
    ml_registry.load()
    
    # OOM PREVENTION FIX: Bulk loading of the 341MB graph structure is deferred 
    # from application boot up to request-time lazy evaluation.
    logger.info("Graph Service infrastructure initialized in lazy-load mode.")
    
    logger.info("Startup complete. Backend ready for lazy-loaded inference.")


# --- Centralized error interceptors ---

@app.exception_handler(FeatureSchemaError)
async def feature_schema_error_handler(request: Request, exc: FeatureSchemaError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "message": str(exc)}
    )


@app.exception_handler(ShapExplainerError)
async def shap_explainer_error_handler(request: Request, exc: ShapExplainerError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "error", "message": str(exc)}
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "message": "Request validation failed.", "errors": exc.errors()}
    )


@app.exception_handler(SQLAlchemyError)
async def db_error_handler(request: Request, exc: SQLAlchemyError):
    logger.exception("Database error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "A database error occurred."}
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "An internal error occurred."}
    )
