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
from app.routers import auth, transactions, vault, honeypot, admin, ops, analytics

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
app.include_router(vault.router)
app.include_router(honeypot.router)
app.include_router(admin.router)
app.include_router(ops.router)
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


@app.on_event("startup")
def on_startup():
    """Executes core framework initializations on boot."""
    logger.info("Syncing relational database schemas...")
    Base.metadata.create_all(bind=engine)
    _ensure_transactions_source_column()
    
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
