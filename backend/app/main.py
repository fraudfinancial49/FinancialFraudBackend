import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db.base import Base, engine
from app.services.feature_pipeline import FeatureSchemaError
from app.services.ml_service import ShapExplainerError
from app.services import ml_service, graph_service as graph_svc_module
from app.routers import auth, transactions, vault, honeypot, admin, ops
import logging
import os

import zipfile
from fastapi import FastAPI, Request, status
# ... (rest of your existing imports)
from app.core.config import settings
from app.db.base import Base, engine
from app.services.feature_pipeline import FeatureSchemaError
from app.services.ml_service import ShapExplainerError
from app.services import ml_service, graph_service as graph_svc_module
from app.routers import auth, transactions, vault, honeypot, admin, ops

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

from huggingface_hub import snapshot_download



logger = logging.getLogger("startup")
def sync_artifacts():
    """Syncs the artifacts folder from Hugging Face."""

    target_dir = "/srv/artifacts"

    logger.info("Syncing artifacts from Hugging Face...")

    

    # This downloads the repo structure to /srv/artifacts

    snapshot_download(

        repo_id="yff49/financialfraudmodel", # CHANGE THIS

        local_dir=target_dir,

        token=os.getenv("HF_TOKEN"), # Ensure this is in Render Env Vars

        ignore_patterns=["*.git*"]

    )

    logger.info("Artifacts successfully synchronized from Hugging Face.")
# --- Middleware & Routes ---
app.add_middleware(
    CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(transactions.router)
app.include_router(vault.router)
app.include_router(honeypot.router)
app.include_router(admin.router)
app.include_router(ops.router)


@app.on_event("startup")
def on_startup():
    # 1. Sync files first
    sync_artifacts_from_drive()
    
    # 2. Database
    Base.metadata.create_all(bind=engine)
    
    # 3. Load models (now safe because files exist)
    ml_service.registry.load()
    graph_svc_module.graph_service.load()
    
    logger.info("Startup complete. %s v%s ready.", settings.APP_NAME, settings.APP_VERSION)

# ... (keep your existing exception handlers below)
# --- Centralized error interceptors: never leak a raw traceback to a caller ---
@app.exception_handler(FeatureSchemaError)
async def feature_schema_error_handler(request: Request, exc: FeatureSchemaError):
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                         content={"status": "error", "message": str(exc)})


@app.exception_handler(ShapExplainerError)
async def shap_explainer_error_handler(request: Request, exc: ShapExplainerError):
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                         content={"status": "error", "message": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                         content={"status": "error", "message": "Request validation failed.",
                                  "errors": exc.errors()})


@app.exception_handler(SQLAlchemyError)
async def db_error_handler(request: Request, exc: SQLAlchemyError):
    logger.exception("Database error")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                         content={"status": "error", "message": "A database error occurred."})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                         content={"status": "error", "message": "An internal error occurred."})
