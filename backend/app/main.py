import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db.base import Base, engine
from app.services.feature_pipeline import FeatureSchemaError
from app.services.ml_service import ShapExplainerError, registry as ml_registry
from app.services import graph_service as graph_svc_module
from app.routers import auth, transactions, vault, honeypot, admin, ops

# Initialize logging configuration once
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# Global Cross-Origin Resource Sharing (CORS) Middleware settings
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Application Route Registration
app.include_router(auth.router)
app.include_router(transactions.router)
app.include_router(vault.router)
app.include_router(honeypot.router)
app.include_router(admin.router)
app.include_router(ops.router)


@app.on_event("startup")
def on_startup():
    """
    Executes core framework initializations. 
    Heavy bulk-artifact downloading is omitted here to preserve RAM headroom.
    """
    logger.info("Syncing relational database schemas...")
    Base.metadata.create_all(bind=engine)
    
    logger.info("Loading metadata registry configurations from Hugging Face Hub...")
    # Pulls the structural JSON metadata definitions needed for lazy-loaded alignment
    ml_registry.load()
    
    logger.info("Initializing Graph Service infrastructure...")
    graph_svc_module.graph_service.load()
    
    logger.info("Startup complete. %s v%s is ready for traffic.", settings.APP_NAME, settings.APP_VERSION)


# --- Centralized Error Interceptors: Preventing Raw Stack Traces From Leaking ---

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
    logger.exception("Database transaction error caught by global interceptor")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "A database error occurred."}
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled runtime error caught by global interceptor")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "An internal error occurred."}
    )  # Fixed missing closing parenthesis here
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.db.base import Base, engine
from app.routers import auth, transactions, vault, honeypot, admin, ops

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router)
app.include_router(transactions.router)
app.include_router(vault.router)
app.include_router(honeypot.router)
app.include_router(admin.router)
app.include_router(ops.router)

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    logger.info("Startup complete. Backend ready for lazy-loaded inference.")

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
