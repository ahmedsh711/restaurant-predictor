import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

import structlog

from src.logging_conf import configure_logging, get_logger, correlation_id_var
from src.model import ZomatoSuccessModel
from src.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from src.config import settings

# Configure logging at import time
configure_logging()
logger = get_logger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header != settings.api_key:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key_header


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup — not per-request."""
    model_path = settings.model_path
    artifacts_dir = settings.artifacts_dir
    model = ZomatoSuccessModel(model_path, artifacts_dir)
    try:
        model.load()
        if model.isloaded:
            logger.info("Model loaded successfully at startup", model_path=model_path)
        else:
            logger.error("Model failed to load at startup", model_path=model_path)
    except Exception as exc:
        logger.error("Unexpected error loading model", error=str(exc))
    app.state.model = model
    yield
    app.state.model = None


app = FastAPI(
    title="Zomato Success Prediction API",
    version="0.2.0",
    description="Predicts restaurant success on Zomato using 48 features from LightGBM + ONNX Runtime",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
    """Attach a correlation ID to every request and return it in the response header."""
    correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    """Return clean 422 with useful message; log as ERROR."""
    logger.error("Validation error", errors=exc.errors())
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> Response:
    """Log traceback but do not leak it to the client."""
    import traceback
    logger.error("Unhandled exception", traceback=traceback.format_exc())
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", summary="Health check")
def health(request: Request) -> dict[str, Any]:
    """Returns 200 only if the model is loaded in memory."""
    model = request.app.state.model
    if model is None or not model.isloaded:
        logger.warning("Health check failed — model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    logger.info("Health check passed")
    return {"status": "ok", "model_loaded": True}


@app.get("/metadata", summary="Model metadata")
def metadata(request: Request) -> dict[str, Any]:
    """Returns model version, framework, and artifact info."""
    model = request.app.state.model
    artifact_hash: str | None = None
    if model is not None and model.isloaded:
        try:
            import hashlib
            with open(settings.model_path, "rb") as f:
                artifact_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            artifact_hash = None
    return {
        "model_version": settings.model_version,
        "framework": "LightGBM + ONNX Runtime",
        "feature_count": 49,
        "artifact_hash": artifact_hash,
        "feature_names": [
            "online_order", "book_table", "votes", "cost_for_two",
            "listed_in_type_*", "rest_type_*", "location_freq",
            "city_freq", "cuisine_count", "cuisine_*"
        ],
    }


@app.post("/predict", response_model=PredictResponse, summary="Single prediction")
def predict(
    request: PredictRequest,
    http_request: Request,
    api_key: str = Depends(get_api_key),
) -> PredictResponse:
    """Predict whether a restaurant will succeed (rating >= 3.75)."""
    model = http_request.app.state.model
    if model is None or not model.isloaded:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = request.model_dump()
    correlation_id = correlation_id_var.get() or str(uuid.uuid4())

    # Warn if inputs look out-of-range
    if features.get("votes", 0) == 0:
        logger.warning("Prediction with zero votes — may be less reliable")

    t0 = time.perf_counter()
    raw = model.predict_one(features)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "Prediction served",
        will_succeed=raw["will_succeed"],
        latency_ms=round(latency_ms, 2),
    )

    return PredictResponse(
        success_probability=raw["success_probability"],
        will_succeed=raw["will_succeed"],
        model_version=settings.model_version,
        correlation_id=correlation_id,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch prediction")
def predict_batch(
    request: BatchPredictRequest,
    http_request: Request,
    api_key: str = Depends(get_api_key),
) -> BatchPredictResponse:
    """Predict success for a list of restaurants."""
    model = http_request.app.state.model
    if model is None or not model.isloaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    correlation_id = correlation_id_var.get() or str(uuid.uuid4())
    results: list[PredictResponse] = []

    t_batch = time.perf_counter()
    for item in request.items:
        features = item.model_dump()
        t0 = time.perf_counter()
        raw = model.predict_one(features)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        results.append(
            PredictResponse(
                success_probability=raw["success_probability"],
                will_succeed=raw["will_succeed"],
                model_version=settings.model_version,
                correlation_id=correlation_id,
                latency_ms=round(latency_ms, 3),
            )
        )

    total_latency = (time.perf_counter() - t_batch) * 1000.0
    logger.info(
        "Batch prediction served",
        batch_size=len(request.items),
        total_latency_ms=round(total_latency, 2),
    )

    return BatchPredictResponse(results=results)