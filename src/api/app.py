"""FastAPI application setup."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import structlog

from src.api.routers import inference, panels
from src.api.schemas import ErrorResponse, HealthResponse
from src.api.database import get_db_manager
from src.config import Config, get_config
from src.inference.pipeline import InferencePipeline

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "solar_detection_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "solar_detection_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)
INFERENCE_COUNT = Counter(
    "solar_detection_inferences_total",
    "Total number of inferences",
    ["model", "result"],
)

# Global state
_pipeline: Optional[InferencePipeline] = None
_config: Optional[Config] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _pipeline, _config

    _config = get_config()
    logger.info("Starting Solar Panel Fault Detection API")

    # Initialize inference pipeline
    sparknet_path = Path(_config.checkpoints.save_dir) / "sparknet_best.pth"
    unet_path = Path(_config.checkpoints.save_dir) / "unet_best.pth"

    _pipeline = InferencePipeline(config=_config)

    # Load models if available
    if sparknet_path.exists():
        _pipeline.load_sparknet(sparknet_path)
        logger.info("Loaded SparkNet model", path=str(sparknet_path))

    if unet_path.exists():
        _pipeline.load_unet(unet_path)
        logger.info("Loaded U-Net model", path=str(unet_path))

    # Set pipeline in inference router
    inference.set_pipeline(_pipeline)

    yield

    logger.info("Shutting down API")


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional configuration object.

    Returns:
        Configured FastAPI application.
    """
    if config is None:
        config = get_config()

    app = FastAPI(
        title="Solar Panel Fault Detection API",
        description="""
        Production-grade API for solar panel fault detection using deep learning.

        ## Features
        - RGB image classification with SparkNet CNN
        - Thermal image segmentation with U-Net
        - Grad-CAM explainability visualizations
        - Severity scoring and risk assessment
        - Panel management and fault history

        ## Models
        - **SparkNet**: Multi-branch CNN with Fire Modules for fault classification
        - **U-Net**: Encoder-decoder for thermal fault segmentation
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging and metrics middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", str(time.time()))

        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
            latency = time.time() - start_time

            # Update metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).inc()
            REQUEST_LATENCY.labels(
                method=request.method, endpoint=request.url.path
            ).observe(latency)

            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                latency_ms=latency * 1000,
            )

            return response

        except Exception as e:
            latency = time.time() - start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500,
            ).inc()

            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                latency_ms=latency * 1000,
            )
            raise

    # Include routers
    app.include_router(inference.router)
    app.include_router(panels.router)

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health Check",
    )
    async def health_check() -> HealthResponse:
        """Check API health status."""
        models_loaded = {
            "sparknet": _pipeline.sparknet is not None if _pipeline else False,
            "unet": _pipeline.unet is not None if _pipeline else False,
        }

        # Check database connection
        database_connected = False
        try:
            db_manager = get_db_manager()
            # Try to get a session and execute a simple query
            session = db_manager.sync_session_factory()
            session.execute("SELECT 1")
            session.close()
            database_connected = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")

        return HealthResponse(
            status="healthy" if database_connected else "degraded",
            version="1.0.0",
            models_loaded=models_loaded,
            database_connected=database_connected,
        )

    # Prometheus metrics endpoint
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics() -> Response:
        """Get Prometheus metrics."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc) if config.logging.level == "DEBUG" else None,
            ).model_dump(),
        )

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=True,
    )
