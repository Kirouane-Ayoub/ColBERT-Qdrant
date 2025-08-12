import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global startup time
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting ColBERT FastAPI server...")

    # Initialize services
    global embedding_service, qdrant_service

    # Initialize embedding service
    embedding_service = EmbeddingService(
        model_name=settings.MODEL_NAME, max_batch_size=settings.MAX_BATCH_SIZE
    )

    # Initialize Qdrant service
    qdrant_service = QdrantService(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        vector_size=settings.VECTOR_SIZE,
    )

    # Load models and connect to services
    try:
        await embedding_service.initialize()
        await qdrant_service.initialize()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    # Update router globals
    import app.api.routes as routes_module

    routes_module.embedding_service = embedding_service
    routes_module.qdrant_service = qdrant_service

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="ColBERT Embedding Service",
    description="FastAPI service for generating ColBERT multivector embeddings with Qdrant integration",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Endpoint not found"})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "ColBERT Embedding Service",
        "version": "1.0.0",
        "uptime_seconds": time.time() - startup_time,
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL,
        reload=True,  # Set to True for development
    )
