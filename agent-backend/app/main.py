"""
FastAPI application entrypoint.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings, setup_logging, get_logger
from app.api import research_router

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")
    
    yield
    
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Agentic AI research workflow system",
    lifespan=lifespan,
)

# Configure CORS
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "llm_provider": settings.llm_provider,
        }
    )


# Include routers
app.include_router(research_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
