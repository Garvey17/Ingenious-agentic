"""
FastAPI application entrypoint.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings, setup_logging, get_logger
from app.api import research_router
from app.api.memory import router as memory_router
from app.api.mcp import router as mcp_router
from app.mcp.client import mcp_client

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
    logger.info(f"MCP Enabled: {settings.enable_mcp}")
    
    # Boot MCP Subprocess if enabled
    if settings.enable_mcp:
        try:
            await mcp_client.connect()
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            # We don't crash here so the rest of the app can still boot
    
    yield
    
    if settings.enable_mcp:
        await mcp_client.disconnect()
        
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
            "memory_enabled": settings.enable_memory,
        }
    )


# Include routers
app.include_router(research_router, prefix="/api")
app.include_router(memory_router)  # prefix already set in memory.py (/api/memory)
app.include_router(mcp_router)     # prefix already set in mcp.py (/api/mcp)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
