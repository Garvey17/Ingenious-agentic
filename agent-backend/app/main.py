"""
FastAPI application entrypoint for Ingenious Agentic.
All API routes are defined here in one place.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware

import app.services as svc
from app.config import settings, setup_logging, get_logger
from app.prompts import ResearchRequest, ResearchResponse, ReportOutput, ReportSection

# Setup logging first
setup_logging()
logger = get_logger(__name__)

# Create the FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Agentic AI research workflow system",
    redirect_slashes=False
)

# Configure CORS
# Hardcoded to allow all origins — credentials must be False when using wildcard.
# This is safe because all API secrets (OpenAI, Tavily) are server-side only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Health Check
# ===========================================================================

@app.get("/health")
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "memory_enabled": settings.enable_memory,
    }


# ===========================================================================
# Research Endpoints
# NOTE: The list route must come BEFORE the /{request_id} routes, otherwise
# FastAPI will try to match "list" as a request_id UUID and fail.
# ===========================================================================

@app.get("/api/research/sessions", response_model=list[str])
async def list_research_sessions():
    """List all active research session IDs (useful for debugging)."""
    return await svc.list_sessions()


@app.post("/api/research/", response_model=ResearchResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new research workflow. Returns immediately with a request_id to poll."""
    request_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"[api] New research task id={request_id} topic='{request.topic}'")

    background_tasks.add_task(
        svc.run_research,
        request_id=request_id,
        goal=request.topic,
        user_id=request.user_id,
        depth=request.depth,
    )

    return ResearchResponse(
        request_id=request_id,
        status="pending",
        topic=request.topic,
        report=None,
        error=None,
        iterations=0,
        created_at=created_at,
        completed_at=None,
    )


@app.get("/api/research/{request_id}/state")
async def get_research_state(request_id: str):
    """
    Returns the full internal ResearchState for a task.
    The frontend polls this to get the active agent step and intermediate results.
    """
    session = await svc.get_session(request_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research state not found: {request_id}",
        )
    return session


@app.get("/api/research/{request_id}", response_model=ResearchResponse)
async def get_research_status(request_id: str):
    """Returns the high-level status and final report for a research task."""
    session = await svc.get_session(request_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research task not found: {request_id}",
        )

    graph_status = session.get("status", "running")
    api_status_map = {
        "running": "in_progress",
        "approved": "completed",
        "failed": "failed",
        "pending": "pending",
    }
    api_status = api_status_map.get(graph_status, "in_progress")

    # Build report object if the task is complete
    report_data = session.get("report")
    report: Optional[ReportOutput] = None
    if report_data and api_status == "completed":
        try:
            sections = [
                ReportSection(
                    title=s.get("title", "Section"),
                    content=s.get("content", ""),
                )
                for s in report_data.get("sections", [])
            ]
            report = ReportOutput(
                topic=report_data.get("topic", session.get("goal", "")),
                executive_summary=report_data.get("executive_summary", ""),
                sections=sections,
                key_findings=report_data.get("key_findings", []),
                recommendations=report_data.get("recommendations", []),
                sources_cited=report_data.get("sources_cited", []),
                word_count=report_data.get("word_count", 0),
            )
        except Exception as e:
            logger.warning(f"[api] Could not parse report for {request_id}: {e}")

    completed_at = None
    if api_status in ("completed", "failed"):
        completed_at = datetime.now(timezone.utc).isoformat()

    return ResearchResponse(
        request_id=request_id,
        status=api_status,
        topic=session.get("goal", ""),
        report=report,
        error=session.get("error"),
        iterations=session.get("iteration", 0),
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=completed_at,
    )


# ===========================================================================
# Memory Endpoints
# ===========================================================================

@app.get("/api/memory/count")
async def count_memories():
    """Returns total number of research memories stored in Qdrant."""
    if not settings.enable_memory:
        return {"count": 0, "memory_enabled": False}
    try:
        from app.memory import Memory
        count = await Memory().get_total_count()
        return {"count": count, "memory_enabled": True}
    except Exception as e:
        logger.error(f"[api/memory] Count failed: {e}")
        return {"count": 0, "memory_enabled": True}


@app.get("/api/memory/search")
async def search_memory(
    q: str = Query(..., description="Topic to search for in past research"),
    top_k: int = Query(5, ge=1, le=20),
    score_threshold: float = Query(0.4, ge=0.0, le=1.0),
):
    """Semantic search across stored research memories in Qdrant."""
    if not settings.enable_memory:
        return {"query": q, "results": [], "count": 0, "memory_enabled": False}
    try:
        from app.memory import Memory
        results = await Memory().search_past_research(
            topic=q, top_k=top_k, score_threshold=score_threshold
        )
        return {"query": q, "results": results, "count": len(results), "memory_enabled": True}
    except Exception as e:
        logger.error(f"[api/memory] Search failed: {e}")
        return {"query": q, "results": [], "count": 0, "memory_enabled": True}


@app.delete("/api/memory/{request_id}")
async def delete_memory(request_id: str):
    """Delete a specific memory entry from Qdrant by request_id."""
    if not settings.enable_memory:
        return {"request_id": request_id, "deleted": False}
    try:
        from app.memory import Memory
        deleted = await Memory().delete_by_id(request_id)
        return {"request_id": request_id, "deleted": deleted}
    except Exception as e:
        logger.error(f"[api/memory] Delete failed: {e}")
        return {"request_id": request_id, "deleted": False}


# ===========================================================================
# MCP Stub (kept for frontend compatibility, MCP code removed)
# ===========================================================================

@app.get("/api/mcp/tools")
async def list_mcp_tools():
    """MCP stub — returns empty tool list since MCP has been removed."""
    return {"mcp_enabled": False, "tools": [], "count": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
    )
