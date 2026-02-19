"""
Research API endpoints — Phase 2 implementation.
POST /api/research/   → triggers the full agent workflow (background task)
GET  /api/research/{id} → returns current status + report
"""

import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from app.prompts.schemas import ResearchRequest, ResearchResponse, ReportOutput, ReportSection
from app.services import session_store
from app.services.orchestrator import run_research
from app.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


# ---------------------------------------------------------------------------
# POST /api/research/
# ---------------------------------------------------------------------------
@router.post("/", response_model=ResearchResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a new research workflow.

    Returns 202 Accepted immediately with a request_id.
    The workflow runs in the background; poll GET /api/research/{id} for results.
    """
    request_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"[api] New research request id={request_id} topic='{request.topic}'")

    # Kick off the workflow in the background
    background_tasks.add_task(
        run_research,
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


# ---------------------------------------------------------------------------
# GET /api/research/{request_id}
# ---------------------------------------------------------------------------
@router.get("/{request_id}", response_model=ResearchResponse)
async def get_research_status(request_id: str):
    """
    Get the current status (and report, if complete) for a research request.
    """
    session = await session_store.get_session(request_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research request not found: {request_id}",
        )

    graph_status = session.get("status", "running")

    # Map internal status → API status
    api_status_map = {
        "running": "in_progress",
        "approved": "completed",
        "failed": "failed",
        "pending": "pending",
    }
    api_status = api_status_map.get(graph_status, "in_progress")

    # Build report object if available
    report_data = session.get("report")
    report: ReportOutput | None = None
    if report_data and api_status == "completed":
        try:
            # Normalise sections
            raw_sections = report_data.get("sections", [])
            sections = [
                ReportSection(
                    title=s.get("title", "Section"),
                    content=s.get("content", ""),
                )
                for s in raw_sections
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
        created_at=datetime.now(timezone.utc).isoformat(),  # placeholder; store in session in Phase 6
        completed_at=completed_at,
    )


# ---------------------------------------------------------------------------
# GET /api/research/  (list all sessions — useful for debugging)
# ---------------------------------------------------------------------------
@router.get("/", response_model=list[str])
async def list_research_sessions():
    """List all active research session IDs (debug endpoint)."""
    return await session_store.list_sessions()
