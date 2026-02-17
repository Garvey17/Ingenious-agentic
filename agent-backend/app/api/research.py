"""
Research API endpoints.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.prompts import ResearchRequest, ResearchResponse
from app.config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/", response_model=ResearchResponse)
async def start_research(request: ResearchRequest):
    """
    Start a new research workflow.
    
    This is a placeholder implementation for Phase 1.
    Full workflow will be implemented in Phase 2.
    
    Args:
        request: Research request parameters
        
    Returns:
        Research response with request ID and status
    """
    logger.info(f"Received research request for topic: {request.topic}")
    
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Create response (placeholder - no actual research yet)
        response = ResearchResponse(
            request_id=request_id,
            status="pending",
            topic=request.topic,
            report=None,
            error=None,
            iterations=0,
            created_at=datetime.utcnow().isoformat(),
            completed_at=None,
        )
        
        logger.info(f"Created research request: {request_id}")
        
        # TODO: Phase 2 - Trigger LangGraph workflow
        # TODO: Phase 2 - Store request in database/state
        # TODO: Phase 5 - Return streaming response
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to start research: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start research: {str(e)}"
        )


@router.get("/{request_id}", response_model=ResearchResponse)
async def get_research_status(request_id: str):
    """
    Get the status of a research request.
    
    This is a placeholder implementation for Phase 1.
    
    Args:
        request_id: Unique request ID
        
    Returns:
        Research response with current status
    """
    logger.info(f"Fetching status for request: {request_id}")
    
    # TODO: Phase 2 - Fetch from database/state
    # For now, return a placeholder response
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Research request not found: {request_id}"
    )
