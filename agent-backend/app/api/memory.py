"""
Memory API — search and manage past research stored in Qdrant.

Endpoints:
  GET  /api/memory/search?q=<query>&top_k=5   — semantic search over stored memories
  GET  /api/memory/count                        — total number of stored memories
  DELETE /api/memory/{request_id}              — remove a specific memory entry

All endpoints return graceful error responses when memory is disabled
or Qdrant is unavailable — they never raise 500 errors.
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from app.config import settings, get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class MemorySearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    count: int
    memory_enabled: bool


class MemoryCountResponse(BaseModel):
    count: int
    memory_enabled: bool


class MemoryDeleteResponse(BaseModel):
    request_id: str
    deleted: bool


# ---------------------------------------------------------------------------
# Helper — lazy retriever
# ---------------------------------------------------------------------------

def _get_retriever():
    """Get retriever instance; return None if memory is disabled or import fails."""
    if not settings.enable_memory:
        return None
    try:
        from app.memory.retriever import MemoryRetriever
        return MemoryRetriever()
    except Exception as e:
        logger.error(f"[memory_api] Could not load MemoryRetriever: {e}")
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/search", response_model=MemorySearchResponse)
async def search_memory(
    q: str = Query(..., description="Search query to find relevant past research"),
    top_k: int = Query(5, ge=1, le=20, description="Maximum number of results"),
    score_threshold: float = Query(0.4, ge=0.0, le=1.0, description="Minimum similarity score"),
):
    """
    Semantic search over past research stored in memory.

    Returns relevant past research ordered by similarity to the query.
    Returns an empty list (not an error) when memory is disabled or Qdrant is unavailable.
    """
    if not settings.enable_memory:
        logger.debug("[memory_api] search called but ENABLE_MEMORY=false")
        return MemorySearchResponse(
            query=q,
            results=[],
            count=0,
            memory_enabled=False,
        )

    retriever = _get_retriever()
    if retriever is None:
        return MemorySearchResponse(
            query=q,
            results=[],
            count=0,
            memory_enabled=True,
        )

    try:
        results = await retriever.retrieve_relevant(
            topic=q,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return MemorySearchResponse(
            query=q,
            results=results,
            count=len(results),
            memory_enabled=True,
        )
    except Exception as e:
        logger.error(f"[memory_api] search error: {e}")
        return MemorySearchResponse(
            query=q,
            results=[],
            count=0,
            memory_enabled=True,
        )


@router.get("/count", response_model=MemoryCountResponse)
async def count_memories():
    """Return the total number of research summaries stored in memory."""
    if not settings.enable_memory:
        return MemoryCountResponse(count=0, memory_enabled=False)

    retriever = _get_retriever()
    if retriever is None:
        return MemoryCountResponse(count=0, memory_enabled=True)

    try:
        count = await retriever.count()
        return MemoryCountResponse(count=count, memory_enabled=True)
    except Exception as e:
        logger.error(f"[memory_api] count error: {e}")
        return MemoryCountResponse(count=0, memory_enabled=True)


@router.delete("/{request_id}", response_model=MemoryDeleteResponse)
async def delete_memory(request_id: str):
    """
    Delete a specific past research entry by request_id.

    Returns deleted=False (not a 404) when memory is disabled or entry not found.
    """
    if not settings.enable_memory:
        return MemoryDeleteResponse(request_id=request_id, deleted=False)

    retriever = _get_retriever()
    if retriever is None:
        return MemoryDeleteResponse(request_id=request_id, deleted=False)

    try:
        deleted = await retriever.delete(request_id)
        if deleted:
            logger.info(f"[memory_api] Deleted memory for request_id={request_id}")
        return MemoryDeleteResponse(request_id=request_id, deleted=deleted)
    except Exception as e:
        logger.error(f"[memory_api] delete error for {request_id}: {e}")
        return MemoryDeleteResponse(request_id=request_id, deleted=False)
