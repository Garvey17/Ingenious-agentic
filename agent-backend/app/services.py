"""
Services module — session storage and the main research runner.
"""

import asyncio
from typing import Optional, List

from app.config import settings, get_logger
from app.graph import ResearchState

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-Memory Session Store
# A plain dictionary keyed by request_id (UUID string).
# ---------------------------------------------------------------------------
_store: dict[str, ResearchState] = {}
_lock = asyncio.Lock()


async def save_session(request_id: str, state: ResearchState) -> None:
    """Saves (or overwrites) a research session."""
    async with _lock:
        _store[request_id] = state


async def get_session(request_id: str) -> Optional[ResearchState]:
    """Returns the session for a request ID, or None if not found."""
    async with _lock:
        return _store.get(request_id)


async def list_sessions() -> List[str]:
    """Returns a list of all active session IDs."""
    async with _lock:
        return list(_store.keys())


# ---------------------------------------------------------------------------
# Research Orchestrator
# ---------------------------------------------------------------------------

async def _load_past_research(goal: str) -> list[dict]:
    """Fetches related past research from Qdrant (only if memory is enabled)."""
    if not settings.enable_memory:
        return []
    try:
        from app.memory import Memory
        results = await Memory().search_past_research(topic=goal, top_k=settings.memory_top_k)
        logger.info(f"[orchestrator] Loaded {len(results)} past research results")
        return results
    except Exception as e:
        logger.warning(f"[orchestrator] Could not load past research: {e}")
        return []


async def _save_to_memory(final_state: ResearchState) -> None:
    """Saves the completed research summary to Qdrant for future lookups."""
    if not settings.enable_memory:
        return

    report = final_state.get("report") or {}
    summary = report.get("executive_summary", "")
    if not summary:
        facts = (final_state.get("insights") or {}).get("key_facts", [])
        summary = " ".join(str(f) for f in facts[:5])

    if not summary:
        return

    try:
        from app.memory import Memory
        await Memory().save_research(
            request_id=final_state["request_id"],
            topic=final_state.get("goal", ""),
            summary=summary,
            sources=final_state.get("sources", []),
            metadata={"depth": final_state.get("depth", "standard"),
                      "iterations": final_state.get("iteration", 0)},
        )
    except Exception as e:
        logger.warning(f"[orchestrator] Could not save to memory: {e}")


async def run_research(request_id: str, goal: str,
                       user_id: str | None = None, depth: str = "standard") -> ResearchState:
    """
    Runs the full research pipeline for a given goal.
    Called as a FastAPI background task — runs asynchronously.
    """
    from app.graph import research_graph

    initial_state: ResearchState = {
        "goal": goal,
        "request_id": request_id,
        "user_id": user_id,
        "depth": depth,
        "max_iterations": settings.max_iterations,
        "plan": [],
        "search_queries": [],
        "sources": [],
        "insights": {},
        "report": {},
        "review": {},
        "iteration": 0,
        "status": "running",
        "current_step": "planner",
        "error": None,
        "past_research": await _load_past_research(goal),
    }

    await save_session(request_id, initial_state)
    logger.info(f"[orchestrator] Starting '{goal[:60]}' (id={request_id})")

    try:
        final_state: ResearchState = await research_graph.ainvoke(initial_state)
        if final_state.get("status") not in ("approved", "failed"):
            final_state["status"] = "approved"
        if final_state.get("status") == "approved":
            await _save_to_memory(final_state)
        await save_session(request_id, final_state)
        logger.info(f"[orchestrator] Done — {request_id} status={final_state.get('status')}")
        return final_state
    except Exception as e:
        logger.error(f"[orchestrator] Unhandled error: {e}")
        error_state = {**initial_state, "status": "failed", "error": str(e)}
        await save_session(request_id, error_state)
        return error_state
