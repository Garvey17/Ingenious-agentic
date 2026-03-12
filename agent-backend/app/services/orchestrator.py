"""
Orchestrator service — runs the full research workflow graph.

Phase 3 additions:
- Before invoking graph: retrieve relevant past research from memory (if ENABLE_MEMORY=true)
- After graph completes successfully: store the research summary in memory
Both operations are fully optional and never raise — the pipeline proceeds normally on failure.
"""

from datetime import datetime
from app.graph.builder import research_graph
from app.graph.state import ResearchState
from app.services import session_store
from app.config.settings import settings
from app.config.logging import get_logger

logger = get_logger(__name__)


def _get_memory_retriever():
    """Lazily import MemoryRetriever to avoid circular imports and import-time errors."""
    try:
        from app.memory.retriever import MemoryRetriever
        return MemoryRetriever()
    except Exception as e:
        logger.error(f"[orchestrator] Could not load MemoryRetriever: {e}")
        return None


async def _retrieve_past_research(goal: str) -> list[dict]:
    """
    Fetch relevant past research from memory.
    Returns empty list on any error or when memory is disabled.
    """
    if not settings.enable_memory:
        return []
    try:
        retriever = _get_memory_retriever()
        if retriever is None:
            return []
        results = await retriever.retrieve_relevant(
            topic=goal,
            top_k=settings.memory_top_k,
        )
        logger.info(f"[orchestrator] Retrieved {len(results)} past memories for context")
        return results
    except Exception as e:
        logger.warning(f"[orchestrator] Memory retrieve failed (non-fatal): {e}")
        return []


async def _store_research(final_state: ResearchState) -> None:
    """
    Store completed research in memory.
    Silently skips on any error or when memory is disabled.
    """
    if not settings.enable_memory:
        return

    report = final_state.get("report") or {}
    summary = report.get("executive_summary", "")
    if not summary:
        # If writer failed, try to build a minimal summary from insights
        insights = final_state.get("insights") or {}
        facts = insights.get("key_facts", [])
        summary = " ".join(str(f) for f in facts[:5]) if facts else ""

    if not summary:
        logger.debug("[orchestrator] No summary to store — skipping memory write")
        return

    try:
        retriever = _get_memory_retriever()
        if retriever is None:
            return
        await retriever.store_research(
            request_id=final_state["request_id"],
            topic=final_state.get("goal", ""),
            summary=summary,
            sources=final_state.get("sources", []),
            metadata={
                "depth": final_state.get("depth", "standard"),
                "iterations": final_state.get("iteration", 0),
            },
        )
    except Exception as e:
        logger.warning(f"[orchestrator] Memory store failed (non-fatal): {e}")


async def run_research(
    request_id: str,
    goal: str,
    user_id: str | None = None,
    depth: str = "standard",
) -> ResearchState:
    """
    Run the full research workflow for a given goal.

    1. Retrieve relevant past research from memory (Phase 3, no-op if disabled)
    2. Initialise ResearchState and save it (status=running)
    3. Invoke the compiled LangGraph
    4. Store the result in memory (Phase 3, no-op if failed/disabled)
    5. Update the session store with the final state
    6. Return the final state
    """
    # Phase 3: pre-fetch relevant memories before graph runs
    past_research = await _retrieve_past_research(goal)

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
        "error": None,
        "past_research": past_research,  # Phase 3: injected into Planner
    }

    # Persist initial state so GET endpoint can return "in_progress" immediately
    await session_store.save_session(request_id, initial_state)

    logger.info(f"[orchestrator] Starting research '{goal[:60]}' (id={request_id})")

    try:
        final_state: ResearchState = await research_graph.ainvoke(initial_state)

        # Ensure status is set correctly
        if final_state.get("status") not in ("approved", "failed"):
            final_state["status"] = "approved"

        # Phase 3: persist successful research to memory
        if final_state.get("status") == "approved":
            await _store_research(final_state)

        await session_store.save_session(request_id, final_state)
        logger.info(
            f"[orchestrator] Completed request {request_id} "
            f"status={final_state.get('status')} "
            f"iterations={final_state.get('iteration', 0)}"
        )
        return final_state

    except Exception as e:
        logger.error(f"[orchestrator] Unhandled error for {request_id}: {e}")
        error_state: ResearchState = {
            **initial_state,
            "status": "failed",
            "error": str(e),
        }
        await session_store.save_session(request_id, error_state)
        return error_state
