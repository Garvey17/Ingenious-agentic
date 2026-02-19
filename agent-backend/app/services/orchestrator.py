"""
Orchestrator service — runs the full research workflow graph.
"""

from datetime import datetime
from app.graph.builder import research_graph
from app.graph.state import ResearchState
from app.services import session_store
from app.config.settings import settings
from app.config.logging import get_logger

logger = get_logger(__name__)


async def run_research(
    request_id: str,
    goal: str,
    user_id: str | None = None,
    depth: str = "standard",
) -> ResearchState:
    """
    Run the full research workflow for a given goal.

    1. Initialises the ResearchState
    2. Saves it to the session store (status=running)
    3. Invokes the compiled LangGraph
    4. Updates the session store with the final state
    5. Returns the final state

    Args:
        request_id: Unique ID for this research session
        goal:       The user's research question / topic
        user_id:    Optional user identifier
        depth:      "quick" | "standard" | "deep"

    Returns:
        Final ResearchState after the graph completes
    """
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
    }

    # Persist initial state so GET endpoint can return "in_progress" immediately
    await session_store.save_session(request_id, initial_state)

    logger.info(f"[orchestrator] Starting research '{goal[:60]}' (id={request_id})")

    try:
        final_state: ResearchState = await research_graph.ainvoke(initial_state)

        # Ensure status is set correctly
        if final_state.get("status") not in ("approved", "failed"):
            final_state["status"] = "approved"

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
