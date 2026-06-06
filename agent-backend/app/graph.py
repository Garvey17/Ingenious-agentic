"""
LangGraph workflow definition for Deep Research Desk.
Defines the shared state and wires the 5 agents together in a pipeline.
"""

from typing import Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from app.config import get_logger

logger = get_logger(__name__)


class ResearchState(TypedDict, total=False):
    """
    The shared data bag that flows between all pipeline steps.
    Every agent reads from and writes back to this same dictionary.
    """
    goal: str
    request_id: str
    user_id: Optional[str]
    max_iterations: int
    depth: str
    plan: list[str]
    search_queries: list[str]
    sources: list[dict[str, Any]]
    insights: dict[str, Any]
    report: dict[str, Any]
    review: dict[str, Any]
    iteration: int
    status: str
    current_step: str        # which agent is running right now (used by the frontend)
    error: Optional[str]
    past_research: list[dict[str, Any]]


def decide_next_step(state: ResearchState) -> str:
    """
    Called after the Critic runs.
    Returns 'end' if the report was approved, or 'researcher' to loop back for revisions.
    """
    status = state.get("status", "running")

    if status == "approved":
        logger.info("[graph] Report approved — finishing workflow")
        return "end"

    if status == "failed":
        logger.warning("[graph] Workflow failed — finishing workflow")
        return "end"

    logger.info(f"[graph] Critic wants revisions (iteration {state.get('iteration', 0)})")
    return "researcher"


def create_workflow():
    """Wires all 5 agents together into a LangGraph pipeline and returns it."""
    from app.agents import (
        run_planner,
        run_researcher,
        run_analyst,
        run_writer,
        run_critic,
    )

    graph = StateGraph(ResearchState)

    # Register each agent as a node
    graph.add_node("planner", run_planner)
    graph.add_node("researcher", run_researcher)
    graph.add_node("analyst", run_analyst)
    graph.add_node("writer", run_writer)
    graph.add_node("critic", run_critic)

    # Connect them in order: planner → researcher → analyst → writer → critic
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "critic")

    # After the critic, either end or loop back to researcher for a revision
    graph.add_conditional_edges(
        "critic",
        decide_next_step,
        {
            "researcher": "researcher",
            "end": END,
        },
    )

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


# Single compiled workflow instance shared across all requests
research_graph = create_workflow()
