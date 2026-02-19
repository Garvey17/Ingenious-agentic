"""
LangGraph workflow builder for the Deep Research Desk.

Graph topology:
  START → planner → researcher → analyst → writer → critic
                        ↑                              |
                        |_____ (revise & iter < max) ──┘
                                                       ↓ (approve)
                                                      END
"""

from langgraph.graph import StateGraph, END

from app.graph.state import ResearchState
from app.agents.planner import run_planner
from app.agents.researcher import run_researcher
from app.agents.analyst import run_analyst
from app.agents.writer import run_writer
from app.agents.critic import run_critic
from app.config.logging import get_logger

logger = get_logger(__name__)


def _route_after_critic(state: ResearchState) -> str:
    """
    Conditional edge: decide what happens after the Critic runs.

    Returns:
        "researcher" — loop back for another research/write cycle
        "end"        — report approved, finish
        "end"        — pipeline failed
    """
    status = state.get("status", "running")

    if status == "approved":
        logger.info("[graph] Critic approved — finishing")
        return "end"

    if status == "failed":
        logger.warning("[graph] Pipeline failed — finishing")
        return "end"

    # Still running → revise
    logger.info(f"[graph] Critic requested revision (iteration {state.get('iteration', 0)})")
    return "researcher"


def build_research_graph():
    """
    Compile and return the LangGraph StateGraph for the research workflow.

    Returns:
        A compiled LangGraph app ready to be invoked with a ResearchState dict.
    """
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("planner", run_planner)
    graph.add_node("researcher", run_researcher)
    graph.add_node("analyst", run_analyst)
    graph.add_node("writer", run_writer)
    graph.add_node("critic", run_critic)

    # Linear edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "critic")

    # Conditional edge from critic
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "researcher": "researcher",  # revision loop
            "end": END,
        },
    )

    compiled = graph.compile()
    logger.info("[graph] Research workflow compiled successfully")
    return compiled


# Singleton compiled graph — built once at import time
research_graph = build_research_graph()
