"""
Shared LangGraph state for the Deep Research Desk workflow.
All agents read from and write to this TypedDict.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """
    Shared state that flows through the entire LangGraph pipeline.

    Fields are populated progressively as each agent runs:
      Planner    → plan, search_queries
      Researcher → sources
      Analyst    → insights
      Writer     → report
      Critic     → review, status
    """

    # --- Input ---
    goal: str                          # The user's research question
    request_id: str                    # Unique ID for this research session
    user_id: Optional[str]             # Optional user identifier
    max_iterations: int                # Maximum revision loops allowed
    depth: str                         # "quick" | "standard" | "deep"

    # --- Planner output ---
    plan: list[str]                    # High-level research plan steps
    search_queries: list[str]          # Specific search queries to execute

    # --- Researcher output ---
    sources: list[dict[str, Any]]      # Raw search results (title, url, content, score)

    # --- Analyst output ---
    insights: dict[str, Any]           # Structured analysis (key_facts, contradictions, etc.)

    # --- Writer output ---
    report: dict[str, Any]             # Structured report (sections, summary, etc.)

    # --- Critic output ---
    review: dict[str, Any]             # Quality review (score, decision, feedback)

    # --- Orchestration ---
    iteration: int                     # Current revision loop count (starts at 0)
    status: str                        # "running" | "approved" | "failed"
    error: Optional[str]               # Error message if something went wrong

    # --- Memory (Phase 3) ---
    past_research: list[dict[str, Any]]  # Relevant past research injected before planning
