"""Prompts module exports."""

from app.prompts.system_prompts import (
    RESEARCHER_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)

from app.prompts.schemas import (
    ResearchSource,
    ResearchOutput,
    Fact,
    Contradiction,
    AnalysisOutput,
    ReportSection,
    ReportOutput,
    CriticFeedback,
    CriticOutput,
    ResearchRequest,
    ResearchResponse,
)

__all__ = [
    # System prompts
    "RESEARCHER_SYSTEM_PROMPT",
    "ANALYST_SYSTEM_PROMPT",
    "WRITER_SYSTEM_PROMPT",
    "CRITIC_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    # Schemas
    "ResearchSource",
    "ResearchOutput",
    "Fact",
    "Contradiction",
    "AnalysisOutput",
    "ReportSection",
    "ReportOutput",
    "CriticFeedback",
    "CriticOutput",
    "ResearchRequest",
    "ResearchResponse",
]
