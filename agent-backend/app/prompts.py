"""
Agent system prompts and API request/response schemas.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ===========================================================================
# API Request / Response Schemas  (used by main.py)
# ===========================================================================

class ResearchRequest(BaseModel):
    """Body of the POST /api/research/ request."""
    topic: str = Field(..., min_length=3)
    user_id: Optional[str] = None
    depth: Literal["quick", "standard", "deep"] = "standard"


class ReportSection(BaseModel):
    title: str
    content: str


class ReportOutput(BaseModel):
    topic: str
    executive_summary: str
    sections: List[ReportSection] = []
    key_findings: List[str] = []
    recommendations: List[str] = []
    sources_cited: List[str] = []
    word_count: int = 0


class ResearchResponse(BaseModel):
    """Response returned by GET /api/research/{id}."""
    request_id: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    topic: str
    report: Optional[ReportOutput] = None
    error: Optional[str] = None
    iterations: int = 0
    created_at: str
    completed_at: Optional[str] = None


# ===========================================================================
# Agent System Prompts
# ===========================================================================

PLANNER_SYSTEM_PROMPT = """You are a Planner Agent that breaks down research topics into actionable steps.

Return a JSON object with this structure:
{
  "sub_topics": ["sub-topic 1", "sub-topic 2"],
  "search_queries": ["query 1", "query 2", "query 3"],
  "research_approach": "brief description of the approach"
}
"""

RESEARCHER_SYSTEM_PROMPT = """You are a Research Agent that gathers information from the web.

Return a JSON object with this structure:
{
  "topic": "research topic",
  "summary": "brief summary of all findings",
  "search_queries_used": ["query 1", "query 2"]
}
"""

ANALYST_SYSTEM_PROMPT = """You are an Analyst Agent that extracts insights from research data.

Return a JSON object with this structure:
{
  "key_facts": [{"statement": "factual statement", "source_url": "URL", "confidence": 0.9}],
  "contradictions": [{"fact_a": "...", "fact_b": "...", "explanation": "..."}],
  "overall_confidence": 0.88,
  "insights": ["insight 1", "insight 2"]
}
"""

WRITER_SYSTEM_PROMPT = """You are a Writer Agent that creates executive-style research reports.

Return a JSON object with this structure:
{
  "topic": "research topic",
  "executive_summary": "2-3 paragraph summary",
  "sections": [{"title": "Section Title", "content": "section content"}],
  "key_findings": ["finding 1", "finding 2"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "sources_cited": ["url1", "url2"],
  "word_count": 1500
}
"""

CRITIC_SYSTEM_PROMPT = """You are a Critic Agent that evaluates research report quality.

Score the report 0.0 to 1.0 and decide whether to approve or request a revision.
- 0.7 and above → approve
- Below 0.7 → revise

Return a JSON object with this structure:
{
  "overall_quality_score": 0.85,
  "decision": "approve",
  "strengths": ["strength 1"],
  "weaknesses": ["weakness 1"],
  "revision_priority": "what to focus on if revision needed"
}
"""
