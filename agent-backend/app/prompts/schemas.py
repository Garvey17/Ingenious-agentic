"""
Pydantic schemas for structured agent outputs.
These schemas enforce type safety and validation across the agent pipeline.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator


class ResearchSource(BaseModel):
    """A single research source from web search."""
    
    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    content: str = Field(..., description="Relevant content/snippet from the source")
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1)"
    )
    
    @validator("url")
    def validate_url(cls, v):
        """Ensure URL is not empty."""
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()


class ResearchOutput(BaseModel):
    """Output from the Researcher agent."""
    
    topic: str = Field(..., description="Research topic")
    sources: List[ResearchSource] = Field(
        default_factory=list,
        description="List of research sources found"
    )
    summary: str = Field(..., description="Brief summary of findings")
    search_queries_used: List[str] = Field(
        default_factory=list,
        description="Search queries that were executed"
    )
    total_sources_found: int = Field(default=0, description="Total sources found")
    
    @validator("sources")
    def validate_sources(cls, v):
        """Ensure at least one source is found."""
        if not v:
            raise ValueError("At least one source must be found")
        return v


class Fact(BaseModel):
    """A single extracted fact."""
    
    statement: str = Field(..., description="The factual statement")
    source_url: str = Field(..., description="Source URL for this fact")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this fact (0-1)"
    )


class Contradiction(BaseModel):
    """A detected contradiction between sources."""
    
    fact_a: str = Field(..., description="First contradicting statement")
    fact_b: str = Field(..., description="Second contradicting statement")
    source_a: str = Field(..., description="Source of first statement")
    source_b: str = Field(..., description="Source of second statement")
    explanation: str = Field(..., description="Explanation of the contradiction")


class AnalysisOutput(BaseModel):
    """Output from the Analyst agent."""
    
    topic: str = Field(..., description="Research topic")
    key_facts: List[Fact] = Field(
        default_factory=list,
        description="Key facts extracted from sources"
    )
    contradictions: List[Contradiction] = Field(
        default_factory=list,
        description="Contradictions found between sources"
    )
    source_quality_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Quality scores for each source URL"
    )
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the analysis"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Key insights derived from analysis"
    )


class ReportSection(BaseModel):
    """A section in the final report."""
    
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")


class ReportOutput(BaseModel):
    """Output from the Writer agent."""
    
    topic: str = Field(..., description="Research topic")
    executive_summary: str = Field(..., description="Executive summary")
    sections: List[ReportSection] = Field(
        default_factory=list,
        description="Report sections"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations based on research"
    )
    sources_cited: List[str] = Field(
        default_factory=list,
        description="List of source URLs cited"
    )
    word_count: int = Field(default=0, description="Total word count")


class CriticFeedback(BaseModel):
    """Feedback on a specific aspect of the report."""
    
    aspect: str = Field(..., description="Aspect being critiqued")
    score: float = Field(..., ge=0.0, le=1.0, description="Score for this aspect")
    feedback: str = Field(..., description="Detailed feedback")
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )


class CriticOutput(BaseModel):
    """Output from the Critic agent."""
    
    overall_quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall quality score (0-1)"
    )
    decision: Literal["approve", "revise"] = Field(
        ...,
        description="Decision to approve or request revision"
    )
    feedback_items: List[CriticFeedback] = Field(
        default_factory=list,
        description="Detailed feedback on different aspects"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Strengths of the report"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Weaknesses of the report"
    )
    revision_priority: Optional[str] = Field(
        None,
        description="What should be prioritized in revision"
    )
    
    @validator("decision")
    def validate_decision(cls, v, values):
        """Ensure decision aligns with quality score."""
        if "overall_quality_score" in values:
            score = values["overall_quality_score"]
            # Auto-approve if score is high enough
            if score >= 0.7 and v == "revise":
                return "approve"
            # Auto-revise if score is too low
            if score < 0.7 and v == "approve":
                return "revise"
        return v


class ResearchRequest(BaseModel):
    """Request to start a research workflow."""
    
    topic: str = Field(..., min_length=3, description="Research topic")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    max_sources: Optional[int] = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of sources to gather"
    )
    depth: Literal["quick", "standard", "deep"] = Field(
        default="standard",
        description="Research depth level"
    )


class ResearchResponse(BaseModel):
    """Response from a research workflow."""
    
    request_id: str = Field(..., description="Unique request ID")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        ...,
        description="Current status"
    )
    topic: str = Field(..., description="Research topic")
    report: Optional[ReportOutput] = Field(None, description="Final report if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    iterations: int = Field(default=0, description="Number of iterations performed")
    created_at: str = Field(..., description="Creation timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
