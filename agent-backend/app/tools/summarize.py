"""
Summarize tool — uses the configured LLM to condense long text.
"""

from pydantic import Field

from app.tools.base import BaseTool, ToolInput, ToolOutput
from app.config.logging import get_logger

logger = get_logger(__name__)


class SummarizeInput(ToolInput):
    """Input for the summarize tool."""
    text: str = Field(..., description="Text to summarize")
    max_words: int = Field(default=150, ge=20, le=500, description="Target word count for summary")
    focus: str = Field(default="", description="Optional focus area or angle for the summary")


class SummarizeOutput(ToolOutput):
    """Output from the summarize tool."""
    summary: str = Field(default="", description="Condensed summary of the input text")
    word_count: int = Field(default=0, description="Actual word count of the summary")


class SummarizeTool(BaseTool):
    """
    LLM-based text summarization tool.
    Condenses long content into concise summaries.
    """

    def __init__(self):
        super().__init__(
            name="summarize",
            description="Summarize long text into a concise, focused summary"
        )

    async def execute(self, input_data: SummarizeInput) -> SummarizeOutput:
        """
        Summarize the given text using the LLM.

        Args:
            input_data: Text and summarization parameters

        Returns:
            SummarizeOutput with the condensed summary
        """
        try:
            from app.models.llm import get_llm

            llm = get_llm()

            focus_clause = f" Focus specifically on: {input_data.focus}." if input_data.focus else ""
            system_prompt = (
                "You are an expert summarizer. Produce clear, accurate, and concise summaries "
                "that preserve the most important facts and insights from the source material."
            )
            user_prompt = (
                f"Summarize the following text in approximately {input_data.max_words} words.{focus_clause}\n\n"
                f"TEXT:\n{input_data.text}\n\n"
                f"SUMMARY:"
            )

            logger.info(f"[summarize] Summarizing {len(input_data.text)} chars → ~{input_data.max_words} words")

            summary = await llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=input_data.max_words * 2,  # tokens ≈ 1.5× words
            )

            summary = summary.strip()
            word_count = len(summary.split())

            logger.info(f"[summarize] Done — {word_count} words")
            return SummarizeOutput(
                success=True,
                data={"summary": summary},
                summary=summary,
                word_count=word_count,
            )

        except Exception as e:
            logger.error(f"[summarize] Error: {e}")
            return SummarizeOutput(success=False, error=str(e), summary="", word_count=0)
