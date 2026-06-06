"""
Web Search and Summarize tools used by the Researcher agent.
"""

from typing import List

from app.config import settings, get_logger
from app.models import get_llm

logger = get_logger(__name__)


class WebSearchTool:
    """Searches the web and returns a list of relevant results."""

    async def run(self, query: str, max_results: int = 5) -> List[dict]:
        """Runs a web search. Uses Tavily if configured, otherwise returns mock results."""
        logger.info(f"[web_search] provider={settings.search_provider} query='{query}'")
        if settings.search_provider == "tavily" and settings.tavily_api_key:
            return await self._use_tavily(query, max_results)
        return self._use_mock_results(query, max_results)

    async def _use_tavily(self, query: str, max_results: int) -> List[dict]:
        """Calls the Tavily search API and formats the results."""
        try:
            from tavily import TavilyClient
            response = TavilyClient(api_key=settings.tavily_api_key).search(
                query=query, max_results=max_results, search_depth="advanced"
            )
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "relevance_score": r.get("score", 0.5),
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            logger.error(f"Tavily failed: {e}. Falling back to mock results.")
            return self._use_mock_results(query, max_results)

    def _use_mock_results(self, query: str, max_results: int) -> List[dict]:
        """Returns fake results so the workflow runs without a real search API key."""
        slug = query.replace(" ", "-").lower()
        return [
            {
                "title": f"Comprehensive Overview: {query}",
                "url": f"https://example.com/overview-{slug}",
                "content": f"Detailed information and background analysis on {query}. This guide explores key findings and current trends.",
                "relevance_score": 0.95,
            },
            {
                "title": f"Latest Research on {query}",
                "url": f"https://research.example.com/{slug}",
                "content": f"New findings and statistics regarding {query}. Studies show major growth and shifts in patterns.",
                "relevance_score": 0.88,
            },
            {
                "title": f"Challenges of {query}",
                "url": f"https://analysis.example.com/{slug}",
                "content": f"An objective look at the risks and limitations of {query} in modern contexts.",
                "relevance_score": 0.79,
            },
        ][:max_results]


class SummarizeTool:
    """Condenses a long piece of text into a short summary using the LLM."""

    async def run(self, text: str, max_words: int = 150, focus: str = "") -> str:
        """Sends the text to the LLM and returns a summary."""
        try:
            focus_clause = f" Focus specifically on: {focus}." if focus else ""
            prompt = (
                f"Summarize the following text in approximately {max_words} words.{focus_clause}\n\n"
                f"TEXT:\n{text}\n\nSUMMARY:"
            )
            summary = await get_llm().ask(
                prompt=prompt,
                system_prompt="You are an expert summarizer. Be clear, accurate, and concise.",
                temperature=0.3,
                max_tokens=max_words * 2,
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Summarize failed: {e}")
            return text[:300] + "..."
