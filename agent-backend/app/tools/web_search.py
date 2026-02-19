"""
Web search tool for the Deep Research Desk.
Supports Tavily API (production) and a placeholder mode (development/testing).
"""

from typing import List
from pydantic import Field

from app.tools.base import BaseTool, ToolInput, ToolOutput
from app.config.settings import settings
from app.config.logging import get_logger

logger = get_logger(__name__)


class WebSearchInput(ToolInput):
    """Input for the web search tool."""
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results to return")


class WebSearchOutput(ToolOutput):
    """Output from the web search tool."""
    results: List[dict] = Field(default_factory=list, description="List of search results")
    query: str = Field(default="", description="The query that was searched")


class WebSearchTool(BaseTool):
    """
    Web search tool that retrieves relevant web pages for a query.

    Uses Tavily API in production, falls back to placeholder data
    when SEARCH_PROVIDER=placeholder (no API key needed).
    """

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information on a given topic"
        )

    async def execute(self, input_data: WebSearchInput) -> WebSearchOutput:
        """
        Execute a web search.

        Args:
            input_data: Search query and parameters

        Returns:
            WebSearchOutput with list of result dicts containing
            title, url, content, and relevance_score
        """
        provider = settings.search_provider
        logger.info(f"[web_search] provider={provider} query='{input_data.query}'")

        if provider == "tavily":
            return await self._search_tavily(input_data)
        elif provider == "placeholder":
            return self._search_placeholder(input_data)
        else:
            logger.warning(f"Unknown search provider '{provider}', using placeholder")
            return self._search_placeholder(input_data)

    # ------------------------------------------------------------------
    # Tavily
    # ------------------------------------------------------------------
    async def _search_tavily(self, input_data: WebSearchInput) -> WebSearchOutput:
        """Call the Tavily search API."""
        try:
            from tavily import TavilyClient  # type: ignore

            api_key = settings.tavily_api_key
            if not api_key:
                logger.warning("TAVILY_API_KEY not set, falling back to placeholder")
                return self._search_placeholder(input_data)

            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=input_data.query,
                max_results=input_data.max_results,
                search_depth="advanced",
                include_answer=False,
            )

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "relevance_score": r.get("score", 0.5),
                }
                for r in response.get("results", [])
            ]

            logger.info(f"[web_search] Tavily returned {len(results)} results")
            return WebSearchOutput(success=True, data=results, results=results, query=input_data.query)

        except Exception as e:
            logger.error(f"[web_search] Tavily error: {e}")
            return WebSearchOutput(success=False, error=str(e), results=[], query=input_data.query)

    # ------------------------------------------------------------------
    # Placeholder (no API key required)
    # ------------------------------------------------------------------
    def _search_placeholder(self, input_data: WebSearchInput) -> WebSearchOutput:
        """Return realistic mock search results for development/testing."""
        query = input_data.query
        mock_results = [
            {
                "title": f"Comprehensive Overview: {query}",
                "url": f"https://example.com/overview-{query.replace(' ', '-').lower()}",
                "content": (
                    f"This article provides a comprehensive overview of {query}. "
                    f"Key aspects include historical context, current developments, "
                    f"and future implications. Experts in the field have noted significant "
                    f"advancements in recent years, with multiple studies confirming "
                    f"the growing importance of this topic across various industries."
                ),
                "relevance_score": 0.95,
            },
            {
                "title": f"Latest Research on {query}",
                "url": f"https://research.example.com/{query.replace(' ', '-').lower()}",
                "content": (
                    f"Recent research on {query} reveals several important findings. "
                    f"Studies conducted in 2024-2025 show a 40% increase in adoption rates. "
                    f"Leading institutions have published peer-reviewed papers highlighting "
                    f"both opportunities and challenges. The consensus among researchers "
                    f"points to transformative potential in the next 5 years."
                ),
                "relevance_score": 0.90,
            },
            {
                "title": f"Industry Impact of {query}",
                "url": f"https://industry.example.com/{query.replace(' ', '-').lower()}",
                "content": (
                    f"The industry impact of {query} cannot be overstated. "
                    f"Major corporations have invested billions in related technologies. "
                    f"Market analysis indicates a compound annual growth rate of 25%. "
                    f"Key players include both established enterprises and innovative startups "
                    f"disrupting traditional business models."
                ),
                "relevance_score": 0.85,
            },
            {
                "title": f"Challenges and Risks: {query}",
                "url": f"https://analysis.example.com/risks-{query.replace(' ', '-').lower()}",
                "content": (
                    f"While {query} presents many opportunities, significant challenges remain. "
                    f"Regulatory frameworks are still evolving, creating uncertainty for stakeholders. "
                    f"Technical limitations, ethical concerns, and implementation costs are "
                    f"frequently cited as barriers. Risk mitigation strategies are being developed "
                    f"by industry consortiums and government bodies."
                ),
                "relevance_score": 0.80,
            },
            {
                "title": f"Future Outlook: {query}",
                "url": f"https://forecast.example.com/{query.replace(' ', '-').lower()}",
                "content": (
                    f"The future of {query} looks promising according to leading analysts. "
                    f"Projections for 2025-2030 suggest widespread adoption across sectors. "
                    f"Emerging technologies will further accelerate development. "
                    f"Policy makers are increasingly recognizing the strategic importance "
                    f"and are developing supportive regulatory environments."
                ),
                "relevance_score": 0.75,
            },
        ]

        results = mock_results[: input_data.max_results]
        logger.info(f"[web_search] Placeholder returned {len(results)} mock results")
        return WebSearchOutput(success=True, data=results, results=results, query=query)
