"""
Researcher Agent — executes web searches and gathers sources.
"""

import asyncio
from app.agents.base_agent import BaseAgent
from app.graph.state import ResearchState
from app.prompts.system_prompts import RESEARCHER_SYSTEM_PROMPT
from app.tools.web_search import WebSearchTool, WebSearchInput
import json
from app.config.logging import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class ResearcherAgent(BaseAgent):
    """Executes search queries and collects research sources."""

    def __init__(self):
        super().__init__(agent_name="researcher", system_prompt=RESEARCHER_SYSTEM_PROMPT)
        self.search_tool = WebSearchTool()

    def _build_prompt(self, input_data: dict) -> str:
        return f"Research topic: {input_data.get('goal', '')}"


async def run_researcher(state: ResearchState) -> ResearchState:
    """LangGraph node function for the Researcher agent."""
    queries = state.get("search_queries", [])
    goal = state.get("goal", "")
    logger.info(f"[researcher] Running {len(queries)} search queries")

    if not queries:
        queries = [goal]

    # Determine results per query based on depth
    depth = state.get("depth", "standard")
    results_per_query = {"quick": 3, "standard": 5, "deep": 8}.get(depth, 5)

    try:
        # Run all searches concurrently
        if settings.enable_mcp:
            from app.mcp.client import mcp_client
            logger.debug("[researcher] Routing searches via MCP")
            tasks = [
                mcp_client.call_tool("web_search", {"query": q, "max_results": results_per_query})
                for q in queries
            ]
        else:
            logger.debug("[researcher] Routing searches via local tool")
            tasks = [
                WebSearchTool().execute(WebSearchInput(query=q, max_results=results_per_query))
                for q in queries
            ]
            
        search_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        all_sources: list[dict] = []
        seen_urls: set[str] = set()

        for output in search_outputs:
            if isinstance(output, Exception):
                logger.warning(f"[researcher] Search error: {output}")
                continue
            
            # Extract results list depending on whether it came from MCP (JSON string) or local (Pydantic model)
            if settings.enable_mcp and isinstance(output, str):
                try:
                    parsed = json.loads(output)
                    results = parsed.get("results", [])
                except Exception as e:
                    logger.warning(f"[researcher] Failed to parse MCP output: {e}")
                    results = []
            else:
                results = output.results
                
            for result in results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_sources.append(result)

        # Sort by relevance score descending
        all_sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Cap at max_sources from settings
        max_sources = settings.max_iterations * 5  # sensible default
        all_sources = all_sources[:max_sources]

        logger.info(f"[researcher] Collected {len(all_sources)} unique sources")

        return {**state, "sources": all_sources}

    except Exception as e:
        logger.error(f"[researcher] Error: {e}")
        return {**state, "status": "failed", "error": f"Researcher failed: {e}"}
