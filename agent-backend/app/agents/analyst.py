"""
Analyst Agent — extracts key facts, detects contradictions, and derives insights.
"""

import json
from app.agents.base_agent import BaseAgent
from app.graph.state import ResearchState
from app.prompts.system_prompts import ANALYST_SYSTEM_PROMPT
from app.config.logging import get_logger

logger = get_logger(__name__)


class AnalystAgent(BaseAgent):
    """Analyses research sources and produces structured insights."""

    def __init__(self):
        super().__init__(agent_name="analyst", system_prompt=ANALYST_SYSTEM_PROMPT)

    def _build_prompt(self, input_data: dict) -> str:
        goal = input_data.get("goal", "")
        sources = input_data.get("sources", [])

        sources_text = "\n\n".join(
            f"[Source {i+1}]\nTitle: {s.get('title', 'N/A')}\n"
            f"URL: {s.get('url', 'N/A')}\n"
            f"Content: {s.get('content', '')[:800]}"
            for i, s in enumerate(sources[:10])  # limit to 10 sources in prompt
        )

        return (
            f"Research Goal: {goal}\n\n"
            f"Sources to Analyse ({len(sources)} total):\n\n"
            f"{sources_text}\n\n"
            "Extract key facts, identify contradictions, assess source quality, "
            "and derive the most important insights."
        )


async def run_analyst(state: ResearchState) -> ResearchState:
    """LangGraph node function for the Analyst agent."""
    logger.info(f"[analyst] Analysing {len(state.get('sources', []))} sources")

    try:
        agent = AnalystAgent()
        prompt = agent._build_prompt({
            "goal": state.get("goal", ""),
            "sources": state.get("sources", []),
        })

        response = await agent.llm.generate(
            prompt=prompt,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        insights = json.loads(response)
        logger.info(
            f"[analyst] Extracted {len(insights.get('key_facts', []))} facts, "
            f"{len(insights.get('insights', []))} insights"
        )

        return {**state, "insights": insights}

    except Exception as e:
        logger.error(f"[analyst] Error: {e}")
        return {**state, "status": "failed", "error": f"Analyst failed: {e}"}
