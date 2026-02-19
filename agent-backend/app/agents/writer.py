"""
Writer Agent — transforms analysis into a structured executive report.
"""

import json
from app.agents.base_agent import BaseAgent
from app.graph.state import ResearchState
from app.prompts.system_prompts import WRITER_SYSTEM_PROMPT
from app.config.logging import get_logger

logger = get_logger(__name__)


class WriterAgent(BaseAgent):
    """Writes a structured research report from insights and sources."""

    def __init__(self):
        super().__init__(agent_name="writer", system_prompt=WRITER_SYSTEM_PROMPT)

    def _build_prompt(self, input_data: dict) -> str:
        goal = input_data.get("goal", "")
        insights = input_data.get("insights", {})
        sources = input_data.get("sources", [])
        review = input_data.get("review", {})

        revision_note = ""
        if review:
            weaknesses = review.get("weaknesses", [])
            priority = review.get("revision_priority", "")
            if weaknesses or priority:
                revision_note = (
                    "\n\nPREVIOUS REVIEW FEEDBACK (address these in your revision):\n"
                    + (f"Priority: {priority}\n" if priority else "")
                    + (f"Weaknesses: {'; '.join(weaknesses)}\n" if weaknesses else "")
                )

        key_facts = insights.get("key_facts", [])
        insights_list = insights.get("insights", [])
        source_urls = [s.get("url", "") for s in sources[:15]]

        return (
            f"Research Goal: {goal}\n\n"
            f"Key Facts ({len(key_facts)} total):\n"
            + "\n".join(f"- {f.get('statement', f)}" for f in key_facts[:10])
            + f"\n\nKey Insights:\n"
            + "\n".join(f"- {i}" for i in insights_list[:8])
            + f"\n\nAvailable Sources: {len(sources)}"
            + revision_note
            + "\n\nWrite a comprehensive, well-structured research report."
        )


async def run_writer(state: ResearchState) -> ResearchState:
    """LangGraph node function for the Writer agent."""
    iteration = state.get("iteration", 0)
    logger.info(f"[writer] Writing report (iteration {iteration})")

    try:
        agent = WriterAgent()
        prompt = agent._build_prompt({
            "goal": state.get("goal", ""),
            "insights": state.get("insights", {}),
            "sources": state.get("sources", []),
            "review": state.get("review", {}),  # feedback from previous critic run
        })

        response = await agent.llm.generate(
            prompt=prompt,
            system_prompt=WRITER_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            temperature=0.6,
        )

        report = json.loads(response)

        # Ensure required fields exist
        report.setdefault("topic", state.get("goal", ""))
        report.setdefault("executive_summary", "")
        report.setdefault("sections", [])
        report.setdefault("key_findings", [])
        report.setdefault("recommendations", [])
        report.setdefault("sources_cited", [s.get("url", "") for s in state.get("sources", [])[:10]])

        logger.info(
            f"[writer] Report written — {len(report.get('sections', []))} sections, "
            f"{report.get('word_count', 0)} words"
        )

        return {**state, "report": report}

    except Exception as e:
        logger.error(f"[writer] Error: {e}")
        return {**state, "status": "failed", "error": f"Writer failed: {e}"}
