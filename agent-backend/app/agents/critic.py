"""
Critic Agent — evaluates report quality and decides to approve or request revision.
"""

import json
from app.agents.base_agent import BaseAgent
from app.graph.state import ResearchState
from app.prompts.system_prompts import CRITIC_SYSTEM_PROMPT
from app.config.settings import settings
from app.config.logging import get_logger

logger = get_logger(__name__)


class CriticAgent(BaseAgent):
    """Evaluates the report and decides whether to approve or request revision."""

    def __init__(self):
        super().__init__(agent_name="critic", system_prompt=CRITIC_SYSTEM_PROMPT)

    def _build_prompt(self, input_data: dict) -> str:
        goal = input_data.get("goal", "")
        report = input_data.get("report", {})
        iteration = input_data.get("iteration", 0)
        max_iterations = input_data.get("max_iterations", 3)

        sections_text = "\n\n".join(
            f"### {s.get('title', '')}\n{s.get('content', '')[:500]}"
            for s in report.get("sections", [])[:5]
        )

        force_approve_note = ""
        if iteration >= max_iterations - 1:
            force_approve_note = (
                "\n\nNOTE: This is the final revision allowed. "
                "Please approve the report even if imperfect, "
                "but provide constructive feedback for future improvement."
            )

        return (
            f"Research Goal: {goal}\n\n"
            f"REPORT TO EVALUATE:\n\n"
            f"Executive Summary:\n{report.get('executive_summary', '')[:600]}\n\n"
            f"Sections:\n{sections_text}\n\n"
            f"Key Findings: {len(report.get('key_findings', []))} items\n"
            f"Recommendations: {len(report.get('recommendations', []))} items\n"
            f"Sources Cited: {len(report.get('sources_cited', []))}\n"
            f"Revision: {iteration + 1} of {max_iterations}"
            + force_approve_note
            + "\n\nEvaluate this report thoroughly and provide your decision."
        )


async def run_critic(state: ResearchState) -> ResearchState:
    """LangGraph node function for the Critic agent."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", settings.max_iterations)
    logger.info(f"[critic] Evaluating report (iteration {iteration + 1}/{max_iterations})")

    try:
        agent = CriticAgent()
        prompt = agent._build_prompt({
            "goal": state.get("goal", ""),
            "report": state.get("report", {}),
            "iteration": iteration,
            "max_iterations": max_iterations,
        })

        response = await agent.llm.generate(
            prompt=prompt,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        review = json.loads(response)

        quality_score = review.get("overall_quality_score", 0.0)
        decision = review.get("decision", "revise")

        # Threshold from settings (default 7.0 on 0-10 scale → 0.7 on 0-1 scale)
        threshold = settings.critic_threshold / 10.0

        # Force approve if we've hit the iteration limit
        if iteration >= max_iterations - 1:
            decision = "approve"
            logger.info(f"[critic] Max iterations reached — forcing approval")

        # Override decision based on score
        if quality_score >= threshold:
            decision = "approve"
        elif decision == "approve" and quality_score < threshold:
            decision = "revise"

        review["decision"] = decision

        logger.info(
            f"[critic] Score={quality_score:.2f} threshold={threshold:.2f} "
            f"decision={decision}"
        )

        new_status = "approved" if decision == "approve" else "running"
        new_iteration = iteration if decision == "approve" else iteration + 1

        return {
            **state,
            "review": review,
            "iteration": new_iteration,
            "status": new_status,
        }

    except Exception as e:
        logger.error(f"[critic] Error: {e}")
        # On critic failure, approve what we have rather than looping forever
        return {
            **state,
            "review": {"decision": "approve", "error": str(e)},
            "status": "approved",
        }
