"""
Planner Agent — breaks the research goal into a plan and search queries.
In Phase 3: also injects relevant past research from memory as context.
"""

import json
from app.agents.base_agent import BaseAgent
from app.graph.state import ResearchState
from app.prompts.system_prompts import PLANNER_SYSTEM_PROMPT
from app.config.logging import get_logger

logger = get_logger(__name__)


class PlannerAgent(BaseAgent):
    """Decomposes the research goal into sub-topics and search queries."""

    def __init__(self):
        super().__init__(agent_name="planner", system_prompt=PLANNER_SYSTEM_PROMPT)

    def _build_prompt(self, input_data: dict) -> str:
        goal = input_data.get("goal", "")
        depth = input_data.get("depth", "standard")
        past_research: list[dict] = input_data.get("past_research", [])

        depth_instructions = {
            "quick": "Generate 2-3 focused search queries. Keep the plan concise.",
            "standard": "Generate 4-6 search queries covering different angles.",
            "deep": "Generate 8-10 comprehensive search queries for thorough coverage.",
        }

        prompt = (
            f"Research Goal: {goal}\n\n"
            f"Depth: {depth}\n"
            f"Instructions: {depth_instructions.get(depth, depth_instructions['standard'])}\n\n"
        )

        # Phase 3: inject past research context when available
        if past_research:
            prompt += "## Relevant Past Research\n"
            prompt += (
                "The following previous research sessions are relevant. "
                "Use them to avoid redundancy and focus on unexplored angles:\n\n"
            )
            for i, mem in enumerate(past_research[:3], 1):  # cap at top 3
                topic = mem.get("topic", "Unknown topic")
                summary = mem.get("summary", "No summary available")
                score = mem.get("score", 0)
                prompt += (
                    f"{i}. **{topic}** (relevance: {score:.2f})\n"
                    f"   {summary[:300]}...\n\n"
                )
            prompt += (
                "Identify GAPS and NEW ANGLES not already covered above.\n\n"
            )

        prompt += "Create a research plan and generate specific search queries."
        return prompt


async def run_planner(state: ResearchState) -> ResearchState:
    """LangGraph node function for the Planner agent."""
    goal = state.get("goal", "")
    past_research = state.get("past_research", [])
    logger.info(
        f"[planner] Starting for goal: {goal[:80]} "
        f"(past_research: {len(past_research)} memories)"
    )

    try:
        agent = PlannerAgent()
        prompt = agent._build_prompt({
            "goal": goal,
            "depth": state.get("depth", "standard"),
            "past_research": past_research,
        })

        response = await agent.llm.generate(
            prompt=prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            temperature=0.5,
        )

        data = json.loads(response)
        plan = data.get("sub_topics", data.get("plan", []))
        search_queries = data.get("search_queries", [])

        if not search_queries:
            # Fallback: use the goal itself as a query
            search_queries = [goal]

        logger.info(f"[planner] Generated {len(search_queries)} queries, {len(plan)} plan steps")

        return {
            **state,
            "plan": plan,
            "search_queries": search_queries,
            "status": "running",
        }

    except Exception as e:
        logger.error(f"[planner] Error: {e}")
        return {**state, "status": "failed", "error": f"Planner failed: {e}"}
