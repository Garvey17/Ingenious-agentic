"""
Agents module — defines the five pipeline steps.
Each run_X function is one node in the LangGraph workflow.
"""

import json

from app.config import get_logger, settings
from app.graph import ResearchState
from app.models import get_llm
from app.prompts import (
    PLANNER_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
)
from app.tools import WebSearchTool

logger = get_logger(__name__)


async def ask_for_json(system_prompt: str, prompt: str, temperature: float = 0.5) -> dict:
    """
    Sends a prompt to the LLM and parses the response as a JSON object.
    All five agents use this instead of keeping a separate Agent class.
    """
    response = await get_llm().ask(
        prompt=prompt,
        system_prompt=system_prompt,
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(response)


# ---------------------------------------------------------------------------
# Pipeline Steps (LangGraph nodes)
# ---------------------------------------------------------------------------

async def run_planner(state: ResearchState) -> ResearchState:
    """Step 1 — Breaks the research goal into a plan and a list of search queries."""
    from app.services import save_session

    state["current_step"] = "planner"
    await save_session(state["request_id"], state)

    goal = state.get("goal", "")
    depth = state.get("depth", "standard")
    past_research = state.get("past_research", [])

    logger.info(f"[planner] Planning: {goal}")

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

    if past_research:
        prompt += "## Relevant Past Research\n"
        for i, mem in enumerate(past_research[:3], 1):
            prompt += f"{i}. **{mem.get('topic', '')}**\n   {mem.get('summary', '')[:300]}...\n\n"
        prompt += "Identify GAPS and NEW ANGLES not already covered above.\n\n"

    prompt += "Create a research plan and generate specific search queries."

    try:
        data = await ask_for_json(PLANNER_SYSTEM_PROMPT, prompt, temperature=0.5)
        state["plan"] = data.get("sub_topics", data.get("plan", []))
        state["search_queries"] = data.get("search_queries", []) or [goal]
        return state
    except Exception as e:
        logger.error(f"[planner] Error: {e}")
        state["status"] = "failed"
        state["error"] = f"Planner failed: {e}"
        return state


async def run_researcher(state: ResearchState) -> ResearchState:
    """Step 2 — Runs the search queries and collects sources."""
    from app.services import save_session

    state["current_step"] = "researcher"
    await save_session(state["request_id"], state)

    queries = state.get("search_queries", []) or [state.get("goal", "")]
    depth = state.get("depth", "standard")
    results_per_query = {"quick": 3, "standard": 5, "deep": 8}.get(depth, 5)

    logger.info(f"[researcher] Running {len(queries)} queries")

    try:
        search_tool = WebSearchTool()
        all_sources, seen_urls = [], set()

        for q in queries:
            for r in await search_tool.run(q, max_results=results_per_query):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_sources.append(r)

        all_sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        state["sources"] = all_sources[:state.get("max_iterations", 3) * 5]
        return state
    except Exception as e:
        logger.error(f"[researcher] Error: {e}")
        state["status"] = "failed"
        state["error"] = f"Researcher failed: {e}"
        return state


async def run_analyst(state: ResearchState) -> ResearchState:
    """Step 3 — Reads through the sources and pulls out key facts and insights."""
    from app.services import save_session

    state["current_step"] = "analyst"
    await save_session(state["request_id"], state)

    goal = state.get("goal", "")
    sources = state.get("sources", [])
    logger.info(f"[analyst] Analysing {len(sources)} sources")

    sources_text = "\n\n".join(
        f"[Source {i+1}]\nTitle: {s.get('title', '')}\nURL: {s.get('url', '')}\nContent: {s.get('content', '')[:800]}"
        for i, s in enumerate(sources[:10])
    )
    prompt = (
        f"Research Goal: {goal}\n\n"
        f"Sources ({len(sources)} total):\n\n{sources_text}\n\n"
        "Extract key facts, identify contradictions, assess quality, and derive insights."
    )

    try:
        state["insights"] = await ask_for_json(ANALYST_SYSTEM_PROMPT, prompt, temperature=0.3)
        return state
    except Exception as e:
        logger.error(f"[analyst] Error: {e}")
        state["status"] = "failed"
        state["error"] = f"Analyst failed: {e}"
        return state


async def run_writer(state: ResearchState) -> ResearchState:
    """Step 4 — Writes the final research report from the analyst's insights."""
    from app.services import save_session

    state["current_step"] = "writer"
    await save_session(state["request_id"], state)

    goal = state.get("goal", "")
    insights = state.get("insights", {})
    sources = state.get("sources", [])
    review = state.get("review", {})

    logger.info("[writer] Writing report")

    revision_note = ""
    if review:
        weaknesses = review.get("weaknesses", [])
        priority = review.get("revision_priority", "")
        if weaknesses or priority:
            revision_note = (
                "\n\nPREVIOUS REVIEW FEEDBACK:\n"
                + (f"Priority: {priority}\n" if priority else "")
                + (f"Weaknesses: {'; '.join(weaknesses)}\n" if weaknesses else "")
            )

    key_facts = insights.get("key_facts", [])
    prompt = (
        f"Research Goal: {goal}\n\n"
        "Key Facts:\n" + "\n".join(f"- {f.get('statement', f)}" for f in key_facts[:10])
        + "\n\nKey Insights:\n" + "\n".join(f"- {i}" for i in insights.get("insights", [])[:8])
        + f"\n\nAvailable Sources: {len(sources)}"
        + revision_note
        + "\n\nWrite a comprehensive, well-structured research report."
    )

    try:
        report = await ask_for_json(WRITER_SYSTEM_PROMPT, prompt, temperature=0.6)
        report.setdefault("topic", goal)
        report.setdefault("executive_summary", "")
        report.setdefault("sections", [])
        report.setdefault("key_findings", [])
        report.setdefault("recommendations", [])
        report.setdefault("sources_cited", [s.get("url", "") for s in sources[:10]])
        state["report"] = report
        return state
    except Exception as e:
        logger.error(f"[writer] Error: {e}")
        state["status"] = "failed"
        state["error"] = f"Writer failed: {e}"
        return state


async def run_critic(state: ResearchState) -> ResearchState:
    """Step 5 — Reviews the report quality and decides whether to approve or revise."""
    from app.services import save_session

    state["current_step"] = "critic"
    await save_session(state["request_id"], state)

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", settings.max_iterations)
    goal = state.get("goal", "")
    report = state.get("report", {})

    logger.info(f"[critic] Review {iteration + 1}/{max_iterations}")

    sections_text = "\n\n".join(
        f"### {s.get('title', '')}\n{s.get('content', '')[:500]}"
        for s in report.get("sections", [])[:5]
    )
    prompt = (
        f"Research Goal: {goal}\n\n"
        f"Executive Summary:\n{report.get('executive_summary', '')[:600]}\n\n"
        f"Sections:\n{sections_text}\n\n"
        f"Revision: {iteration + 1} of {max_iterations}\n\n"
        "Evaluate this report and provide your decision."
    )

    try:
        review = await ask_for_json(CRITIC_SYSTEM_PROMPT, prompt, temperature=0.3)

        quality_score = review.get("overall_quality_score", 0.0)
        threshold = settings.critic_threshold / 10.0

        # Force approve if we've hit the max loops or quality is high enough
        if iteration >= max_iterations - 1 or quality_score >= threshold:
            review["decision"] = "approve"
        else:
            review["decision"] = review.get("decision", "revise")

        state["review"] = review
        if review["decision"] == "approve":
            state["status"] = "approved"
        else:
            state["status"] = "running"
            state["iteration"] = iteration + 1

        return state
    except Exception as e:
        logger.error(f"[critic] Error: {e}")
        # If the critic crashes, approve so we don't loop forever
        state["review"] = {"decision": "approve", "error": str(e)}
        state["status"] = "approved"
        return state
