"""
Agents module — defines the five pipeline steps.
Each run_X function is one node in the LangGraph workflow.
"""

import json
import re

from app.config import get_logger, settings
from app.graph import ResearchState
from app.models import get_llm
from app.prompts import (
    PLANNER_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
)
from app.tools import WebSearchTool, wayback_lookup

logger = get_logger(__name__)


def _historical_page_request(goal: str) -> tuple[str, str] | None:
    """Returns the page URL and requested date for historical page questions."""
    date_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", goal)
    if not date_match:
        return None

    lower_goal = goal.lower()
    url_match = re.search(r"https?://[^\s<>]+", goal, re.IGNORECASE)
    domain_match = re.search(
        r"(?<![@\w-])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s]*)?",
        goal,
        re.IGNORECASE,
    )
    page_context = any(
        phrase in lower_goal
        for phrase in ("hacker news", "web page", "webpage", "website", "homepage", "web site")
    ) or bool(url_match or domain_match)
    if not page_context:
        return None

    if "hacker news" in lower_goal:
        url = "https://news.ycombinator.com/"
    elif url_match:
        url = url_match.group(0).rstrip(".,?!;:)")
    elif domain_match:
        url = domain_match.group(0).rstrip(".,?!;:)")
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
    else:
        url = ""

    return url, "".join(date_match.groups())


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

    if _historical_page_request(goal):
        state["plan"] = ["Retrieve the Wayback snapshot for the requested page and date."]
        state["search_queries"] = []
        return state

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
        historical_request = _historical_page_request(state.get("goal", ""))
        if historical_request:
            url, date = historical_request
            if not url:
                result = {
                    "found": False,
                    "snapshot_url": "",
                    "snapshot_timestamp": "",
                    "top_stories": [],
                    "message": "Could not identify the page URL, so the historical page could not be verified.",
                }
                logger.info("[wayback_lookup] skipped: no page URL could be identified")
            else:
                result = await wayback_lookup(url, date)

            state["wayback_result"] = result
            if not result["found"]:
                # Do not let current web search results substitute for missing historical evidence.
                state["sources"] = []
                return state

            titles = result.get("top_stories", [])
            content = (
                f"Wayback snapshot timestamp: {result['snapshot_timestamp']}\n"
                + ("Top stories:\n" + "\n".join(f"- {title}" for title in titles) if titles else "")
            )
            state["sources"] = [{
                "title": f"Wayback snapshot ({result['snapshot_timestamp']})",
                "url": result["snapshot_url"],
                "content": content,
                "relevance_score": 1.0,
            }]
            return state

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

    wayback_result = state.get("wayback_result")
    if wayback_result:
        state["insights"] = {
            "key_facts": [
                {
                    "statement": title,
                    "source_url": wayback_result.get("snapshot_url", ""),
                    "confidence": 1.0,
                }
                for title in wayback_result.get("top_stories", [])
            ],
            "insights": [],
        }
        return state

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
    wayback_result = state.get("wayback_result")

    logger.info("[writer] Writing report")

    if wayback_result and not wayback_result.get("found"):
        message = "I could not verify what the page said on the requested date."
        detail = wayback_result.get("message", "")
        state["report"] = {
            "topic": goal,
            "executive_summary": message,
            "sections": [{"title": "Verification", "content": f"{message} {detail}".strip()}],
            "key_findings": [],
            "recommendations": [],
            "sources_cited": [],
            "word_count": len(message.split()),
        }
        return state

    if wayback_result:
        timestamp = wayback_result.get("snapshot_timestamp", "")
        snapshot_url = wayback_result.get("snapshot_url", "")
        titles = wayback_result.get("top_stories", [])
        requested = _historical_page_request(goal)
        requested_date = requested[1] if requested else ""
        snapshot_date = timestamp[:8]

        if titles:
            first_story = titles[0]
            requested_label = (
                f"{requested_date[:4]}-{requested_date[4:6]}-{requested_date[6:8]}"
                if requested_date else "the requested date"
            )
            snapshot_label = (
                f"{snapshot_date[:4]}-{snapshot_date[4:6]}-{snapshot_date[6:8]}"
                if len(snapshot_date) == 8 else timestamp
            )
            if requested_date == snapshot_date:
                summary = f'The first Hacker News story in the snapshot is "{first_story}".'
            else:
                summary = (
                    f"The closest snapshot for {requested_label} is dated {snapshot_label}. "
                    f'Its first Hacker News story is "{first_story}", but it does not verify '
                    f"the page state on {requested_label}."
                )
            sections = [
                {"title": "Stories in the snapshot", "content": "\n".join(titles)},
                {
                    "title": "Snapshot source",
                    "content": f"Snapshot timestamp: {timestamp}\nSnapshot URL: {snapshot_url}",
                },
            ]
            findings = [f"First story in the archived snapshot: {first_story}."]
        else:
            summary = "I could not verify the page contents from the available snapshot."
            sections = [{
                "title": "Snapshot source",
                "content": f"Snapshot timestamp: {timestamp}\nSnapshot URL: {snapshot_url}",
            }]
            findings = []

        state["report"] = {
            "topic": goal,
            "executive_summary": summary,
            "sections": sections,
            "key_findings": findings,
            "recommendations": [],
            "sources_cited": [snapshot_url] if snapshot_url else [],
            "word_count": len(summary.split()),
        }
        return state

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

    if state.get("wayback_result"):
        state["review"] = {"decision": "approve", "reason": "Report uses Wayback snapshot evidence."}
        state["status"] = "approved"
        return state

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
