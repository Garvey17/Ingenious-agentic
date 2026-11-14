"""Compare the bare model with the Wayback-enabled research agent."""

import asyncio
import json
from pathlib import Path

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.config import settings
from app.services import run_research
from bare_model import ask_bare_model


CASES_PATH = Path(__file__).with_name("cases.json")
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
JUDGE = GPTModel(model=settings.openai_model, api_key=settings.openai_api_key)


def _format_score(score) -> str:
    if score is None:
        return "N/A"
    return f"{score:.3f}"


def _measure_case(answer: str, question: str, context: list[str], ground_truth: str, criteria: str):
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth or None,
        retrieval_context=context,
    )
    faithfulness = FaithfulnessMetric(model=JUDGE, async_mode=False, include_reason=True)
    faithfulness.measure(test_case)
    faithfulness_score = faithfulness.score

    exact_match_score = None
    if ground_truth:
        exact_match = GEval(
            name="ExactHeadlineMatch",
            criteria=criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=JUDGE,
            async_mode=False,
        )
        exact_match.measure(test_case)
        exact_match_score = exact_match.score

    return faithfulness_score, exact_match_score


def test_osint_stumper_comparison():
    asyncio.run(_run_comparison())


async def _run_comparison():
    for case in CASES:
        bare_answer = await ask_bare_model(case["question"])
        agent_state = await run_research(
            request_id=f"osint-stumper-{case['id']}",
            goal=case["question"],
            depth="quick",
        )
        report = agent_state.get("report", {})
        agent_answer = "\n\n".join(
            [report.get("executive_summary", "")]
            + [section.get("content", "") for section in report.get("sections", [])]
        ).strip()

        wayback_result = agent_state.get("wayback_result", {})
        context = list(case.get("reference_context", "") and [case["reference_context"]] or [])
        context.extend(wayback_result.get("top_stories", []))
        if case.get("ground_truth"):
            context.append(case["ground_truth"])
        if not context:
            context = ["No snapshot headline has been entered as ground truth yet."]

        criteria = (
            "Does the answer reproduce the ground-truth headline exactly as the #1 story, "
            "without substituting a different headline?"
            if case["id"] == "hn_closest_to_20230315"
            else "Does the answer clearly say it could not verify the page and avoid guessing a headline?"
        )
        bare_scores = await asyncio.to_thread(
            _measure_case,
            bare_answer,
            case["question"],
            context,
            case.get("ground_truth", ""),
            criteria,
        )
        agent_scores = await asyncio.to_thread(
            _measure_case,
            agent_answer,
            case["question"],
            context,
            case.get("ground_truth", ""),
            criteria,
        )

        print(f"\n=== {case['id']} ===")
        print(f"Ground truth: {case.get('ground_truth') or '[fill from snapshot]'}")
        print(f"Bare model answer:\n{bare_answer}")
        print(
            "Bare model scores: "
            f"Faithfulness={_format_score(bare_scores[0])}, "
            f"Exact headline / abstention GEval={_format_score(bare_scores[1])}"
        )
        print(f"Agent answer:\n{agent_answer}")
        print(
            "Agent scores: "
            f"Faithfulness={_format_score(agent_scores[0])}, "
            f"Exact headline / abstention GEval={_format_score(agent_scores[1])}"
        )

        if case["id"] == "hn_no_nearby_snapshot_19900101":
            assert "could not verify" in agent_answer.lower()
            assert not wayback_result.get("found")
        elif wayback_result.get("found"):
            assert case["snapshot_url"] in agent_answer
            assert wayback_result["snapshot_timestamp"] in agent_answer
