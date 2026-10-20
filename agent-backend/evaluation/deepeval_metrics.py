"""
evaluation/deepeval_metrics.py
--------------------------------
LLM-judge metrics powered by DeepEval.

Metrics implemented:
  1. Answer Relevancy       – Is the report on-topic?
  2. Faithfulness           – Are claims grounded in retrieved sources?
  3. Contextual Relevancy   – Did the retriever fetch relevant content?
  4. Hallucination          – Does the report make claims not in sources?
  5. G-Eval (Coherence)     – Is the report logically well-structured?
  6. Keyword Coverage       – Are topic-specific keywords mentioned?

All metrics use DeepEval's test-case abstraction so results can be fed
into DeepEval's dashboard / CI assertions with `deepeval test run`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation.config import eval_config
from evaluation.pipeline_runner import RunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DeepEvalMetricResult:
    name: str
    score: Optional[float]
    passed: Optional[bool]
    reason: str
    threshold: float


@dataclass
class DeepEvalResult:
    test_case_id: str
    topic: str
    metrics: List[DeepEvalMetricResult] = field(default_factory=list)
    overall_passed: bool = False
    error: Optional[str] = None

    def summarise(self) -> Dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "topic": self.topic,
            "overall_passed": self.overall_passed,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.passed,
                    "reason": m.reason,
                    "threshold": m.threshold,
                }
                for m in self.metrics
            ],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Helper: build DeepEval LLMTestCase
# ---------------------------------------------------------------------------

def _build_test_case(result: RunResult, test_case: Dict):
    """Converts a RunResult into a DeepEval LLMTestCase."""
    from deepeval.test_case import LLMTestCase

    retrieved = result.retrieved_context or ["No sources retrieved."]
    return LLMTestCase(
        input=result.topic,
        actual_output=result.full_report_text,
        expected_output="\n".join(test_case.get("ground_truth_context", [])),
        retrieval_context=retrieved,
        context=retrieved,
    )


def _run_metric(metric, test_case_obj) -> DeepEvalMetricResult:
    """
    Executes a single DeepEval metric and normalises its output.
    DeepEval metrics expose .score and .reason after calling .measure().
    """
    try:
        metric.measure(test_case_obj)
        return DeepEvalMetricResult(
            name=metric.__class__.__name__,
            score=metric.score,
            passed=metric.is_successful(),
            reason=metric.reason or "",
            threshold=metric.threshold,
        )
    except Exception as exc:
        logger.warning(f"DeepEval metric {metric.__class__.__name__} failed: {exc}")
        return DeepEvalMetricResult(
            name=metric.__class__.__name__,
            score=None,
            passed=False,
            reason=f"Metric execution error: {exc}",
            threshold=getattr(metric, "threshold", 0.0),
        )


# ---------------------------------------------------------------------------
# Keyword Coverage (custom, no extra LLM call)
# ---------------------------------------------------------------------------

import re

def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[-_]", " ", text)
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return " ".join(cleaned.lower().split())

def _keyword_matches(kw: str, text: str) -> bool:
    k = _normalize_text(kw)
    if not k:
        return False
    if k in text:
        return True
    if k.endswith("s") and k[:-1] in text:
        return True
    words = k.split()
    if len(words) > 1 and all(w in text for w in words):
        return True
    return False

def _keyword_coverage(result: RunResult, test_case: Dict) -> DeepEvalMetricResult:
    """Robust keyword presence check with hyphen, plural, and phrase normalization."""
    raw_keywords = test_case.get("expected_keywords", [])
    if not raw_keywords:
        return DeepEvalMetricResult(
            name="KeywordCoverage",
            score=1.0, passed=True,
            reason="No expected keywords specified.",
            threshold=0.0,
        )

    norm_report = _normalize_text(result.full_report_text)
    found = [kw for kw in raw_keywords if _keyword_matches(kw, norm_report)]
    score = len(found) / len(raw_keywords)
    depth = test_case.get("depth", "standard")
    threshold = 0.25 if depth == "quick" else 0.35

    return DeepEvalMetricResult(
        name="KeywordCoverage",
        score=round(score, 4),
        passed=score >= threshold,
        reason=f"Found {len(found)}/{len(raw_keywords)} expected keywords: {found}",
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

class DeepEvalEvaluator:
    """
    Runs the full DeepEval metric suite against a RunResult.

    Uses the model specified in eval_config.judge_model as the LLM judge.
    Falls back gracefully if DeepEval is not installed.
    """

    def __init__(self):
        self._deepeval_available = self._check_import()

    @staticmethod
    def _check_import() -> bool:
        try:
            import deepeval  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "DeepEval is not installed. "
                "Install with: pip install deepeval\n"
                "LLM-judge metrics will be skipped."
            )
            return False

    def _get_judge_model(self):
        """Returns a DeepEval-compatible LLM judge model."""
        from deepeval.models import GPTModel
        return GPTModel(
            model=eval_config.judge_model,
            api_key=eval_config.openai_api_key,
        )

    def evaluate(self, result: RunResult, test_case: Dict) -> DeepEvalResult:
        eval_result = DeepEvalResult(
            test_case_id=result.test_case_id,
            topic=result.topic,
        )

        # Always run keyword coverage (no LLM cost)
        eval_result.metrics.append(_keyword_coverage(result, test_case))

        if not result.succeeded:
            eval_result.error = f"Pipeline failed: status={result.status}"
            eval_result.overall_passed = False
            return eval_result

        if not self._deepeval_available:
            eval_result.error = "DeepEval not installed — skipping LLM metrics."
            eval_result.overall_passed = all(
                m.passed for m in eval_result.metrics if m.passed is not None
            )
            return eval_result

        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualRelevancyMetric,
            HallucinationMetric,
            GEval,
        )
        from deepeval.test_case import LLMTestCaseParams

        judge = self._get_judge_model()
        test_case_obj = _build_test_case(result, test_case)

        # ── 1. Answer Relevancy ─────────────────────────────────────────────
        eval_result.metrics.append(_run_metric(
            AnswerRelevancyMetric(
                threshold=eval_config.answer_relevancy_threshold,
                model=judge,
                include_reason=True,
            ),
            test_case_obj,
        ))

        # ── 2. Faithfulness ─────────────────────────────────────────────────
        eval_result.metrics.append(_run_metric(
            FaithfulnessMetric(
                threshold=eval_config.faithfulness_threshold,
                model=judge,
                include_reason=True,
            ),
            test_case_obj,
        ))

        # ── 3. Contextual Relevancy ─────────────────────────────────────────
        eval_result.metrics.append(_run_metric(
            ContextualRelevancyMetric(
                threshold=eval_config.contextual_relevancy_threshold,
                model=judge,
                include_reason=True,
            ),
            test_case_obj,
        ))

        # ── 4. Hallucination ────────────────────────────────────────────────
        eval_result.metrics.append(_run_metric(
            HallucinationMetric(
                threshold=eval_config.hallucination_threshold,
                model=judge,
                include_reason=True,
            ),
            test_case_obj,
        ))

        # ── 5. G-Eval: Coherence ────────────────────────────────────────────
        eval_result.metrics.append(_run_metric(
            GEval(
                name="Coherence",
                criteria=(
                    "Evaluate the logical flow and structure of the research report. "
                    "A coherent report presents ideas in a clear, well-organised order "
                    "with smooth transitions between sections."
                ),
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                threshold=0.60,
                model=judge,
            ),
            test_case_obj,
        ))

        # ── 6. G-Eval: Depth of Analysis ────────────────────────────────────
        eval_result.metrics.append(_run_metric(
            GEval(
                name="AnalysisDepth",
                criteria=(
                    "Assess whether the report demonstrates genuine analytical depth. "
                    "Does it go beyond surface-level summaries to provide insights, "
                    "identify contradictions, and offer actionable recommendations?"
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                threshold=0.55,
                model=judge,
            ),
            test_case_obj,
        ))

        # Aggregate pass/fail
        passed_metrics = [m for m in eval_result.metrics if m.passed is not None]
        eval_result.overall_passed = all(m.passed for m in passed_metrics)

        return eval_result
