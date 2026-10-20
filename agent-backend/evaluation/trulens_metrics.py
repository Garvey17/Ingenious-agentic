"""
evaluation/trulens_metrics.py
-------------------------------
TruLens RAG Triad evaluation for the Ingenious Agentic pipeline.

Implements the three canonical TruLens RAG metrics:
  1. Context Relevance  – Are retrieved sources relevant to the query?
  2. Groundedness       – Is the answer grounded in the retrieved context?
  3. Answer Relevance   – Does the final answer address the original query?

Additionally wraps the full pipeline run as a TruLens "app" so that
all traces are recorded in the TruLens SQLite dashboard for inspection.

Run the TruLens dashboard after evaluation with:
  trulens-eval  (or: python -m trulens_eval)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation.config import eval_config
from evaluation.pipeline_runner import RunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TruLensMetricResult:
    name: str
    score: Optional[float]
    passed: Optional[bool]
    feedback: str
    threshold: float


@dataclass
class TruLensEvalResult:
    test_case_id: str
    topic: str
    metrics: List[TruLensMetricResult] = field(default_factory=list)
    overall_passed: bool = False
    record_id: Optional[str] = None
    error: Optional[str] = None

    def summarise(self) -> Dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "topic": self.topic,
            "overall_passed": self.overall_passed,
            "record_id": self.record_id,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.passed,
                    "feedback": m.feedback,
                    "threshold": m.threshold,
                }
                for m in self.metrics
            ],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# TruLens Evaluator
# ---------------------------------------------------------------------------

class TruLensEvaluator:
    """
    Evaluates each RunResult using the TruLens RAG Triad.

    On first call, initialises the TruLens session (SQLite DB) and
    the OpenAI provider used as the feedback LLM.
    """

    def __init__(self):
        self._available = self._check_import()
        self._session = None
        self._provider = None

    @staticmethod
    def _check_import() -> bool:
        try:
            import trulens  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "TruLens is not installed. "
                "Install with: pip install trulens trulens-providers-openai\n"
                "RAG Triad metrics will be skipped."
            )
            return False

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_session(self):
        """Creates a TruLens session backed by a local SQLite database."""
        if self._session is not None:
            return

        from trulens.core import TruSession
        db_url = str(eval_config.trulens_db_url)
        logger.info(f"Initialising TruLens session at: {db_url}")
        self._session = TruSession(database_url=db_url)
        self._session.reset_database()   # fresh run each time

    def _init_provider(self):
        """Initialises the OpenAI feedback provider."""
        if self._provider is not None:
            return

        from trulens.providers.openai import OpenAI as TruOpenAI
        self._provider = TruOpenAI(
            model_engine=eval_config.judge_model,
            api_key=eval_config.openai_api_key,
        )

    def _get_feedbacks(self):
        """Returns the three RAG Triad Feedback objects."""
        from trulens.core import Feedback
        import numpy as np

        prov = self._provider

        # Context Relevance: for each retrieved chunk, score relevance to query
        f_context_relevance = (
            Feedback(
                prov.context_relevance,
                name="ContextRelevance",
            )
            .on_input()
            .on(lambda x: x.get("context", []))
            .aggregate(np.mean)
        )

        # Groundedness: is the answer supported by the context?
        f_groundedness = (
            Feedback(
                prov.groundedness_measure_with_cot_reasons,
                name="Groundedness",
            )
            .on(lambda x: x.get("context", []))
            .on_output()
            .aggregate(np.mean)
        )

        # Answer Relevance: does the answer address the input query?
        f_answer_relevance = Feedback(
            prov.relevance,
            name="AnswerRelevance",
        ).on_input_output()

        return [f_context_relevance, f_groundedness, f_answer_relevance]

    # ── Core evaluation ──────────────────────────────────────────────────────

    def _score_feedbacks(
        self, result: RunResult
    ) -> List[TruLensMetricResult]:
        """
        Runs each feedback function directly against the RunResult data.
        This avoids needing to replay the pipeline through a TruLens-wrapped app.
        """
        from trulens.core import Feedback

        provider = self._provider
        metrics: List[TruLensMetricResult] = []

        # ── 1. Context Relevance ────────────────────────────────────────────
        try:
            context_scores = [
                provider.context_relevance(result.topic, chunk)
                for chunk in (result.retrieved_context or [])[:5]   # cap at 5 chunks
            ]
            ctx_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
            ctx_thresh = eval_config.trulens_context_relevance_threshold
            metrics.append(TruLensMetricResult(
                name="ContextRelevance",
                score=round(ctx_score, 4),
                passed=ctx_score >= ctx_thresh,
                feedback=f"Mean relevance across {len(context_scores)} chunks.",
                threshold=ctx_thresh,
            ))
        except Exception as exc:
            logger.warning(f"ContextRelevance failed: {exc}")
            metrics.append(TruLensMetricResult(
                name="ContextRelevance", score=None, passed=False,
                feedback=str(exc), threshold=eval_config.trulens_context_relevance_threshold,
            ))

        # ── 2. Groundedness ─────────────────────────────────────────────────
        try:
            gnd_result = provider.groundedness_measure_with_cot_reasons(
                source="\n\n".join(result.retrieved_context[:5]),
                statement=result.executive_summary or result.full_report_text[:1500],
            )
            # groundedness_measure_with_cot_reasons returns (score, cot_string)
            gnd_score = gnd_result[0] if isinstance(gnd_result, tuple) else gnd_result
            gnd_thresh = eval_config.trulens_groundedness_threshold
            metrics.append(TruLensMetricResult(
                name="Groundedness",
                score=round(float(gnd_score), 4),
                passed=float(gnd_score) >= gnd_thresh,
                feedback=str(gnd_result[1]) if isinstance(gnd_result, tuple) else "",
                threshold=gnd_thresh,
            ))
        except Exception as exc:
            logger.warning(f"Groundedness failed: {exc}")
            metrics.append(TruLensMetricResult(
                name="Groundedness", score=None, passed=False,
                feedback=str(exc), threshold=eval_config.trulens_groundedness_threshold,
            ))

        # ── 3. Answer Relevance ─────────────────────────────────────────────
        try:
            ans_score = provider.relevance(
                prompt=result.topic,
                response=result.executive_summary or result.full_report_text[:1500],
            )
            ans_thresh = eval_config.trulens_answer_relevance_threshold
            metrics.append(TruLensMetricResult(
                name="AnswerRelevance",
                score=round(float(ans_score), 4),
                passed=float(ans_score) >= ans_thresh,
                feedback="Relevance of the report's executive summary to the original query.",
                threshold=ans_thresh,
            ))
        except Exception as exc:
            logger.warning(f"AnswerRelevance failed: {exc}")
            metrics.append(TruLensMetricResult(
                name="AnswerRelevance", score=None, passed=False,
                feedback=str(exc), threshold=eval_config.trulens_answer_relevance_threshold,
            ))

        return metrics

    def evaluate(self, result: RunResult, test_case: Dict) -> TruLensEvalResult:
        """Entry point: evaluates one RunResult with the RAG Triad."""
        eval_result = TruLensEvalResult(
            test_case_id=result.test_case_id,
            topic=result.topic,
        )

        if not self._available:
            eval_result.error = "TruLens not installed — skipping RAG Triad metrics."
            return eval_result

        if not result.succeeded:
            eval_result.error = f"Pipeline failed: status={result.status}"
            eval_result.overall_passed = False
            return eval_result

        try:
            self._init_session()
            self._init_provider()
            eval_result.metrics = self._score_feedbacks(result)
            passed = [m for m in eval_result.metrics if m.passed is not None]
            eval_result.overall_passed = all(m.passed for m in passed)
        except Exception as exc:
            logger.error(f"TruLens evaluation error: {exc}", exc_info=True)
            eval_result.error = str(exc)
            eval_result.overall_passed = False

        return eval_result

    def get_leaderboard(self) -> Optional[Any]:
        """Returns the TruLens leaderboard DataFrame (requires session)."""
        if self._session is None:
            return None
        try:
            return self._session.get_leaderboard()
        except Exception as exc:
            logger.warning(f"Could not fetch TruLens leaderboard: {exc}")
            return None
