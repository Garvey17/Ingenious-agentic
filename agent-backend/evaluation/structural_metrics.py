"""
evaluation/structural_metrics.py
----------------------------------
Rule-based (deterministic, no LLM cost) structural quality checks.

These run first and act as a fast quality gate before the more expensive
LLM-judge metrics (DeepEval / TruLens).

Metrics:
  - Report Completeness Score
  - Schema Compliance
  - Source Coverage
  - Critic Self-Score Gate
  - Agent Step Coverage
  - Iteration Efficiency
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation.config import eval_config
from evaluation.pipeline_runner import RunResult

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    name: str
    score: float          # normalised 0.0 – 1.0
    passed: bool
    details: str
    raw_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralEvalResult:
    test_case_id: str
    topic: str
    metrics: List[MetricScore] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False

    def add(self, metric: MetricScore):
        self.metrics.append(metric)
        self._recalculate()

    def _recalculate(self):
        if not self.metrics:
            return
        self.overall_score = sum(m.score for m in self.metrics) / len(self.metrics)
        self.passed = all(m.passed for m in self.metrics)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def check_report_completeness(result: RunResult, test_case: Dict) -> MetricScore:
    """
    Checks whether all required report fields are populated and meet
    minimum size thresholds defined in the test case and eval_config.
    """
    issues = []
    scores = []

    # Executive summary
    es = result.executive_summary
    es_words = _count_words(es)
    es_ok = es_words >= eval_config.min_executive_summary_words
    scores.append(1.0 if es_ok else max(0.0, es_words / eval_config.min_executive_summary_words))
    if not es_ok:
        issues.append(f"Executive summary too short ({es_words} words, need {eval_config.min_executive_summary_words})")

    # Sections
    min_sections = test_case.get("min_sections", eval_config.min_section_count)
    section_count = len(result.sections)
    sections_ok = section_count >= min_sections
    scores.append(1.0 if sections_ok else section_count / min_sections)
    if not sections_ok:
        issues.append(f"Only {section_count}/{min_sections} sections present")

    # Key findings
    min_findings = test_case.get("min_findings", eval_config.min_key_findings)
    findings_count = len(result.key_findings)
    findings_ok = findings_count >= min_findings
    scores.append(1.0 if findings_ok else findings_count / min_findings)
    if not findings_ok:
        issues.append(f"Only {findings_count}/{min_findings} key findings present")

    # Recommendations
    min_recs = test_case.get("min_recommendations", eval_config.min_recommendations)
    recs_count = len(result.recommendations)
    recs_ok = recs_count >= min_recs
    scores.append(1.0 if recs_ok else recs_count / min_recs)
    if not recs_ok:
        issues.append(f"Only {recs_count}/{min_recs} recommendations present")

    # Word count
    full_words = _count_words(result.full_report_text)
    wc_ok = full_words >= eval_config.min_word_count
    scores.append(1.0 if wc_ok else full_words / eval_config.min_word_count)
    if not wc_ok:
        issues.append(f"Total report word count too low ({full_words}, need {eval_config.min_word_count})")

    score = sum(scores) / len(scores)
    passed = score >= 0.8 and not issues

    return MetricScore(
        name="report_completeness",
        score=round(score, 4),
        passed=passed,
        details="; ".join(issues) if issues else "All completeness checks passed.",
        raw_values={
            "executive_summary_words": es_words,
            "section_count": section_count,
            "findings_count": findings_count,
            "recommendations_count": recs_count,
            "full_word_count": full_words,
        },
    )


def check_schema_compliance(result: RunResult) -> MetricScore:
    """
    Verifies the report dict has all expected keys and correct types.
    Checks both the report and the analyst insights schemas.
    """
    issues = []

    # Report schema
    REQUIRED_REPORT_KEYS = {
        "topic": str,
        "executive_summary": str,
        "sections": list,
        "key_findings": list,
        "recommendations": list,
        "sources_cited": list,
    }
    report = result.report or {}
    for key, expected_type in REQUIRED_REPORT_KEYS.items():
        val = report.get(key)
        if val is None:
            issues.append(f"report.{key} is missing")
        elif not isinstance(val, expected_type):
            issues.append(f"report.{key} has wrong type (got {type(val).__name__}, expected {expected_type.__name__})")

    # Analyst insights schema
    REQUIRED_INSIGHT_KEYS = {"key_facts": list, "insights": list}
    insights = result.insights or {}
    for key, expected_type in REQUIRED_INSIGHT_KEYS.items():
        val = insights.get(key)
        if val is None:
            issues.append(f"insights.{key} is missing")
        elif not isinstance(val, expected_type):
            issues.append(f"insights.{key} has wrong type")

    # Review schema
    REQUIRED_REVIEW_KEYS = {"overall_quality_score": (int, float), "decision": str}
    review = result.review or {}
    for key, expected_type in REQUIRED_REVIEW_KEYS.items():
        val = review.get(key)
        if val is None:
            issues.append(f"review.{key} is missing")
        elif not isinstance(val, expected_type):
            issues.append(f"review.{key} has wrong type")

    score = max(0.0, 1.0 - len(issues) * 0.1)
    return MetricScore(
        name="schema_compliance",
        score=round(score, 4),
        passed=len(issues) == 0,
        details="; ".join(issues) if issues else "All schema checks passed.",
        raw_values={"violation_count": len(issues), "violations": issues},
    )


def check_source_coverage(result: RunResult, test_case: Dict) -> MetricScore:
    """
    Measures whether sources were retrieved and cited in the report.
    """
    min_sources = test_case.get("min_sources", 2)
    retrieved_count = len(result.sources)
    cited_count = len(result.sources_cited)

    retrieval_ok = retrieved_count >= min_sources
    citation_ok = cited_count >= 1

    issues = []
    if not retrieval_ok:
        issues.append(f"Only {retrieved_count} sources retrieved (need {min_sources})")
    if not citation_ok:
        issues.append("No sources cited in the final report")

    # Dedup ratio (URL uniqueness)
    all_urls = [s.get("url", "") for s in result.sources]
    unique_urls = len(set(u for u in all_urls if u))
    dedup_ratio = unique_urls / max(len(all_urls), 1)

    score = (
        (0.5 * min(retrieved_count / max(min_sources, 1), 1.0))
        + (0.3 * (1.0 if citation_ok else 0.0))
        + (0.2 * dedup_ratio)
    )

    return MetricScore(
        name="source_coverage",
        score=round(score, 4),
        passed=retrieval_ok and citation_ok,
        details="; ".join(issues) if issues else "Source coverage is adequate.",
        raw_values={
            "retrieved_count": retrieved_count,
            "cited_count": cited_count,
            "unique_urls": unique_urls,
            "dedup_ratio": round(dedup_ratio, 4),
        },
    )


def check_critic_quality_score(result: RunResult) -> MetricScore:
    """
    Validates the pipeline's own internal quality gate (the Critic agent score).
    """
    score = result.critic_quality_score

    if score is None:
        return MetricScore(
            name="critic_quality_gate",
            score=0.0,
            passed=False,
            details="Critic quality score is absent from the review.",
        )

    passed = score >= eval_config.min_critic_quality_score
    return MetricScore(
        name="critic_quality_gate",
        score=round(score, 4),
        passed=passed,
        details=(
            f"Critic score {score:.3f} {'≥' if passed else '<'} "
            f"threshold {eval_config.min_critic_quality_score}"
        ),
        raw_values={"critic_score": score, "threshold": eval_config.min_critic_quality_score},
    )


def check_agent_step_coverage(result: RunResult) -> MetricScore:
    """
    Verifies that all 5 expected pipeline steps executed.
    Combines live step traces with presence of step artifacts in final state
    to prevent false negatives when fast steps finish between poll intervals.
    """
    observed_steps = {t.step_name for t in result.step_traces}

    # Add steps confirmed by their produced artifacts
    if result.plan:
        observed_steps.add("planner")
    if result.sources:
        observed_steps.add("researcher")
    if result.insights:
        observed_steps.add("analyst")
    if result.report:
        observed_steps.add("writer")
    if result.review:
        observed_steps.add("critic")

    EXPECTED_STEPS = {"planner", "researcher", "analyst", "writer", "critic"}
    missing = EXPECTED_STEPS - observed_steps

    coverage = len(observed_steps & EXPECTED_STEPS) / len(EXPECTED_STEPS)
    passed = len(missing) == 0

    return MetricScore(
        name="agent_step_coverage",
        score=round(coverage, 4),
        passed=passed,
        details=(
            f"Missing steps: {missing}" if missing
            else f"All {len(EXPECTED_STEPS)} pipeline steps executed."
        ),
        raw_values={
            "observed_steps": list(observed_steps),
            "missing_steps": list(missing),
        },
    )


def check_iteration_efficiency(result: RunResult) -> MetricScore:
    """
    Rewards converging quickly (fewer critic revision loops).
    1 iteration = perfect; max_iterations = lowest passing score.
    """
    max_iter = 3  # from config default
    iterations = max(result.iterations + 1, 1)  # 0-indexed → 1-indexed
    score = max(0.0, 1.0 - ((iterations - 1) / max_iter))
    passed = iterations <= max_iter

    return MetricScore(
        name="iteration_efficiency",
        score=round(score, 4),
        passed=passed,
        details=f"Completed in {iterations} iteration(s) (max allowed: {max_iter}).",
        raw_values={"iterations": iterations, "max_iterations": max_iter},
    )


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class StructuralEvaluator:
    """Runs all structural metrics against a set of RunResults."""

    def evaluate(
        self, result: RunResult, test_case: Dict
    ) -> StructuralEvalResult:
        eval_result = StructuralEvalResult(
            test_case_id=result.test_case_id,
            topic=result.topic,
        )

        if not result.succeeded:
            eval_result.metrics.append(MetricScore(
                name="pipeline_execution",
                score=0.0,
                passed=False,
                details=f"Pipeline did not complete successfully: status={result.status}, error={result.error}",
            ))
            return eval_result

        eval_result.add(check_report_completeness(result, test_case))
        eval_result.add(check_schema_compliance(result))
        eval_result.add(check_source_coverage(result, test_case))
        eval_result.add(check_critic_quality_score(result))
        eval_result.add(check_agent_step_coverage(result))
        eval_result.add(check_iteration_efficiency(result))

        return eval_result
