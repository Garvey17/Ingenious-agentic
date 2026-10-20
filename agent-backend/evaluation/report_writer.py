"""
evaluation/report_writer.py
-----------------------------
Generates a structured evaluation report in JSON and prints a
human-readable summary to stdout with colour coding.

Combines results from:
  - StructuralEvaluator
  - DeepEvalEvaluator
  - TruLensEvaluator
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.config import eval_config
from evaluation.pipeline_runner import RunResult
from evaluation.structural_metrics import StructuralEvalResult
from evaluation.deepeval_metrics import DeepEvalResult
from evaluation.trulens_metrics import TruLensEvalResult

logger = logging.getLogger(__name__)

# ANSI colours — disabled automatically on non-TTY outputs
try:
    import sys
    _COLOUR = sys.stdout.isatty()
except Exception:
    _COLOUR = False

RED    = "\033[91m" if _COLOUR else ""
GREEN  = "\033[92m" if _COLOUR else ""
YELLOW = "\033[93m" if _COLOUR else ""
CYAN   = "\033[96m" if _COLOUR else ""
BOLD   = "\033[1m"  if _COLOUR else ""
RESET  = "\033[0m"  if _COLOUR else ""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CombinedTestResult:
    test_case_id: str
    topic: str
    depth: str
    latency_seconds: float
    iterations: int
    pipeline_status: str
    structural: Optional[StructuralEvalResult]
    deepeval: Optional[DeepEvalResult]
    trulens: Optional[TruLensEvalResult]

    @property
    def overall_passed(self) -> bool:
        checks = []
        if self.structural:
            checks.append(self.structural.passed)
        if self.deepeval and not (self.deepeval.error and "not installed" in self.deepeval.error):
            checks.append(self.deepeval.overall_passed)
        if self.trulens and not (self.trulens.error and "not installed" in self.trulens.error):
            checks.append(self.trulens.overall_passed)
        return all(checks) if checks else False


@dataclass
class EvaluationReport:
    run_timestamp: str
    backend_url: str
    judge_model: str
    test_case_count: int
    passed_count: int
    failed_count: int
    pass_rate: float
    mean_latency_seconds: float
    results: List[CombinedTestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Report Writer
# ---------------------------------------------------------------------------

class ReportWriter:

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or eval_config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Aggregation ──────────────────────────────────────────────────────────

    def combine(
        self,
        run_results: List[RunResult],
        structural_results: List[StructuralEvalResult],
        deepeval_results: List[DeepEvalResult],
        trulens_results: List[TruLensEvalResult],
    ) -> EvaluationReport:
        """Merges all evaluator outputs into one EvaluationReport."""

        # Index by test_case_id for alignment
        struct_map  = {r.test_case_id: r for r in structural_results}
        de_map      = {r.test_case_id: r for r in deepeval_results}
        tru_map     = {r.test_case_id: r for r in trulens_results}

        combined: List[CombinedTestResult] = []
        for rr in run_results:
            combined.append(CombinedTestResult(
                test_case_id=rr.test_case_id,
                topic=rr.topic,
                depth=rr.depth,
                latency_seconds=round(rr.latency_seconds, 2),
                iterations=rr.iterations,
                pipeline_status=rr.status,
                structural=struct_map.get(rr.test_case_id),
                deepeval=de_map.get(rr.test_case_id),
                trulens=tru_map.get(rr.test_case_id),
            ))

        passed     = [c for c in combined if c.overall_passed]
        latencies  = [rr.latency_seconds for rr in run_results if rr.latency_seconds > 0]

        return EvaluationReport(
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            backend_url=eval_config.backend_url,
            judge_model=eval_config.judge_model,
            test_case_count=len(combined),
            passed_count=len(passed),
            failed_count=len(combined) - len(passed),
            pass_rate=round(len(passed) / max(len(combined), 1), 4),
            mean_latency_seconds=round(sum(latencies) / max(len(latencies), 1), 2),
            results=combined,
        )

    # ── JSON serialisation ───────────────────────────────────────────────────

    def _serialise(self, report: EvaluationReport) -> Dict[str, Any]:
        """Converts the report to a JSON-serialisable dict."""
        out: Dict[str, Any] = {
            "run_timestamp": report.run_timestamp,
            "backend_url": report.backend_url,
            "judge_model": report.judge_model,
            "summary": {
                "test_case_count": report.test_case_count,
                "passed": report.passed_count,
                "failed": report.failed_count,
                "pass_rate": report.pass_rate,
                "mean_latency_seconds": report.mean_latency_seconds,
            },
            "results": [],
        }

        for c in report.results:
            entry: Dict[str, Any] = {
                "test_case_id": c.test_case_id,
                "topic": c.topic,
                "depth": c.depth,
                "pipeline_status": c.pipeline_status,
                "latency_seconds": c.latency_seconds,
                "iterations": c.iterations,
                "overall_passed": c.overall_passed,
            }

            if c.structural:
                entry["structural"] = {
                    "overall_score": c.structural.overall_score,
                    "passed": c.structural.passed,
                    "metrics": [
                        {
                            "name": m.name, "score": m.score,
                            "passed": m.passed, "details": m.details,
                            "raw_values": m.raw_values,
                        }
                        for m in c.structural.metrics
                    ],
                }

            if c.deepeval:
                entry["deepeval"] = c.deepeval.summarise()

            if c.trulens:
                entry["trulens"] = c.trulens.summarise()

            out["results"].append(entry)

        return out

    def save_json(self, report: EvaluationReport) -> Path:
        """Writes the full report to a timestamped JSON file."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self.results_dir / f"eval_report_{ts}.json"
        data = self._serialise(report)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"Evaluation report saved to: {path}")
        return path

    # ── Console output ───────────────────────────────────────────────────────

    def print_summary(self, report: EvaluationReport) -> None:
        """Prints a rich human-readable summary to stdout."""
        SEP = "-" * 70

        print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
        print(f"{BOLD}{CYAN}  INGENIOUS AGENTIC - EVALUATION REPORT{RESET}")
        print(f"{BOLD}{CYAN}{'='*70}{RESET}")
        print(f"  Timestamp : {report.run_timestamp}")
        print(f"  Backend   : {report.backend_url}")
        print(f"  LLM Judge : {report.judge_model}")
        print(f"{SEP}")

        # ── High-level summary ───────────────────────────────────────────────
        pass_colour = GREEN if report.pass_rate >= 0.8 else (YELLOW if report.pass_rate >= 0.5 else RED)
        print(f"\n{BOLD}  OVERALL SUMMARY{RESET}")
        print(f"  Test Cases  : {report.test_case_count}")
        print(f"  Passed      : {pass_colour}{report.passed_count}{RESET}")
        print(f"  Failed      : {RED if report.failed_count else GREEN}{report.failed_count}{RESET}")
        print(f"  Pass Rate   : {pass_colour}{report.pass_rate*100:.1f}%{RESET}")
        print(f"  Mean Latency: {report.mean_latency_seconds:.1f}s\n")
        print(f"{SEP}")

        # ── Per-test breakdown ───────────────────────────────────────────────
        for c in report.results:
            status_colour = GREEN if c.overall_passed else RED
            status_label  = "[PASS]" if c.overall_passed else "[FAIL]"

            print(f"\n  {BOLD}[{c.test_case_id}]{RESET} {c.topic[:60]}")
            print(f"  Status   : {status_colour}{status_label}{RESET}  "
                  f"| Pipeline: {c.pipeline_status} | Depth: {c.depth} "
                  f"| Latency: {c.latency_seconds}s | Iter: {c.iterations}")

            # Structural
            if c.structural:
                sc = c.structural.overall_score
                s_col = GREEN if c.structural.passed else RED
                print(f"  {BOLD}Structural{RESET}  [{s_col}{'PASS' if c.structural.passed else 'FAIL'}{RESET}] score={sc:.3f}")
                for m in c.structural.metrics:
                    m_col = GREEN if m.passed else RED
                    print(f"    * {m.name:<28} score={m.score:.3f}  [{m_col}{'PASS' if m.passed else 'FAIL'}{RESET}]  {m.details[:60]}")

            # DeepEval
            if c.deepeval:
                de_col = GREEN if c.deepeval.overall_passed else RED
                print(f"  {BOLD}DeepEval{RESET}    [{de_col}{'PASS' if c.deepeval.overall_passed else 'FAIL'}{RESET}]")
                if c.deepeval.error:
                    print(f"    (!) {YELLOW}{c.deepeval.error}{RESET}")
                for m in c.deepeval.metrics:
                    s_str = f"{m.score:.3f}" if m.score is not None else "N/A "
                    m_col = GREEN if m.passed else RED
                    print(f"    * {m.name:<28} score={s_str}  [{m_col}{'PASS' if m.passed else 'FAIL'}{RESET}]")

            # TruLens
            if c.trulens:
                tru_col = GREEN if c.trulens.overall_passed else RED
                print(f"  {BOLD}TruLens{RESET}     [{tru_col}{'PASS' if c.trulens.overall_passed else 'FAIL'}{RESET}]")
                if c.trulens.error:
                    print(f"    (!) {YELLOW}{c.trulens.error}{RESET}")
                for m in c.trulens.metrics:
                    s_str = f"{m.score:.3f}" if m.score is not None else "N/A "
                    m_col = GREEN if m.passed else RED
                    print(f"    * {m.name:<28} score={s_str}  [{m_col}{'PASS' if m.passed else 'FAIL'}{RESET}]")

        print(f"\n{SEP}")
        print(f"  Full JSON report saved in: {eval_config.results_dir}/")
        print(f"  TruLens DB  : {eval_config.trulens_db_url}")
        print(f"  DeepEval CI : deepeval test run evaluation/")
        print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")
