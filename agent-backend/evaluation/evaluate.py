"""
evaluate.py  ← ENTRY POINT
----------------------------
Run the full Ingenious Agentic evaluation suite.

Usage:
    python evaluate.py [OPTIONS]

Options:
    --ids TD001,TD002      Run only these test case IDs (comma-separated)
    --skip-deepeval        Skip DeepEval LLM-judge metrics
    --skip-trulens         Skip TruLens RAG Triad metrics
    --no-structural        Skip rule-based structural metrics
    --backend-url URL      Override backend URL (default: http://localhost:8000)
    --depth quick|standard|deep   Override depth for all test cases

Prerequisites:
    1. Start the backend server:
           python -m uvicorn app.main:app --reload
    2. Activate the venv and install evaluation dependencies (see below).
    3. Run this script from the agent-backend/ directory:
           python evaluate.py

Required pip packages:
    deepeval
    trulens
    trulens-providers-openai
    httpx
    python-dotenv
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# ── Path bootstrap ────────────────────────────────────────────────────────────
# __file__ is  agent-backend/evaluation/evaluate.py
# .parent      → agent-backend/evaluation/
# .parent.parent → agent-backend/          ← the package root we need
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Internal imports ─────────────────────────────────────────────────────────
from evaluation.config import eval_config
from evaluation.pipeline_runner import PipelineRunner, RunResult
from evaluation.structural_metrics import StructuralEvaluator, StructuralEvalResult
from evaluation.deepeval_metrics import DeepEvalEvaluator, DeepEvalResult
from evaluation.trulens_metrics import TruLensEvaluator, TruLensEvalResult
from evaluation.report_writer import ReportWriter

# ── Windows Console UTF-8 Fix ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Ingenious Agentic — Multi-Agent System Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of test case IDs to run (e.g. td_001,td_003). Runs all by default.",
    )
    parser.add_argument(
        "--skip-deepeval",
        action="store_true",
        help="Skip DeepEval LLM-judge metrics (saves API cost).",
    )
    parser.add_argument(
        "--skip-trulens",
        action="store_true",
        help="Skip TruLens RAG Triad metrics.",
    )
    parser.add_argument(
        "--no-structural",
        action="store_true",
        help="Skip deterministic structural quality checks.",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=None,
        help="Override the backend URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default=None,
        help="Override research depth for all test cases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load test cases and validate config without hitting the backend. "
            "Useful for verifying installation."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(ids_filter: Optional[List[str]] = None) -> List[dict]:
    path = eval_config.dataset_path
    if not path.exists():
        logger.error(f"Dataset not found at: {path}")
        sys.exit(1)

    with path.open(encoding="utf-8") as f:
        dataset = json.load(f)

    if ids_filter:
        ids_lower = [i.lower() for i in ids_filter]
        dataset = [tc for tc in dataset if tc["id"].lower() in ids_lower]
        if not dataset:
            logger.error(f"No test cases matched the requested IDs: {ids_filter}")
            sys.exit(1)

    logger.info(f"Loaded {len(dataset)} test case(s) from {path.name}")
    return dataset


# ---------------------------------------------------------------------------
# Main async evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(args) -> int:
    """
    Orchestrates the full evaluation pipeline.
    Returns an exit code: 0 = all passed, 1 = failures detected.
    """
    # ── Config overrides ─────────────────────────────────────────────────────
    if args.backend_url:
        eval_config.backend_url = args.backend_url

    # ── Validate config ──────────────────────────────────────────────────────
    require_keys = not (args.skip_deepeval and args.skip_trulens) and not args.dry_run
    try:
        eval_config.validate(require_llm_keys=require_keys)
    except ValueError as exc:
        logger.error(f"Configuration error: {exc}")
        return 1

    logger.info(f"Backend URL : {eval_config.backend_url}")
    logger.info(f"Judge model : {eval_config.judge_model}")
    logger.info(f"DeepEval    : {'SKIPPED' if args.skip_deepeval else 'ENABLED'}")
    logger.info(f"TruLens     : {'SKIPPED' if args.skip_trulens else 'ENABLED'}")
    logger.info(f"Structural  : {'SKIPPED' if args.no_structural else 'ENABLED'}")

    # ── Load test cases ──────────────────────────────────────────────────────
    ids_filter = [i.strip() for i in args.ids.split(",")] if args.ids else None
    test_cases = load_dataset(ids_filter)

    # Apply depth override
    if args.depth:
        for tc in test_cases:
            tc["depth"] = args.depth

    # ── Dry-run mode ─────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Configuration and dataset are valid.")
        print(f"  Would evaluate {len(test_cases)} test case(s):")
        for tc in test_cases:
            print(f"    [{tc['id']}] {tc['topic'][:60]} (depth={tc['depth']})")
        return 0

    # ── Step 1: Run the live pipeline ────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 1/4 - Running Live Pipeline")
    print("="*70)
    runner = PipelineRunner()
    run_results: List[RunResult] = await runner.run_all(test_cases)

    succeeded = [r for r in run_results if r.succeeded]
    logger.info(f"Pipeline complete: {len(succeeded)}/{len(run_results)} succeeded.")

    # ── Step 2: Structural metrics ───────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 2/4 - Structural Quality Checks")
    print("="*70)
    structural_results: List[StructuralEvalResult] = []
    if not args.no_structural:
        evaluator = StructuralEvaluator()
        tc_map = {tc["id"]: tc for tc in test_cases}
        for rr in run_results:
            sr = evaluator.evaluate(rr, tc_map.get(rr.test_case_id, {}))
            structural_results.append(sr)
            logger.info(f"  [{rr.test_case_id}] Structural: score={sr.overall_score:.3f} | passed={sr.passed}")
    else:
        logger.info("Structural metrics skipped.")

    # ── Step 3: DeepEval LLM-judge metrics ───────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 3/4 - DeepEval LLM-Judge Metrics")
    print("="*70)
    deepeval_results: List[DeepEvalResult] = []
    if not args.skip_deepeval:
        de_evaluator = DeepEvalEvaluator()
        tc_map = {tc["id"]: tc for tc in test_cases}
        for rr in run_results:
            de_result = de_evaluator.evaluate(rr, tc_map.get(rr.test_case_id, {}))
            deepeval_results.append(de_result)
            logger.info(
                f"  [{rr.test_case_id}] DeepEval: passed={de_result.overall_passed} "
                f"| metrics={len(de_result.metrics)}"
            )
    else:
        logger.info("DeepEval metrics skipped.")

    # ── Step 4: TruLens RAG Triad ────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 4/4 - TruLens RAG Triad")
    print("="*70)
    trulens_results: List[TruLensEvalResult] = []
    if not args.skip_trulens:
        tru_evaluator = TruLensEvaluator()
        tc_map = {tc["id"]: tc for tc in test_cases}
        for rr in run_results:
            tru_result = tru_evaluator.evaluate(rr, tc_map.get(rr.test_case_id, {}))
            trulens_results.append(tru_result)
            logger.info(
                f"  [{rr.test_case_id}] TruLens: passed={tru_result.overall_passed} "
                f"| metrics={len(tru_result.metrics)}"
            )

        # Print TruLens leaderboard if available
        leaderboard = tru_evaluator.get_leaderboard()
        if leaderboard is not None:
            print("\n-- TruLens Leaderboard --")
            print(leaderboard.to_string())
    else:
        logger.info("TruLens metrics skipped.")

    # ── Combine & Report ─────────────────────────────────────────────────────
    writer = ReportWriter()
    report = writer.combine(run_results, structural_results, deepeval_results, trulens_results)
    json_path = writer.save_json(report)
    writer.print_summary(report)

    # ── Exit code ────────────────────────────────────────────────────────────
    if report.failed_count > 0:
        logger.warning(f"{report.failed_count} test case(s) failed.")
        return 1

    logger.info("All test cases passed.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    exit_code = asyncio.run(run_evaluation(args))
    sys.exit(exit_code)
