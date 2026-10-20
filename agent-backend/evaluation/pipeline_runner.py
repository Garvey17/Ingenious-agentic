"""
evaluation/pipeline_runner.py
------------------------------
Calls the live Ingenious Agentic backend, waits for completion,
and returns structured RunResult objects for downstream metric modules.

Works by hitting the real FastAPI endpoints:
  POST /api/research/   → start task
  GET  /api/research/{id}/state → poll for completion
  GET  /api/research/{id}       → fetch final report
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from evaluation.config import eval_config

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AgentStepTrace:
    """Captures the intermediate state of one agent step."""
    step_name: str
    status: str
    data_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Complete output of one end-to-end pipeline run."""
    test_case_id: str
    topic: str
    depth: str
    status: str                          # "completed" | "failed" | "timeout"
    report: Optional[Dict[str, Any]]     # raw report dict from the backend
    sources: List[Dict[str, Any]]        # raw sources list
    insights: Optional[Dict[str, Any]]   # raw analyst insights
    review: Optional[Dict[str, Any]]     # critic review
    iterations: int
    latency_seconds: float
    plan: List[str] = field(default_factory=list) # planner output
    step_traces: List[AgentStepTrace] = field(default_factory=list)
    error: Optional[str] = None

    # ── Convenience accessors ────────────────────────────────────────────────
    @property
    def executive_summary(self) -> str:
        return (self.report or {}).get("executive_summary", "")

    @property
    def sections(self) -> List[Dict]:
        return (self.report or {}).get("sections", [])

    @property
    def key_findings(self) -> List[str]:
        return (self.report or {}).get("key_findings", [])

    @property
    def recommendations(self) -> List[str]:
        return (self.report or {}).get("recommendations", [])

    @property
    def sources_cited(self) -> List[str]:
        return (self.report or {}).get("sources_cited", [])

    @property
    def word_count(self) -> int:
        return (self.report or {}).get("word_count", 0)

    @property
    def full_report_text(self) -> str:
        """Flattens the report into a single string for LLM-based metrics."""
        if not self.report:
            return ""
        parts = [self.executive_summary]
        for s in self.sections:
            parts.append(f"## {s.get('title', '')}\n{s.get('content', '')}")
        parts.extend(self.key_findings)
        parts.extend(self.recommendations)
        return "\n\n".join(p for p in parts if p)

    @property
    def retrieved_context(self) -> List[str]:
        """Returns source content snippets — used as RAG retrieval context."""
        return [
            s.get("content", "") for s in self.sources if s.get("content")
        ]

    @property
    def critic_quality_score(self) -> Optional[float]:
        return (self.review or {}).get("overall_quality_score")

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.report is not None


# ---------------------------------------------------------------------------
# Backend Client
# ---------------------------------------------------------------------------

class BackendClient:
    """Thin async client for the Ingenious Agentic FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or eval_config.backend_url).rstrip("/")
        self.timeout = timeout

    async def health_check(self) -> bool:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
            except Exception as exc:
                logger.error(f"Health check failed: {exc}")
                return False

    async def start_research(self, topic: str, depth: str = "standard") -> str:
        """Starts a research task and returns the request_id."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/research/",
                json={"topic": topic, "depth": depth},
            )
            resp.raise_for_status()
            return resp.json()["request_id"]

    async def get_state(self, request_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/research/{request_id}/state")
            resp.raise_for_status()
            return resp.json()

    async def get_report(self, request_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/research/{request_id}")
            resp.raise_for_status()
            return resp.json()

    async def poll_until_done(
        self,
        request_id: str,
        step_traces: List[AgentStepTrace],
    ) -> str:
        """
        Polls the state endpoint until the pipeline finishes.
        Returns the final status string.
        """
        terminal_statuses = {"approved", "failed"}
        seen_steps: set = set()

        for _ in range(eval_config.max_poll_attempts):
            await asyncio.sleep(eval_config.poll_interval_seconds)
            try:
                state = await self.get_state(request_id)
            except Exception as exc:
                logger.warning(f"Poll error: {exc}")
                continue

            current_step = state.get("current_step", "")
            status = state.get("status", "running")

            # Record a new agent step transition
            if current_step and current_step not in seen_steps:
                seen_steps.add(current_step)
                step_traces.append(
                    AgentStepTrace(
                        step_name=current_step,
                        status=status,
                        data_snapshot={
                            "iteration": state.get("iteration", 0),
                            "sources_count": len(state.get("sources", [])),
                            "has_insights": bool(state.get("insights")),
                            "has_report": bool(state.get("report")),
                        },
                    )
                )
                logger.info(f"  → Step: {current_step} | Status: {status}")

            if status in terminal_statuses:
                return status

        logger.warning(f"Polling timeout for {request_id}")
        return "timeout"


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Orchestrates end-to-end pipeline runs for a list of test cases
    and returns a list of RunResult objects.
    """

    def __init__(self, client: Optional[BackendClient] = None):
        self.client = client or BackendClient()

    async def _run_single(
        self, test_case_id: str, topic: str, depth: str
    ) -> RunResult:
        step_traces: List[AgentStepTrace] = []
        start_time = time.monotonic()

        logger.info(f"\n{'='*60}")
        logger.info(f"Running test case [{test_case_id}]: {topic!r} (depth={depth})")

        # 1. Verify backend is reachable
        if not await self.client.health_check():
            return RunResult(
                test_case_id=test_case_id,
                topic=topic,
                depth=depth,
                status="failed",
                report=None,
                sources=[],
                insights=None,
                review=None,
                iterations=0,
                latency_seconds=0.0,
                error="Backend is not reachable. Start the server first.",
            )

        # 2. Submit the research request
        try:
            request_id = await self.client.start_research(topic, depth)
            logger.info(f"  Request ID: {request_id}")
        except Exception as exc:
            return RunResult(
                test_case_id=test_case_id, topic=topic, depth=depth,
                status="failed", report=None, sources=[], insights=None,
                review=None, iterations=0, latency_seconds=0.0,
                error=f"Failed to start research: {exc}",
            )

        # 3. Poll until terminal state
        final_status = await self.client.poll_until_done(request_id, step_traces)
        latency = time.monotonic() - start_time

        # 4. Fetch full internal state for metric extraction
        try:
            final_state = await self.client.get_state(request_id)
        except Exception as exc:
            logger.error(f"Could not fetch final state: {exc}")
            final_state = {}

        api_status = "completed" if final_status == "approved" else final_status

        result = RunResult(
            test_case_id=test_case_id,
            topic=topic,
            depth=depth,
            status=api_status,
            report=final_state.get("report"),
            sources=final_state.get("sources", []),
            insights=final_state.get("insights"),
            review=final_state.get("review"),
            plan=final_state.get("plan", []),
            iterations=final_state.get("iteration", 0),
            latency_seconds=latency,
            step_traces=step_traces,
            error=final_state.get("error"),
        )

        logger.info(
            f"  Done: status={api_status} | "
            f"iterations={result.iterations} | "
            f"latency={latency:.1f}s | "
            f"sections={len(result.sections)}"
        )
        return result

    async def run_all(self, test_cases: List[Dict[str, Any]]) -> List[RunResult]:
        """Runs all test cases sequentially (to avoid rate-limit issues)."""
        results = []
        for tc in test_cases:
            result = await self._run_single(
                test_case_id=tc["id"],
                topic=tc["topic"],
                depth=tc.get("depth", "standard"),
            )
            results.append(result)
        return results
