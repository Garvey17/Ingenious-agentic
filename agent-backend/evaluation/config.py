"""
evaluation/config.py
--------------------
Centralised configuration for the Ingenious Agentic evaluation suite.
Reads from the same .env file as the main application so no duplicate
secrets are needed.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow imports from the agent-backend root (app.*) even when running
# the evaluation scripts from inside the evaluation/ sub-folder.
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from dotenv import load_dotenv

load_dotenv(_BACKEND_ROOT / ".env")


# ---------------------------------------------------------------------------
# Core Settings
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # ── API keys (pulled from environment) ──────────────────────────────────
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )

    # ── Backend URL ─────────────────────────────────────────────────────────
    backend_url: str = field(
        default_factory=lambda: os.getenv("EVAL_BACKEND_URL", "http://localhost:8000")
    )

    # ── LLM judge model ─────────────────────────────────────────────────────
    # The model used by DeepEval / TruLens to *score* the outputs.
    # Defaults to gpt-4o (strong judge); can be overridden via env var.
    judge_model: str = field(
        default_factory=lambda: os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "datasets" / "golden_dataset.json"
    )

    # ── Polling ─────────────────────────────────────────────────────────────
    poll_interval_seconds: float = 5.0
    max_poll_attempts: int = 120          # 120 × 5 s = 10 minutes timeout

    # ── Quality gate thresholds (0–1 scale) ─────────────────────────────────
    # DeepEval G-Eval / faithfulness thresholds
    faithfulness_threshold: float = 0.60
    answer_relevancy_threshold: float = 0.65
    contextual_relevancy_threshold: float = 0.55
    hallucination_threshold: float = 0.50  # Modern DeepEval: 1.0 is no hallucination, threshold is min score

    # Structural quality thresholds (rule-based, no LLM needed)
    min_executive_summary_words: int = 50
    min_section_count: int = 2
    min_key_findings: int = 2
    min_recommendations: int = 1
    min_word_count: int = 200

    # Critic quality score gate (the system's own internal score)
    min_critic_quality_score: float = 0.65

    # TruLens RAG triad thresholds
    trulens_context_relevance_threshold: float = 0.50
    trulens_groundedness_threshold: float = 0.55
    trulens_answer_relevance_threshold: float = 0.60

    # ── Output ──────────────────────────────────────────────────────────────
    results_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "results"
    )

    @property
    def trulens_db_url(self) -> str:
        db_file = (self.results_dir / "trulens_eval.db").resolve().as_posix()
        return f"sqlite:///{db_file}"

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Convenience helpers ──────────────────────────────────────────────────
    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_google_key(self) -> bool:
        return bool(self.google_api_key)

    def validate(self, require_llm_keys: bool = True):
        """Raises ValueError if required config is missing."""
        if require_llm_keys and not self.has_openai_key and not self.has_google_key:
            raise ValueError(
                "At least one LLM API key (OPENAI_API_KEY or GOOGLE_API_KEY) "
                "must be set in the .env file to run evaluations."
            )


# Singleton used by all evaluation modules
eval_config = EvalConfig()
