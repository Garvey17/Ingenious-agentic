# Ingenious Agentic — Evaluation Suite

A professional, three-tier evaluation framework for the multi-agent research pipeline.

---

## Architecture

```
evaluation/
├── evaluate.py              ← ENTRY POINT
├── config.py                ← Thresholds, API keys, paths
├── pipeline_runner.py       ← Calls the live backend & collects RunResults
├── structural_metrics.py    ← Tier 1: Rule-based structural checks (no LLM cost)
├── deepeval_metrics.py      ← Tier 2: DeepEval LLM-judge metrics
├── trulens_metrics.py       ← Tier 3: TruLens RAG Triad
├── report_writer.py         ← Combines results → JSON + console summary
├── datasets/
│   └── golden_dataset.json  ← Curated test cases with ground-truth context
└── results/                 ← Generated JSON reports + TruLens SQLite DB
```

---

## Evaluation Tiers

### Tier 1 — Structural (Rule-Based, Zero LLM Cost)

Fast deterministic checks that run first.

| Metric | What it checks |
|---|---|
| `report_completeness` | Word counts, section count, findings, recommendations |
| `schema_compliance` | Report / insights / review JSON key presence and types |
| `source_coverage` | Sources retrieved, URLs cited, deduplication |
| `critic_quality_gate` | The pipeline's own internal Critic agent score |
| `agent_step_coverage` | All 5 nodes (planner → researcher → analyst → writer → critic) executed |
| `iteration_efficiency` | Convergence speed (fewer revision loops = higher score) |

### Tier 2 — DeepEval (LLM-as-Judge)

Uses GPT-4o as a judge model to assess semantic quality.

| Metric | Threshold | What it measures |
|---|---|---|
| `AnswerRelevancyMetric` | 0.65 | Is the report on-topic for the research query? |
| `FaithfulnessMetric` | 0.60 | Are claims grounded in the retrieved sources? |
| `ContextualRelevancyMetric` | 0.55 | Did the retriever fetch relevant source chunks? |
| `HallucinationMetric` | ≤ 0.40 | Does the report fabricate facts not in sources? |
| `GEval (Coherence)` | 0.60 | Is the report logically well-structured? |
| `GEval (AnalysisDepth)` | 0.55 | Does the report go beyond surface-level summaries? |
| `KeywordCoverage` | 0.60 | Are topic-specific keywords present? (rule-based) |

### Tier 3 — TruLens RAG Triad

The canonical RAG evaluation framework.

| Metric | Threshold | What it measures |
|---|---|---|
| `ContextRelevance` | 0.50 | Are the retrieved source chunks relevant to the query? |
| `Groundedness` | 0.55 | Is the answer supported by the retrieved context? |
| `AnswerRelevance` | 0.60 | Does the report's executive summary address the query? |

TruLens traces are saved to `results/trulens_eval.db`. View the dashboard with:
```bash
trulens-eval
```

---

## Installation

Install the following packages into the project's virtual environment:

```bash
pip install deepeval trulens trulens-providers-openai httpx python-dotenv
```

> **Note:** The existing `requirements.txt` packages (`openai`, `fastapi`, etc.) must already be installed.

---

## Usage

### Prerequisites

1. **Start the backend server** (in a separate terminal):
   ```bash
   cd agent-backend
   .venv\Scripts\activate        # Windows
   python -m uvicorn app.main:app --reload
   ```

2. **Ensure `.env` is configured** with at least `OPENAI_API_KEY`.

### Run the full suite (all 5 test cases, all 3 tiers)

```bash
cd agent-backend
.venv\Scripts\activate
python evaluation\evaluate.py
```

### Common options

```bash
# Run a single test case only
python evaluation\evaluate.py --ids td_001

# Run two cases at quick depth (faster, cheaper)
python evaluation\evaluate.py --ids td_001,td_002 --depth quick

# Skip LLM-judge metrics (Tier 2 + 3) — pure structural checks only
python evaluation\evaluate.py --skip-deepeval --skip-trulens

# Override backend URL (e.g. staging)
python evaluation\evaluate.py --backend-url http://staging.example.com:8000

# Validate installation without hitting the backend
python evaluation\evaluate.py --dry-run
```

---

## Output

After a run you'll find:
- **Console**: colour-coded per-metric pass/fail summary
- **`results/eval_report_<timestamp>.json`**: full machine-readable report
- **`results/trulens_eval.db`**: TruLens SQLite trace database

### JSON report structure

```jsonc
{
  "run_timestamp": "2026-10-18T22:00:00Z",
  "summary": {
    "test_case_count": 5,
    "passed": 4,
    "failed": 1,
    "pass_rate": 0.80,
    "mean_latency_seconds": 42.3
  },
  "results": [
    {
      "test_case_id": "td_001",
      "overall_passed": true,
      "structural": { "overall_score": 0.95, ... },
      "deepeval":    { "overall_passed": true, "metrics": [...] },
      "trulens":     { "overall_passed": true, "metrics": [...] }
    }
  ]
}
```

---

## CI Integration

DeepEval results can be integrated into a pytest / CI pipeline:

```bash
deepeval test run evaluation/
```

The exit code of `evaluate.py` is `0` if all tests pass, `1` if any fail — suitable for CI quality gates.

---

## Adding Test Cases

Edit [`datasets/golden_dataset.json`](datasets/golden_dataset.json). Each entry requires:

```jsonc
{
  "id": "td_006",
  "topic": "Your research topic",
  "depth": "standard",                     // quick | standard | deep
  "expected_keywords": ["keyword1", ...],  // for KeywordCoverage
  "ground_truth_context": ["fact 1", ...], // for Faithfulness / Groundedness
  "min_sections": 3,
  "min_findings": 3,
  "min_recommendations": 2,
  "min_sources": 2
}
```
