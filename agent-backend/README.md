# Deep Research Desk — Backend

A multi-agent AI system that autonomously researches any topic and writes a comprehensive report.

## What it does

1. **Planner** — Breaks your question into sub-topics and generates search queries  
2. **Researcher** — Runs web searches and gathers sources  
3. **Analyst** — Extracts facts, spots contradictions, and scores confidence  
4. **Writer** — Produces a structured executive report  
5. **Critic** — Scores the report; if quality is too low it loops back for revision  

## Project Structure

```
agent-backend/
├── app/
│   ├── main.py       # FastAPI app — all API routes live here
│   ├── config.py     # Settings (from .env) and logging setup
│   ├── models.py     # OpenAI / Gemini LLM and Embedding wrappers
│   ├── prompts.py    # System prompts + Pydantic validation schemas
│   ├── agents.py     # Agent class + 5 workflow node functions
│   ├── graph.py      # LangGraph state machine definition
│   ├── tools.py      # Web search (Tavily / mock) and Summarize tools
│   ├── memory.py     # Qdrant vector memory client + retriever
│   └── services.py   # Session store + research orchestrator
├── requirements.txt
├── .env              # Your local secrets (not committed to git)
└── .env.example      # Template — copy to .env and fill in keys
```

## Quick Start

### 1. Install dependencies
```bash
cd agent-backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Open .env and add your API keys
```

The key settings you need to fill in:
| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |
| `TAVILY_API_KEY` | Tavily search key — or set `SEARCH_PROVIDER=placeholder` to skip |
| `QDRANT_HOST` / `QDRANT_PORT` | Only needed if `ENABLE_MEMORY=true` |

### 3. Run the server
```bash
python -m uvicorn app.main:app --reload
```

Server starts at **http://localhost:8000**

### 4. Explore the API docs
Open **http://localhost:8000/docs** in your browser for interactive Swagger docs.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/research/` | Start a research task |
| `GET` | `/api/research/{id}` | Get task status + report |
| `GET` | `/api/research/{id}/state` | Get live agent step details |
| `GET` | `/api/memory/count` | Count stored memories |
| `GET` | `/api/memory/search?q=...` | Search past research |
| `DELETE` | `/api/memory/{id}` | Delete a memory entry |

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `SEARCH_PROVIDER` | `tavily` | `tavily` or `placeholder` (no key needed) |
| `ENABLE_MEMORY` | `false` | Requires Qdrant running locally |
| `MAX_ITERATIONS` | `3` | Max revision loops |
| `CRITIC_THRESHOLD` | `7.0` | Quality score (0–10) before auto-approving |

---

## Tech Stack

- **FastAPI** — REST API
- **LangGraph** — Agent workflow orchestration  
- **OpenAI** — Primary LLM (GPT-4 Turbo) and embeddings  
- **Qdrant** — Vector memory database (optional)  
- **Tavily** — Web search API (optional — falls back to placeholder)
