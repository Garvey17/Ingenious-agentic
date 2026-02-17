# Deep Research Desk - Backend

A production-style Agentic AI system for autonomous research workflows.

## 🎯 Project Overview

Deep Research Desk is a multi-agent AI system that collaborates to conduct comprehensive research on any topic. A team of specialized AI agents work together to:

- 🔍 Search the web for relevant information
- 📊 Analyze findings and extract insights
- ✍️ Generate structured executive reports
- 🔄 Critique and iteratively improve outputs

## 🏗️ Architecture

### Orchestration Layer
- **LangGraph** controls the workflow state and agent execution order
- Graph flow: `START → Planner → Researcher → Analyst → Writer → Critic`
- Iterative loop: If quality score is low, loop back to Researcher

### Agent Design (CrewAI-style roles)
Each agent is a specialized role with structured JSON outputs:

- **Researcher Agent**: Searches the web and gathers sources (via MCP tools)
- **Analyst Agent**: Extracts facts, identifies contradictions, scores confidence
- **Writer Agent**: Converts insights into executive-style reports
- **Critic Agent**: Evaluates quality and decides on approval/revision

### MCP (Model Context Protocol)
All external tools are accessed through MCP servers:
- Agents → MCP Client → MCP Servers → Tools
- Initial tools: `search_web`, `store_memory`, `retrieve_memory`

### Tech Stack
- **Python** - Core language
- **FastAPI** - REST API framework
- **LangGraph** - Workflow orchestration
- **OpenAI Agents SDK** - Agent implementation
- **Gemini/GPT** - LLM provider
- **Qdrant** - Vector memory database
- **Tavily** - Web search API

## 📁 Project Structure

```
agent-backend/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── api/
│   │   └── research.py         # Research endpoints
│   ├── graph/
│   │   ├── state.py            # Shared workflow state
│   │   ├── builder.py          # LangGraph setup
│   │   └── nodes/              # Graph nodes
│   ├── agents/
│   │   ├── base_agent.py       # Base agent class
│   │   └── ...                 # Specialized agents
│   ├── mcp/
│   │   ├── client.py           # MCP client
│   │   └── servers/            # MCP servers
│   ├── memory/
│   │   ├── qdrant_client.py    # Vector DB client
│   │   └── retriever.py        # Memory retrieval
│   ├── models/
│   │   ├── llm.py              # LLM abstraction
│   │   └── embeddings.py       # Embedding models
│   ├── prompts/
│   │   ├── system_prompts.py   # Agent prompts
│   │   └── schemas.py          # Pydantic schemas
│   └── config/
│       ├── settings.py         # Configuration
│       └── logging.py          # Logging setup
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- API Keys:
  - Google Gemini API key (or OpenAI)
  - Tavily API key

### Installation

1. **Clone the repository**
```bash
cd agent-backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Run the application**
```bash
# Development mode
python app/main.py

# Or with uvicorn
uvicorn app.main:app --reload
```

6. **Using Docker (optional)**
```bash
cd docker
docker-compose up -d
```

## 🔧 Configuration

Edit `.env` file with your settings:

```env
# LLM Provider
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_api_key_here

# Search API
TAVILY_API_KEY=your_tavily_key_here

# Qdrant (if using Docker)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Start Research
```bash
POST /api/research/
{
  "topic": "Artificial General Intelligence",
  "depth": "standard",
  "max_sources": 10
}
```

### Get Research Status
```bash
GET /api/research/{request_id}
```

## 🗺️ Development Roadmap

### ✅ Phase 1 — Core Infrastructure (Current)
- [x] Project bootstrap
- [x] Configuration layer
- [x] LLM abstraction (Gemini/OpenAI)
- [x] Pydantic schemas
- [x] Base agent class
- [x] FastAPI skeleton

### 🔄 Phase 2 — Agent Workflow Engine
- [ ] LangGraph workflow
- [ ] Researcher agent
- [ ] Analyst agent
- [ ] Writer agent
- [ ] Critic agent
- [ ] Orchestrator

### 🔄 Phase 3 — Memory + Vector DB
- [ ] Qdrant integration
- [ ] Memory manager
- [ ] Reflection loop

### 🔄 Phase 4 — MCP Integration
- [ ] MCP server implementation
- [ ] Tool integration

### 🔄 Phase 5 — API Enhancements
- [ ] Streaming responses
- [ ] Authentication
- [ ] WebSocket support

### 🔄 Phase 6 — Production Features
- [ ] State persistence
- [ ] Advanced logging
- [ ] Observability
- [ ] Token streaming

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=app
```

## 📝 License

MIT License

## 🤝 Contributing

This is a learning/portfolio project. Feel free to fork and experiment!

---

**Built with ❤️ using LangGraph, FastAPI, and Gemini AI**
