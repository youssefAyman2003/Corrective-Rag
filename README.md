# Corrective RAG (CRAG) Service

An advanced Corrective Retrieval-Augmented Generation (CRAG) system that evaluates retrieved documents for relevance and dynamically self-corrects when retrieved evidence is insufficient. Integrates query refinement, Tavily web search fallback, and grounded LLM generation.

![Corrective RAG](corrective%20Rag.png)

---

## 🌟 Key Features

- **Text Generation (LLM)**: Powered by Groq using **`llama-3.3-70b-versatile`**.
- **Embeddings**: Powered by Ollama using **`bge-m3:latest`**.
- **Self-Correction & Evaluation**: Binary relevance grading of retrieved documents.
- **Dynamic Web Search**: If documents lack sufficient relevance, the query is rewritten and searched on Tavily before generation.
- **Modular Production Architecture**: Clean separation into `config/`, `src/core/`, `src/graph/`, `src/models/`, and `src/services/`.
- **FastAPI Backend**: RESTful endpoints for health checks, indexing URLs, and running CRAG queries.
- **Modern UI/UX**: Interactive web application in `frontend/` with real-time pipeline stepper, relevance decision badges, document inspection drawer, and live LangGraph visualization.
- **Docker Ready**: Production-grade `Dockerfile` and `docker-compose.yml`.

---

## 📁 Project Structure

```text
Corrective-Rag/
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized Pydantic application settings
├── frontend/
│   ├── index.html           # Modern glassmorphism UI layout
│   ├── style.css            # Dark mode, responsive design, animations
│   └── app.js               # Reactive client interacting with backend
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── embeddings.py    # OllamaEmbeddings factory (bge-m3:latest)
│   │   ├── generator.py     # RAG prompt and answer generator
│   │   ├── grader.py        # Document relevance grader (structured output)
│   │   ├── llm.py           # ChatGroq LLM factory (llama-3.3-70b-versatile)
│   │   ├── rewriter.py      # Question rewriter for web search query optimization
│   │   ├── search.py        # Tavily search integration with safe fallback
│   │   └── vectorstore.py   # Document loading, text splitting, FAISS index management
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── edges.py         # Conditional routing logic
│   │   ├── nodes.py         # Graph node implementations
│   │   └── workflow.py      # LangGraph StateGraph assembly and compilation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py       # Pydantic request, response, and grading models
│   │   └── state.py         # LangGraph GraphState TypedDict
│   └── services/
│       ├── __init__.py
│       └── crag_service.py  # Service layer coordinating indexing and query execution
├── Notebook/
│   └── 2-CorrectiveRAG.ipynb# Original experimental notebook
├── docker-compose.yml       # Docker Compose service definition
├── Dockerfile               # Multi-stage production container definition
├── main.py                  # FastAPI server entrypoint (serves API & /ui)
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # Pinned requirements for container and pip
├── .env.example             # Example environment configuration
└── README.md                # Documentation
```

---

## ⚙️ Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Configure your environment variables in `.env`:
```env
# Groq API Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.0

# Ollama Embeddings Configuration
OLLAMA_MODEL=bge-m3:latest
OLLAMA_BASE_URL=http://localhost:11434

# Tavily API Configuration (Web Search Fallback)
TAVILY_API_KEY=tvly-your_tavily_api_key_here
TAVILY_K=3

# Server Settings
HOST=0.0.0.0
PORT=8000
VECTORSTORE_PATH=data/faiss_index
```

---

## 🚀 Getting Started Locally

### 1. Ensure Ollama is Running with BGE-M3
```bash
ollama pull bge-m3:latest
ollama serve
```

### 2. Install Dependencies
Using `uv`:
```bash
uv sync
```
Or using standard `pip`:
```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI Server
```bash
uv run python main.py
```
Or:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger API documentation will be available at:
`http://localhost:8000/docs`

---

## 🐳 Running with Docker

### Using Docker Compose (Recommended)
This starts both the FastAPI backend and an Ollama container:
```bash
docker compose up --build -d
```
Then pull the embedding model inside the Ollama container:
```bash
docker compose exec ollama ollama pull bge-m3:latest
```

### Using Standalone Docker
```bash
docker build -t corrective-rag .
docker run -p 8000:8000 --env-file .env corrective-rag
```

---

## 📡 API Reference

### 1. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "groq_model": "llama-3.3-70b-versatile",
  "ollama_model": "bge-m3:latest",
  "vectorstore_ready": true
}
```

### 2. Query Corrective RAG
`POST /query`
**Request Body**:
```json
{
  "question": "What are the types of agent memory?"
}
```
**Response**:
```json
{
  "question": "What are the types of agent memory?",
  "generation": "The types of agent memory include Sensory Memory, Short-Term Memory (STM) / Working Memory, and Long-Term Memory (LTM). Long-term memory is further divided into Explicit / Declarative memory and Implicit / Procedural memory.",
  "web_search_used": false,
  "documents": [
    "...retrieved document chunk..."
  ]
}
```

### 3. Index Web Documents
`POST /index`
**Request Body**:
```json
{
  "urls": [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
  ]
}
```

### 4. Graph Architecture Diagram
`GET /graph`
Returns the Mermaid diagram of the StateGraph workflow.
