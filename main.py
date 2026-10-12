import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse

from config.settings import get_settings
from src.models.schemas import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)
from src.services.crag_service import crag_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("crag-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for startup and shutdown.
    Attempts to initialize vectorstore or load persisted index on startup.
    """
    settings = get_settings()
    logger.info("Starting Corrective RAG API...")
    logger.info("Configured Groq model: %s", settings.groq_model)
    logger.info("Configured Ollama model: %s (base url: %s)", settings.ollama_model, settings.ollama_base_url)

    # Attempt to load existing index if available
    try:
        from src.core.vectorstore import load_vectorstore
        vs = load_vectorstore()
        if vs is not None:
            crag_service.initialize()
            logger.info("Persisted vectorstore index found and loaded successfully.")
        else:
            logger.info("No pre-existing vectorstore found. Vectorstore will initialize on first query or /index call.")
    except Exception as exc:
        logger.warning("Could not auto-load vectorstore on startup: %s", exc)

    yield

    logger.info("Shutting down Corrective RAG API.")


app = FastAPI(
    title="Corrective RAG (CRAG) API",
    description="Production-ready Corrective Retrieval-Augmented Generation API powered by Groq (LLaMA 3.3 70B) and Ollama Embeddings (BGE-M3).",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static UI if present
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")


@app.get("/", tags=["General"])
async def root(request: Request):
    """Root endpoint providing service information or redirecting browser to /ui."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and frontend_dir.exists():
        return RedirectResponse(url="/ui")

    settings = get_settings()
    return {
        "name": "Corrective RAG (CRAG) Service",
        "status": "online",
        "version": "1.0.0",
        "ui": "/ui",
        "models": {
            "llm": f"Groq ({settings.groq_model})",
            "embeddings": f"Ollama ({settings.ollama_model})",
        },
        "endpoints": {
            "ui": "/ui",
            "health": "/health",
            "query": "/query (POST)",
            "index": "/index (POST)",
            "graph": "/graph (GET)",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check() -> HealthResponse:
    """Health check endpoint to inspect configuration and readiness."""
    settings = get_settings()
    is_ready = crag_service.is_ready()
    return HealthResponse(
        status="healthy",
        groq_model=settings.groq_model,
        ollama_model=settings.ollama_model,
        vectorstore_ready=is_ready,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG"],
)
async def query_crag(request: QueryRequest) -> QueryResponse:
    """
    Submits a query to the Corrective RAG workflow.

    Workflow:
    1. Retrieve relevant chunks from local FAISS vectorstore.
    2. Grade document relevance using Groq (llama-3.3-70b-versatile).
    3. If any document is not relevant, rewrite the query and invoke Tavily web search.
    4. Generate the final answer using Groq with accumulated context.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        response = crag_service.query(request.question)
        return response
    except Exception as exc:
        logger.exception("Error executing CRAG query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CRAG execution failed: {str(exc)}",
        )


@app.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    tags=["Indexing"],
)
async def index_documents(request: IndexRequest) -> IndexResponse:
    """
    Scrapes documents from given URLs, chunks them, and builds or updates
    the FAISS vectorstore using Ollama embeddings (bge-m3:latest).
    """
    try:
        response = crag_service.index_urls(request.urls)
        return response
    except Exception as exc:
        logger.exception("Error indexing documents: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {str(exc)}",
        )


@app.get("/graph", response_class=PlainTextResponse, tags=["RAG"])
async def get_graph():
    """Returns the LangGraph Mermaid diagram definition."""
    try:
        mermaid_code = crag_service.get_graph_mermaid()
        if not mermaid_code:
            return "flowchart TD\n  START --> retrieve\n  retrieve --> grade_documents\n  grade_documents --> transform_query\n  grade_documents --> generate\n  transform_query --> web_search_node\n  web_search_node --> generate\n  generate --> END"
        return mermaid_code
    except Exception as exc:
        logger.error("Error retrieving graph diagram: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


if __name__ == "__main__":
    settings = get_settings()
    print("\n" + "=" * 65)
    print(" 🚀 Corrective RAG (CRAG) is starting up!")
    print(f" 👉 Open Web UI in Browser:   http://localhost:{settings.port}/ui")
    print(f"                             (or http://127.0.0.1:{settings.port}/ui)")
    print(f" 👉 API Documentation:        http://localhost:{settings.port}/docs")
    print(f" ⚠️  Note: Do NOT type '0.0.0.0' in Windows browsers (ERR_ADDRESS_INVALID)")
    print("=" * 65 + "\n")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
