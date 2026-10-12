import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from config.settings import Settings, get_settings
from src.models.schemas import (
    GradeDocuments,
    QueryRequest,
    QueryResponse,
    IndexRequest,
    IndexResponse,
    HealthResponse,
)
from src.models.state import GraphState
from src.graph.edges import decide_to_generate
from src.graph.workflow import build_crag_app
from main import app


def test_settings_defaults():
    settings = get_settings()
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.ollama_model == "bge-m3:latest"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert len(settings.default_urls) == 3


def test_schemas_validation():
    # GradeDocuments
    grade = GradeDocuments(binary_score="yes")
    assert grade.binary_score == "yes"

    # QueryRequest
    req = QueryRequest(question="What is CRAG?")
    assert req.question == "What is CRAG?"

    # QueryResponse
    res = QueryResponse(
        question="What is CRAG?",
        generation="CRAG is Corrective RAG.",
        web_search_used=False,
        documents=["chunk1"],
    )
    assert res.web_search_used is False
    assert len(res.documents) == 1


def test_decide_to_generate_edges():
    # When web_search is 'Yes', route to transform_query
    state_search: GraphState = {
        "question": "test query",
        "generation": "",
        "web_search": "Yes",
        "documents": [],
    }
    assert decide_to_generate(state_search) == "transform_query"

    # When web_search is 'No', route to generate
    state_generate: GraphState = {
        "question": "test query",
        "generation": "",
        "web_search": "No",
        "documents": [],
    }
    assert decide_to_generate(state_generate) == "generate"


def test_graph_compilation_with_mocks():
    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    mock_grader = MagicMock()
    mock_generator = MagicMock()
    mock_rewriter = MagicMock()
    mock_search = MagicMock()

    app_instance = build_crag_app(
        retriever=mock_retriever,
        llm=mock_llm,
        grader=mock_grader,
        generator=mock_generator,
        rewriter=mock_rewriter,
        search_tool=mock_search,
    )
    assert app_instance is not None
    mermaid_diagram = app_instance.get_graph().draw_mermaid()
    assert "retrieve" in mermaid_diagram
    assert "grade_documents" in mermaid_diagram
    assert "transform_query" in mermaid_diagram
    assert "web_search_node" in mermaid_diagram
    assert "generate" in mermaid_diagram


def test_fastapi_endpoints():
    client = TestClient(app)

    # Test root endpoint
    res_root = client.get("/")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["status"] == "online"
    assert "Groq (llama-3.3-70b-versatile)" in data_root["models"]["llm"]
    assert "Ollama (bge-m3:latest)" in data_root["models"]["embeddings"]

    # Test health endpoint
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert data_health["groq_model"] == "llama-3.3-70b-versatile"
    assert data_health["ollama_model"] == "bge-m3:latest"

    # Test empty question validation
    res_bad_query = client.post("/query", json={"question": "   "})
    assert res_bad_query.status_code == 400

    # Test graph endpoint
    res_graph = client.get("/graph")
    assert res_graph.status_code == 200
    assert "retrieve" in res_graph.text

    # Test query endpoint with mocked CRAG execution
    mock_crag_response = QueryResponse(
        question="What is memory?",
        generation="Memory is storage.",
        web_search_used=False,
        documents=["sample doc"],
    )
    with patch("main.crag_service.query", return_value=mock_crag_response):
        res_query = client.post("/query", json={"question": "What is memory?"})
        assert res_query.status_code == 200
        data_query = res_query.json()
        assert data_query["question"] == "What is memory?"
        assert data_query["generation"] == "Memory is storage."
        assert data_query["web_search_used"] is False

    # Test index endpoint with mocked indexing
    mock_index_response = IndexResponse(
        message="Indexed successfully",
        urls_indexed=["https://example.com"],
        chunks_count=5,
    )
    with patch("main.crag_service.index_urls", return_value=mock_index_response):
        res_index = client.post("/index", json={"urls": ["https://example.com"]})
        assert res_index.status_code == 200
        data_index = res_index.json()
        assert data_index["chunks_count"] == 5
