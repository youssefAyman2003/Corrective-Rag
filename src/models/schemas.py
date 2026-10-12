from typing import List, Optional
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class QueryRequest(BaseModel):
    """Request schema for querying the Corrective RAG pipeline."""

    question: str = Field(..., min_length=1, description="Question to be answered by the CRAG system")


class QueryResponse(BaseModel):
    """Response schema returned by the Corrective RAG pipeline."""

    question: str
    generation: str
    web_search_used: bool
    documents: List[str] = Field(default_factory=list)


class IndexRequest(BaseModel):
    """Request schema for indexing documents from URLs into FAISS."""

    urls: List[str] = Field(
        default_factory=list,
        description="List of web page URLs to scrape and index. Defaults to preconfigured URLs if empty.",
    )


class IndexResponse(BaseModel):
    """Response schema after indexing documents."""

    message: str
    urls_indexed: List[str]
    chunks_count: int


class HealthResponse(BaseModel):
    """Health status and configuration info."""

    status: str
    groq_model: str
    ollama_model: str
    vectorstore_ready: bool
