import logging
from typing import Any, Dict, List, Optional
from config.settings import get_settings
from src.core.vectorstore import (
    build_vectorstore,
    get_retriever,
    get_vectorstore,
    load_vectorstore,
)
from src.graph.workflow import build_crag_app
from src.models.schemas import IndexResponse, QueryResponse

logger = logging.getLogger(__name__)


class CRAGService:
    """
    High-level service coordinating vector indexing, state graph execution,
    and response synthesis for the Corrective RAG pipeline.
    """

    def __init__(self):
        self.settings = get_settings()
        self._app: Optional[Any] = None
        self._retriever: Optional[Any] = None

    def initialize(self, force_reindex: bool = False, auto_build: bool = False) -> None:
        """
        Initializes the vectorstore and compiles the CRAG graph.
        """
        logger.info("Initializing CRAG Service...")
        vs = None
        if not force_reindex:
            vs = load_vectorstore()

        if vs is None and auto_build:
            logger.info("No persisted index found; building vectorstore from default URLs...")
            try:
                vs = build_vectorstore(self.settings.default_urls)
            except Exception as exc:
                logger.warning("Could not auto-build vectorstore on init: %s", exc)

        self._retriever = get_retriever(vectorstore=vs, auto_build=False)
        self._app = build_crag_app(retriever=self._retriever)
        logger.info("CRAG Service initialized successfully.")

    @property
    def app(self):
        """Returns the compiled LangGraph application, initializing lazily if needed."""
        if self._app is None:
            self.initialize()
        return self._app

    def is_ready(self) -> bool:
        """Checks if the vectorstore is ready."""
        return get_vectorstore(auto_build=False) is not None

    def index_urls(self, urls: Optional[List[str]] = None) -> IndexResponse:
        """
        Ingests and chunks documents from specified URLs and builds/rebuilds the FAISS index.
        """
        target_urls = urls if urls and len(urls) > 0 else self.settings.default_urls
        logger.info("Re-indexing vectorstore with URLs: %s", target_urls)

        vs = build_vectorstore(target_urls, persist=True)
        self._retriever = get_retriever(vectorstore=vs)
        # Re-compile workflow with updated retriever
        self._app = build_crag_app(retriever=self._retriever)

        # Count total documents/vectors in FAISS
        total_chunks = vs.index.ntotal if hasattr(vs, "index") else len(target_urls)

        return IndexResponse(
            message=f"Successfully indexed {len(target_urls)} URLs.",
            urls_indexed=target_urls,
            chunks_count=total_chunks,
        )

    def query(self, question: str) -> QueryResponse:
        """
        Executes a question through the Corrective RAG LangGraph workflow.
        """
        app = self.app
        initial_state = {"question": question.strip()}

        logger.info("Invoking CRAG pipeline for question: '%s'", question)
        final_state: Dict[str, Any] = app.invoke(initial_state)

        generation = final_state.get("generation", "No generation produced.")
        web_search_val = final_state.get("web_search", "No")
        web_search_used = str(web_search_val).strip().lower() == "yes"

        raw_docs = final_state.get("documents", [])
        doc_contents = []
        for doc in raw_docs:
            if hasattr(doc, "page_content"):
                doc_contents.append(doc.page_content)
            else:
                doc_contents.append(str(doc))

        return QueryResponse(
            question=question,
            generation=generation,
            web_search_used=web_search_used,
            documents=doc_contents,
        )

    def get_graph_mermaid(self) -> str:
        """Returns the Mermaid graph representation of the workflow."""
        try:
            return self.app.get_graph().draw_mermaid()
        except Exception as exc:
            logger.warning("Could not draw mermaid graph: %s", exc)
            return ""


# Global singleton instance
crag_service = CRAGService()
