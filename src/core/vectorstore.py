import logging
import os
from pathlib import Path
from typing import Any, List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import get_settings
from src.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_VECTORSTORE_INSTANCE: Optional[FAISS] = None


def load_and_split_urls(
    urls: List[str],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
):
    """
    Scrapes documents from given URLs and splits them into chunks.
    """
    settings = get_settings()
    c_size = chunk_size if chunk_size is not None else settings.chunk_size
    c_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    logger.info("Loading %d URLs: %s", len(urls), urls)
    docs_list = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            docs_list.extend(docs)
        except Exception as exc:
            logger.error("Failed to load URL '%s': %s", url, exc)

    if not docs_list:
        logger.warning("No documents were loaded from the provided URLs.")
        return []

    logger.info("Loaded %d raw document objects. Splitting text...", len(docs_list))
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=c_size, chunk_overlap=c_overlap
        )
    except Exception as exc:
        logger.warning(
            "Could not initialize tiktoken splitter (%s); using default RecursiveCharacterTextSplitter",
            exc,
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size, chunk_overlap=c_overlap
        )

    doc_splits = text_splitter.split_documents(docs_list)
    logger.info("Created %d chunks from %d URLs", len(doc_splits), len(urls))
    return doc_splits


def build_vectorstore(
    urls: Optional[List[str]] = None,
    persist: bool = True,
) -> FAISS:
    """
    Builds a new FAISS vectorstore from provided or default URLs and optionally persists it.
    """
    global _VECTORSTORE_INSTANCE
    settings = get_settings()
    target_urls = urls if urls and len(urls) > 0 else settings.default_urls

    doc_splits = load_and_split_urls(target_urls)
    embeddings = get_embeddings()

    if not doc_splits:
        # Fallback to an empty or placeholder document so vectorstore can still initialize
        from langchain_core.documents import Document
        doc_splits = [
            Document(
                page_content="Placeholder document for Corrective RAG vectorstore initialization.",
                metadata={"source": "system_init"},
            )
        ]

    logger.info("Building FAISS vectorstore with %d document chunks...", len(doc_splits))
    vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embeddings)

    if persist:
        save_vectorstore(vectorstore, settings.vectorstore_path)

    _VECTORSTORE_INSTANCE = vectorstore
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: str) -> None:
    """Saves FAISS index to the local filesystem."""
    try:
        persist_dir = Path(path)
        persist_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(persist_dir))
        logger.info("FAISS vectorstore saved locally to '%s'", path)
    except Exception as exc:
        logger.error("Failed to persist FAISS vectorstore to '%s': %s", path, exc)


def load_vectorstore(path: Optional[str] = None) -> Optional[FAISS]:
    """Loads FAISS index from the local filesystem if available."""
    global _VECTORSTORE_INSTANCE
    settings = get_settings()
    target_path = path or settings.vectorstore_path

    if Path(target_path).exists() and (Path(target_path) / "index.faiss").exists():
        try:
            logger.info("Loading existing FAISS vectorstore from '%s'...", target_path)
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(
                target_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            _VECTORSTORE_INSTANCE = vectorstore
            return vectorstore
        except Exception as exc:
            logger.error("Failed to load FAISS index from '%s': %s", target_path, exc)
    return None


def get_vectorstore(auto_build: bool = True) -> Optional[FAISS]:
    """
    Returns the currently active FAISS vectorstore.
    Loads from disk or builds automatically if not currently in memory.
    """
    global _VECTORSTORE_INSTANCE
    if _VECTORSTORE_INSTANCE is not None:
        return _VECTORSTORE_INSTANCE

    # Try loading from local storage
    vectorstore = load_vectorstore()
    if vectorstore is not None:
        return vectorstore

    if auto_build:
        return build_vectorstore()

    return None


class DummyRetriever:
    """Fallback retriever when no FAISS index is loaded yet."""

    def invoke(self, query: str) -> List[Document]:
        logger.warning(
            "Vectorstore index is not initialized. Returning empty documents. "
            "Call POST /index to build the vectorstore index."
        )
        return []


def get_retriever(
    k: Optional[int] = None,
    vectorstore: Optional[FAISS] = None,
    auto_build: bool = False,
) -> Any:
    """
    Returns a VectorStoreRetriever from the active or supplied vectorstore.
    Falls back to DummyRetriever if no index is loaded.
    """
    settings = get_settings()
    k_val = k if k is not None else settings.retriever_k

    vs = vectorstore or get_vectorstore(auto_build=auto_build)
    if vs is None:
        logger.info("No active vectorstore found; using DummyRetriever.")
        return DummyRetriever()

    return vs.as_retriever(search_kwargs={"k": k_val})
