import logging
from typing import Any, Dict
from langchain_core.documents import Document
from src.models.state import GraphState

logger = logging.getLogger(__name__)


def retrieve_node(state: GraphState, retriever: Any) -> Dict[str, Any]:
    """
    Retrieve documents relevant to the current question.

    Args:
        state: Current GraphState.
        retriever: VectorStoreRetriever.

    Returns:
        Updated state dictionary with retrieved documents.
    """
    logger.info("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    logger.info("Retrieved %d documents for question: '%s'", len(documents), question)
    return {"documents": documents, "question": question}


def grade_documents_node(state: GraphState, grader: Any) -> Dict[str, Any]:
    """
    Determines whether retrieved documents are relevant to the question.
    Filters out irrelevant documents and flags web_search if any doc is irrelevant.

    Args:
        state: Current GraphState.
        grader: Document relevance grader chain.

    Returns:
        Updated state dictionary with filtered documents and web_search flag.
    """
    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs = []
    web_search = "No"

    for d in documents:
        doc_content = d.page_content if hasattr(d, "page_content") else str(d)
        try:
            score = grader.invoke({"question": question, "document": doc_content})
            grade = getattr(score, "binary_score", None) or (
                score.get("binary_score") if isinstance(score, dict) else str(score)
            )
        except Exception as exc:
            logger.error("Error grading document: %s. Defaulting to relevant.", exc)
            grade = "yes"

        if str(grade).strip().lower() == "yes":
            logger.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"

    # If no documents are relevant, also ensure web_search is triggered
    if not filtered_docs:
        logger.info("---NO RELEVANT DOCUMENTS FOUND; TRIGGERING WEB SEARCH---")
        web_search = "Yes"

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }


def transform_query_node(state: GraphState, rewriter: Any) -> Dict[str, Any]:
    """
    Transforms the query to formulate an improved version for web search.

    Args:
        state: Current GraphState.
        rewriter: Question rewriter chain.

    Returns:
        Updated state dictionary with rewritten question.
    """
    logger.info("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state.get("documents", [])

    better_question = rewriter.invoke({"question": question})
    logger.info("Transformed query: '%s' -> '%s'", question, better_question)
    return {"documents": documents, "question": better_question}


def web_search_node(state: GraphState, search_tool: Any) -> Dict[str, Any]:
    """
    Executes web search for the rewritten question and appends results.

    Args:
        state: Current GraphState.
        search_tool: Search tool (e.g., TavilySearchResults or SafeWebSearchWrapper).

    Returns:
        Updated state dictionary with appended web search document.
    """
    logger.info("---WEB SEARCH---")
    question = state["question"]
    documents = list(state.get("documents", []))

    try:
        search_results = search_tool.invoke({"query": question})
        if isinstance(search_results, list):
            web_results = "\n\n".join(
                [d.get("content", str(d)) for d in search_results if isinstance(d, dict)]
            )
        else:
            web_results = str(search_results)
    except Exception as exc:
        logger.error("Web search invocation failed: %s", exc)
        web_results = f"[Web search failed: {exc}]"

    web_doc = Document(
        page_content=web_results,
        metadata={"source": "tavily_web_search", "query": question},
    )
    documents.append(web_doc)
    logger.info("Appended web search results to document list. Total docs: %d", len(documents))

    return {"documents": documents, "question": question}


def generate_node(state: GraphState, rag_chain: Any) -> Dict[str, Any]:
    """
    Generates answer using RAG chain given retrieved/web-searched documents.

    Args:
        state: Current GraphState.
        rag_chain: RAG answer generation chain.

    Returns:
        Updated state dictionary with generated answer.
    """
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])

    generation = rag_chain.invoke({"context": documents, "question": question})
    logger.info("Generated answer for question '%s'", question)
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }
