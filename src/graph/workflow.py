import logging
from typing import Any, Optional
from langgraph.graph import END, START, StateGraph
from src.core.generator import get_rag_chain
from src.core.grader import get_retrieval_grader
from src.core.llm import get_llm
from src.core.rewriter import get_question_rewriter
from src.core.search import get_web_search_tool
from src.core.vectorstore import get_retriever
from src.graph.edges import decide_to_generate
from src.graph.nodes import (
    generate_node,
    grade_documents_node,
    retrieve_node,
    transform_query_node,
    web_search_node,
)
from src.models.state import GraphState

logger = logging.getLogger(__name__)


def build_crag_app(
    retriever: Optional[Any] = None,
    llm: Optional[Any] = None,
    grader: Optional[Any] = None,
    generator: Optional[Any] = None,
    rewriter: Optional[Any] = None,
    search_tool: Optional[Any] = None,
):
    """
    Constructs and compiles the Corrective RAG (CRAG) StateGraph workflow.

    Args:
        retriever: Optional pre-configured vectorstore retriever.
        llm: Optional Groq LLM instance.
        grader: Optional document relevance grading chain.
        generator: Optional RAG answer generation chain.
        rewriter: Optional query rewriter chain.
        search_tool: Optional web search tool.

    Returns:
        Compiled LangGraph application.
    """
    # Initialize components if not provided
    shared_llm = llm or get_llm()
    shared_retriever = retriever or get_retriever()
    shared_grader = grader or get_retrieval_grader(shared_llm)
    shared_generator = generator or get_rag_chain(shared_llm)
    shared_rewriter = rewriter or get_question_rewriter(shared_llm)
    shared_search = search_tool or get_web_search_tool()

    workflow = StateGraph(GraphState)

    # Define nodes with closures binding their dependencies
    def retrieve_step(state: GraphState):
        return retrieve_node(state, shared_retriever)

    def grade_step(state: GraphState):
        return grade_documents_node(state, shared_grader)

    def transform_query_step(state: GraphState):
        return transform_query_node(state, shared_rewriter)

    def web_search_step(state: GraphState):
        return web_search_node(state, shared_search)

    def generate_step(state: GraphState):
        return generate_node(state, shared_generator)

    workflow.add_node("retrieve", retrieve_step)
    workflow.add_node("grade_documents", grade_step)
    workflow.add_node("transform_query", transform_query_step)
    workflow.add_node("web_search_node", web_search_step)
    workflow.add_node("generate", generate_step)

    # Define graph edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    logger.info("Compiled Corrective RAG LangGraph workflow.")
    return app
