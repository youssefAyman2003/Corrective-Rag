from typing import List, Sequence
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of our Corrective RAG graph.

    Attributes:
        question: The user input or rewritten question.
        generation: The generated LLM answer.
        web_search: Flag ("Yes" or "No") indicating if web search was triggered.
        documents: List of retrieved / filtered / web-searched documents.
    """

    question: str
    generation: str
    web_search: str
    documents: List[Document]
