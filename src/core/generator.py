import logging
from typing import Sequence
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from src.core.llm import get_llm

logger = logging.getLogger(__name__)

# Standard RAG prompt equivalent to 'rlm/rag-prompt'
RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

RAG_HUMAN_PROMPT = """Question: {question}

Context: {context}

Answer:"""


def format_docs(docs: Sequence[Document | str]) -> str:
    """Format a sequence of documents into a single concatenated string."""
    formatted = []
    for doc in docs:
        if isinstance(doc, Document):
            formatted.append(doc.page_content)
        elif isinstance(doc, str):
            formatted.append(doc)
        elif hasattr(doc, "page_content"):
            formatted.append(str(doc.page_content))
        else:
            formatted.append(str(doc))
    return "\n\n".join(formatted)


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Returns the RAG prompt template.
    Attempts to pull 'rlm/rag-prompt' from LangChain Hub, falling back to local template.
    """
    try:
        from langchain import hub
        prompt = hub.pull("rlm/rag-prompt")
        logger.info("Successfully pulled 'rlm/rag-prompt' from LangChain Hub")
        return prompt
    except Exception as exc:
        logger.warning(
            "Could not pull 'rlm/rag-prompt' from hub (%s); using local standard prompt template.",
            exc,
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_HUMAN_PROMPT),
            ]
        )


def get_rag_chain(llm=None) -> Runnable:
    """
    Creates and returns the RAG answer generation chain:
    rag_chain = prompt | llm | StrOutputParser()
    """
    if llm is None:
        llm = get_llm()

    prompt = get_rag_prompt()
    return prompt | llm | StrOutputParser()
