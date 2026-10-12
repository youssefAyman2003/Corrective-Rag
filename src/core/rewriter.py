import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from src.core.llm import get_llm

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You a question re-writer that converts an input question to a better version that is optimized
for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""


def get_question_rewriter(llm=None) -> Runnable:
    """
    Creates and returns the question re-writer chain.
    Transforms an ambiguous or retrieval-specific question into an optimized web query.
    """
    if llm is None:
        llm = get_llm()

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM_PROMPT),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    return re_write_prompt | llm | StrOutputParser()
