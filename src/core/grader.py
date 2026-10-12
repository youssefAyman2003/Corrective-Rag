import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from src.core.llm import get_llm
from src.models.schemas import GradeDocuments

logger = logging.getLogger(__name__)

GRADE_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""


def get_retrieval_grader(llm=None) -> Runnable:
    """
    Creates and returns the document relevance grading chain.
    Uses structured output with GradeDocuments model.
    """
    if llm is None:
        llm = get_llm()

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GRADE_SYSTEM_PROMPT),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    return grade_prompt | structured_llm_grader
