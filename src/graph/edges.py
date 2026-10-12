import logging
from typing import Literal
from src.models.state import GraphState

logger = logging.getLogger(__name__)


def decide_to_generate(state: GraphState) -> Literal["transform_query", "generate"]:
    """
    Assesses graded documents to decide whether to transform query and do web search
    or directly generate an answer.

    Args:
        state: Current GraphState.

    Returns:
        Next node name: 'transform_query' or 'generate'.
    """
    logger.info("---ASSESS GRADED DOCUMENTS---")
    web_search = state.get("web_search", "No")

    if web_search == "Yes":
        logger.info(
            "---DECISION: DOCUMENTS NOT RELEVANT (OR EMPTY), ROUTE TO TRANSFORM QUERY & WEB SEARCH---"
        )
        return "transform_query"
    else:
        logger.info("---DECISION: RELEVANT DOCUMENTS PRESENT, ROUTE TO GENERATE---")
        return "generate"
