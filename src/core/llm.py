import logging
import os
from langchain_groq import ChatGroq
from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatGroq:
    """
    Initializes and returns a ChatGroq LLM instance.
    Defaults to model 'llama-3.3-70b-versatile'.
    """
    settings = get_settings()
    model = model_name or settings.groq_model
    temp = settings.groq_temperature if temperature is None else temperature
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.warning("GROQ_API_KEY is not set. Groq LLM calls will fail if an API key is required.")

    logger.info("Initializing Groq LLM with model='%s', temperature=%s", model, temp)
    return ChatGroq(
        model=model,
        temperature=temp,
        groq_api_key=api_key or None,
    )
