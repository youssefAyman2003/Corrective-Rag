import logging
from langchain_ollama import OllamaEmbeddings
from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_embeddings() -> OllamaEmbeddings:
    """
    Initializes and returns OllamaEmbeddings.
    Configured with model (bge-m3:latest) and Ollama base URL.
    """
    settings = get_settings()
    logger.info(
        "Initializing Ollama embeddings with model='%s' and base_url='%s'",
        settings.ollama_model,
        settings.ollama_base_url,
    )
    return OllamaEmbeddings(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )
