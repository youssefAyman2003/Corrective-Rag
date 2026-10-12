from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Groq LLM Configuration
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.0, alias="GROQ_TEMPERATURE")

    # Ollama Embeddings Configuration
    ollama_model: str = Field(default="bge-m3:latest", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # Tavily Search Configuration
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    tavily_k: int = Field(default=3, alias="TAVILY_K")

    # Document Chunking and Retrieval
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=0, alias="CHUNK_OVERLAP")
    retriever_k: int = Field(default=4, alias="RETRIEVER_K")

    # Default URLs to index if none specified
    default_urls: List[str] = Field(
        default=[
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ],
        alias="DEFAULT_URLS",
    )

    # Vectorstore Persistence Directory
    vectorstore_path: str = Field(default="data/faiss_index", alias="VECTORSTORE_PATH")

    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
