from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration using Pydantic BaseSettings.

    Automatically loads from environment variables or .env file.
    Production deployments should use environment variables.
    """

    # API Configuration
    app_name: str = "Document Summarizer API"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Database Configuration
    database_url: str = "postgresql://user:password@localhost:5432/doc_summarizer"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Cache Configuration (In-Memory)
    cache_ttl: int = 3600  # 1 hour in seconds

    # Vector Database Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents"
    qdrant_api_key: Optional[str] = None
    vector_dimension: int = 1536  # OpenAI ada-002 embedding dimension
    similarity_threshold: float = 0.8
    max_similar_documents: int = 5

    # LLM Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-ada-002"
    max_tokens_per_request: int = 4000
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Feature flags
    enable_ai_summarization: bool = True

    # Processing Configuration
    max_document_size: int = 50 * 1024 * 1024  # 50MB
    supported_formats: list[str] = ["pdf", "docx", "txt"]
    async_task_timeout: int = 300  # 5 minutes

    # Long-term Memory Configuration
    enable_memory: bool = True
    memory_similarity_boost: float = 0.1  # Boost confidence when similar docs found
    memory_context_window: int = 3  # Number of similar docs to consider

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance to avoid re-reading environment variables."""
    return Settings()
