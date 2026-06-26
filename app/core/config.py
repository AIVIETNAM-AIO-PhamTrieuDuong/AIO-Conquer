from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load application, Redis, vector memory, and model configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI-compatible LLM endpoint (set in .env)
    ninerouter_url: str
    ninerouter_model: str
    ninerouter_key: Optional[str] = None

    redis_url: str = "redis://localhost:6379"
    redis_vector_url: str = "redis://localhost:6380"
    redis_vector_index: str = "aio_conquer_vector_memory"
    redis_domain_url: str = "redis://localhost:6381"
    session_ttl: int = 3600
    max_history_turns: int = 5

    pinecone_api_key: Optional[str] = None
    pinecone_index: Optional[str] = None


settings = Settings()
