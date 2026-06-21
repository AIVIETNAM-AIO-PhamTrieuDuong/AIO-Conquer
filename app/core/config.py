from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI-compatible LLM endpoint (set in .env)
    ninerouter_url: str
    ninerouter_model: str
    ninerouter_key: Optional[str] = None

    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 3600
    max_history_turns: int = 5

    pinecone_api_key: Optional[str] = None
    pinecone_index: Optional[str] = None
    embed_model: str = "text-embedding-3-small"


settings = Settings()
