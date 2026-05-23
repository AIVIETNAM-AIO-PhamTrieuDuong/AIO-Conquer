from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 9Router
    ninerouter_url: str = "http://45.128.222.24:20128/v1"
    ninerouter_model: str = "ragas_experiments"
    ninerouter_key: Optional[str] = None

    # Legacy keys kept so existing .env files don't break on startup
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"

    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 3600
    max_history_turns: int = 5

    pinecone_api_key: Optional[str] = None
    pinecone_index: Optional[str] = None
    embed_model: str = "text-embedding-004"


settings = Settings()
