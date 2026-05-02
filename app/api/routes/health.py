from fastapi import APIRouter
from app.model.llm_client import llm
from app.memory.redis_client import memory

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    gemini_ok = await llm.is_alive()
    redis_ok = await memory.is_alive()
    return {
        "status": "ok" if (gemini_ok and redis_ok) else "degraded",
        "gemini": "up" if gemini_ok else "down",
        "redis": "up" if redis_ok else "down",
    }
