from fastapi import APIRouter
from app.memory.redis_client import memory
from app.core.pipeline import SESSION_ID
from app.retrieval.embedder import embed

router = APIRouter(prefix="/dev", tags=["dev"])


@router.post("/reset")
async def reset_session() -> dict:
    """Clear conversation history. Dev-only — reset before a new demo run."""
    await memory.clear(SESSION_ID)
    return {"status": "ok", "message": "Session cleared."}


@router.get("/embed-test")
async def embed_test() -> dict:
    text = "Hello from fastembed test"
    vectors = await embed([text])
    v = vectors[0]
    return {
        "text": text,
        "dim": len(v),
        "preview": v[:5],
    }
