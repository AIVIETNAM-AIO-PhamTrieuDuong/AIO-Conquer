from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.memory.eda_store import eda_store
from app.memory.vector_store import vector_store
from app.core.pipeline import SESSION_ID, reset_conversation_thread
from app.retrieval.embedder import embed

router = APIRouter(prefix="/dev", tags=["dev"])


@router.post("/reset")
async def reset_session(thread_id: str = SESSION_ID) -> dict:
    """Clear checkpointed conversation history for a LangGraph thread."""
    await reset_conversation_thread(thread_id)
    return {"status": "ok", "message": "Thread cleared.", "thread_id": thread_id}


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


# ---------------------------------------------------------------------------
# EDA RAG inspection — verify chunking + embedding landed in Redis
# ---------------------------------------------------------------------------


@router.get("/eda/active")
async def eda_active() -> dict:
    """Show the active EDA job_id bound to the default session."""
    job_id = await eda_store.get_active_eda(SESSION_ID)
    status = await eda_store.get_eda_status(job_id) if job_id else None
    return {"session_id": SESSION_ID, "active_job_id": job_id, "status": status}


async def _resolve_job_id(job_id: str | None) -> str:
    """Use the given job_id, or fall back to the session's active job."""
    resolved = job_id or await eda_store.get_active_eda(SESSION_ID)
    if not resolved:
        raise HTTPException(status_code=404, detail="No job_id given and no active EDA job.")
    return resolved


@router.get("/eda/chunks/{job_id}")
async def eda_chunks(job_id: str, preview_chars: int = 200) -> dict:
    """Inspect EDA summary chunks stored in Redis vector memory."""
    chunks = await vector_store.list_chunks(
        job_id=job_id,
        memory_types=["eda_summary"],
    )
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No vector chunks stored for job {job_id}.",
        )
    return {
        "job_id": job_id,
        "num_chunks": len(chunks),
        "embedding_dim": chunks[0].get("embedding_dim", 0),
        "chunks": [
            {
                "index": i,
                "key": chunk.get("key"),
                "memory_type": chunk.get("memory_type"),
                "source_type": chunk.get("source_type"),
                "title": chunk.get("title"),
                "length": len(chunk.get("text", "")),
                "preview": chunk.get("text", "")[:preview_chars],
                "embedding_dim": chunk.get("embedding_dim", 0),
            }
            for i, chunk in enumerate(chunks)
        ],
    }


class SearchRequest(BaseModel):
    question: str
    job_id: str | None = None
    top_k: int = 3
    preview_chars: int = 300


@router.post("/eda/search")
async def eda_search(req: SearchRequest) -> dict:
    """Semantic search test against Redis vector EDA summary memory."""
    job_id = await _resolve_job_id(req.job_id)
    results = await vector_store.search(
        job_id=job_id,
        query=req.question,
        memory_types=["eda_summary"],
        top_k=req.top_k,
    )
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No vector chunks stored for job {job_id}.",
        )
    return {
        "job_id": job_id,
        "question": req.question,
        "top_k": req.top_k,
        "total_results": len(results),
        "results": [
            {
                "index": i,
                "score": result.get("score"),
                "distance": result.get("distance"),
                "memory_type": result.get("memory_type"),
                "source_type": result.get("source_type"),
                "preview": result.get("text", "")[:req.preview_chars],
            }
            for i, result in enumerate(results)
        ],
    }


@router.post("/eda/retrieve")
async def eda_retrieve(req: SearchRequest) -> dict:
    """Run the Redis vector retrieval path used by EDA memory."""
    job_id = await _resolve_job_id(req.job_id)
    results = await vector_store.search(
        job_id=job_id,
        query=req.question,
        memory_types=["eda_summary"],
        top_k=req.top_k,
    )
    return {
        "job_id": job_id,
        "question": req.question,
        "backend": "redis-vector",
        "results": [item.get("text", "")[: req.preview_chars] for item in results],
    }
