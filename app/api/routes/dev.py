from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.memory.redis_client import memory
from app.memory.eda_store import eda_store
from app.core.pipeline import SESSION_ID
from app.retrieval.embedder import embed
from app.retrieval.retriever import retriever, _cosine_similarity

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
    """Inspect chunks + embeddings stored in Redis for a job (text preview only)."""
    chunks, embeddings = await eda_store.get_eda_chunks(job_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks stored for job {job_id}.")
    return {
        "job_id": job_id,
        "num_chunks": len(chunks),
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "chunks": [
            {
                "index": i,
                "length": len(c),
                "preview": c[:preview_chars],
                "embedding_preview": embeddings[i][:5],
            }
            for i, c in enumerate(chunks)
        ],
    }


class SearchRequest(BaseModel):
    question: str
    job_id: str | None = None
    top_k: int = 3
    preview_chars: int = 300


@router.post("/eda/search")
async def eda_search(req: SearchRequest) -> dict:
    """Semantic search test: embed the question, cosine-rank Redis chunks, return top-k."""
    job_id = await _resolve_job_id(req.job_id)
    chunks, embeddings = await eda_store.get_eda_chunks(job_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks stored for job {job_id}.")

    query_embedding = (await embed([req.question]))[0]
    scored = sorted(
        (
            {
                "index": i,
                "score": round(_cosine_similarity(query_embedding, emb), 4),
                "preview": chunks[i][:req.preview_chars],
            }
            for i, emb in enumerate(embeddings)
        ),
        key=lambda x: x["score"],
        reverse=True,
    )
    return {
        "job_id": job_id,
        "question": req.question,
        "top_k": req.top_k,
        "total_chunks": len(chunks),
        "results": scored[: req.top_k],
    }


@router.post("/eda/retrieve")
async def eda_retrieve(req: SearchRequest) -> dict:
    """Run the real retriever path (Pinecone if enabled, else Redis cosine fallback)."""
    job_id = await _resolve_job_id(req.job_id)
    chunks, embeddings = await eda_store.get_eda_chunks(job_id)
    query_embedding = (await embed([req.question]))[0]
    results = await retriever.search(
        session_id=SESSION_ID,
        query_embedding=query_embedding,
        top_k=req.top_k,
        fallback_chunks=chunks,
        fallback_embeddings=embeddings,
    )
    return {
        "job_id": job_id,
        "question": req.question,
        "backend": "pinecone" if retriever._enabled else "redis-fallback",
        "results": [c[: req.preview_chars] for c in results],
    }
