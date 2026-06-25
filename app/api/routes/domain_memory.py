from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.memory.vector_store import vector_store
from app.retrieval.chunker import fixed_size_chunk

router = APIRouter(prefix="/domain-memory", tags=["domain-memory"])


class DomainMemoryWriteRequest(BaseModel):
    """Represent custom domain knowledge to append for one dataset."""

    text: str
    title: str | None = None
    source_id: str | None = None
    metrics: list[str] | None = None
    features: list[str] | None = None
    constraints: list[str] | None = None


class DomainMemorySearchRequest(BaseModel):
    """Represent a semantic query against dataset-scoped domain memory."""

    query: str
    top_k: int = 3


@router.post("/{job_id}")
async def append_domain_memory(
    job_id: str,
    request: DomainMemoryWriteRequest,
) -> dict:
    """Append custom domain knowledge to Redis vector memory."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Domain memory text is required.")

    chunks = fixed_size_chunk(text)
    source_id = request.source_id or str(uuid.uuid4())
    records = await vector_store.upsert_texts(
        job_id=job_id,
        memory_type="domain_custom",
        texts=chunks,
        source_type="custom",
        source_id=source_id,
        title=request.title or "",
        metadata={
            "metrics": request.metrics or [],
            "features": request.features or [],
            "constraints": request.constraints or [],
        },
    )
    return {
        "job_id": job_id,
        "source_id": source_id,
        "memory_type": "domain_custom",
        "chunks_written": len(records),
    }


@router.post("/{job_id}/search")
async def search_domain_memory(
    job_id: str,
    request: DomainMemorySearchRequest,
) -> dict:
    """Search generated and custom domain memory for one dataset."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required.")

    results = await vector_store.search(
        job_id=job_id,
        query=query,
        memory_types=["domain_generated", "domain_custom"],
        top_k=request.top_k,
    )
    return {
        "job_id": job_id,
        "query": query,
        "top_k": request.top_k,
        "results": results,
    }
