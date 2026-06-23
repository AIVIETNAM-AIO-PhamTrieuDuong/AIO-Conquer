from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings

_pinecone_executor = ThreadPoolExecutor(max_workers=1)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class PineconeRetriever:
    def __init__(self) -> None:
        self._enabled = bool(settings.pinecone_api_key and settings.pinecone_index)
        if self._enabled:
            try:
                from pinecone import Pinecone
                self._pc = Pinecone(api_key=settings.pinecone_api_key)
                self._index = self._pc.Index(settings.pinecone_index)
            except ImportError:
                self._enabled = False

    # ------------------------------------------------------------------
    # Legacy — kept as-is
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        """Return relevant context as plain text, or empty string if disabled."""
        if not self._enabled:
            return ""
        # TODO: embed query with EMBED_MODEL, search index, join top-k texts
        return ""

    # ------------------------------------------------------------------
    # EDA chunk upsert
    # ------------------------------------------------------------------

    async def upsert_chunks(
        self,
        job_id: str,
        session_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if not self._enabled:
            return
        vectors = [
            {
                "id": f"{session_id}_{job_id}_{i}",
                "values": emb,
                "metadata": {"text": chunk, "session_id": session_id, "job_id": job_id},
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _pinecone_executor,
            lambda: self._index.upsert(vectors=vectors),
        )

    # ------------------------------------------------------------------
    # EDA chunk search
    # ------------------------------------------------------------------

    async def search(
        self,
        session_id: str,
        query_embedding: list[float],
        top_k: int = 3,
        fallback_chunks: list[str] | None = None,
        fallback_embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        if self._enabled:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                _pinecone_executor,
                lambda: self._index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    filter={"session_id": session_id},
                    include_metadata=True,
                ),
            )
            return [
                m.metadata["text"]
                for m in results.matches
                if m.metadata.get("text")
            ]

        # Fallback: cosine similarity in-memory
        if not fallback_chunks or not fallback_embeddings:
            return []
        scores = [_cosine_similarity(query_embedding, e) for e in fallback_embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [fallback_chunks[i] for i in top_indices]


retriever = PineconeRetriever()
