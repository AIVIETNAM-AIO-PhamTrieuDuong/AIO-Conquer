from __future__ import annotations

import json
import redis.asyncio as aioredis

from app.core.config import settings


class EDAStore:
    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._client

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def set_eda_status(self, job_id: str, status: str) -> None:
        client = await self._get_client()
        await client.setex(f"eda:{job_id}:status", settings.session_ttl, status)

    async def get_eda_status(self, job_id: str) -> str | None:
        client = await self._get_client()
        return await client.get(f"eda:{job_id}:status")

    # ------------------------------------------------------------------
    # Result payload (for /eda/result API)
    # ------------------------------------------------------------------

    async def set_eda_result(self, job_id: str, payload: dict) -> None:
        client = await self._get_client()
        await client.setex(
            f"eda:{job_id}:result",
            settings.session_ttl,
            json.dumps(payload, ensure_ascii=False),
        )

    async def get_eda_result(self, job_id: str) -> dict | None:
        client = await self._get_client()
        raw = await client.get(f"eda:{job_id}:result")
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------
    # Active job per session
    # ------------------------------------------------------------------

    async def set_active_eda(self, session_id: str, job_id: str) -> None:
        client = await self._get_client()
        await client.setex(f"eda:active:{session_id}", settings.session_ttl, job_id)

    async def get_active_eda(self, session_id: str) -> str | None:
        client = await self._get_client()
        return await client.get(f"eda:active:{session_id}")

    # ------------------------------------------------------------------
    # Chunks + embeddings (Pinecone fallback)
    # ------------------------------------------------------------------

    async def set_eda_chunks(
        self, job_id: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        client = await self._get_client()
        payload = [{"text": c, "embedding": e} for c, e in zip(chunks, embeddings)]
        await client.setex(
            f"eda:{job_id}:chunks",
            settings.session_ttl,
            json.dumps(payload, ensure_ascii=False),
        )

    async def get_eda_chunks(
        self, job_id: str
    ) -> tuple[list[str], list[list[float]]]:
        client = await self._get_client()
        raw = await client.get(f"eda:{job_id}:chunks")
        if not raw:
            return [], []
        items = json.loads(raw)
        chunks = [item["text"] for item in items]
        embeddings = [item["embedding"] for item in items]
        return chunks, embeddings


eda_store = EDAStore()
