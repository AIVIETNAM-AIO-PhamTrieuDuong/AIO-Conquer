from __future__ import annotations

import json
import redis.asyncio as aioredis

from app.core.config import settings

INDEX_FIELD = "__index__"


class DomainKnowledgeStore:
    """Persist generated domain knowledge (multivariate use-cases) in Redis.

    Backed by a dedicated Redis instance separate from operational and vector
    memory. Each job is stored as one Redis Hash at ``multivariate:{job_id}``:

    - one field per use-case record (field name = record id, value = item JSON),
    - one ``__index__`` field holding a compact menu ``[{id, variable_a,
      variable_b, metric, confidence}]`` for cheap LLM-driven record selection.

    No chunking or embedding — selection happens application-side after reading
    the lightweight index, then fetching only the chosen records by id.
    """

    def __init__(self) -> None:
        """Initialize the lazy Redis connection for the domain knowledge store."""
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(settings.redis_domain_url, decode_responses=True)
        return self._client

    @staticmethod
    def _key(job_id: str) -> str:
        return f"multivariate:{job_id}"

    async def set_multivariate(self, job_id: str, items: list[dict]) -> None:
        """Store the use-case array as a Hash with a compact ``__index__`` menu."""
        client = await self._get_client()
        key = self._key(job_id)

        mapping: dict[str, str] = {}
        index: list[dict] = []
        for position, item in enumerate(items):
            record_id = str(position)
            mapping[record_id] = json.dumps(item, ensure_ascii=False)
            pair = item.get("comparison_pair", {})
            evaluation = item.get("evaluation", {})
            index.append(
                {
                    "id": record_id,
                    "variable_a": pair.get("variable_a", ""),
                    "variable_b": pair.get("variable_b", ""),
                    "metric": evaluation.get("proposed_analysis_metric", ""),
                    "confidence": evaluation.get("confidence_score"),
                }
            )
        mapping[INDEX_FIELD] = json.dumps(index, ensure_ascii=False)

        pipe = client.pipeline()
        pipe.delete(key)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, settings.session_ttl)
        await pipe.execute()

    async def get_index(self, job_id: str) -> list[dict]:
        """Return the lightweight selection menu for a job (no full payloads)."""
        client = await self._get_client()
        raw = await client.hget(self._key(job_id), INDEX_FIELD)
        return json.loads(raw) if raw else []

    async def get_records(self, job_id: str, ids: list[str]) -> list[dict]:
        """Fetch only the chosen full use-case records by id, preserving order."""
        if not ids:
            return []
        client = await self._get_client()
        raws = await client.hmget(self._key(job_id), [str(i) for i in ids])
        return [json.loads(raw) for raw in raws if raw]

    async def get_multivariate(self, job_id: str) -> list[dict]:
        """Return every use-case record for a job, ordered by the index."""
        index = await self.get_index(job_id)
        if not index:
            return []
        return await self.get_records(job_id, [entry["id"] for entry in index])


domain_store = DomainKnowledgeStore()
