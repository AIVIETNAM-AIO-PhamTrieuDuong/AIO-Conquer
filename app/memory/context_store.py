from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings

SCHEMA_VERSION = "1"
MAX_RECORDS = 100


class BaseContextMemory:
    """Share Redis helpers for one scoped meta-memory category."""

    _client: aioredis.Redis | None = None

    def __init__(self, memory_name: str) -> None:
        """Initialize the memory category key suffix."""
        self.memory_name = memory_name

    async def _get_client(self) -> aioredis.Redis:
        """Return the shared operational Redis client."""
        if BaseContextMemory._client is None:
            BaseContextMemory._client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
        return BaseContextMemory._client

    def _key(self, scope_id: str) -> str:
        """Build the Redis key for this memory category and scope."""
        return f"meta:{scope_id}:{self.memory_name}"

    async def _append_record(
        self,
        *,
        scope_id: str,
        thread_id: str,
        run_id: str,
        source_node: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one JSON record to a trimmed Redis list."""
        record = self._record(
            scope_id=scope_id,
            thread_id=thread_id,
            run_id=run_id,
            source_node=source_node,
            payload=payload,
        )
        client = await self._get_client()
        key = self._key(scope_id)
        await client.rpush(key, json.dumps(record, ensure_ascii=False))
        await client.ltrim(key, -MAX_RECORDS, -1)
        await client.expire(key, settings.session_ttl)
        return record

    async def _read_records(
        self,
        *,
        scope_id: str,
        limit: int = MAX_RECORDS,
    ) -> list[dict[str, Any]]:
        """Read recent JSON records from a Redis list."""
        client = await self._get_client()
        raw_items = await client.lrange(self._key(scope_id), -limit, -1)
        return [json.loads(item) for item in raw_items]

    async def _set_snapshot(
        self,
        *,
        scope_id: str,
        thread_id: str,
        run_id: str,
        source_node: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Replace the current snapshot for this memory category."""
        record = self._record(
            scope_id=scope_id,
            thread_id=thread_id,
            run_id=run_id,
            source_node=source_node,
            payload=payload,
        )
        client = await self._get_client()
        await client.setex(
            self._key(scope_id),
            settings.session_ttl,
            json.dumps(record, ensure_ascii=False),
        )
        return record

    async def _get_snapshot(self, *, scope_id: str) -> dict[str, Any]:
        """Read the current JSON snapshot for this memory category."""
        client = await self._get_client()
        raw = await client.get(self._key(scope_id))
        return json.loads(raw) if raw else {}

    def _record(
        self,
        *,
        scope_id: str,
        thread_id: str,
        run_id: str,
        source_node: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Wrap a payload with common meta-memory metadata."""
        record = {
            "schema_version": SCHEMA_VERSION,
            "scope_id": scope_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_node": source_node,
        }
        record.update(payload)
        return record


class ToolMemory(BaseContextMemory):
    """Persist compact tool requests, results, warnings, and provenance."""

    def __init__(self) -> None:
        """Initialize tool memory under its Redis key suffix."""
        super().__init__("tool_memory")

    async def append(
        self,
        *,
        scope_id: str,
        thread_id: str,
        run_id: str,
        source_node: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one compact tool-memory record."""
        return await self._append_record(
            scope_id=scope_id,
            thread_id=thread_id,
            run_id=run_id,
            source_node=source_node,
            payload=record,
        )

    async def read(self, scope_id: str, limit: int = MAX_RECORDS) -> list[dict]:
        """Read recent tool-memory records for a scope."""
        return await self._read_records(scope_id=scope_id, limit=limit)


class AgentWorkingMemory(BaseContextMemory):
    """Persist the latest agent working-memory snapshot."""

    def __init__(self) -> None:
        """Initialize working memory under its Redis key suffix."""
        super().__init__("agent_working_memory")

    async def save(
        self,
        *,
        scope_id: str,
        thread_id: str,
        run_id: str,
        source_node: str,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Replace the current working-memory snapshot."""
        return await self._set_snapshot(
            scope_id=scope_id,
            thread_id=thread_id,
            run_id=run_id,
            source_node=source_node,
            payload=snapshot,
        )

    async def read(self, scope_id: str) -> dict[str, Any]:
        """Read the current working-memory snapshot for a scope."""
        return await self._get_snapshot(scope_id=scope_id)


class CuratedContextMemory(BaseContextMemory):
    """Persist reusable system-curated facts and decisions."""

    def __init__(self) -> None:
        """Initialize curated context memory under its Redis key suffix."""
        super().__init__("curated_context")

    async def append(
        self,
        *,
        scope_id: str,
        thread_id: str,
        run_id: str,
        source_node: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one curated context record."""
        return await self._append_record(
            scope_id=scope_id,
            thread_id=thread_id,
            run_id=run_id,
            source_node=source_node,
            payload=record,
        )

    async def read(self, scope_id: str, limit: int = MAX_RECORDS) -> list[dict]:
        """Read recent curated context records for a scope."""
        return await self._read_records(scope_id=scope_id, limit=limit)


class ErrorMemory(BaseContextMemory):
    """Persist recoverable failures, warnings, and retry context."""

    def __init__(self) -> None:
        """Initialize error memory under its Redis key suffix."""
        super().__init__("error_memory")

    async def append(
        self,
        *,
        scope_id: str,
        thread_id: str,
        run_id: str,
        source_node: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one recoverable error or warning record."""
        return await self._append_record(
            scope_id=scope_id,
            thread_id=thread_id,
            run_id=run_id,
            source_node=source_node,
            payload=record,
        )

    async def read(self, scope_id: str, limit: int = MAX_RECORDS) -> list[dict]:
        """Read recent error-memory records for a scope."""
        return await self._read_records(scope_id=scope_id, limit=limit)


class ContextStore:
    """Coordinate the four operational Redis meta-memory categories."""

    def __init__(self) -> None:
        """Initialize the individual memory category stores."""
        self.tool_memory = ToolMemory()
        self.agent_working_memory = AgentWorkingMemory()
        self.curated_context = CuratedContextMemory()
        self.error_memory = ErrorMemory()

    async def load_meta_memory(self, scope_id: str) -> dict[str, Any]:
        """Load all meta-memory categories for a scope."""
        return {
            "tool_memory": await self.tool_memory.read(scope_id, limit=20),
            "agent_working_memory": await self.agent_working_memory.read(
                scope_id
            ),
            "curated_context": await self.curated_context.read(
                scope_id,
                limit=20,
            ),
            "error_memory": await self.error_memory.read(scope_id, limit=20),
        }


context_store = ContextStore()
