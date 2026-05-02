import json
import redis.asyncio as aioredis
from app.core.config import settings


class SessionMemory:
    def __init__(self):
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._client

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    async def get_history(self, session_id: str) -> list[dict]:
        client = await self._get_client()
        raw = await client.get(self._key(session_id))
        if raw is None:
            return []
        return json.loads(raw)

    async def append(self, session_id: str, question: str, answer: str) -> None:
        client = await self._get_client()
        history = await self.get_history(session_id)
        history.append({"q": question, "a": answer})
        history = history[-settings.max_history_turns:]
        await client.setex(
            self._key(session_id),
            settings.session_ttl,
            json.dumps(history, ensure_ascii=False),
        )

    async def clear(self, session_id: str) -> None:
        client = await self._get_client()
        await client.delete(self._key(session_id))

    async def is_alive(self) -> bool:
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # EDA result storage
    # ------------------------------------------------------------------

    def _eda_key(self, job_id: str) -> str:
        return f"eda:{job_id}:result"

    def _eda_status_key(self, job_id: str) -> str:
        return f"eda:{job_id}:status"

    def _eda_active_key(self, session_id: str) -> str:
        return f"eda:active:{session_id}"

    async def set_eda_result(self, job_id: str, payload: dict) -> None:
        client = await self._get_client()
        await client.setex(
            self._eda_key(job_id),
            settings.session_ttl,
            json.dumps(payload, ensure_ascii=False),
        )

    async def get_eda_result(self, job_id: str) -> dict | None:
        client = await self._get_client()
        raw = await client.get(self._eda_key(job_id))
        return json.loads(raw) if raw else None

    async def set_eda_status(self, job_id: str, status: str) -> None:
        client = await self._get_client()
        await client.setex(self._eda_status_key(job_id), settings.session_ttl, status)

    async def get_eda_status(self, job_id: str) -> str | None:
        client = await self._get_client()
        return await client.get(self._eda_status_key(job_id))

    async def set_active_eda(self, session_id: str, job_id: str) -> None:
        client = await self._get_client()
        await client.setex(self._eda_active_key(session_id), settings.session_ttl, job_id)

    async def get_active_eda(self, session_id: str) -> str | None:
        client = await self._get_client()
        return await client.get(self._eda_active_key(session_id))


memory = SessionMemory()
