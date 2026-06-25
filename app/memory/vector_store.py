from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_redis import RedisVectorStore as LangChainRedisVectorStore
from redisvl.query.filter import Tag

from app.core.config import settings

SCHEMA_VERSION = "1"
VECTOR_KEY_PREFIX = "vector"
_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_vector_executor = ThreadPoolExecutor(max_workers=2)


class FastEmbedEmbeddings(Embeddings):
    """Provide LangChain's embedding interface using the app FastEmbed model."""

    def __init__(self) -> None:
        """Initialize the embedding adapter with lazy model loading."""
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents for LangChain vector ingestion."""
        return [vector.tolist() for vector in self._get_model().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string for LangChain vector search."""
        return self.embed_documents([text])[0]

    def _get_model(self):
        """Return the lazy FastEmbed model instance."""
        if self._model is None:
            self._model = TextEmbedding(_EMBEDDING_MODEL)

        return self._model


class VectorMemoryStore:
    """Wrap LangChain RedisVectorStore for dataset-scoped vector memory."""

    def __init__(self) -> None:
        """Initialize lazy LangChain vector store and Redis client handles."""
        self._embeddings = FastEmbedEmbeddings()
        self._store: LangChainRedisVectorStore | None = None

    def _get_store(self) -> LangChainRedisVectorStore:
        """Return the initialized LangChain Redis vector store."""
        if self._store is None:
            self._store = LangChainRedisVectorStore(
                embeddings=self._embeddings,
                index_name=settings.redis_vector_index,
                key_prefix=VECTOR_KEY_PREFIX,
                redis_url=settings.redis_vector_url,
                ttl=settings.session_ttl,
                distance_metric="COSINE",
                indexing_algorithm="FLAT",
                metadata_schema=[
                    {"name": "schema_version", "type": "tag"},
                    {"name": "job_id", "type": "tag"},
                    {"name": "memory_type", "type": "tag"},
                    {"name": "source_type", "type": "tag"},
                    {"name": "source_id", "type": "tag"},
                    {"name": "chunk_id", "type": "tag"},
                    {"name": "title", "type": "text"},
                    {"name": "metadata_json", "type": "text"},
                ],
            )
        return self._store

    async def upsert_texts(
        self,
        *,
        job_id: str,
        memory_type: str,
        texts: list[str],
        source_type: str,
        source_id: str | None = None,
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Write text chunks to LangChain Redis vector memory."""
        if not texts:
            return []

        resolved_source_id = source_id or str(uuid.uuid4())
        keys = [
            self._record_id(memory_type, job_id, resolved_source_id, index)
            for index, _ in enumerate(texts)
        ]
        metadatas = [
            self._metadata(
                job_id=job_id,
                memory_type=memory_type,
                source_type=source_type,
                source_id=resolved_source_id,
                chunk_id=keys[index],
                title=title,
                metadata=metadata,
            )
            for index, _ in enumerate(texts)
        ]
        await self._run_blocking(
            self._get_store().add_texts,
            texts=texts,
            metadatas=metadatas,
            keys=keys,
        )
        return [
            self._record_from_payload(
                key=f"{VECTOR_KEY_PREFIX}:{keys[index]}",
                text=text,
                payload=metadatas[index],
            )
            for index, text in enumerate(texts)
        ]

    async def search(
        self,
        *,
        job_id: str,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return nearest vector memory records for one dataset job."""
        if not query.strip():
            return []

        store = self._get_store()
        all_results: list[tuple[Any, float]] = []
        for filter_expression in self._filter_expressions(job_id, memory_types):
            results = await self._run_blocking(
                store.similarity_search_with_score,
                query,
                k=top_k,
                filter=filter_expression,
            )
            all_results.extend(results)

        seen: set[str] = set()
        records: list[dict[str, Any]] = []
        for document, distance in sorted(all_results, key=lambda item: item[1]):
            record = self._record_from_document(document, float(distance))
            dedupe_key = record.get("key") or record.get("chunk_id", "")
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append(record)
            if len(records) >= top_k:
                break
        return records

    async def list_chunks(
        self,
        *,
        job_id: str,
        memory_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Inspect vector memory records stored for one dataset job."""
        return await self._run_blocking(
            self._list_chunks_sync,
            job_id,
            memory_types,
            limit,
        )

    def _list_chunks_sync(
        self,
        job_id: str,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Scan Redis hashes and decode matching vector memory records."""
        redis = self._get_store().config.redis()
        allowed_types = set(memory_types or [])
        records: list[dict[str, Any]] = []
        pattern = f"{VECTOR_KEY_PREFIX}:*:{job_id}:*"

        for key in redis.scan_iter(match=pattern, count=100):
            raw = redis.hgetall(key)
            record = self._record_from_hash(key, raw)
            if allowed_types and record.get("memory_type") not in allowed_types:
                continue
            records.append(record)
            if len(records) >= limit:
                break
        return records

    def _metadata(
        self,
        *,
        job_id: str,
        memory_type: str,
        source_type: str,
        source_id: str,
        chunk_id: str,
        title: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build LangChain document metadata for one vector chunk."""
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "memory_type": memory_type,
            "source_type": source_type,
            "source_id": source_id,
            "chunk_id": chunk_id,
            "title": title,
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        }

    def _record_id(
        self,
        memory_type: str,
        job_id: str,
        source_id: str,
        index: int,
    ) -> str:
        """Build a key suffix passed to LangChain RedisVectorStore."""
        return f"{memory_type}:{job_id}:{source_id}:{index}"

    def _filter_expressions(
        self,
        job_id: str,
        memory_types: list[str] | None,
    ) -> Iterable[Any]:
        """Build RedisVL filters used by LangChain RedisVectorStore."""
        job_filter = Tag("job_id") == job_id
        if not memory_types:
            yield job_filter
            return

        for memory_type in memory_types:
            yield job_filter & (Tag("memory_type") == memory_type)

    def _record_from_document(self, document: Any, distance: float) -> dict[str, Any]:
        """Convert one LangChain Document result into graph-safe memory data."""
        record = self._record_from_payload(
            key=document.metadata.get("chunk_id", ""),
            text=document.page_content,
            payload=document.metadata,
        )
        record["distance"] = round(distance, 6)
        record["score"] = round(1.0 - distance, 6)
        return record

    def _record_from_hash(
        self,
        key: bytes | str,
        raw: dict[Any, Any],
    ) -> dict[str, Any]:
        """Decode one Redis hash created by LangChain RedisVectorStore."""
        decoded = {
            self._decode(field): value
            for field, value in raw.items()
            if self._decode(field) != self._get_store().config.embedding_field
        }
        payload = {
            field: self._decode(value)
            for field, value in decoded.items()
            if isinstance(value, (bytes, str))
        }
        return self._record_from_payload(
            key=self._decode(key),
            text=payload.get(self._get_store().config.content_field, ""),
            payload=payload,
        )

    def _record_from_payload(
        self,
        *,
        key: str,
        text: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert stored metadata into the API memory record shape."""
        memory_metadata = self._load_metadata(payload.get("metadata_json"))
        return {
            "key": key,
            "chunk_id": str(payload.get("chunk_id", "")),
            "schema_version": str(payload.get("schema_version", "")),
            "job_id": str(payload.get("job_id", "")),
            "memory_type": str(payload.get("memory_type", "")),
            "source_type": str(payload.get("source_type", "")),
            "source_id": str(payload.get("source_id", "")),
            "title": str(payload.get("title", "")),
            "text": text,
            "metadata": memory_metadata,
            "embedding_dim": self._embedding_dim(),
        }

    def _load_metadata(self, value: Any) -> dict[str, Any]:
        """Decode stored metadata JSON into a dictionary."""
        if value is None:
            return {}
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str) and value:
            payload = json.loads(value)
            return payload if isinstance(payload, dict) else {}
        if isinstance(value, dict):
            return value
        return {}

    def _embedding_dim(self) -> int:
        """Return the configured embedding dimension when available."""
        return int(self._get_store().config.embedding_dimensions or 0)

    async def _run_blocking(self, func, *args, **kwargs):
        """Run LangChain Redis operations away from the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _vector_executor,
            lambda: func(*args, **kwargs),
        )

    def _decode(self, value: bytes | str) -> str:
        """Decode Redis bytes while leaving strings unchanged."""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value


vector_store = VectorMemoryStore()
