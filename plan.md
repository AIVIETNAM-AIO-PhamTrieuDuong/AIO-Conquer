# Plan: Tách EDA Storage + RAG cho EDA Context

## Vấn đề hiện tại

`SessionMemory` trong `redis_client.py` đang gánh 2 việc:
- `session:{id}` → lịch sử hội thoại
- `eda:{job_id}:*` → toàn bộ EDA output

`node_load_eda_context` trong `pipeline.py` đang dump **toàn bộ** `summary_md` vào prompt,
không filter → bloat context window khi file lớn.

## Mục tiêu

1. Tách biệt storage: hội thoại ↔ EDA output
2. EDA output sau khi tạo ra → chunk fixed-size → embed local model → lưu vào Pinecone
3. QA agent dùng RAG: embed câu hỏi → retrieve top-k chunks → inject vào context → generate
4. API key chỉ dùng cho LLM generation, không dùng cho embedding

---

## Định hướng triển khai: Simple RAG

Không over-engineer. Toàn bộ pipeline RAG gồm 3 bước thẳng:

```
INDEX TIME (khi upload xong EDA):
  summary_md → chunk() → embed() → store()

QUERY TIME (khi user hỏi):
  question → embed() → search() → top-k chunks → prompt → LLM → answer
```

### Chunking — fixed-size, không cần hiểu cấu trúc

```
chunk_size = 1000 ký tự
overlap    = 100  ký tự

"ABCDE...Z" → ["ABCDE...1000", "951...1950", "1901...2900", ...]
```

Overlap đủ để câu không bị đứt giữa 2 chunk liền kề.

### Embedding — local ONNX, không gọi API

Model: `BAAI/bge-small-en-v1.5`
- 384 dimensions
- ~130MB, chạy tốt trên CPU
- Download từ HuggingFace lần đầu, cache lại

### Retrieval — cosine similarity

```
query_vec = embed(question)
scores    = cosine_similarity(query_vec, all_chunk_vecs)
top_k     = argsort(scores)[-3:]   # lấy 3 chunks liên quan nhất
context   = "\n\n".join(top_k_texts)
```

Nếu có Pinecone → dùng Pinecone (đã có sẵn index, ANN search nhanh hơn).
Nếu không có Pinecone → load chunks + embeddings từ Redis, tính cosine tại chỗ (đủ dùng khi file nhỏ).

### RAG trong QA agent

Node `node_load_eda_context` trong `pipeline.py` trở thành:

```
Cũ: get full summary_md → inject

Mới:
  1. Lấy job_id active của session
  2. embed(question)            ← local, không tốn API
  3. retrieve top-k chunks      ← Pinecone hoặc Redis fallback
  4. format thành "Reference Document:\n{chunk1}\n\n{chunk2}\n..."
  5. inject vào prompt như cũ
```

Agent không thay đổi logic generate hay parse — chỉ thay đổi cách lấy context.

---

## Kiến trúc mới

```
[Upload Excel]
      │
      ▼
EDA Pipeline
      │── analyze_and_clean_data()
      │── generate_summary_md()
      │── call_llm_for_insight()
      │
      ├─→ Redis : eda:{job_id}:status / result       (API polling)
      │
      └─→ chunker → embedder → store
                                 ├─→ Pinecone        (nếu có key)
                                 └─→ Redis chunks+vecs (fallback)


[User hỏi]
      │
      ▼
QA Pipeline (LangGraph)
      │
      ├── node_load_history      → Redis session:{id}
      │
      ├── node_load_eda_context  → embed(question)
      │                            → retrieve top-k chunks
      │                            → format context string
      ├── node_generate          → LLM (9Router, dùng API key)
      │
      ├── node_parse
      │
      └── node_save_memory       → Redis session:{id}
```

---

## Các bước triển khai

### Bước 1 — `requirements.txt`

Thêm:
```
fastembed>=0.4.0
```
- ONNX-based, không cần PyTorch
- Tự download model từ HuggingFace lần đầu chạy

---

### Bước 2 — `app/core/config.py`

Xóa field `embed_model` (không còn gọi API embedding).

---

### Bước 3 — `app/memory/eda_store.py` (tạo mới)

Class `EDAStore` tách khỏi `SessionMemory`:

```
set_eda_status(job_id, status)
get_eda_status(job_id) -> str | None
set_eda_result(job_id, payload)           # cho /eda/result API
get_eda_result(job_id) -> dict | None
set_active_eda(session_id, job_id)
get_active_eda(session_id) -> str | None
set_eda_chunks(job_id, chunks, embeddings) # lưu text + vecs (fallback Pinecone)
get_eda_chunks(job_id) -> tuple[list[str], list[list[float]]]
```

Key schema Redis:
```
eda:{job_id}:status
eda:{job_id}:result
eda:{job_id}:chunks      ← JSON: [{text, embedding}, ...]
eda:active:{session_id}
```

---

### Bước 4 — `app/memory/redis_client.py`

Xóa toàn bộ EDA methods. Giữ: `get_history`, `append`, `clear`, `is_alive`.

---

### Bước 5 — `app/retrieval/chunker.py` (tạo mới)

```python
def fixed_size_chunk(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]
```

---

### Bước 6 — `app/retrieval/embedder.py` (tạo mới)

```python
# Singleton: load model một lần khi import
# Thread pool 1 worker vì CPU-bound

async def embed(texts: list[str]) -> list[list[float]]
```

Model: `BAAI/bge-small-en-v1.5`, 384 dims.

---

### Bước 7 — `app/retrieval/retriever.py`

Thêm vào `PineconeRetriever`:

```python
async def upsert_chunks(job_id, session_id, chunks, embeddings) -> None
async def search(session_id, query_embedding, top_k=3) -> list[str]
```

Fallback (Pinecone off): nhận chunks + embeddings từ EDAStore, tính cosine tại chỗ:

```python
def _cosine_top_k(query_vec, chunk_vecs, texts, top_k) -> list[str]
```

---

### Bước 8 — `app/core/eda_pipeline.py`

Sau khi có `summary_md`, thêm:

```
summary_md
  → chunker.fixed_size_chunk()
  → embedder.embed(chunks)
  → eda_store.set_eda_chunks(job_id, chunks, embeddings)   # Redis fallback
  → retriever.upsert_chunks(job_id, session_id, ...)       # Pinecone nếu có
```

---

### Bước 9 — `app/core/pipeline.py`

`node_load_eda_context`:

```
1. job_id = await eda_store.get_active_eda(SESSION_ID)
2. query_vec = await embedder.embed([state["question"]])[0]
3. chunks = await retriever.search(SESSION_ID, query_vec, top_k=3)
4. return {"context": "\n\n".join(chunks)}
```

---

### Bước 10 — `app/api/routes/eda.py`

Đổi import từ `memory` sang `eda_store`.

---

## Thứ tự implement

| # | File | Loại |
|---|------|-------|
| 1 | `requirements.txt` | Sửa |
| 2 | `app/core/config.py` | Sửa |
| 3 | `app/memory/eda_store.py` | Tạo mới |
| 4 | `app/memory/redis_client.py` | Sửa |
| 5 | `app/retrieval/chunker.py` | Tạo mới |
| 6 | `app/retrieval/embedder.py` | Tạo mới |
| 7 | `app/retrieval/retriever.py` | Sửa |
| 8 | `app/core/eda_pipeline.py` | Sửa |
| 9 | `app/core/pipeline.py` | Sửa |
| 10 | `app/api/routes/eda.py` | Sửa |
