# XAI QA Backend

A QA backend with explainable AI output, built on FastAPI + LangGraph + Gemini + Redis.

---

## Architecture

```
[load_history] → [generate] → [parse] → [save_memory] → END
```

All pipeline nodes live in `app/core/pipeline.py::_build_graph()`.  
New features are added by inserting nodes — existing nodes are never modified.

### Layer overview

| Layer | Role |
|---|---|
| **API** (`app/api/`) | HTTP interface — routes, schemas, middleware |
| **Core** (`app/core/`) | LangGraph pipeline, config |
| **Model** (`app/model/`) | Gemini async client, prompt templates |
| **Memory** (`app/memory/`) | Redis — conversation history per session |
| **Retrieval** (`app/retrieval/`) | Pinecone stub — not yet active |
| **Validation** (`app/validation/`) | JSON → `QAResponse` parsing |

---

## Tech Stack

| Concern | Choice |
|---|---|
| API framework | FastAPI |
| Pipeline orchestration | LangGraph (`StateGraph`) |
| LLM | Google Gemini (`gemini-2.5-flash`) |
| Short-term memory | Redis 7 |
| Vector store | Pinecone (stub — not yet active) |
| Runtime | Python 3.11 |

---

## Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ask.py          # POST /ask
│   │   │   ├── health.py       # GET /health
│   │   │   └── dev.py          # GET|DELETE /dev/history
│   │   ├── middleware/
│   │   └── schemas.py          # AskRequest, QAResponse
│   │
│   ├── core/
│   │   ├── pipeline.py         # Graph definition + all nodes (entry point)
│   │   └── config.py           # Settings from .env
│   │
│   ├── model/
│   │   ├── llm_client.py       # Gemini async client
│   │   └── prompts/
│   │       └── qa_system.py    # Base prompt + build_prompt()
│   │
│   ├── memory/
│   │   └── redis_client.py     # get_history(), append()
│   │
│   ├── retrieval/
│   │   └── retriever.py        # Pinecone stub — fill when ready
│   │
│   ├── validation/
│   │   └── parser.py           # JSON → QAResponse
│   │
│   └── main.py
│
├── Dockerfile.api              # FastAPI service
├── docker-compose.yml          # Orchestrates redis + api
├── .env.example
└── README.md
```

---

## Getting Started

### 1. Lấy Google API Key

Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey) và tạo API key.

### 2. Cấu hình environment

```bash
cp .env.example .env
```

Mở `.env` và điền API key:

```env
GOOGLE_API_KEY=your-google-api-key-here
GEMINI_MODEL=gemini-2.5-flash
```

Biến `REDIS_URL` được docker-compose tự inject cho internal networking — không cần sửa.

### 3. Build và start stack

```bash
docker compose up -d --build
```

Hai service khởi động theo thứ tự:
1. `redis` — sẵn sàng khi ping OK
2. `api` — chờ redis healthy rồi mới start

### 4. Kiểm tra liveness

```bash
curl http://localhost:8000/health
```

Response mẫu:

```json
{"status": "ok", "gemini": "up", "redis": "up"}
```

### 5. Gửi câu hỏi

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tính đạo hàm của x^2 + 3x"}'
```

### 6. Reset lịch sử hội thoại (dev)

```bash
curl -X DELETE http://localhost:8000/dev/history
```

### Dừng stack

```bash
docker compose down
```

Dữ liệu Redis được giữ trong Docker volume (`redis_data`).  
Để xóa luôn volume: `docker compose down -v`

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root — kiểm tra backend đang chạy |
| `POST` | `/ask` | Gửi câu hỏi, nhận câu trả lời |
| `GET` | `/health` | Liveness check (Gemini + Redis) |
| `DELETE` | `/dev/history` | Xóa lịch sử hội thoại hiện tại |

### Request / Response

**`POST /ask`**

```json
// Request
{ "question": "Tính đạo hàm của x^2 + 3x" }

// Response
{
  "answer": "...",
  "explanation": "...",
  "fol": "...",          // optional — first-order logic form
  "cot": ["..."],        // optional — chain-of-thought steps
  "premises": ["..."],   // optional — premises used
  "confidence": 0.95     // optional — [0.0, 1.0]
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(bắt buộc)* | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model ID |
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL (tự inject bởi docker-compose) |
| `SESSION_TTL` | `3600` | TTL lịch sử hội thoại (giây) |
| `MAX_HISTORY_TURNS` | `5` | Số lượt hội thoại giữ trong memory |
| `PINECONE_API_KEY` | — | Pinecone API key (chưa dùng) |
| `PINECONE_INDEX` | — | Pinecone index name (chưa dùng) |
| `EMBED_MODEL` | `text-embedding-004` | Embedding model (chưa dùng) |
