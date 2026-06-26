# Plan: Persist Multivariate Use-Case Dictionary vào Domain Knowledge Store

## Mục tiêu

Thay cơ chế **ghi file tạm** hiện tại (`{tempdir}/eda/{job_id}/multivariate.json`) bằng cơ chế
**persist có cấu trúc** vào store, để QA agent dùng lại cho câu hỏi đa biến.

---

## Quyết định kiến trúc (đã phân tích)

### ❌ KHÔNG dựng MongoDB

| Tiêu chí | Thực tế | Kết luận |
|---|---|---|
| Khối lượng | 1 array ~20–80 item / `job_id` | Redis thừa sức |
| Vòng đời | Cả app đang TTL `session_ttl` (1h), ephemeral | Chưa cần durable DB |
| Query | (a) lấy hết theo job, (b) semantic tìm cặp biến | Đã có đủ hạ tầng |
| Cross-dataset query | Chưa có nhu cầu | Chưa cần Mongo |

→ Thêm Mongo lúc này = thêm container + driver `motor` + lifecycle cho việc Redis đang làm tốt.
**Chỉ thêm Mongo khi** cần query ad-hoc xuyên nhiều dataset HOẶC persist vĩnh viễn (knowledge base
tích luỹ qua thời gian). Để ngỏ — không làm lần này.

### ❌ KHÔNG tách key-value theo từng field

Tách mỗi field thành 1 Redis key → phân mảnh, mất atomic, retrieval thành N lệnh GET.
→ Lưu **cả array thành 1 JSON value** (đúng như `set_eda_result`).

### ✅ Hybrid dual-write trên 2 Redis có sẵn (đúng pattern `run_eda` hiện tại)

| Tầng | Store | Key / memory_type | Dùng để |
|---|---|---|---|
| Source of truth | operational `redis:6379` | `eda:{job_id}:multivariate` (JSON array) | Lấy toàn bộ, load context, dev download |
| Semantic index | `redis-vector:6380` | `memory_type="multivariate_usecase"` | Agent tìm cặp biến liên quan câu hỏi |

Mỗi **item use-case = 1 unit retrieval** (KHÔNG `fixed_size_chunk` cắt ngang item).

---

## Quyết định còn để ngỏ (mặc định + lý do)

| Vấn đề | Mặc định chọn | Ghi chú |
|---|---|---|
| Có cần semantic index ngay? | **CÓ** (dual-write) | Dataset rộng → N×N item nhiều, không nhét hết vào context được. Vector cho phép lọc top-k theo câu hỏi. Nếu chỉ muốn tối giản: làm Bước A trước, hoãn Bước B. |
| Searchable text cho embed | `"{var_a} vs {var_b} | {proposed_metric} | {business_value}"` | Đủ ngắn, đủ ngữ nghĩa để match câu hỏi người dùng |
| Full item lưu ở đâu trong vector | `metadata_json` của record | Search trả về → agent đọc full item từ metadata, không cần GET lại blob |
| Giữ file tạm không? | **Bỏ** ghi file là source of truth; vẫn giữ `multivariate_raw.txt` để debug | Blob Redis thay vai trò |
| TTL | Theo `session_ttl` như mọi key khác | Nhất quán; nâng lên nếu sau này cần "knowledge" sống lâu hơn |

---

## Các bước triển khai

### Bước A — `app/memory/eda_store.py` (SỬA): thêm multivariate blob

```python
async def set_eda_multivariate(self, job_id: str, items: list[dict]) -> None:
    client = await self._get_client()
    await client.setex(
        f"eda:{job_id}:multivariate",
        settings.session_ttl,
        json.dumps(items, ensure_ascii=False),
    )

async def get_eda_multivariate(self, job_id: str) -> list[dict]:
    client = await self._get_client()
    raw = await client.get(f"eda:{job_id}:multivariate")
    return json.loads(raw) if raw else []
```

Cập nhật danh sách key trong docstring/comment store:
`eda:{job_id}:status | result | chunks | multivariate`.

---

### Bước B — `app/core/eda_pipeline.py` (SỬA): dual-write trong `run_eda`

Trong block `try` của multivariate (sau `items = parse_multivariate(raw)`), **thay**
`eda_corr.write_json_file(job_id, items)` bằng:

```python
# 1) Source of truth — operational Redis blob
await eda_store.set_eda_multivariate(job_id, items)

# 2) Semantic index — 1 embedding / use-case item (skip nếu rỗng)
if items:
    search_texts = [_usecase_search_text(it) for it in items]
    await vector_store.upsert_texts(
        job_id=job_id,
        memory_type="multivariate_usecase",
        texts=search_texts,
        source_type="generated",
        source_id="multivariate",
        title="Multivariate use-case",
        metadata={"session_id": session_id, "items": items},  # full payload
    )
```

> Lưu ý metadata: hiện `upsert_texts` nhận **1 dict metadata chung** cho mọi text. Để mỗi record
> mang đúng item của nó, cần truyền metadata **per-text**. → Xem Bước C.

Thêm helper trong `eda_pipeline.py`:
```python
def _usecase_search_text(item: dict) -> str:
    pair = item.get("comparison_pair", {})
    ev = item.get("evaluation", {})
    return (
        f"{pair.get('variable_a','')} vs {pair.get('variable_b','')} | "
        f"{ev.get('proposed_analysis_metric','')} | {item.get('business_value','')}"
    ).strip()
```

Block `except` (multivariate fail) → đổi `write_json_file(job_id, [])`
thành `await eda_store.set_eda_multivariate(job_id, [])` (giữ graceful degrade).

---

### Bước C — `app/memory/vector_store.py` (SỬA nhỏ): hỗ trợ metadata per-text

`upsert_texts` hiện áp **1** dict metadata cho tất cả chunk. Để mỗi use-case record mang full item
riêng, cho phép `metadata` nhận `list[dict]` (1 phần tử / text) bên cạnh dict chung:

```python
metadata: dict | list[dict] | None = None
...
def _meta_for(index: int) -> dict | None:
    if isinstance(metadata, list):
        return metadata[index] if index < len(metadata) else {}
    return metadata
```

Giữ tương thích ngược: các call hiện tại truyền dict → không đổi hành vi.
*(Phương án thay thế nếu muốn tránh sửa vector_store: nhét full item vào ngay searchable text /
title. Nhưng để metadata sạch hơn → ưu tiên sửa nhẹ ở đây.)*

---

### Bước D — `app/api/routes/dev.py` (SỬA): đọc từ Redis thay vì file

`GET /dev/eda/multivariate/{job_id}` → trả `await eda_store.get_eda_multivariate(job_id)`
dạng `JSONResponse` (404 nếu rỗng), thay cho `FileResponse` đọc file tạm.

---

### Bước E — (Sau này, ngoài phạm vi) wire vào QA agent

Khi cần dùng thật trong chat pipeline:
- Node load `eda:{job_id}:multivariate` (toàn bộ, nếu nhỏ) HOẶC
- `vector_store.search(job_id, query, memory_types=["multivariate_usecase"], top_k=...)` để lấy
  cặp biến liên quan → đọc full item từ `metadata["items"]` / `metadata` của record.

---

## Thứ tự implement

| # | File | Loại |
|---|------|------|
| A | `app/memory/eda_store.py` (set/get multivariate) | Sửa |
| B | `app/core/eda_pipeline.py` (dual-write + helper) | Sửa |
| C | `app/memory/vector_store.py` (metadata per-text) | Sửa nhỏ |
| D | `app/api/routes/dev.py` (đọc Redis) | Sửa |

*(KHÔNG thêm container, KHÔNG đụng docker-compose, KHÔNG thêm dependency.)*

---

## Rủi ro / lưu ý

- **TTL 1h**: blob + vector đều hết hạn theo `session_ttl`. Nếu muốn knowledge sống lâu hơn report,
  cần TTL riêng cho multivariate → cân nhắc khi wire vào agent.
- **Vector store dùng chung index** với eda_summary/domain_generated → filter bằng
  `memory_type="multivariate_usecase"` (đã có cơ chế filter). Không lẫn.
- **Embedding cost**: mỗi job thêm ~20–80 embedding (FastEmbed local, rẻ). Chấp nhận được.
- **metadata size**: nhét full item array vào metadata vector record hơi lặp dữ liệu với blob.
  Đổi lại: search 1 lần ra ngay full item, không cần round-trip. Chấp nhận đánh đổi này.
