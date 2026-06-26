# Multivariate Use-Case Dictionary + Domain Store

Tài liệu mô tả feature "multivariate use-case" đã triển khai: cách sinh ra
dictionary, cách build index, cách hai bên (dev endpoint + QA agent) lấy index,
và cách hệ thống map từ index ra **đúng record → đúng cột → đúng feature** cho
tool thống kê.

## 1. Tổng quan

Sau khi EDA chạy xong, hệ thống sinh một **Multivariate Use-Case Dictionary**:
một mảng các "use-case" phân tích đa biến (cặp biến + test thống kê + hướng dẫn
diễn giải). Mảng này quá nặng để nhét hết vào prompt của agent, nên ta lưu nó
vào một **Redis store riêng (domain store, port 6381)** kèm một **index gọn nhẹ**.
Khi user hỏi, agent chỉ đọc index → để LLM chọn `id` liên quan → fetch đúng vài
record → bind cặp cột vào tool thống kê.

Quan trọng — đừng nhầm với vector store:

| | Vector store (6380) | Domain store (6381) — feature này |
|---|---|---|
| Dữ liệu | "domain insight" theo từng cột, chunk + embed | mảng multivariate use-case (JSON nguyên vẹn) |
| Cách tìm | semantic search (cosine distance) | LLM đọc index → chọn `id` → HMGET |
| Node | `node_load_domain_context` | `node_route_multivariate` |
| Có embedding? | Có (`embedding_dim: 384`) | Không |

## 2. Sinh dictionary (trong `run_eda`)

File: `app/core/eda_pipeline.py` (bước multivariate, được bọc `try/except` riêng
để lỗi ở đây **không bao giờ** làm hỏng core EDA result).

```
df_clean
  → eda_corr.build_association(df_clean)      # tính truth table từ CỘT THẬT
  → build_multivariate_prompt(summary_md, truth_table_md, target_col)
  → llm.generate_text(...)                    # LLM sinh JSON array
  → parse_multivariate(raw)                   # → list[dict]
  → domain_store.set_multivariate(job_id, items)   # lưu Redis 6381
```

- `app/core/eda_corr.py` — `build_association` tính ma trận association hỗn hợp
  (Spearman / Cramér's V / Eta) và render `truth_table_md` với **tên cột thật**
  của dataframe (``| `Column A` | `Column B` | Score | Type |``).
- `app/model/prompts/multivariate.py` — prompt ép LLM dùng `"exact column name
  from profile"` và trả về **raw JSON array** (không markdown fence).
- `app/validation/multivariate_parser.py` — `parse_multivariate` parse JSON về
  `list[dict]`.

### Schema mỗi record

```json
{
  "comparison_pair": { "variable_a": "Attrition", "variable_b": "Job Role" },
  "evaluation": { "confidence_score": 4, "proposed_analysis_metric": "Attrition Rate by Job Role" },
  "business_value": "…",
  "sympy_calculation_code": "…",
  "metrics_and_significance": { "statistical_test_type": "Chi-Square", "expected_metrics": ["…"] },
  "interpretation_instructions": "…"
}
```

## 3. Cách build index + lưu trữ

File: `app/memory/domain_store.py` (`DomainKnowledgeStore`).

Mỗi job lưu thành **một Redis Hash** tại key `multivariate:{job_id}`, TTL =
`session_ttl`. Trong hash có 2 loại field:

1. **Field full record** — mỗi use-case một field, field name = `id` (vị trí
   trong mảng, dạng chuỗi `"0"`, `"1"`, …), value = JSON của record đó.
2. **Field `__index__`** — "menu" gọn để LLM chọn, value là JSON array:

```json
[
  { "id": "0", "variable_a": "Attrition", "variable_b": "Job Role",
    "metric": "Attrition Rate by Job Role", "confidence": 4 },
  { "id": "1", "variable_a": "Attrition", "variable_b": "Gender",
    "metric": "Attrition Rate by Gender", "confidence": 4 }
]
```

Index chỉ giữ 4 trường (`variable_a`, `variable_b`, `metric`, `confidence`) +
`id`, đủ để LLM xét liên quan mà **không phải tải full payload nặng**.

`set_multivariate` ghi atomic qua pipeline: `delete(key)` → `hset(mapping)` →
`expire(key, ttl)`.

### API của store

| Hàm | Trả về | Dùng cho |
|---|---|---|
| `set_multivariate(job_id, items)` | — | Ghi cả hash + `__index__` |
| `get_index(job_id)` | `list[dict]` (menu nhẹ) | Bước chọn id |
| `get_records(job_id, ids)` | `list[dict]` (full, đúng thứ tự `ids`) | Fetch sau khi chọn |
| `get_multivariate(job_id)` | `list[dict]` (toàn bộ, theo index) | Lấy hết |

> **Lưu ý nội tại nhất quán:** `index` được dựng từ chính `records`, nên
> `menu ↔ index ↔ record ↔ id` luôn khớp 100%. LLM chỉ trả `id`, và `id` được
> validate `if record_id in valid_ids`. Vì vậy bước "chọn id → fetch record"
> **không thể chọn nhầm record** và **không cần fuzzy matching**.

## 4. Hai bên lấy index

### 4.1 Dev endpoints (kiểm tra thủ công)

File: `app/api/routes/dev.py`.

| Endpoint | Mục đích |
|---|---|
| `GET /dev/eda/multivariate/{job_id}` | Full records (`get_multivariate`) |
| `GET /dev/eda/multivariate/{job_id}/index` | Chỉ menu `__index__` (`get_index`) |
| `GET /dev/eda/truth-table/{job_id}` | Tải `truth_table.md` |

```bash
curl http://localhost:8000/dev/eda/multivariate/<job_id>/index
```

### 4.2 QA agent (đường chính)

File: `app/graph/nodes.py` — node `node_route_multivariate`, nối cứng trong
graph **sau** `load_meta_memory` (`load_meta_memory → route_multivariate →
column_metadata`, xem `app/core/pipeline.py`). Đặt sau `load_meta_memory` là cố
ý: node đó reset `tool_memory`/`error_memory`/`agent_working_memory`, nên
route phải chạy sau để tool record không bị xoá. Node **luôn chạy**, nhưng việc
chọn record thì **query-driven** (theo câu hỏi user).

```
node_route_multivariate(state):
  job_id  = state["dataset_id"]
  index   = domain_store.get_index(job_id)           # đọc menu nhẹ (không phải tool)
  ids     = _select_usecase_ids(question, index)     # LLM chọn id theo câu hỏi
  request = ToolRequest(domain.usecase_lookup, {job_id, ids, available_columns})
  result  = domain_usecase_tool.invoke(request)      # TOOL: fetch + resolve cột
  → multivariate_requirements / multivariate_selected / multivariate_index  (xem mục 5)
```

`_select_usecase_ids` (nodes.py): dựng text menu từ index, nhét **câu hỏi user**
+ menu vào prompt, bắt LLM trả `{"ids": ["0","4"]}` (tối đa 5, most-relevant
first). Lọc id hợp lệ, dedup, fail-closed về `[]` nếu parse lỗi.

## 5. Từ index → đúng record → đúng cột → đúng feature

Đây là phần "đảm bảo đúng cột". Có 2 ranh giới tách biệt:

### Ranh giới A — `id` → record (an toàn tuyệt đối)
LLM trả về **số `id`**. `id` chỉ là chìa khóa; `get_records` mở ra **full
record**, và tên cột nằm **bên trong** `record["comparison_pair"]`. Vì index
sinh từ record nên không có khả năng lệch. Không cần fuzzy.

### Ranh giới B — tên cột trong record → cột thật trong CSV
Bước này **được đóng gói trong tool `domain.usecase_lookup`**
(`app/tools/domain_usecase.py`), theo đúng envelope `ToolRequest`/`ToolResult`
như mọi MVP tool khác. Tool fetch record theo `ids` rồi **resolve** cặp cột
trong `comparison_pair` với cột thật của dataset (`available_columns` =
numeric + categorical từ `_eda_column_groups`):

- Cột khớp (case-insensitive) → `data["columns"]` (resolved).
- Cột không khớp → `data["unresolved_columns"]` + một `warning` (không còn
  "âm thầm fallback" như trước).
- Test type → `data["association_method"]` (`spearman`/`pearson` nếu khớp).

Node ghi kết quả vào **`multivariate_requirements`** (tách khỏi
`domain_requirements` của vector store):

```
multivariate_requirements = {
  "features": [...resolved columns...],
  "association_method": "spearman" | "pearson" | "",
  "unresolved_columns": [...]
}
```

Khi node thống kê chạy, `_feature_hints(state)` **merge** features từ cả hai
nguồn (`multivariate_requirements` ưu tiên trước, rồi `domain_requirements`),
`_domain_feature_columns` khớp với cột thật, `_association_columns` chọn cặp
cột, và `_association_method` ưu tiên test type từ use-case.

**Tại sao thường khớp đúng:** `truth_table_md`, dataset profile, và
`num_stats`/`cat_stats` đều sinh ra từ **cùng một `df_clean`**, lại thêm prompt
ép `"exact column name"`. Nên `variable_a/b` thường == tên cột thật → match
exact chạy đúng, **không cần fuzzy**.

**Rủi ro còn lại (chỉ một):** LLM không tuân lệnh và tự đổi tên (vd `JobRole` →
`Job Role`). Giờ trường hợp này **không còn im lặng**: tool đẩy tên lệch vào
`unresolved_columns` + `warnings` (và `error_memory` qua envelope tool), nên
debug thấy ngay thay vì fallback ngầm.

## 6. State của LangGraph

File: `app/graph/schema.py`. Bước này ghi vào global state:

| Field | Nội dung |
|---|---|
| `multivariate_index` | Menu `__index__` mà LLM đã nhìn để chọn (debug/replay) |
| `multivariate_selected` | Các record đầy đủ đã được chọn |
| `multivariate_requirements` | `features` (cột resolved) + `association_method` + `unresolved_columns` — **tách riêng** khỏi `domain_requirements` (vector store) để không lẫn nguồn |

Tất cả khởi tạo rỗng trong `run_qa_pipeline` (`app/core/pipeline.py`). Vì là một
tool đúng chuẩn, lượt chạy còn được ghi vào `tool_requests` / `tool_results` /
`tool_memory` (và `error_memory` nếu có cảnh báo). Đối chiếu `multivariate_index`
(LLM thấy gì) với `multivariate_selected` (LLM chọn gì) để soi quyết định agent.

## 7. File liên quan

| File | Vai trò |
|---|---|
| `app/core/eda_corr.py` | Association matrix + truth table từ cột thật |
| `app/model/prompts/multivariate.py` | Prompt sinh use-case dictionary |
| `app/validation/multivariate_parser.py` | Parse JSON array → `list[dict]` |
| `app/core/eda_pipeline.py` | Gọi bước multivariate trong `run_eda` |
| `app/memory/domain_store.py` | Lưu hash + build/đọc `__index__` |
| `app/tools/domain_usecase.py` | Tool `domain.usecase_lookup`: fetch record + resolve cột |
| `app/graph/nodes.py` | `node_route_multivariate`, `_select_usecase_ids`, `_apply_tool_result`, `_feature_hints` |
| `app/graph/schema.py` | `multivariate_index`, `multivariate_selected`, `multivariate_requirements` |
| `app/core/pipeline.py` | Nối node vào graph (sau `load_meta_memory`) + init state |
| `app/api/routes/dev.py` | Dev endpoints kiểm tra |
