# Plan: Multivariate Use-Case Dictionary (EDA Pipeline v2)

## Mục tiêu

Enhance EDA pipeline để sinh **thêm 1 output mới** ngoài report hiện tại: một **Multivariate
Use-Case Dictionary** (mảng JSON các cặp biến + use-case phân tích) giúp QA agent xử lý câu hỏi
đa biến (Target × Feature, Metric × Category…).

**Giữ nguyên:**
- Toàn bộ logic EDA gốc (`analyze_and_clean_data` → `generate_summary_md` → `call_llm_for_insight`).
- Chunking `summary_md` → embed → lưu vector store Redis (`eda:{job_id}:chunks`) như cũ.

**Thêm mới:**
- Bước tính **Association Truth Table** đa kiểu (Spearman / Cramér's V / Eta).
- Bước LLM sinh **JSON Use-Case Dictionary** từ 2 input: Dataset Profile + Correlation Truth Table.

**KHÔNG đổi cấu trúc:** `eda_pipeline` giữ nguyên dạng **async tuyến tính** (`run_eda`), tách biệt
với LangGraph của pipeline chat chính. 2 bước mới chỉ là 2 lời gọi hàm nối thêm vào `run_eda`,
**không** chuyển sang `StateGraph`.

---

## Quyết định đã chốt (mặc định — nói nếu muốn đổi)

| Vấn đề | Lựa chọn | Ghi chú |
|--------|----------|---------|
| Correlation | **Tính full mixed-type** theo `run_corr_v2.py` | Spearman (num×num), Cramér's V (cat×cat), Eta (mixed) |
| TARGET detection | **Auto-detect heuristic** | Cột boolean 2 giá trị hoặc tên khớp `churn\|attrition\|target\|label\|y\|default\|fraud`. Cho override sau qua param. |
| Lưu JSON | **Tạm thời ghi ra FILE**, chưa lưu Redis | Ghi `multivariate.json` vào temp dir của job; tải về qua dev endpoint. Lưu Redis tính sau. |
| Thời điểm chạy | **Trong `run_eda`, tự động** sau `summary_md` | Nối thêm vào cuối hàm async, 1 lần upload xong hết. |

---

## Luồng mới — `run_eda` async tuyến tính (giữ nguyên kiểu hiện tại)

```
[POST /eda/analyze] → background task → run_eda(job_id, file_path, session_id)

  # ===== GIỮ NGUYÊN =====
  df_clean, num_stats, cat_stats, date_stats, profile_text, high_corr
        = analyze_and_clean_data(df)              [CPU, executor]
  summary_md = generate_summary_md(...)           [fast, executor]
  summary_md += await call_llm_for_insight(...)   [LLM]
  await eda_store.set_eda_result(job_id, payload)
  chunks = fixed_size_chunk(summary_md)
  embeddings = await embed(chunks)
  await eda_store.set_eda_chunks(job_id, chunks, embeddings)   # vector store
  await retriever.upsert_chunks(...)

  # ===== THÊM MỚI (nối tiếp, cùng hàm) =====
  assoc = await loop.run_in_executor(None, eda_corr.build_association, df_clean)  [CPU]
        → matrix + truth_table_md + col roles + target_col
  raw = await llm.generate_text(build_multivariate_prompt(                        [LLM]
            summary_md, assoc["truth_table_md"], assoc["target_col"]))
  items = parse_multivariate(raw)                 # list[dict], fallback []
  # Ghi thẳng ra file (CHƯA lưu Redis) — cùng temp dir với cleaned.csv:
  #   {tempdir}/eda/{job_id}/multivariate.json
  write_json_file(job_id, items)

  await eda_store.set_eda_status(job_id, "done")
```

Đường dẫn file dùng lại `tmp_dir = {tempdir}/eda/{job_id}` đã có sẵn trong `run_eda` (chỗ ghi
`cleaned.csv`). Lưu thêm `multivariate.json` (và tùy chọn `truth_table.md`) vào đó.

Xử lý lỗi: 2 bước mới bọc trong `try/except` riêng — fail thì set field rỗng (`truth_table_md=""`,
`multivariate=[]`) + log, **vẫn** `set_eda_status("done")`. EDA gốc + chunking không bị ảnh hưởng
(graceful degrade, giống pattern fallback của `call_llm_for_insight` hiện tại).

---

## Các bước triển khai

### Bước 1 — `requirements.txt`
Thêm `scipy>=1.11.0` (cần `scipy.stats.chi2_contingency` cho Cramér's V). Hiện chỉ có numpy/pandas.

---

### Bước 2 — `app/core/eda_corr.py` (TẠO MỚI)

Port logic từ `run_corr_v2.py`, bỏ phần I/O file, trả về object thuần để node dùng:

```python
def cramers_v(col_a, col_b) -> float
def eta_squared(categorical_col, numeric_col) -> float

def classify_columns(df) -> tuple[list[str], list[str], list[str], list[str]]
    # → (valid_cols, numeric_cols, categorical_cols, excluded_cols)
    # loại high-cardinality id (tên khớp id/index/_id/name/email…, hoặc unique_ratio > 0.5)

def detect_target(df, categorical_cols) -> str | None
    # boolean 2-value hoặc tên khớp churn|attrition|target|label|y|default|fraud

def build_association(df) -> dict
    # {
    #   "matrix": DataFrame,           # full NxN, score 0..1
    #   "truth_table_md": str,         # markdown bảng cặp (threshold >= 0.20)
    #   "valid_cols", "numeric_cols", "categorical_cols", "target_col",
    #   "pairs": [(a, b, score, type), ...]   # upper-triangle, sorted desc
    # }
```

- Ngưỡng truth table: `>= 0.20` (script gốc để 0.00 — nâng lên để cắt nhiễu cho LLM).
- `truth_table_md` chính là **input document #2** cho bước multivariate.

---

### Bước 3 — `app/model/prompts/multivariate.py` (TẠO MỚI)

Đưa prompt "Semantic Multivariate Potential Analysis" của bạn vào đây dưới dạng template:

```python
def build_multivariate_prompt(dataset_profile_md: str, truth_table_md: str, target_col: str | None) -> str
```

- `dataset_profile_md` = `summary_md` (đã có column metadata, types, top-5, stats).
- `truth_table_md` = output node correlation.
- Chèn `target_col` đã detect vào để ép rule "TARGET phải ghép mọi cột, score ≥ 4".
- Giữ nguyên JSON Schema + SymPy Integration Rule + Pair Evaluation Logic trong prompt.

---

### Bước 4 — Parse JSON array (`app/validation/`)

Prompt yêu cầu **raw JSON array** (không markdown fence). Lưu ý:
- **Không dùng** `generate` (json_mode) vì `response_format=json_object` ép top-level là object,
  không nhận array. Dùng `generate_text` (plain) với `max_tokens` cao (~4096+).
- Viết `parse_multivariate(raw: str) -> list[dict]`: strip ```` ```json ```` fence nếu LLM lỡ thêm,
  `json.loads`, validate tối thiểu (mỗi item có `comparison_pair`, `evaluation.confidence_score`).
  Parse fail → trả `[]` + log (graceful).
- (Tùy chọn) Pydantic model `MultivariateUseCase` để validate schema chặt.

---

### Bước 5 — Ghi file (KHÔNG đụng Redis lúc này)

Thêm helper (trong `eda_pipeline.py` hoặc `eda_corr.py`):
```python
def write_json_file(job_id: str, items: list[dict]) -> str
    # ghi {tempdir}/eda/{job_id}/multivariate.json, trả về path
def multivariate_path(job_id: str) -> str
    # helper để dev endpoint biết đường dẫn file
```

**Redis KHÔNG đổi** so với hiện tại — không thêm key `multivariate`:
```
session:{session_id}
eda:active:{session_id}
eda:{job_id}:status
eda:{job_id}:result
eda:{job_id}:chunks
```
(Khi nào cần persist/đưa cho agent thì mới thêm `eda:{job_id}:multivariate` sau.)

---

### Bước 6 — `app/core/eda_pipeline.py` (SỬA — nối thêm vào `run_eda`, KHÔNG đổi cấu trúc)

- Giữ nguyên `run_eda` async tuyến tính + các hàm `analyze_and_clean_data`,
  `generate_summary_md`, `call_llm_for_insight`.
- Sau khối chunking hiện có, **nối thêm**: `eda_corr.build_association` (qua `run_in_executor`
  vì CPU-bound) → `llm.generate_text(build_multivariate_prompt(...))` → `parse_multivariate` →
  `write_json_file(job_id, items)` (ghi ra file, **không** Redis).
- Bọc `try/except` riêng cho cụm mới để không làm hỏng flow gốc.

---

### Bước 7 — `app/api/schemas.py`

- Không bắt buộc đổi (JSON chưa vào `EDAResult`). Có thể bỏ qua bước này lúc này.

---

### Bước 8 — `app/api/routes/eda.py`

- Không đổi. `/eda/result/{job_id}` giữ nguyên (JSON tải qua dev endpoint, không nhét vào result).

---

### Bước 9 — `app/api/routes/dev.py` (SỬA — endpoint tải JSON)

- `GET /dev/eda/multivariate/{job_id}` — **tải file** `multivariate.json` về bằng `FileResponse`
  (`media_type="application/json"`, `filename=multivariate_{job_id}.json`). 404 nếu file chưa có.
- (Tùy chọn) `GET /dev/eda/truth-table/{job_id}` — tải/xem `truth_table.md` nếu có ghi kèm.

---

### Bước 10 — (Sau này, chưa làm)

Khi cần đưa cho QA agent: thêm `eda:{job_id}:multivariate` vào Redis (Bước 5 cũ) + node load
trong `pipeline.py`. Ngoài phạm vi lần này.

---

## Thứ tự implement

| # | File | Loại |
|---|------|------|
| 1 | `requirements.txt` (scipy) | Sửa |
| 2 | `app/core/eda_corr.py` (corr + `write_json_file`) | Tạo mới |
| 3 | `app/model/prompts/multivariate.py` | Tạo mới |
| 4 | `app/validation/` parse_multivariate | Tạo mới |
| 5 | `app/core/eda_pipeline.py` (nối thêm vào `run_eda`) | Sửa |
| 6 | `app/api/routes/dev.py` (endpoint tải JSON) | Sửa |

*(KHÔNG đụng `eda_store.py`, `schemas.py`, `routes/eda.py` lần này — JSON chỉ ghi file + tải qua dev.)*

---

## Rủi ro / lưu ý

- **scipy nặng**: build Docker lâu hơn + image to hơn. Chấp nhận được.
- **LLM JSON không hợp lệ**: prompt phức tạp, model nhỏ dễ trả JSON lỗi/cụt token. Cần
  `parse_multivariate` chịu lỗi + `max_tokens` đủ lớn. Có thể thêm 1 retry.
- **TARGET auto-detect sai** trên dataset mơ hồ: chấp nhận default heuristic, để ngỏ override
  qua param `target_column` ở `/eda/analyze` nếu cần chính xác.
- **N×N association** O(N²) cặp × chi-square: với N cột vừa phải (<50) thì ổn; dataset rất nhiều
  cột nên cap N hoặc chạy executor (đã tính).
```
