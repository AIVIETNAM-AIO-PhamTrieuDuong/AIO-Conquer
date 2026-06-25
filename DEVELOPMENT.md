# Development Guide — Procedural Extension Pattern

Pipeline hiện tại là base tối giản: nhận câu hỏi → gọi Ollama → trả JSON.
Mọi tính năng mới đều được thêm vào theo **cùng một pattern**: thêm field vào State → viết node → nối edge vào graph.
Không phá vỡ node cũ, không breaking changes.

---

## Cấu trúc graph hiện tại

```
[load_history] → [load_eda_context] → [column_metadata]
    → [missingness_summary] → [type_compatibility]
    → [basic_statistical_summary] → [statistical_association]
    → [custom_metric]
    → [generate] → [parse] → [save_memory] → END
```

`app/core/pipeline.py` là file chính khi mở rộng pipeline. Nếu thay đổi public
request schema, dependency, dev endpoint, hoặc memory contract thì cập nhật
file liên quan tối thiểu theo scope được duyệt.

## LangGraph global state

`app/core/pipeline.py` dùng `GraphState` làm state chung của graph. Các node
chỉ return dict chứa field chúng cập nhật để LangGraph merge vào state hiện
tại.

Các field hiện tại:

- `question`: câu hỏi đầu vào.
- `session_id`: LangGraph `thread_id` và EDA context đang hoạt động.
- `history`: lịch sử hội thoại dạng `{"q": ..., "a": ...}` từ LangGraph
  checkpoint state.
- `context`: EDA summary context đang hoạt động, hoặc chuỗi rỗng.
- `eda_result`: payload EDA có cấu trúc của job đang hoạt động.
- `dataset_id`: job id / dataset id đang hoạt động.
- `dataset_file_path`: đường dẫn CSV đã làm sạch từ EDA memory.
- `tool_requests`: danh sách `ToolRequest` dạng dict đã gọi trong graph.
- `tool_results`: danh sách `ToolResult` dạng dict đã trả về trong graph.
- `statistical_findings`: kết quả thống kê có cấu trúc cho downstream nodes.
- `warnings`: cảnh báo từ tool hoặc node trong graph run hiện tại.
- `prompt`: prompt cuối cùng đã gửi tới LLM.
- `raw_response`: raw JSON text từ LLM.
- `response`: dict JSON-safe đã parse từ `QAResponse`, hoặc `None` trước bước
  parse.

---

## Pattern: Thêm tính năng mới

### Bước 1 — Thêm field vào GraphState

```python
class GraphState(TypedDict):
    question: str
    history: list[dict]
    raw_response: str
    response: Optional[QAResponse]
    # --- thêm vào đây ---
    your_new_field: str
```

### Bước 2 — Viết node mới

Node nhận state, trả về **dict chỉ gồm key nó thay đổi**.
LangGraph merge phần còn lại tự động — không cần return toàn bộ state.

```python
async def node_your_feature(state: GraphState) -> dict:
    result = do_something(state["question"])
    return {"your_new_field": result}
```

### Bước 3 — Nối vào graph

```python
g.add_node("your_feature", node_your_feature)

# Chèn vào giữa 2 node có sẵn — không đụng node khác
g.add_edge("load_history", "your_feature")   # thay vì load_history → generate
g.add_edge("your_feature", "generate")
```

---

## Thêm Tool nội bộ (Z3, SymPy, calculator...)

Tool là node thực hiện tác vụ ngoài LLM. Viết hàm async trong `app/tools/`, gọi trong node.

`app/tools/dataset_profile.py` hiện có `DatasetProfileTool` cho các tool
profiling tabular deterministic sau, dùng chung `ToolRequest` / `ToolResult`:

- `tabular.dataset_profile`
- `tabular.column_metadata`
- `tabular.missingness_summary`
- `tabular.type_compatibility`

`app/tools/statistics.py` hiện có `StatisticalAnalysisTool` cho các tool
statistical deterministic sau, dùng chung `ToolRequest` / `ToolResult`:

- `stats.correlation`
- `stats.basic_summary`
- `stats.custom_metric`

QA graph hiện gọi các tool này bằng node riêng sau `load_eda_context` nếu EDA
memory có `cleaned_file_path`. Mỗi node append request/result vào
`tool_requests` và `tool_results`, ghi finding vào `statistical_findings`, gom
warnings vào `warnings`, và thêm summary ngắn vào `context` trước `generate`.

```python
# app/tools/z3_solver.py
from z3 import Solver, Real, sat

def solve_constraints(constraints: dict) -> dict:
    s = Solver()
    # ... thêm constraint vào solver
    return {"sat": s.check() == sat, "model": str(s.model())}
```

Z3 là blocking (không async native) — bọc bằng `run_in_executor`:

```python
# pipeline.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.tools.z3_solver import solve_constraints

_executor = ThreadPoolExecutor(max_workers=2)

async def node_z3(state: GraphState) -> dict:
    loop = asyncio.get_event_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(_executor, solve_constraints, state["constraints"]),
        timeout=10.0,
    )
    return {"tool_result": result}
```

Dùng `add_conditional_edges` nếu tool chỉ chạy trong một số trường hợp:

```python
def route_after_parse(state: GraphState) -> str:
    if state.get("needs_z3"):
        return "z3"
    return "save_memory"

g.add_conditional_edges("parse", route_after_parse, {"z3": "z3", "save_memory": "save_memory"})
g.add_edge("z3", "save_memory")
```

> **FastMCP**: chỉ cần khi tool là microservice độc lập cần reuse cross-service. Đa số case không cần.
> **`@tool` decorator (LangChain)**: chỉ dùng khi muốn LLM tự quyết gọi tool — không dùng trong pipeline này.

---

## MoE (Mixture of Experts)

MoE trong pipeline này chỉ là gọi `llm.generate()` nhiều lần với prompt khác nhau.
Không cần thay đổi gì về memory hay session.

### Graph sau khi thêm MoE

```
[load_history] → [route] → [physics_expert]  ↘
                        ↘ [rules_expert]   → [aggregate] → [parse] → [save_memory] → END
```

### Bước 1 — Thêm field vào State

```python
class GraphState(TypedDict):
    ...
    expert: str                        # "physics" | "rules"
    expert_response: str               # raw output từ expert được chọn
```

### Bước 2 — Prompt files

```
app/model/prompts/
    qa_system.py       ← hiện tại
    router.py          ← build_router_prompt()
    expert_physics.py  ← build_physics_prompt()
    expert_rules.py    ← build_rules_prompt()
```

Mỗi file chỉ chứa text và build function, không có logic.

### Bước 3 — Viết nodes

```python
from app.model.prompts.router import build_router_prompt
from app.model.prompts.expert_physics import build_physics_prompt
from app.model.prompts.expert_rules import build_rules_prompt

async def node_route(state: GraphState) -> dict:
    prompt = build_router_prompt(state["question"])
    expert = await llm.generate(prompt)   # trả về "physics" hoặc "rules"
    return {"expert": expert.strip()}

async def node_physics_expert(state: GraphState) -> dict:
    prompt = build_physics_prompt(state["question"], history=state["history"])
    return {"expert_response": await llm.generate(prompt)}

async def node_rules_expert(state: GraphState) -> dict:
    prompt = build_rules_prompt(state["question"], history=state["history"])
    return {"expert_response": await llm.generate(prompt)}
```

### Bước 4 — Nối vào graph

```python
def route_to_expert(state: GraphState) -> str:
    return state["expert"]   # "physics" | "rules"

g.add_node("route", node_route)
g.add_node("physics", node_physics_expert)
g.add_node("rules", node_rules_expert)

g.add_edge("load_history", "route")
g.add_conditional_edges("route", route_to_expert, {"physics": "physics", "rules": "rules"})
g.add_edge("physics", "parse")
g.add_edge("rules", "parse")
```

---

## Retrieval với Pinecone

Retriever hiện tại là stub — trả `""` nếu `PINECONE_API_KEY` không set.
Khi cần retrieval, thêm field `context` vào State và thêm node:

```python
class GraphState(TypedDict):
    ...
    context: str   # retrieved docs, mặc định ""
```

```python
async def node_retrieve(state: GraphState) -> dict:
    context = await retriever.retrieve(state["question"])
    return {"context": context}
```

`build_prompt()` đã có sẵn param `context=""` — chỉ cần truyền vào:

```python
async def node_generate(state: GraphState) -> dict:
    prompt = build_prompt(state["question"], context=state["context"], history=state["history"])
    return {"raw_response": await llm.generate(prompt)}
```

Fill logic embedding vào `app/retrieval/retriever.py` khi sẵn sàng:

```python
async def retrieve(self, query: str, top_k: int = 3) -> str:
    embedding = await embed(query)
    results = self._index.query(vector=embedding, top_k=top_k, include_metadata=True)
    chunks = [m["metadata"]["text"] for m in results["matches"]]
    return "\n\n".join(chunks)
```

---

## Short-term Memory

Conversation memory dùng LangGraph checkpointer với Redis backend
(`RedisSaver`). `/ask` nhận `thread_id` optional; nếu client không gửi thì
dùng active EDA `job_id` từ `/eda/analyze`; nếu chưa có active EDA job thì
fallback về `SESSION_ID = "default"`. `thread_id` được truyền vào
`config={"configurable": {"thread_id": ...}}` khi gọi graph, đồng thời được
dùng làm `session_id` để EDA active context đi cùng conversation thread.
Nếu Redis hiện tại không hỗ trợ Redis Stack search commands như `FT._LIST`,
pipeline fallback sang LangGraph in-process checkpointer để `/ask` vẫn chạy
theo `thread_id` trong process hiện tại.

`history` nằm trong checkpointed `GraphState`, không còn được đọc/ghi qua
custom Redis key `session:{session_id}` trong QA graph path. `node_load_history`
chỉ normalize history đã được LangGraph restore, còn `node_save_memory` append
turn cuối dạng `{"q": question, "a": answer}` và trim theo
`MAX_HISTORY_TURNS`.

`app/api/routes/dev.py` dùng `reset_conversation_thread()` để xóa checkpoints
của thread được chọn. Redis checkpointer retention theo backend mặc định;
`SESSION_TTL` vẫn áp dụng cho EDA/session stores hiện có.

Cấu hình trong `.env`:

```
SESSION_TTL=3600        # giây — áp dụng cho EDA/session stores hiện có
MAX_HISTORY_TURNS=5     # giữ tối đa 5 lượt Q&A gần nhất
```

---

scratchpad

## Agentic workflow cho phân tích đa biến

Phần này mô tả pattern mở rộng pipeline cho bài toán phân tích đa biến.
Đây là thiết kế mục tiêu, không phải mô tả tính năng đã được triển khai.
Workflow vẫn tuân theo nguyên tắc chung: dữ liệu đi qua `GraphState`, mỗi agent
là một node có trách nhiệm hẹp, và router dùng conditional edges để chọn
đường chạy.

### Graph mục tiêu

```text
[load_history] → [augment_context] → [domain_router]
                                          ↓
                     ┌────────────────────┼────────────────────┐
                     ↓                    ↓                    ↓
             [statistical_agent]   [feature_agent]      [domain_agent]
               [internal]           [internal]           [internal]
                     └────────────────────┼────────────────────┘
                                          ↓
                                  [update_whiteboard]
                                          ↓
                                  [save_memory] → END
```

Router chỉ kích hoạt các agent cần thiết. `quality_gate` yêu cầu human review
khi kết quả có rủi ro cao, thiếu dữ liệu, mâu thuẫn giữa các agent, hoặc cần
quyết định nghiệp vụ mà hệ thống không được tự suy diễn.

### State cho agentic workflow

Các field nên dùng kiểu dữ liệu có cấu trúc để agent và tool không phải parse
free-form text:

```python
class GraphState(TypedDict):
    ...
    domain: str
    analysis_plan: list[str]
    selected_agents: list[str]
    dataset_profile: dict
    statistical_findings: list[dict]
    feature_findings: list[dict]
    domain_context: list[dict]
    whiteboard: dict
    tool_calls: list[dict]
```

Không lưu chain-of-thought hoặc suy luận ẩn trong state. `whiteboard` chỉ chứa
artifact có thể kiểm tra: giả thuyết, biến số, assumptions, nguồn context,
kết quả tool, cảnh báo, quyết định đã được duyệt và câu hỏi còn mở.

### Tool-calling theo statistical awareness

`statistical_agent` chọn tool dựa trên loại biến, phân phối, kích thước mẫu,
missingness và assumptions của phép kiểm định. Tool phải trả kết quả có cấu
trúc, bao gồm input summary, method, assumptions, metrics, warnings và error.

Nhóm tool tối thiểu:

- Data profiling: kiểu biến, missing values, cardinality và outlier summary.
- Dependency analysis: correlation, covariance và association cho biến
  numeric/categorical phù hợp.
- Statistical tests: normality, homoscedasticity, independence và kiểm định
  giả thuyết được router cho phép.

Agent không được chọn phép kiểm định chỉ từ tên câu hỏi. Trước khi gọi tool,
agent phải ghi `method`, `required_inputs` và `assumptions` vào
`analysis_plan`.

```python
async def node_statistical_agent(state: GraphState) -> dict:
    plan = build_statistical_plan(
        state["question"],
        state["dataset_profile"],
        state["domain_context"],
    )
    results = await statistical_tools.execute(plan)
    return {
        "analysis_plan": plan["steps"],
        "statistical_findings": results,
        "tool_calls": plan["tool_calls"],
    }
```

### Tool-calling theo feature awareness

`feature_agent` đánh giá vai trò và chất lượng của feature trước khi mô hình
hóa. Agent phải phân biệt target, predictor, identifier, timestamp, group,
confounder và biến có nguy cơ leakage.

Nhóm kiểm tra chính:

- Feature typing và semantic role detection.
- Missingness mechanism và missingness theo subgroup.
- Leakage, duplicate signal và post-outcome feature detection.
- Multicollinearity, redundancy và interaction candidates.
- Scaling, encoding, transformation và domain constraints.
- Stability theo thời gian, cohort hoặc data source.

Feature proposal phải giữ provenance: feature gốc, phép biến đổi, lý do,
constraint và cảnh báo. 

### Scratchpad/whiteboard cho context augmentation

`augment_context` khởi tạo whiteboard từ câu hỏi, history, dataset profile và
domain context được retrieval. Mỗi agent chỉ cập nhật namespace của mình để
tránh ghi đè kết quả của agent khác.

```python
whiteboard = {
    "objective": "",
    "dataset": {},
    "domain": {
        "facts": [],
        "constraints": [],
        "sources": [],
    },
    "hypotheses": [],
    "statistics": {
        "methods": [],
        "findings": [],
        "warnings": [],
    },
    "features": {
        "roles": {},
        "transformations": [],
        "warnings": [],
    },
    "decisions": [],
    "open_questions": [],
}
```

Context augmentation phải lưu nguồn và scope của mỗi fact. Retrieved context
không được coi là đúng mặc định; agent phải đánh dấu conflict, stale context
hoặc fact không đủ bằng chứng. Whiteboard là context làm việc có giới hạn,
không thay thế Redis conversation history và không lưu raw sensitive data nếu
không cần cho kết quả cuối.

### Dynamic workflow theo domain

`domain_router` tạo execution plan từ objective, dataset profile, domain và
risk level. Mapping domain phải explicit và có fallback:

```python
def route_domain_agents(state: GraphState) -> list[str]:
    selected = ["statistical_agent", "feature_agent"]
    if state["domain"] in {"finance", "healthcare", "legal"}:
        selected.append("domain_agent")
    return selected
```

Domain agent cung cấp constraints, terminology, acceptable assumptions và
validation rules; agent này không thay statistical evidence. Nếu domain chưa
được hỗ trợ, router dùng workflow tổng quát, ghi limitation vào whiteboard và
không tự tạo rule nghiệp vụ.

Dynamic routing phải có giới hạn số vòng lặp và số tool call. Agent chỉ được
re-plan khi có tool error, evidence conflict, failed assumption hoặc human
feedback mới. Mỗi lần re-plan phải ghi lý do vào `decisions`.

### Quy tắc tổng hợp kết quả

`synthesize` chỉ sử dụng finding có provenance từ tool, domain context hoặc
human decision. Kết quả cuối phải phân biệt:

- Evidence đã quan sát.
- Statistical inference và assumptions.
- Feature/domain interpretation.
- Uncertainty, limitations và unresolved questions.
- Human-approved decisions, nếu có.

Không đưa raw whiteboard, hidden reasoning hoặc context không liên quan vào
response cuối.

---

## Checklist khi thêm tính năng

- [ ] Thêm field vào `GraphState` nếu cần truyền data qua nodes
- [ ] Viết `node_xxx()` — chỉ return dict của key nó thay đổi
- [ ] Thêm prompt file vào `app/model/prompts/` nếu cần prompt mới
- [ ] Thêm `add_node` + `add_edge` (hoặc `add_conditional_edges`) vào `_build_graph()`
- [ ] Không sửa node đang hoạt động — thêm node mới và chèn vào graph
- [ ] Dùng `run_in_executor` nếu tool là blocking (Z3, SymPy...)
- [ ] Tool thống kê kiểm tra assumptions và trả provenance/warnings có cấu trúc
- [ ] Feature agent kiểm tra semantic role, leakage và domain constraints
- [ ] Whiteboard chỉ lưu artifact có thể kiểm tra, không lưu chain-of-thought
- [ ] Dynamic routing có fallback, giới hạn vòng lặp và giới hạn tool call
- [ ] High-risk hoặc ambiguous decision đi qua human review
- [ ] Test bằng `curl http://localhost:8000/ask` sau mỗi thay đổi

---

## Verification Notes

- 2026-06-25: Đã chuyển conversation memory của QA graph sang LangGraph
  Redis checkpointer (`RedisSaver`) theo `thread_id`; `/ask` nhận
  `thread_id` optional, graph invoke truyền `configurable.thread_id`, và
  `history` được append trong checkpointed `GraphState` thay vì custom Redis
  key `session:{session_id}`. Đã chạy `python -m py_compile
  app\api\schemas.py app\core\pipeline.py app\graph\nodes.py
  app\api\routes\dev.py` thành công.
- 2026-06-25: Đã xử lý Redis không có Redis Stack/RediSearch command
  `FT._LIST` bằng fallback sang LangGraph in-process checkpointer khi
  `RedisSaver.setup()` báo unknown `FT.*` command.
- 2026-06-25: `/ask` hiện trả về final `GraphState` thay vì `QAResponse`
  trực tiếp; `response` trong state là dict JSON-safe để tránh LangGraph
  checkpoint deserialize warning cho unregistered `QAResponse`.
- 2026-06-25: `/ask` mặc định dùng active EDA `job_id` làm LangGraph
  `thread_id` và bind active EDA job vào thread đó trước khi gọi graph. Đã
  chạy `python -m py_compile app\api\routes\ask.py` thành công.
- 2026-06-24: Sau khi nối profiling tool nodes vào QA graph, đã chạy
  `python -m py_compile app\graph\schema.py app\graph\nodes.py
  app\core\pipeline.py` thành công. Node-level smoke test với CSV tạm và stub
  service ngoài xác nhận 3 tool nodes append `tool_requests` / `tool_results`
  và type compatibility trả compatible cho case compare category + numeric.
- 2026-06-24: Đã hoàn tất migration profiling tools vào `DatasetProfileTool`
  trong `app/tools/dataset_profile.py`; `app/tools/statistics.py` không còn
  export compatibility alias và được giữ cho statistical analysis tools.
- 2026-06-24: Đã hoàn tất Milestone 1 statistical tools trong
  `StatisticalAnalysisTool`: `stats.correlation`, `stats.basic_summary`, và
  `stats.custom_metric`. Đã chạy `python -m py_compile app\tools\statistics.py
  app\tools\__init__.py app\graph\nodes.py app\core\pipeline.py`, direct
  temp-CSV smoke test cho 3 tool, và graph-node smoke test với service stubs.
