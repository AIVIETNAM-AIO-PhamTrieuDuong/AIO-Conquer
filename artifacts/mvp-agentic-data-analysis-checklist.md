# MVP Agentic Data Analysis Checklist

This checklist defines the MVP milestones for a LangGraph-based, JSON-first,
Python-tool-based data analysis workflow.

The MVP focuses on:

- Tool schemas and deterministic helper tools.
- LangGraph agent orchestration.
- Redis memory key/API contracts.
- A query agent that reads structured EDA memory and updates global state.
- A feature agent that calls statistical tools and updates global state.
- A domain agent that adds actionable requirements and ambiguity handling.

## Tech Stack

- **Agent orchestration:** LangGraph coordinates agents, graph state,
  conditional routing, tool execution nodes, and state transitions.
- **Memory backend:** Redis stores all memory categories, including
  conversation memory, dataset memory, tool memory, agent working memory,
  curated context memory, and error memory.
- **Tool runtime:** Python-based tools run approved CSV/tabular and statistical
  operations through the unified tool interface.

## Global State Note

The global state schema is provisional. It should remain easy to change while
the team validates agent responsibilities, tool outputs, memory boundaries, and
handoff contracts.

For the MVP, every agent or tool that writes to global state must document:

- Which state keys it reads.
- Which state keys it writes.
- Whether the write replaces, appends, or merges data.
- What provenance is attached to the update.
- What downstream node consumes the update.

## MVP Scope

### In Scope

- JSON-first contracts for tool requests, tool results, agent requirements,
  and state updates.
- Deterministic Python helper tools for CSV/tabular handling and statistical
  computation.
- LangGraph used as the agent orchestration framework.
- Redis key/API contracts configured by purpose rather than one generic memory
  bucket.
- Query, feature, and domain agents with explicit tool usage.
- Global state updates after each agent/tool stage.
- Validation for single-variable and multivariate scenarios.

### Out Of Scope

- Arbitrary unrestricted Python execution.
- Finalized production global state schema.
- UI changes unless required for existing endpoint verification.
- Hardcoded dataset-specific behavior.
- Broad modeling stack beyond what the MVP tools require.
- Model-native LLM tool calling as a required dependency.

## Milestone 1: Tool Schemas And Deterministic Helper Tools

Goal: define the tool schemas first, then implement deterministic helper tools
for statistics and CSV/tabular data handling. MVP tools should be callable by
name with validated JSON inputs and should not require model-native tool
calling.

### Required Tools

- [ ] CSV/dataframe loader tool.
- [ ] Dataset profile tool.
- [ ] Column metadata tool.
- [ ] Missingness summary tool.
- [ ] Type compatibility tool.
- [ ] Correlation/statistical association tool.
- [ ] Basic statistical summary tool.
- [ ] Deterministic custom metric helper for approved simple calculations.

### Tool Interface Tasks

- [ ] Define `ToolRequest` JSON schema.
- [ ] Define `ToolResult` JSON schema.
- [ ] Require `tool_name`, `request_id`, `caller`, `purpose`, `inputs`, and
  `expected_output_schema` in every request.
- [ ] Require `status`, `data`, `summary`, `warnings`, `error`, and
  `provenance` in every result.
- [ ] Normalize error types: invalid column, incompatible type, insufficient
  rows, missing values, timeout, invalid code, invalid result shape, and
  unsupported method.
- [ ] Require every tool result to identify dataset id, source columns, rows
  used, and preprocessing notes when applicable.
- [ ] Support approved Python backends such as `pandas`, `numpy`, `scipy`, and
  `sympy` without restricting the MVP to Sympy.
- [ ] Prefer deterministic helper functions over generated Python execution
  for the first MVP pass.
- [ ] Add a future extension point for constrained Python execution without
  making it part of the MVP critical path.

### Suggested Request Shape

```json
{
  "tool_name": "stats.correlation",
  "request_id": "string",
  "caller": "feature_agent",
  "purpose": "Measure relationship between two numeric columns.",
  "inputs": {
    "dataset_id": "active",
    "columns": ["sales", "profit"],
    "method": "pearson"
  },
  "expected_output_schema": "ToolResult"
}
```

### Suggested Result Shape

```json
{
  "tool_name": "stats.correlation",
  "request_id": "string",
  "status": "ok",
  "data": {
    "method": "pearson",
    "correlation": 0.72,
    "p_value": 0.001
  },
  "summary": "sales and profit have a positive relationship.",
  "warnings": [],
  "error": null,
  "provenance": {
    "dataset_id": "active",
    "columns": ["sales", "profit"],
    "rows_used": 1000,
    "missing_rows_dropped": 0
  }
}
```

### Exit Criteria

- [ ] Every MVP tool can be called through the same request/result envelope.
- [ ] Tool failures are machine-readable.
- [ ] Tool results can be written into global state without parsing free-form
  text.
- [ ] No MVP path depends on unrestricted generated Python code.

## Milestone 2: Redis Memory Key/API Contracts

Goal: define Redis key patterns and API contracts before expanding agent logic.
Each memory kind should have a purpose, owner, lifecycle, and predictable
serialization format while LangGraph orchestrates read/write timing.

### Memory Types

- [ ] Conversation memory: user questions and final assistant answers.
- [ ] Dataset memory: active dataset id, cleaned file path, shape, profile, and
  metadata.
- [ ] Tool memory: tool requests, tool results, warnings, and provenance.
- [ ] Agent working memory: active requirements, assumptions, intermediate
  findings, and unresolved questions.
- [ ] Curated context memory: validated reusable facts and decisions.
- [ ] Error memory: recoverable failures, blocked tasks, and retry context.

### LangGraph And Redis Tasks

- [ ] Define Redis key pattern for each memory type.
- [ ] Define read API for each memory type.
- [ ] Define write API for each memory type.
- [ ] Define update semantics: replace, append, merge, or expire.
- [ ] Decide which memory fields belong in LangGraph state for one run.
- [ ] Decide which memory fields persist outside the run in Redis.
- [ ] Define read/write policy for each memory type.
- [ ] Define TTL or lifecycle for temporary memory.
- [ ] Define merge behavior for concurrent or repeated writes.
- [ ] Add provenance requirements for memory writes.
- [ ] Define compaction rules for large tool outputs or long context.
- [ ] Add `schema_version` or equivalent metadata to memory payloads.

### Provisional State Fields

These fields are a starting point, not a final schema:

```python
class AnalysisState(TypedDict):
    question: str
    dataset_id: str
    dataset_profile: dict
    domain_requirements: dict
    query_plan: dict
    feature_plan: dict
    tool_requests: list[dict]
    tool_results: list[dict]
    findings: list[dict]
    warnings: list[str]
    global_state_updates: list[dict]
    response: dict
```

### Exit Criteria

- [ ] Each memory type has a clear purpose.
- [ ] Agents know where to read context from.
- [ ] Agents know where to write outputs.
- [ ] Temporary working context is separated from curated reusable context.
- [ ] Redis memory payloads include schema/version metadata.

## Milestone 3: Query Agent Reads Structured EDA Memory

Goal: complete the query agent so it reads structured EDA memory, maps user
wording to dataset columns, uses lightweight helper tools when needed, and
updates global state.

### Responsibilities

- [ ] Parse the user question into retrieval intent.
- [ ] Load structured EDA memory: `num_stats`, `cat_stats`, shape, profile text,
  cleaned file path, and summary metadata.
- [ ] Identify candidate columns from structured dataset memory.
- [ ] Retrieve relevant column metadata, dataset profile, and prior findings
  without relying on Markdown-only parsing.
- [ ] Detect whether the question is single-variable or multivariate.
- [ ] Select tools needed for tabular retrieval or lightweight validation.
- [ ] Emit `ToolRequest` JSON for every tool call.
- [ ] Consume `ToolResult` JSON.
- [ ] Update global state with query intent, candidate columns, retrieval
  evidence, confidence, warnings, and open questions.

### Tools Used

- [ ] Dataset profile tool.
- [ ] Column metadata tool.
- [ ] Type compatibility tool.
- [ ] CSV/dataframe preview or filtered retrieval tool.
- [ ] Optional text retrieval tool for stored Markdown/profile context as a
  supplement, not the source of truth.

### Global State Updates

- [ ] `query_plan`
- [ ] `candidate_columns`
- [ ] `retrieved_context`
- [ ] `tool_requests`
- [ ] `tool_results`
- [ ] `warnings`
- [ ] `open_questions`

### Exit Criteria

- [ ] Query agent can map user wording to likely dataset columns.
- [ ] Query agent can retrieve structured EDA context for downstream agents.
- [ ] Query agent does not fabricate unavailable columns or metrics.
- [ ] Query agent writes machine-readable updates to global state.

## Milestone 4: Feature Agent Calls Statistical Tools

Goal: complete the feature agent so it evaluates candidate features by calling
deterministic statistical tools, records assumptions and warnings, and updates
global state.

### Responsibilities

- [ ] Consume query agent output and domain agent requirements.
- [ ] Validate feature compatibility for requested analysis.
- [ ] Determine required statistical computations.
- [ ] Select appropriate Python-based tools.
- [ ] Generate `ToolRequest` JSON for statistical computation.
- [ ] Consume `ToolResult` JSON.
- [ ] Produce feature findings with metric values, assumptions, warnings, and
  provenance.
- [ ] Update global state with feature roles, statistical findings, and
  unresolved issues.

### Tools Used

- [ ] Missingness summary tool.
- [ ] Type compatibility tool.
- [ ] Correlation/statistical association tool.
- [ ] Basic statistical summary tool.
- [ ] Deterministic custom metric helper for approved simple calculations.

### Global State Updates

- [ ] `feature_plan`
- [ ] `feature_roles`
- [ ] `statistical_findings`
- [ ] `tool_requests`
- [ ] `tool_results`
- [ ] `warnings`
- [ ] `open_questions`

### Exit Criteria

- [ ] Feature agent can compute or request required statistics for candidate
  features.
- [ ] Feature agent records assumptions and warnings.
- [ ] Feature agent returns structured findings that downstream synthesis can
  consume.
- [ ] Feature agent updates global state without relying on Markdown parsing.

## Milestone 5: Domain Agent Requirements And Ambiguity Handling

Goal: complete the domain agent so it adds actionable requirements and
ambiguity handling for the query and feature agents.

The domain agent should be domain-augmenting, not domain-hardcoded. It may infer
business meaning, constraints, and analysis intent from the question, dataset
profile, memory, and retrieved context, but it must preserve uncertainty when
the domain is unclear.

### Responsibilities

- [ ] Interpret the user question into an analysis objective.
- [ ] Identify domain terms and map them to candidate dataset concepts.
- [ ] Define actionable requirements for the query agent.
- [ ] Define actionable requirements for the feature agent.
- [ ] Identify required metrics, comparisons, filters, groups, and time windows.
- [ ] Identify ambiguity, missing context, or risky assumptions.
- [ ] Decide when the answer should ask for clarification instead of forcing an
  analysis path.
- [ ] Select tools needed to inspect dataset context or prior memory.
- [ ] Update global state with requirements, assumptions, constraints, and
  unresolved questions.

### Tools Used

- [ ] Dataset profile tool.
- [ ] Column metadata tool.
- [ ] Context/memory retrieval tool.
- [ ] Type compatibility tool when requirements imply specific variables.

### Global State Updates

- [ ] `domain_requirements`
- [ ] `analysis_objective`
- [ ] `required_metrics`
- [ ] `required_filters`
- [ ] `candidate_concepts`
- [ ] `assumptions`
- [ ] `warnings`
- [ ] `open_questions`

### Exit Criteria

- [ ] Domain agent produces requirements that query and feature agents can
  consume directly.
- [ ] Domain agent preserves ambiguity instead of forcing unsupported
  assumptions.
- [ ] Domain agent writes structured global state updates.
- [ ] Domain agent does not hardcode behavior for a single dataset.

## MVP Acceptance Checklist

- [ ] Essential statistics and CSV/tabular tools are defined.
- [ ] All MVP tools use one request/result interface.
- [ ] LangGraph is used as the agent orchestration framework.
- [ ] Redis key/API contracts are separated by purpose.
- [ ] All memory kinds are backed by Redis with schema/version metadata.
- [ ] Query agent reads structured EDA memory and updates global state.
- [ ] Feature agent calls statistical tools and updates
  global state.
- [ ] Domain agent converts user intent into actionable requirements and
  handles ambiguity before updating global state.
- [ ] Global state schema is treated as provisional during MVP development.
- [ ] Tool outputs and agent outputs are JSON-first.
- [ ] Errors and warnings are machine-readable.
- [ ] No downstream stage depends on parsing Markdown tables.
- [ ] No Python execution is unrestricted.

## Recommended MVP Build Order

```text
1. Tool schemas and deterministic helper tools
2. Redis memory key/API contracts
3. Query agent reads structured EDA memory
4. Feature agent calls statistical tools
5. Domain agent adds requirements and ambiguity handling
6. End-to-end global state update validation
```
