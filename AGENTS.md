# AGENTS.md

This file defines the mandatory working rules for coding agents in this
repository. It applies to the entire repository unless a more specific
`AGENTS.md` exists in a subdirectory.

## 1. Instruction Priority

Follow instructions in this order:

1. The user's explicit request.
2. This `AGENTS.md`.
3. `DEVELOPMENT.md` and other repository documentation.
4. Existing code, tests, and established local patterns.

If instructions conflict, stop before making the conflicting change and state
the conflict clearly. Do not silently choose a broader interpretation.

## 2. Required Context

Before planning or editing code:

1. Read this `AGENTS.md`.
2. Read `DEVELOPMENT.md`.
3. Inspect the target source file and the directly relevant tests or callers.
4. Check the current Git branch and working-tree status.

Naming compatibility for this repository:

- References to `AGENT.md` mean this canonical `AGENTS.md`.
- References to `DEVELOPMENTS.md` mean the existing canonical
  `DEVELOPMENT.md`.
- Do not create duplicate singular or plural variants merely to satisfy a
  filename reference.

Use these files as primary context. Verify their claims against the current
code when necessary, and preserve existing architecture and conventions.

## 3. Scope Discipline
- Implement only what the user explicitly requests.
- Never ask about, propose, recommend, or suggest new features unless the user explicitly asks for feature ideas or recommendations.
- Do not add optional enhancements, speculative abstractions, unrelated cleanup, opportunistic refactors, or future-facing hooks.
- Fix only defects required to complete the requested work.
- Do not broaden acceptance criteria without explicit user approval.
- Don't assume. Don't hide confusion. Surface tradeoffs.
- Before implementing:
  - State your assumptions explicitly. If uncertain, ask.
  - If multiple interpretations exist, present them - don't pick silently.
  - If a simpler approach exists, say so. Push back when warranted.
  - If something is unclear, stop. Name what's confusing. Ask.
- When ambiguity cannot be resolved safely, ask only the minimum clarifying question needed to proceed. Do not use that question to expand scope.
- For huge code base changes, refactor, impactful or big features, always decompose user's requests to smallest WBS and ask for confirmation from user before execution

## 4. File-Creation Rule

- Never create a new source file, test file, configuration file, migration,  script, fixture, generated artifact, or document unless the user explicitly
  requests creation of that file or the requested deliverable unavoidably is  that file.
- A request for a behavior change is not permission to create supporting
  files.
- Prefer modifying an existing appropriate file when the request permits it.
- Do not split code into new modules merely for style or organization.
- Do not create placeholder files, empty scaffolding, or documentation that
  was not requested.
- If completion genuinely requires a new file and the user did not authorize
  one, stop before creating it and request explicit approval.

## 5. Edit Boundary

Every code modification must satisfy all of the following:
- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
- It is bounded to one existing function or one existing class.
- For a class-bounded edit, changes may affect multiple methods only when all changed methods belong to that one class and are necessary for the request.
- Do not combine unrelated function-bounded and class-bounded changes.
- Do not make module-wide rewrites, cross-file source edits, broad search and   replace operations, or repository-wide formatting changes.
- Do not change imports, module constants, decorators, schemas, or top-level registration unless they are part of the same required function/class boundary or the user explicitly authorizes a broader change.

For work that inherently requires multiple source files:

1. Do not edit all of those files in one step.
2. Present a detailed, ordered plan with one source file and one function/class  boundary per step.
3. Obtain explicit approval for the multi-file scope before editing.
4. Complete and verify each step independently.

Documentation updates required by Section 7 are not source-code edits, but
they must remain narrowly limited to recording the completed work.

## 6. Large Changes And New Features

A change is large when it introduces a new feature, changes public behavior or architecture, spans multiple source files, changes a shared contract, requires a migration, or has substantial regression risk.
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
- The test: Every changed line should trace directly to the user's request.

Before implementing a large change or new feature:

1. State that the work qualifies as a large change.
2. Always suggest checking out a separate Git branch before implementation.
3. Use the repository branch convention when one exists; otherwise suggest a short descriptive branch name.
4. Do not switch branches, create branches, commit, or discard work without explicit user authorization.
5. Inspect the working tree and protect unrelated user changes.
6. Provide a detailed plan when needed. The plan must identify:
   - requested outcome and explicit non-goals;
   - affected source files;
   - the single function or class boundary for each step;
   - dependency and execution order;
   - verification for each step;
   - documentation updates required at completion;
   - rollback or compatibility concerns when relevant.
7. Wait for approval of any multi-file or otherwise boundary-expanding plan
   before editing.

Do not suggest a branch for small, single-boundary maintenance changes unless
the user asks.

## 7. Completion Documentation

After completing a large change or new feature, always update:

- `DEVELOPMENT.md`, with the implemented behavior, architecture or workflow
  impact, configuration changes, and verification performed.
- This `AGENTS.md`, but only when the completed work changes agent-relevant
  repository context, commands, constraints, architecture, or working rules.

The user's phrase `DEVELOPMENTS.md and AGENT.md` maps to the canonical files above. Do not create duplicate filenames.

Do not record proposals as completed work. Documentation must describe the final verified implementation. Keep documentation edits limited to existing
relevant sections when possible.

If a large change is complete but this `AGENTS.md` needs no contextual or rule change, add a concise dated entry under `Repository Context Updates` stating that the guidance was reviewed and remains valid.

## 8. Implementation Rules
Define success criteria. Loop until verified.
Transform tasks into verifiable goals:
"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


- Preserve the repository's current architecture and local coding style.
- Follow the procedural extension pattern in `DEVELOPMENT.md`.
- Keep changes minimal and directly traceable to the user's request.
- Do not modify working code merely to make it cleaner.
- Do not introduce dependencies unless explicitly required by the request.
- Do not change public APIs, stored data formats, environment variables, or
  runtime behavior beyond the requested scope.
- Preserve backward compatibility unless the user explicitly requests a
  breaking change.
- Never overwrite, revert, or discard unrelated working-tree changes.
- Comments should explain non-obvious constraints, not narrate obvious code.
- Never automatically check for dependencies in requirements file. Always assume that you have sufficent available dependencies to  implement new codes.

### Python Standards

- All new or modified Python code must comply with PEP 8.
- Preserve established repository conventions where PEP 8 permits multiple
  valid styles.
- Keep imports organized, use four spaces for indentation, use descriptive
  names, and keep lines within PEP 8 limits unless an unavoidable construct
  would become less readable.
- Every new or modified Python function, method, and class must have a
  meaningful docstring.
- Docstrings must follow PEP 257 conventions and describe the callable's or
  class's purpose. Include arguments, return values, raised exceptions, side
  effects, or invariants when they are not obvious from the signature.
- Do not add empty, redundant, or mechanically generated docstrings that only
  restate the function, method, or class name.
- When editing an existing undocumented Python function, method, or class, add
  or improve its docstring within the same approved function/class boundary.

## 9. Verification

After each implementation step:

1. Review the diff for scope and boundary compliance.
2. Run the narrowest relevant existing tests, checks, or reproduction command.
3. Confirm that only the intended source file and function/class boundary
   changed.
4. Confirm that no unrequested files were created.
5. Report verification failures accurately; do not claim success when checks
   were not run or did not pass.

For large changes, run focused checks after every step and the relevant broader
suite after all approved steps are complete.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 10. Communication

- Be direct and concise.
- Report assumptions that affect behavior.
- Do not market optional improvements or end with feature suggestions.
- Do not ask whether the user wants additional features.
- For plans, distinguish requested work, non-goals, risks, and verification.
- At completion, summarize only the requested changes, files changed, and
  checks performed.
- If blocked by these rules, identify the exact rule and the minimum approval
  needed to continue.

## 11. Repository Context Updates

Add concise dated entries here only when a completed large change requires
agent-relevant context that does not belong more naturally in
`DEVELOPMENT.md`.

- 2026-06-14: Established repository-wide agent rules. `AGENTS.md` is the
  canonical agent instruction file, and `DEVELOPMENT.md` is the canonical
  development guide.
- 2026-06-22: Reviewed after `GraphState` replaced `QAState` in the QA
  LangGraph pipeline; existing agent guidance remains valid.
- 2026-06-24: Reviewed after deterministic column metadata, missingness
  summary, and type compatibility tools were added; existing agent guidance
  remains valid.
- 2026-06-24: Reviewed after QA LangGraph added deterministic tabular tool
  nodes for column metadata, missingness summary, and type compatibility;
  existing agent guidance remains valid.
- 2026-06-24: Reviewed after profiling tool migration completed in
  `DatasetProfileTool`; `app/tools/statistics.py` no longer exports a
  profiling compatibility alias and existing agent guidance remains valid.
- 2026-06-24: Reviewed after Milestone 1 statistical tools were added in
  `StatisticalAnalysisTool` and wired into the QA LangGraph; existing agent
  guidance remains valid.
- 2026-06-25: Reviewed after QA conversation memory moved to LangGraph Redis
  checkpointing by `thread_id`; existing agent guidance remains valid.
- 2026-06-25: Reviewed after QA conversation memory added an in-process
  checkpointer fallback for Redis instances without Redis Stack `FT.*`
  commands; existing agent guidance remains valid.
- 2026-06-25: Reviewed after `/ask` response changed to final `GraphState`
  with JSON-safe `response` dict; existing agent guidance remains valid.
- 2026-06-25: Reviewed after QA conversation memory switched from
  `AsyncRedisSaver` to synchronous `RedisSaver`; existing agent guidance
  remains valid.
- 2026-06-25: Reviewed after `/ask` default thread selection changed to the
  active EDA `job_id`; existing agent guidance remains valid.
