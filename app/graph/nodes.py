
from app.memory.redis_client import memory
from app.model.openai_client import llm
from app.model.prompts.qa_system import build_prompt
from app.validation.parser import parse_response
from app.graph.schema import GraphState
# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def node_load_history(state: GraphState) -> dict:
    """Load recent conversation history for the active graph session."""
    return {"history": await memory.get_history(state["session_id"])}


async def node_load_eda_context(state: GraphState) -> dict:
    """Load the active EDA summary context for the graph session."""
    job_id = await memory.get_active_eda(state["session_id"])
    if not job_id:
        return {"context": ""}
    result = await memory.get_eda_result(job_id)
    if not result:
        return {"context": ""}
    return {"context": result.get("summary_md", "")}


async def node_generate(state: GraphState) -> dict:
    """Build the QA prompt and request a raw model response."""
    context = state.get("context", "")
    history = state.get("history", [])
    prompt = build_prompt(state["question"], context=context, history=history)
    max_tokens = 4096 if context else 1024
    raw_response = await llm.generate(prompt, max_tokens=max_tokens)
    return {"prompt": prompt, "raw_response": raw_response}


async def node_parse(state: GraphState) -> dict:
    """Parse the raw LLM JSON text into the API response schema."""
    return {"response": parse_response(state["raw_response"])}


async def node_save_memory(state: GraphState) -> dict:
    """Persist the final answer in session memory when parsing succeeds."""
    r = state["response"]
    if r:
        await memory.append(state["session_id"], state["question"], r.answer)
    return {}