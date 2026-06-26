"""LLM generation and response parsing nodes for the QA graph."""

from app.graph.nodes.common import dump_model
from app.graph.schema import GraphState
from app.model.openai_client import llm
from app.model.prompts.qa_system import build_prompt
from app.validation.parser import parse_response


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
    return {"response": dump_model(parse_response(state["raw_response"]))}
