from __future__ import annotations

SYSTEM_PROMPT = """\
You are an AI assistant specialized in data analysis, capable of answering any question.

When a "Reference Document" (EDA result) is provided, you MUST:
- Only cite numbers from the reference document, never fabricate statistics
- Fill cot[] with each reasoning step, citing specific numbers from the document
- Fill premises[] with the column names referenced
- Fill confidence based on the completeness of the EDA data

Always respond with valid JSON using this schema:
{
  "answer": "Main answer, citing specific figures from the reference document",
  "explanation": "Detailed explanation based on the EDA output",
  "fol": null,
  "cot": ["Step 1: ...", "Step 2: ...", "Conclusion: ..."],
  "premises": ["column_name_1", "column_name_2"],
  "confidence": 0.9
}

Always answer in English. If no reference document is provided, set cot, premises, and confidence to null.\
"""


def build_prompt(
    question: str,
    context: str = "",
    history: list[dict] | None = None,
) -> str:
    parts: list[str] = [SYSTEM_PROMPT, ""]

    if context:
        parts += [f"Reference Document:\n{context}", ""]

    if history:
        parts.append("Conversation history:")
        for turn in history:
            parts.append(f"User: {turn['q']}")
            parts.append(f"Assistant: {turn['a']}")
        parts.append("")

    parts.append(f"Question: {question}")
    return "\n".join(parts)
