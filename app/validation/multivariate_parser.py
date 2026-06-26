"""Parse the Multivariate Use-Case Dictionary (raw JSON array) from the LLM.

The prompt asks for a bare JSON array with no markdown fence, but small models
sometimes wrap it or add stray text. Be lenient; never raise — return [] on failure.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def _extract_array(raw: str) -> str:
    """Strip code fences / prose and isolate the outermost JSON array."""
    text = raw.strip()

    # Remove ```json ... ``` or ``` ... ``` fences if present.
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    text = fenced.strip()

    # Slice from the first '[' to the last ']' (drops leading/trailing prose).
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _is_valid_item(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    pair = item.get("comparison_pair")
    evaluation = item.get("evaluation")
    return (
        isinstance(pair, dict)
        and "variable_a" in pair
        and "variable_b" in pair
        and isinstance(evaluation, dict)
        and "confidence_score" in evaluation
    )


def parse_multivariate(raw: str) -> list[dict]:
    """Return a list of use-case dicts, or [] if parsing fails."""
    if not raw or not raw.strip():
        return []

    text = _extract_array(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("parse_multivariate failed | error=%s | raw=%r", exc, raw[:500])
        return []

    if not isinstance(data, list):
        logger.error("parse_multivariate: expected a JSON array, got %s", type(data).__name__)
        return []

    valid = [item for item in data if _is_valid_item(item)]
    if len(valid) != len(data):
        logger.warning(
            "parse_multivariate: dropped %d malformed items of %d",
            len(data) - len(valid),
            len(data),
        )
    return valid
