from __future__ import annotations
import json
import logging
import re
from app.api.schemas import QAResponse

logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> str:
    """Strip markdown code fences and extract the first JSON object/array."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    if fenced != text:
        return fenced.strip()
    # Fallback: find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _regex_extract(raw: str) -> QAResponse | None:
    """Last-resort: pull answer/explanation via regex when JSON is truncated."""
    answer_m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if not answer_m:
        return None
    answer = answer_m.group(1)
    expl_m = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    explanation = expl_m.group(1) if expl_m else ""
    logger.warning("parse_response used regex fallback — JSON was truncated")
    return QAResponse(answer=answer, explanation=explanation)


def parse_response(raw: str) -> QAResponse:
    text = _extract_json(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("parse_response failed | error=%s | raw=%r", exc, raw[:500])
        fallback = _regex_extract(raw)
        if fallback:
            return fallback
        return QAResponse(answer="?", explanation=raw.strip())

    return QAResponse(
        answer=str(data.get("answer", "?")),
        explanation=str(data.get("explanation", "")),
        fol=data.get("fol") or None,
        cot=data.get("cot") or None,
        premises=data.get("premises") or None,
        confidence=_clamp(data.get("confidence")),
    )


def _clamp(val: object) -> float | None:
    if val is None:
        return None
    try:
        return max(0.0, min(1.0, float(val)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
