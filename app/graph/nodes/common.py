"""Shared helpers for QA graph node modules."""


def append_context(context: str, addition: str) -> str:
    """Append a structured summary to existing prompt context."""
    if not addition:
        return context
    if not context:
        return addition
    return f"{context}\n\n{addition}"


def as_list(value: object) -> list[str]:
    """Normalize metadata values to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether any term appears in normalized text."""
    return any(term in text for term in terms)


def dump_model(model: object) -> dict:
    """Return a dict from a Pydantic model across supported versions."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def extend_unique(target: list[str], values: list[str]) -> None:
    """Append unique non-empty values while preserving existing order."""
    existing = {item.lower() for item in target}
    for value in values:
        normalized = value.strip()
        if normalized and normalized.lower() not in existing:
            target.append(normalized)
            existing.add(normalized.lower())


def unique_values(values: list[str]) -> list[str]:
    """Return unique non-empty strings while preserving order."""
    unique = []
    seen = set()
    for value in values:
        normalized = str(value).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            unique.append(normalized)
            seen.add(key)
    return unique
