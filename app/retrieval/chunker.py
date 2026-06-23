from __future__ import annotations


def fixed_size_chunk(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
