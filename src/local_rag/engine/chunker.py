from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks using recursive character splitting."""
    if not text.strip():
        return []

    base_meta = metadata or {}

    if len(text) <= chunk_size:
        return [Chunk(text=text, index=0, metadata=dict(base_meta))]

    raw_chunks = _recursive_split(text, chunk_size, _SEPARATORS)
    merged = _apply_overlap(raw_chunks, chunk_overlap)
    filtered = [c for c in merged if len(c) >= min_chunk_size]

    if not filtered and text.strip():
        filtered = [text]

    return [
        Chunk(text=t, index=i, metadata=dict(base_meta))
        for i, t in enumerate(filtered)
    ]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split text trying each separator in order."""
    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if len(part) > chunk_size:
                        remaining_seps = separators[separators.index(sep) + 1:]
                        if remaining_seps:
                            chunks.extend(_recursive_split(part, chunk_size, remaining_seps))
                        else:
                            for i in range(0, len(part), chunk_size):
                                chunks.append(part[i:i + chunk_size])
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap from the end of the previous chunk to the start of the next."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
        result.append(overlap_text + chunks[i])
    return result
