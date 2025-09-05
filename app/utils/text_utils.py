from __future__ import annotations

import json
import re
from typing import Any, Iterable, Tuple


def chunk_markdown(markdown_text: str, chunk_size_tokens: int = 3000) -> list[str]:
    """Split markdown into roughly token-sized chunks.

    Heuristic: ~4 chars/token. Splits around headers when possible.
    """
    chars_per_chunk = max(1, chunk_size_tokens * 4)

    chunks = re.split(r"(#{1,6}\s+.*?\n)", markdown_text)

    result_chunks: list[str] = []
    current = ""

    for piece in chunks:
        if len(current) + len(piece) < chars_per_chunk:
            current += piece
        else:
            if current:
                result_chunks.append(current)
            current = piece

    if current:
        result_chunks.append(current)

    if not result_chunks or (result_chunks and min(len(c) for c in result_chunks) > chars_per_chunk):
        result_chunks = [
            markdown_text[i: i + chars_per_chunk] for i in range(0, len(markdown_text), chars_per_chunk)
        ]

    return result_chunks


def extract_json_payload(text: str) -> Tuple[dict[str, Any] | list[Any] | None, str | None]:
    """Extract a JSON object/array from LLM text.

    Supports fenced blocks ```json ... ``` or plain JSON content, and trims common prefixes like 'json'.
    Returns (parsed, raw_json_str) or (None, None) if not found/parsable.
    """
    raw = None

    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        raw = text[start:end].strip() if end != -1 else text[start:].strip()
    else:
        t = text.strip()
        if t.startswith("json"):
            raw = t[len("json"):].strip()
        elif t.startswith("{") or t.startswith("["):
            raw = t

    if not raw:
        return None, None

    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def dedupe_stable(items: Iterable[Any]) -> list[Any]:
    """Stable de-duplication for lists of strings or dicts."""
    out: list[Any] = []
    seen: set[Any] = set()

    for it in items:
        key: Any
        if isinstance(it, dict):
            key = tuple(sorted((k, str(v)) for k, v in it.items()))
        else:
            key = it
        if key not in seen:
            seen.add(key)
            out.append(it)

    return out
