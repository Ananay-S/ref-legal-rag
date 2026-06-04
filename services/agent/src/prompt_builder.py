"""
Prompt builder — groups retrieval chunks by doc_id, dedupes by chunk_id,
sorts by chunk_index, then renders the context block and user prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "system_prompt.md"


def load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_context_block(chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """
    1. Dedupe by chunk_id (full id — whole id uniqueness).
    2. Group by doc_id (base id).
    3. Within each group, sort ascending by chunk_index.
    4. Render a clean context block string.

    Returns (context_block_str, ordered_chunks_list).
    """
    # Step 1 — Dedupe by full chunk_id
    seen_ids: set[str] = set()
    unique: list[dict[str, Any]] = []
    for chunk in chunks:
        cid = str(chunk.get("chunk_id") or chunk.get("id") or "")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(chunk)

    # Step 2 & 3 — Group by doc_id then sort by chunk_index within each group
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in unique:
        doc_id = str(chunk.get("doc_id") or "unknown")
        groups.setdefault(doc_id, []).append(chunk)
    for doc_id in groups:
        groups[doc_id].sort(key=lambda c: int(c.get("chunk_index") or 0))

    # Step 4 — Render
    ordered: list[dict[str, Any]] = []
    lines: list[str] = []
    for doc_id, doc_chunks in groups.items():
        lines.append(f"### Document: {doc_id}")
        for chunk in doc_chunks:
            chunk_index = chunk.get("chunk_index", "?")
            text = (chunk.get("text") or "").strip()
            lines.append(f"[Chunk {chunk_index}] {text}")
            ordered.append(chunk)
        lines.append("")

    context_block = "\n".join(lines).strip()
    return context_block, ordered


def build_user_prompt(
    *,
    context_block: str,
    history: list[dict[str, str]],
    query: str,
) -> str:
    """
    Compose the user-side prompt:
    - Retrieved context at the top.
    - Last ≤5 conversation turns (labelled with recency weighting note).
    - Current query last (highest focus).
    """
    parts: list[str] = []

    parts.append("## Retrieved Legal Context\n")
    parts.append(context_block if context_block else "_No relevant context found._")

    if history:
        parts.append("\n## Conversation History (oldest → newest, focus on last)")
        for turn in history:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"**{role_label}:** {turn['content']}")

    parts.append(f"\n## Current Question\n{query}")

    return "\n".join(parts)
