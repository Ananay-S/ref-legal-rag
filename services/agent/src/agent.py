"""
Agent core — retrieves context, builds prompt, calls Gemini, logs the call.
Reads config from .env at project root.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from logger import log_llm_call
from prompt_builder import build_context_block, build_user_prompt, load_system_prompt

# ─── Load .env from project root ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_ROOT / ".env")

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
RETRIEVAL_URL: str = os.getenv("RETRIEVAL_URL", "http://localhost:8000/search")
TOP_K: int = int(os.getenv("TOP_K", "15"))
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
INCLUDE_NEIGHBORS: bool = os.getenv("INCLUDE_NEIGHBORS", "false").lower() == "true"

_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
_SYSTEM_PROMPT = load_system_prompt()


# ─── Retrieval ─────────────────────────────────────────────────────────────────

def _fetch_chunks(query: str) -> list[dict[str, Any]]:
    """POST to the retrieval service and return raw result chunks."""
    payload = json.dumps({
        "query": query,
        "top_k": TOP_K,
        "rerank_top_n": RERANK_TOP_N,
        "include_neighbors": INCLUDE_NEIGHBORS,
    }).encode("utf-8")

    req = urllib.request.Request(
        RETRIEVAL_URL,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("results") or []
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"Retrieval service error: {exc}") from exc


# ─── Agent call ────────────────────────────────────────────────────────────────

def run(query: str, history: list[dict[str, str]]) -> dict[str, Any]:
    """
    Main entry point.

    Args:
        query:   The current user question.
        history: Last ≤5 turns — list of {"role": "user"|"assistant", "content": str}.

    Returns dict with keys: answer, chunks, log_file.
    """
    # 1. Retrieve
    raw_chunks = _fetch_chunks(query)

    # 2. Build context (deduped, grouped by doc_id, sorted by chunk_index)
    context_block, ordered_chunks = build_context_block(raw_chunks)

    # 3. Build prompts
    user_prompt = build_user_prompt(
        context_block=context_block,
        history=history,
        query=query,
    )

    # 4. Call Gemini (single non-streaming call)
    response = _CLIENT.models.generate_content(
        model=LLM_MODEL,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )
    answer = (response.text or "").strip()

    # 5. Extract token usage
    usage = getattr(response, "usage_metadata", None)
    input_tokens: int = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens: int = getattr(usage, "candidates_token_count", 0) or 0

    # 6. Log
    log_path = log_llm_call(
        query=query,
        history=history,
        context_chunks=ordered_chunks,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_text=answer,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return {
        "answer": answer,
        "chunks": ordered_chunks,
        "log_file": str(log_path),
    }
