"""
Debug logger — writes one JSON file per LLM call to data/debug_logs/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


# Gemini 1.5 Flash pricing (USD per token, as of 2025)
_INPUT_COST_PER_TOKEN = 0.075 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 0.30 / 1_000_000

_LOG_DIR = Path(__file__).resolve().parents[4] / "data" / "debug_logs"


def log_llm_call(
    *,
    query: str,
    history: list[dict[str, str]],
    context_chunks: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    input_tokens: int,
    output_tokens: int,
) -> Path:
    """Write a timestamped log file and return its path."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    cost_usd = (input_tokens * _INPUT_COST_PER_TOKEN) + (output_tokens * _OUTPUT_COST_PER_TOKEN)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    slug = query[:30].strip().replace(" ", "_").replace("/", "-")
    filename = _LOG_DIR / f"{int(time.time())}_{slug}.json"

    log: dict[str, Any] = {
        "timestamp": ts,
        "query": query,
        "history": history,
        "retrieved_context": context_chunks,
        "prompts": {
            "system": system_prompt,
            "user": user_prompt,
        },
        "response": response_text,
        "token_usage": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        },
        "estimated_cost_usd": round(cost_usd, 8),
    }

    filename.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return filename
