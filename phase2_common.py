import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


VECTOR_SIZE = 1024
DEFAULT_COLLECTION_NAME = "legal_chunks"
LOCAL_STORE_FILENAME = "legal_chunks.json"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def stable_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:{chunk_index}"


def dense_vector_from_text(text: str, size: int = VECTOR_SIZE) -> list[float]:
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * size

    counts = Counter(tokens)
    vector = [0.0] * size
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % size
        weight = 1.0 + math.log1p(count)
        vector[index] += weight

        alt_index = int.from_bytes(digest[4:8], "big") % size
        if alt_index != index:
            vector[alt_index] += weight * 0.5

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def sparse_vector_from_text(text: str, max_terms: int = 128) -> dict[str, list[float]]:
    tokens = tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)
    items: list[tuple[int, float]] = []
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big")
        weight = 1.0 + math.log1p(count)
        items.append((index, weight))

    items.sort(key=lambda item: (-item[1], item[0]))
    items = items[:max_terms]
    return {
        "indices": [index for index, _ in items],
        "values": [value for _, value in items],
    }


def dense_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def sparse_similarity(left: dict[str, Sequence[float]], right: dict[str, Sequence[float]]) -> float:
    left_indices = left.get("indices", [])
    left_values = left.get("values", [])
    right_indices = right.get("indices", [])
    right_values = right.get("values", [])
    if not left_indices or not right_indices:
        return 0.0

    right_lookup = {index: value for index, value in zip(right_indices, right_values)}
    return sum(value * right_lookup.get(index, 0.0) for index, value in zip(left_indices, left_values))


def hybrid_similarity(
    dense_left: Sequence[float],
    sparse_left: dict[str, Sequence[float]],
    dense_right: Sequence[float],
    sparse_right: dict[str, Sequence[float]],
) -> float:
    return 0.4 * dense_similarity(dense_left, dense_right) + 0.6 * sparse_similarity(sparse_left, sparse_right)


def candidate_text_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0

    text_set = set(text_tokens)
    overlap = sum(1 for token in query_tokens if token in text_set)
    coverage = overlap / len(query_tokens)
    density = overlap / math.sqrt(len(text_tokens))

    query_norm = normalize_text(query).lower()
    text_norm = normalize_text(text).lower()
    phrase_bonus = 0.0
    if query_norm and query_norm in text_norm:
        phrase_bonus = 0.25
    elif all(token in text_norm for token in query_tokens[: min(3, len(query_tokens))]):
        phrase_bonus = 0.1

    return min(1.0, coverage * 0.6 + min(0.3, density * 0.1) + phrase_bonus)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])
