import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional, Sequence

from phase2_common import (
    candidate_text_score,
    dense_vector_from_text,
    sparse_vector_from_text,
)


class HybridEmbedder:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 10) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout

    def embed_texts(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        if self._use_remote():
            try:
                return self._embed_remote(texts)
            except Exception:
                pass
        return [self._embed_local(text) for text in texts]

    def embed_query(self, text: str) -> dict[str, Any]:
        return self.embed_texts([text])[0]

    def _embed_local(self, text: str) -> dict[str, Any]:
        return {
            "dense": dense_vector_from_text(text),
            "sparse": sparse_vector_from_text(text),
        }

    def _embed_remote(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        payload = {"texts": list(texts)}
        url = f"{self.base_url}/embed"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        items = response_payload.get("embeddings") or response_payload.get("result") or response_payload.get("data")
        if not isinstance(items, list):
            items = [items]
        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(
                    {
                        "dense": item.get("dense") or item.get("dense_vector") or item.get("vector") or [],
                        "sparse": item.get("sparse") or item.get("sparse_vector") or {"indices": [], "values": []},
                    }
                )
            else:
                normalized.append(self._embed_local(str(item)))
        return normalized

    def _use_remote(self) -> bool:
        return bool(self.base_url and urllib.parse.urlparse(self.base_url).scheme in {"http", "https"})


class HybridReranker:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 10) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout

    def rerank(self, query: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []

        if self._use_remote():
            try:
                return self._rerank_remote(query, candidates)
            except Exception:
                pass

        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            text = candidate.get("text") or candidate.get("payload", {}).get("text", "")
            score = candidate_text_score(query, text)
            scored.append({**candidate, "rerank_score": score})
        scored.sort(key=lambda item: (-item["rerank_score"], item.get("id", "")))
        return scored

    def _rerank_remote(self, query: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "texts": [candidate.get("text") or candidate.get("payload", {}).get("text", "") for candidate in candidates],
        }
        url = f"{self.base_url}/rerank"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        scores = response_payload.get("scores") or response_payload.get("result") or response_payload.get("data") or []
        if not isinstance(scores, list):
            scores = [scores]

        scored: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            scored.append({**candidate, "rerank_score": float(score)})
        scored.sort(key=lambda item: (-item["rerank_score"], item.get("id", "")))
        return scored

    def _use_remote(self) -> bool:
        return bool(self.base_url and urllib.parse.urlparse(self.base_url).scheme in {"http", "https"})
