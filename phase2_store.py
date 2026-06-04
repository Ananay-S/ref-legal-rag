import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from phase2_common import (
    DEFAULT_COLLECTION_NAME,
    LOCAL_STORE_FILENAME,
    dense_similarity,
    load_json,
    now_iso,
    repo_root,
    sparse_similarity,
    write_json_atomic,
)


@dataclass(frozen=True)
class QdrantPoint:
    id: str
    vector: dict[str, Any]
    payload: dict[str, Any]


class HybridVectorStore:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        data_dir: Optional[Any] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        timeout: int = 10,
    ) -> None:
        self.qdrant_url = (qdrant_url or "").rstrip("/")
        self.data_dir = Path(data_dir) if data_dir is not None else repo_root() / "data" / "qdrant"
        self.collection_name = collection_name
        self.timeout = timeout
        self.local_path = self.data_dir / LOCAL_STORE_FILENAME

    def ensure_collection(self) -> None:
        if self._use_remote():
            payload = {
                "vectors": {"dense": {"size": 1024, "distance": "Cosine"}},
                "sparse_vectors": {"sparse": {}},
            }
            try:
                self._request("PUT", f"/collections/{self.collection_name}", payload)
                return
            except RuntimeError:
                pass

        state = self._load_local_state()
        if state is None:
            self._save_local_state(
                {
                    "collection_name": self.collection_name,
                    "created_at": now_iso(),
                    "points": [],
                }
            )

    def delete_document(self, doc_id: str) -> None:
        if self._use_remote():
            payload = {
                "filter": {
                    "must": [
                        {
                            "key": "doc_id",
                            "match": {"value": doc_id},
                        }
                    ]
                }
            }
            try:
                self._request("POST", f"/collections/{self.collection_name}/points/delete", payload)
                return
            except RuntimeError:
                pass

        state = self._load_local_state()
        if state is None:
            return
        points = [point for point in state.get("points", []) if point.get("payload", {}).get("doc_id") != doc_id]
        state["points"] = points
        state["updated_at"] = now_iso()
        self._save_local_state(state)

    def upsert_points(self, points: Sequence[dict[str, Any]]) -> None:
        if self._use_remote():
            import uuid
            qdrant_points = []
            for p in points:
                u_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(p["id"])))
                qdrant_points.append({**p, "id": u_id})
            payload = {"points": qdrant_points}
            try:
                self._request("PUT", f"/collections/{self.collection_name}/points?wait=true", payload)
                return
            except RuntimeError:
                pass

        state = self._load_local_state() or {
            "collection_name": self.collection_name,
            "created_at": now_iso(),
            "points": [],
        }
        by_id = {point["id"]: point for point in state.get("points", [])}
        for point in points:
            by_id[str(point["id"])] = {
                "id": str(point["id"]),
                "vector": point["vector"],
                "payload": point["payload"],
            }
        state["points"] = list(by_id.values())
        state["updated_at"] = now_iso()
        self._save_local_state(state)

    def search(self, vector_name: str, vector: Any, limit: int = 10, query_filter: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        if self._use_remote():
            payload: dict[str, Any] = {
                "query": vector,
                "using": vector_name,
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
            }
            if query_filter is not None:
                payload["filter"] = query_filter
            try:
                response = self._request("POST", f"/collections/{self.collection_name}/points/query", payload)
                result_data = response.get("result", {})
                points_list = result_data.get("points", []) if isinstance(result_data, dict) else (result_data if isinstance(result_data, list) else [])
                return [self._normalize_remote_point(point) for point in points_list]
            except RuntimeError:
                pass

        points = self._load_points()
        scored: list[dict[str, Any]] = []
        for point in points:
            dense = point.get("vector", {}).get("dense", [])
            sparse = point.get("vector", {}).get("sparse", {})
            if vector_name == "dense":
                score = dense_similarity(vector, dense)
            elif vector_name == "sparse":
                score = sparse_similarity(vector, sparse)
            else:
                raise ValueError(f"Unsupported vector name: {vector_name}")
            scored.append({"id": point["id"], "payload": point["payload"], "vector": point["vector"], "score": score})

        scored.sort(key=lambda item: (-item["score"], item["id"]))
        return scored[:limit]

    def retrieve(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        id_set = {str(item) for item in ids}
        if self._use_remote():
            import uuid
            qdrant_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, item)) for item in id_set]
            payload = {"ids": qdrant_ids, "with_payload": True, "with_vector": False}
            try:
                response = self._request("POST", f"/collections/{self.collection_name}/points/retrieve", payload)
                result = response.get("result") or response.get("points") or []
                return [self._normalize_remote_point(point) for point in result]
            except RuntimeError:
                pass

        points = self._load_points()
        lookup = {point["id"]: point for point in points}
        return [lookup[point_id] for point_id in ids if point_id in lookup]

    def scroll_doc_points(self, doc_id: str) -> list[dict[str, Any]]:
        if self._use_remote():
            payload = {"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}, "with_payload": True, "with_vector": False, "limit": 1000}
            try:
                response = self._request("POST", f"/collections/{self.collection_name}/points/scroll", payload)
                result = response.get("result") or []
                return [self._normalize_remote_point(point) for point in result]
            except RuntimeError:
                pass
        return [point for point in self._load_points() if point.get("payload", {}).get("doc_id") == doc_id]

    def load_all_points(self) -> list[dict[str, Any]]:
        return self._load_points()

    def _normalize_remote_point(self, point: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(point.get("id")),
            "vector": point.get("vector") or {},
            "payload": point.get("payload") or {},
            "score": point.get("score", 0.0),
        }

    def _load_points(self) -> list[dict[str, Any]]:
        state = self._load_local_state()
        if not state:
            return []
        return list(state.get("points", []))

    def _load_local_state(self) -> Optional[dict[str, Any]]:
        data = load_json(self.local_path, default=None)
        if data is None:
            return None
        if data.get("collection_name") not in {None, self.collection_name}:
            return None
        return data

    def _save_local_state(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.local_path, state)

    def _use_remote(self) -> bool:
        return bool(self.qdrant_url and urllib.parse.urlparse(self.qdrant_url).scheme in {"http", "https"})

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if not self._use_remote():
            raise RuntimeError("Remote Qdrant URL is not configured")

        url = f"{self.qdrant_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method.upper())
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Qdrant request failed: {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qdrant request failed: {exc.reason}") from exc
        if not content:
            return {}
        return json.loads(content)


def point_payload_for_text(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_id: str,
    prev_chunk_id: Optional[str],
    next_chunk_id: Optional[str],
    text: str,
    metadata: dict[str, Any],
    source_chunk_id: Any = None,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "chunk_id": chunk_id,
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,
        "source_chunk_id": source_chunk_id,
        "appellant": metadata.get("appellant", "Not Found"),
        "respondent": metadata.get("respondent", "Not Found"),
        "year": metadata.get("year"),
        "month": metadata.get("month"),
        "day": metadata.get("day"),
        "court": metadata.get("court", "Supreme Court of India"),
        "jurisdiction": metadata.get("jurisdiction", "Not Found"),
        "text": text,
    }
