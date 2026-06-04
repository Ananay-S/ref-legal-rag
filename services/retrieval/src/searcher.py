from pathlib import Path
import sys
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_models import HybridEmbedder, HybridReranker  # noqa: E402
from phase2_store import HybridVectorStore  # noqa: E402
from chunk_expander import expand_neighbors  # noqa: E402


class HybridSearcher:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        embed_url: Optional[str] = None,
        data_dir: Optional[str] = None,
        collection_name: str = "legal_chunks",
    ) -> None:
        self.store = HybridVectorStore(qdrant_url=qdrant_url, data_dir=data_dir, collection_name=collection_name)
        self.embedder = HybridEmbedder(base_url=embed_url)
        self.reranker = HybridReranker()

    def search(self, query: str, top_k: int = 20, rerank_top_n: int = 5, include_neighbors: bool = False) -> dict[str, Any]:
        query_embedding = self.embedder.embed_query(query)
        dense_hits = self.store.search("dense", query_embedding["dense"], limit=max(top_k * 2, rerank_top_n * 2))
        sparse_hits = self.store.search("sparse", query_embedding["sparse"], limit=max(top_k * 2, rerank_top_n * 2))

        merged = self._merge_hits(dense_hits, sparse_hits, query_embedding)
        total_candidates = len(merged)
        merged.sort(key=lambda item: (-item["score"], item["id"]))
        candidates = merged[: max(top_k * 2, rerank_top_n * 2)]

        reranked = self.reranker.rerank(query, candidates)
        final = reranked[:rerank_top_n]

        if include_neighbors:
            final = expand_neighbors(final, self.store)

        results = [self._serialize_result(result) for result in final]
        return {
            "results": results,
            "query": query,
            "total_candidates": total_candidates,
            "returned": len(results),
        }

    def _merge_hits(
        self,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
        query_embedding: dict[str, Any],
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        dense_lookup = {item["id"]: item for item in dense_hits}
        sparse_lookup = {item["id"]: item for item in sparse_hits}
        candidate_ids = list(dict.fromkeys([*dense_lookup.keys(), *sparse_lookup.keys()]))

        for point_id in candidate_ids:
            dense_hit = dense_lookup.get(point_id)
            sparse_hit = sparse_lookup.get(point_id)
            source = dense_hit or sparse_hit or {}
            payload = source.get("payload", {})
            vector = source.get("vector", {})
            dense_vector = vector.get("dense", [])
            sparse_vector = vector.get("sparse", {})
            score = 0.0
            if dense_hit and sparse_hit:
                score = 0.4 * float(dense_hit.get("score", 0.0)) + 0.6 * float(sparse_hit.get("score", 0.0))
            elif dense_hit:
                score = float(dense_hit.get("score", 0.0))
            elif sparse_hit:
                score = float(sparse_hit.get("score", 0.0))
            by_id[point_id] = {
                "id": point_id,
                "score": score,
                "payload": payload,
                "vector": vector,
                "text": payload.get("text", ""),
                "dense_score": float(dense_hit.get("score", 0.0)) if dense_hit else 0.0,
                "sparse_score": float(sparse_hit.get("score", 0.0)) if sparse_hit else 0.0,
                "query_dense": query_embedding["dense"],
                "query_sparse": query_embedding["sparse"],
                "point_dense": dense_vector,
                "point_sparse": sparse_vector,
            }
        return list(by_id.values())

    def _serialize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        payload = result.get("payload", {})
        response = {
            "chunk_id": payload.get("chunk_id") or result.get("id"),
            "doc_id": payload.get("doc_id"),
            "chunk_index": payload.get("chunk_index"),
            "score": result.get("rerank_score", result.get("score", 0.0)),
            "text": payload.get("text", ""),
        }
        if "neighbors" in result:
            response["neighbors"] = result["neighbors"]
            response["prev_chunk_id"] = result.get("prev_chunk_id")
            response["next_chunk_id"] = result.get("next_chunk_id")
        return response
