from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_store import HybridVectorStore  # noqa: E402


def expand_neighbors(
    results: list[dict[str, Any]],
    store: HybridVectorStore,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for result in results:
        payload = result.get("payload", {})
        prev_id = payload.get("prev_chunk_id")
        next_id = payload.get("next_chunk_id")
        neighbor_ids = [chunk_id for chunk_id in [prev_id, next_id] if chunk_id]
        neighbors = store.retrieve(neighbor_ids) if neighbor_ids else []
        
        # Strip vectors from neighbors to reduce noise in the API output
        neighbor_lookup = {}
        for neighbor in neighbors:
            n_copy = dict(neighbor)
            n_copy.pop("vector", None)
            neighbor_lookup[n_copy["id"]] = n_copy

        expanded.append(
            {
                **result,
                "prev_chunk_id": prev_id,
                "next_chunk_id": next_id,
                "neighbors": {
                    "prev": neighbor_lookup.get(prev_id),
                    "next": neighbor_lookup.get(next_id),
                },
            }
        )
    return expanded

