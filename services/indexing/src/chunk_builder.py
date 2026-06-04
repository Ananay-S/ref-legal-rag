from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_common import normalize_text, stable_chunk_id
from phase2_store import point_payload_for_text


def build_chunk_records(document: dict[str, Any]) -> list[dict[str, Any]]:
    doc_id = str(document["id"])
    metadata = document.get("metadata", {})
    source_chunks = document.get("chunks") or []

    records: list[dict[str, Any]] = []
    total = len(source_chunks)
    for index, source_chunk in enumerate(source_chunks, start=1):
        text = _chunk_text(source_chunk)
        if not text:
            continue
        chunk_id = stable_chunk_id(doc_id, index)
        prev_chunk_id = stable_chunk_id(doc_id, index - 1) if index > 1 else None
        next_chunk_id = stable_chunk_id(doc_id, index + 1) if index < total else None
        payload = point_payload_for_text(
            doc_id=doc_id,
            chunk_index=index,
            chunk_id=chunk_id,
            prev_chunk_id=prev_chunk_id,
            next_chunk_id=next_chunk_id,
            text=text,
            metadata=metadata,
            source_chunk_id=source_chunk.get("cid") or source_chunk.get("paragraph_id") or index,
        )
        records.append(
            {
                "id": chunk_id,
                "vector": {},
                "payload": payload,
                "doc_id": doc_id,
                "chunk_index": index,
                "chunk_id": chunk_id,
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
                "text": text,
            }
        )
    return records


def _chunk_text(source_chunk: dict[str, Any]) -> str:
    text = source_chunk.get("text")
    if text is None:
        text = source_chunk.get("data")
    return normalize_text(str(text or ""))
