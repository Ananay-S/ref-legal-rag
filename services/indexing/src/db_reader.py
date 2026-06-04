import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Union


def read_documents(db_path: Union[str, Path]) -> Iterator[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute("SELECT id, document FROM documents ORDER BY id")
        for doc_id, document_json in cursor:
            yield _normalize_document(doc_id, document_json)
    finally:
        conn.close()


def _normalize_document(doc_id: str, document_json: str) -> dict[str, Any]:
    document = json.loads(document_json)
    document.setdefault("id", doc_id)
    document.setdefault("metadata", {})
    if "chunks" not in document and "data" in document:
        document["chunks"] = document["data"]
    document.setdefault("chunks", [])
    return document
