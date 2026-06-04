import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_common import iter_batches  # noqa: E402
from phase2_models import HybridEmbedder  # noqa: E402
from phase2_store import HybridVectorStore  # noqa: E402

from chunk_builder import build_chunk_records  # noqa: E402
from db_reader import read_documents  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("indexing")


def index_documents(
    db_path: str,
    qdrant_url: Optional[str],
    embed_url: Optional[str],
    data_dir: Optional[str],
    batch_size: int = 8,
) -> dict[str, int]:
    store = HybridVectorStore(qdrant_url=qdrant_url, data_dir=data_dir)
    store.ensure_collection()
    embedder = HybridEmbedder(base_url=embed_url)

    indexed_documents = 0
    indexed_points = 0

    for document in read_documents(db_path):
        doc_id = str(document["id"])
        chunk_records = build_chunk_records(document)
        if not chunk_records:
            logger.info("Skipping document %s because it has no chunks", doc_id)
            continue

        logger.info("Indexing document %s with %s chunks", doc_id, len(chunk_records))
        store.delete_document(doc_id)

        for batch in iter_batches(chunk_records, batch_size):
            embeddings = embedder.embed_texts([record["text"] for record in batch])
            points = []
            for record, embedding in zip(batch, embeddings):
                point = {
                    "id": record["id"],
                    "vector": embedding,
                    "payload": record["payload"],
                }
                points.append(point)
            store.upsert_points(points)
            indexed_points += len(points)

        indexed_documents += 1

    return {"documents": indexed_documents, "points": indexed_points}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Phase 1 documents into a hybrid vector store")
    parser.add_argument("--db-path", default=str(ROOT / "data" / "db" / "legal.db"), help="Path to the Phase 1 SQLite database")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant base URL")
    parser.add_argument("--embed-url", default="", help="Optional embedding service URL")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "qdrant"), help="Local fallback directory for vector storage")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size")
    args = parser.parse_args()

    result = index_documents(
        db_path=args.db_path,
        qdrant_url=args.qdrant_url,
        embed_url=args.embed_url or None,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    logger.info("Indexing complete: %s", result)


if __name__ == "__main__":
    main()
