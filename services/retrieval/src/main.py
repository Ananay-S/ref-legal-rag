import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from searcher import HybridSearcher  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("retrieval")


def make_handler(searcher: HybridSearcher):
    class RetrievalHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(200, {"status": "ok"})
                return
            self._write_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/search":
                self._write_json(404, {"error": "not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")
            except Exception as exc:  # pragma: no cover - request parsing guard
                self._write_json(400, {"error": f"invalid JSON body: {exc}"})
                return

            query = str(payload.get("query", "")).strip()
            if not query:
                self._write_json(400, {"error": "query is required"})
                return

            top_k = int(payload.get("top_k", 20))
            rerank_top_n = int(payload.get("rerank_top_n", 5))
            include_neighbors = bool(payload.get("include_neighbors", False))
            result = searcher.search(query, top_k=top_k, rerank_top_n=rerank_top_n, include_neighbors=include_neighbors)
            self._write_json(200, result)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            logger.info("%s - %s", self.address_string(), format % args)

        def _write_json(self, status_code: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return RetrievalHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase-2 retrieval API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the HTTP server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the HTTP server to")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant base URL")
    parser.add_argument("--embed-url", default="", help="Optional embedding service URL")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "qdrant"), help="Local fallback directory for vector storage")
    parser.add_argument("--collection-name", default="legal_chunks", help="Vector collection name")
    args = parser.parse_args()

    searcher = HybridSearcher(
        qdrant_url=args.qdrant_url,
        embed_url=args.embed_url or None,
        data_dir=args.data_dir,
        collection_name=args.collection_name,
    )
    handler = make_handler(searcher)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logger.info("Retrieval service listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - service shutdown
        logger.info("Shutting down retrieval service")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

