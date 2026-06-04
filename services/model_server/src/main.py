import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_models import HybridEmbedder, HybridReranker  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model-server")


def make_handler(mode: str):
    embedder = HybridEmbedder()
    reranker = HybridReranker()

    class ModelHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(200, {"status": "ok", "mode": mode})
                return
            self._write_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if mode == "embed" and self.path == "/embed":
                payload = self._read_json()
                texts = payload.get("texts")
                if isinstance(texts, str):
                    texts = [texts]
                texts = list(texts or [])
                embeddings = embedder.embed_texts(texts)
                self._write_json(200, {"embeddings": embeddings})
                return

            if mode == "rerank" and self.path == "/rerank":
                payload = self._read_json()
                query = str(payload.get("query", ""))
                texts = list(payload.get("texts") or [])
                candidates = [{"text": text} for text in texts]
                ranked = reranker.rerank(query, candidates)
                self._write_json(200, {"scores": [item["rerank_score"] for item in ranked]})
                return

            self._write_json(404, {"error": "not found"})

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            logger.info("%s - %s", self.address_string(), format % args)

        def _read_json(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            return json.loads(raw or "{}")

        def _write_json(self, status_code: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ModelHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight phase-2 model server")
    parser.add_argument("--mode", choices=("embed", "rerank"), required=True, help="Server mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind to")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.mode))
    logger.info("Model server (%s) listening on http://%s:%s", args.mode, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - service shutdown
        logger.info("Shutting down model server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

