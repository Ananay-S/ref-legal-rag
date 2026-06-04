#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    print("=========================================================")
    print("📡 INITIALIZING PHASE 2 HYBRID RETRIEVAL API SERVICES")
    print("=========================================================")

    # Binding and microservice address mapping configurations
    host = "0.0.0.0"
    port = "8000"
    qdrant_url = "http://localhost:6333"
    embed_url = "http://localhost:8081"       # Dense footprint extraction vector mapping
    collection_name = "legal_chunks"
    data_dir = ROOT / "data" / "qdrant"

    script_target = ROOT / "services" / "retrieval" / "src" / "main.py"

    cmd = [
        sys.executable,
        str(script_target),
        "--host", host,
        "--port", port,
        "--qdrant-url", qdrant_url,
        "--embed-url", embed_url,
        "--collection-name", collection_name,
        "--data-dir", str(data_dir)
    ]

    print(f"Retrieval Service Endpoint active at: http://{host}:{port}")
    print(f"Executing Engine Run Commands:\n{' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Gracefully halting engine processes. Retrieval server down.")
    except subprocess.CalledProcessError as err:
        print(f"\n❌ Critical Server Crash. Exit code: {err.returncode}", file=sys.stderr)
        sys.exit(err.returncode)

if __name__ == "__main__":
    main()