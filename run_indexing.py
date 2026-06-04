#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

# Enforce Python path validation for project root imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    print("=========================================================")
    print("🚀 LAUNCHING PHASE 2 INGESTION & PIPELINE VECTORIZATION")
    print("=========================================================")

    # Production context parameters 
    db_path = ROOT / "data" / "db" / "legal.db"
    qdrant_url = "http://localhost:6333"
    embed_url = "http://localhost:8081"  # Port mapped to bge-m3 service container
    data_dir = ROOT / "data" / "qdrant"
    batch_size = "8"                     # Bounded footprint to prevent token overflow/OMMs

    # Construct clean path to the service execution target
    script_target = ROOT / "services" / "indexing" / "src" / "main.py"

    if not db_path.exists():
        print(f"❌ Error: Database source file not found at: {db_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(script_target),
        "--db-path", str(db_path),
        "--qdrant-url", qdrant_url,
        "--embed-url", embed_url,
        "--data-dir", str(data_dir),
        "--batch-size", batch_size
    ]

    print(f"Executing Ingestion Core Loop Commands:\n{' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Ingestion loop processing completed cleanly.")
    except subprocess.CalledProcessError as err:
        print(f"\n❌ Pipeline execution failure during vectorization. Exit code: {err.returncode}", file=sys.stderr)
        sys.path.pop(0)
        sys.exit(err.returncode)

if __name__ == "__main__":
    main()