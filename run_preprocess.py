import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "services" / "preprocessing" / "src" / "main.py"


if __name__ == "__main__":
    try:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--pdf-dir",
                str(ROOT / "dataset"),
                "--db-path",
                str(ROOT / "data" / "db" / "legal.db"),
                "--output-dir",
                str(ROOT / "data" / "preprocessed"),
            ],
            check=True,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
