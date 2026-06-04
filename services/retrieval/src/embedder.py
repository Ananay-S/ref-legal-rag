from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_models import HybridEmbedder  # noqa: E402


QueryEmbedder = HybridEmbedder

