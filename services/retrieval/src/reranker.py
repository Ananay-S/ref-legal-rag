from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_models import HybridReranker  # noqa: E402


QueryReranker = HybridReranker

