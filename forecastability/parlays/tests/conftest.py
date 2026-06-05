from __future__ import annotations

import sys
from pathlib import Path


PARLAY_ROOT = Path(__file__).resolve().parents[1]
if str(PARLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(PARLAY_ROOT))
