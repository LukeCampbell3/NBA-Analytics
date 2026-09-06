#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.scripts.build_sequential_pa_final_validation_report_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
