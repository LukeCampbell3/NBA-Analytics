#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.mlb.unified.promotion_validation import run_promotion_validation


if __name__ == "__main__":
    result = run_promotion_validation(ROOT)
    print(json.dumps({
        "historical_validation": result["certification"]["status"],
        "eligible_slates": result["certification"]["eligible_slates"],
        "selected_singles": result["certification"]["selected_singles"],
        "production_state": result["production_status"]["state"],
    }, indent=2))
