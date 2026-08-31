#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.unified.pipeline import export_payload, run, write_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fail-closed unified MLB shadow artifact")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "sports/mlb/web/data")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "sports/mlb/web/data/unified_predictions.json")
    parser.add_argument("--run-date")
    args = parser.parse_args()
    result = run(args.data_dir)
    write_payload(export_payload(result, run_date=args.run_date), args.output)
    print(f"unified shadow: candidates={len(result.candidates)} singles={len(result.singles)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
