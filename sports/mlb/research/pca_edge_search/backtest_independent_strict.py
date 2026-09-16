from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import backtest_independent_support as integrity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    cfg = integrity.base.SearchConfig()
    report, rows = integrity.base.run_analysis(args.input, cfg, strict_same_book=True)
    payload = {
        "status": "DEVELOPMENT_BACKTEST_INDEPENDENT_SUPPORT_STRICT_ONLY",
        "semantics": {
            "astar_success": "EDGE_FOUND in a prior-data PCA residual basin; never a win label",
            "settlement": "derived after frozen selection from Actual versus Market_Line",
            "support_unit": "unique historical player-game; alternate lines from one event cannot multiply support",
        },
        "config": integrity.base.asdict(cfg),
        "primary_same_book_two_sided": report,
        "limitations": [
            "Real sportsbook prices are concentrated on a small number of acquisition/slate dates; date-level uncertainty dominates row count.",
            "Processed prediction-gap lineage is still being audited; this result is not leakage-certified.",
            "Research result only; no production authority.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    out = rows.copy()
    if not out.empty:
        out["analysis"] = "primary_same_book_two_sided"
    out.to_csv(args.output_csv, index=False)
    print("STRICT_INDEPENDENT_SUPPORT_REPORT=" + json.dumps(payload, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
