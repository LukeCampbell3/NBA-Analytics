from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.common import (
    augment_with_snapshot_prices,
    compute_price_quality_frame,
    merge_selected_with_candidate_pool,
    summarize_price_provenance,
)
from research.run_improvement_discovery import (
    _load_candidate_pool_rows,
    _load_selected_rows,
    _resolve_daily_runs_dirs,
    _resolve_selected_board_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit timestamp-safe price provenance and edge defendability.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-board-csv", type=Path, action="append", default=[])
    parser.add_argument("--candidate-pool-csv", type=Path, action="append", default=[])
    parser.add_argument("--daily-runs-dir", type=Path, action="append", default=[])
    parser.add_argument("--selected-variant", type=str, default="baseline_no_rebound_diagnostics")
    parser.add_argument("--broad-walk-forward", action="store_true")
    return parser.parse_args()


def run_price_provenance_audit(
    *,
    selected_board_paths: list[Path],
    candidate_pool_paths: list[Path],
    daily_runs_dirs: list[Path],
    output_dir: Path,
    selected_variant: str = "baseline_no_rebound_diagnostics",
    broad_walk_forward: bool = False,
) -> dict[str, Any]:
    output_dir.resolve().mkdir(parents=True, exist_ok=True)
    selected_rows, used_selected_paths = _load_selected_rows(
        selected_board_paths,
        selected_variant=selected_variant,
        broad_walk_forward=bool(broad_walk_forward),
    )
    candidate_pool_rows, used_candidate_paths = _load_candidate_pool_rows(
        selected_rows=selected_rows,
        candidate_pool_paths=candidate_pool_paths,
        daily_runs_dirs=daily_runs_dirs,
    )
    candidate_pool_rows = augment_with_snapshot_prices(candidate_pool_rows)
    selected_rows = merge_selected_with_candidate_pool(selected_rows, candidate_pool_rows)

    candidate_ledger = compute_price_quality_frame(candidate_pool_rows, record_scope="candidate")
    selected_ledger = compute_price_quality_frame(selected_rows, record_scope="selected")
    ledger = pd.concat([candidate_ledger, selected_ledger], ignore_index=True, sort=False)

    audit_csv = output_dir / "price_provenance_audit.csv"
    summary_json = output_dir / "price_provenance_audit_summary.json"
    selected_untrusted_csv = output_dir / "selected_rows_price_untrusted.csv"

    ledger.to_csv(audit_csv, index=False)
    if "edge_price_untrusted_flag" in selected_ledger.columns:
        selected_untrusted = selected_ledger.loc[selected_ledger["edge_price_untrusted_flag"].astype(bool)].copy()
    else:
        selected_untrusted = selected_ledger.copy()
    selected_untrusted.to_csv(selected_untrusted_csv, index=False)

    summary = summarize_price_provenance(ledger)
    defendability = ledger.get("edge_defendability_tier", pd.Series("", index=ledger.index)).astype(str)
    summary.update(
        {
            "rows_where_edge_cannot_be_validated": int(ledger["price_gap_blocks_validation"].astype(bool).sum()),
            "rows_that_would_be_edge_defendable": int(defendability.eq("EDGE_DEFENDABLE").sum()),
            "rows_that_would_be_edge_price_dependent": int(defendability.eq("EDGE_PRICE_DEPENDENT").sum()),
            "rows_that_would_fail_price": int(defendability.eq("EDGE_FAILS_PRICE").sum()),
            "input_paths": {
                "selected_board_csvs": [str(path) for path in used_selected_paths],
                "candidate_pool_csvs": [str(path) for path in used_candidate_paths],
                "daily_runs_dirs": [str(path) for path in daily_runs_dirs],
            },
            "output_paths": {
                "price_provenance_audit_csv": str(audit_csv),
                "price_provenance_audit_summary_json": str(summary_json),
                "selected_rows_price_untrusted_csv": str(selected_untrusted_csv),
            },
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    selected_board_paths = _resolve_selected_board_paths(list(args.selected_board_csv), broad_walk_forward=bool(args.broad_walk_forward))
    daily_runs_dirs = _resolve_daily_runs_dirs(list(args.daily_runs_dir), selected_board_paths, broad_walk_forward=bool(args.broad_walk_forward))
    summary = run_price_provenance_audit(
        selected_board_paths=selected_board_paths,
        candidate_pool_paths=list(args.candidate_pool_csv),
        daily_runs_dirs=daily_runs_dirs,
        output_dir=args.output_dir,
        selected_variant=str(args.selected_variant),
        broad_walk_forward=bool(args.broad_walk_forward),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
