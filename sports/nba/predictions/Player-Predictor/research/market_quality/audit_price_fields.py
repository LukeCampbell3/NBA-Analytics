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
    summarize_price_quality,
)
from research.run_improvement_discovery import (
    _load_candidate_pool_rows,
    _load_selected_rows,
    _resolve_daily_runs_dirs,
    _resolve_selected_board_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit selector price-field availability for market-quality research.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-board-csv", type=Path, action="append", default=[])
    parser.add_argument("--candidate-pool-csv", type=Path, action="append", default=[])
    parser.add_argument("--daily-runs-dir", type=Path, action="append", default=[])
    parser.add_argument("--selected-variant", type=str, default="baseline_no_rebound_diagnostics")
    parser.add_argument("--broad-walk-forward", action="store_true")
    return parser.parse_args()


def run_price_field_audit(
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

    candidate_audit = compute_price_quality_frame(candidate_pool_rows, record_scope="candidate")
    selected_audit = compute_price_quality_frame(selected_rows, record_scope="selected")
    audit_rows = pd.concat([candidate_audit, selected_audit], ignore_index=True, sort=False)

    availability_csv = output_dir / "price_field_availability.csv"
    summary_json = output_dir / "price_field_availability_summary.json"
    missing_examples_csv = output_dir / "missing_price_examples.csv"

    audit_rows.to_csv(availability_csv, index=False)
    missing_examples = audit_rows.loc[
        audit_rows["missing_price_flag"] | audit_rows["invalid_price_flag"] | audit_rows["stale_price_flag"]
    ].copy()
    missing_examples.head(200).to_csv(missing_examples_csv, index=False)

    summary = summarize_price_quality(audit_rows)
    summary["input_paths"] = {
        "selected_board_csvs": [str(path) for path in used_selected_paths],
        "candidate_pool_csvs": [str(path) for path in used_candidate_paths],
        "daily_runs_dirs": [str(path) for path in daily_runs_dirs],
    }
    summary["output_paths"] = {
        "price_field_availability_csv": str(availability_csv),
        "price_field_availability_summary_json": str(summary_json),
        "missing_price_examples_csv": str(missing_examples_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    selected_board_paths = _resolve_selected_board_paths(list(args.selected_board_csv), broad_walk_forward=bool(args.broad_walk_forward))
    daily_runs_dirs = _resolve_daily_runs_dirs(list(args.daily_runs_dir), selected_board_paths, broad_walk_forward=bool(args.broad_walk_forward))
    summary = run_price_field_audit(
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
