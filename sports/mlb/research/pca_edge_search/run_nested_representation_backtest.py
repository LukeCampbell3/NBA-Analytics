from __future__ import annotations

import argparse
from pathlib import Path

from sports.mlb.research.pca_edge_search.nested_representation_certificate_backtest import run


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def main() -> None:
    root = repo_root()

    parser = argparse.ArgumentParser(
        description=(
            "Run the nested chronological MLB PCA/A* certificate backtest directly "
            "from Python. No GitHub Actions workflow is required."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "sports/mlb/data/predictions/calibration/historical_pool_universe_2026.csv",
        help="Historical universe CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "sports/mlb/data/evaluation/pca_astar_edge/direct_nested_representation_summary.json",
        help="Backtest summary JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "sports/mlb/data/evaluation/pca_astar_edge/direct_nested_representation_scored_rows.csv",
        help="Contract-level scored CSV.",
    )
    args = parser.parse_args()

    report = run(args.input, args.output_json, args.output_csv)

    selected = report["inner_development_selection"]["selected_representation"]
    holdout = report["outer_holdout_selected_representation"]
    strict = holdout["strict_local"]["metrics"]
    cert = holdout["astar_certificate"]["metrics"]

    print("\nDIRECT BACKTEST COMPLETE")
    print(f"selected_representation={selected}")
    print(
        "strict_local: "
        f"plays={strict.get('plays', 0)} "
        f"net_units={strict.get('net_units')} "
        f"roi={strict.get('roi_per_play')}"
    )
    print(
        "astar_certificate: "
        f"plays={cert.get('plays', 0)} "
        f"net_units={cert.get('net_units')} "
        f"roi={cert.get('roi_per_play')}"
    )
    print(f"summary_json={args.output_json}")
    print(f"scored_csv={args.output_csv}")


if __name__ == "__main__":
    main()
