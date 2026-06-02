from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.common import candidate_identity_columns
from scripts.post_process_market_plays import compute_final_board


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return candidate_identity_columns(frame.copy())


def _build_source_consistency_report(production_rows: pd.DataFrame, candidate_rows: pd.DataFrame) -> dict[str, Any]:
    production_rows = _normalize_frame(production_rows)
    candidate_rows = _normalize_frame(candidate_rows)

    production_ids = set(production_rows.get("candidate_id", pd.Series("", index=production_rows.index)).astype(str).fillna(""))
    candidate_ids = set(candidate_rows.get("candidate_id", pd.Series("", index=candidate_rows.index)).astype(str).fillna(""))
    unmatched_production_ids = sorted([value for value in production_ids if value and value not in candidate_ids])

    selected_source_profiles = production_rows.get("source_candidate_pool_csv", pd.Series("", index=production_rows.index)).fillna("").astype(str)
    candidate_source_profiles = candidate_rows.get("source_candidate_pool_csv", pd.Series("", index=candidate_rows.index)).fillna("").astype(str)

    report = {
        "total_production_rows": int(len(production_rows)),
        "total_candidate_rows": int(len(candidate_rows)),
        "production_rows_in_candidate_pool": int(len(production_rows) - len(unmatched_production_ids)),
        "production_rows_missing_from_candidate_pool": int(len(unmatched_production_ids)),
        "production_candidate_ids_missing": unmatched_production_ids[:25],
        "unique_production_source_profiles": sorted(set(selected_source_profiles.str.strip().replace("", pd.NA).dropna().astype(str).tolist())),
        "unique_candidate_source_profiles": sorted(set(candidate_source_profiles.str.strip().replace("", pd.NA).dropna().astype(str).tolist())),
        "production_source_profile_counts": selected_source_profiles.value_counts(dropna=False).to_dict(),
        "candidate_source_profile_counts": candidate_source_profiles.value_counts(dropna=False).to_dict(),
    }
    return report


def _build_source_consistency_markdown(report: dict[str, Any]) -> list[str]:
    lines = [
        "# Price Defense Shadow Source Consistency Report",
        "",
        f"- Production rows: {report['total_production_rows']}",
        f"- Candidate rows: {report['total_candidate_rows']}",
        f"- Production rows missing from candidate pool: {report['production_rows_missing_from_candidate_pool']}",
        f"- Production source profiles: {', '.join(report['unique_production_source_profiles']) or 'none'}",
        f"- Candidate source profiles: {', '.join(report['unique_candidate_source_profiles']) or 'none'}",
        "",
        "This audit report verifies that the production final board can be traced back to the same candidate pool source paths.",
    ]
    return lines


def _ensure_minimum_candidate_columns(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    defaults = {
        "gap_percentile": 0.0,
        "final_confidence": 0.0,
        "market_books": 0,
        "history_rows": 0,
        "thompson_ev": 0.0,
        "ev_adjusted": 0.0,
        "expected_win_rate": 0.5,
        "abs_edge": 0.0,
        "edge": 0.0,
        "belief_uncertainty": 1.0,
        "recommendation": "pass",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)

    if "line" not in frame.columns and "market_line" in frame.columns:
        frame["line"] = frame["market_line"]
    if "market_line" not in frame.columns and "line" in frame.columns:
        frame["market_line"] = frame["line"]
    return frame


def _build_shadow_board(
    candidate_rows: pd.DataFrame,
    production_board_size: int,
    append_max_extra_plays: int = 1,
    append_agreement_min: int = 1,
    append_edge_percentile_min: float = 0.90,
) -> pd.DataFrame:
    if candidate_rows.empty:
        return candidate_rows.copy()
    candidate_rows = _ensure_minimum_candidate_columns(candidate_rows)
    base_size = int(max(0, production_board_size))
    max_total_plays = max(base_size, 0)
    return compute_final_board(
        candidate_rows.copy(),
        american_odds=-110,
        min_ev=-1.0,
        min_final_confidence=0.0,
        min_recommendation="pass",
        selection_mode="edge_append_shadow",
        ranking_mode="edge_append_shadow",
        max_plays_per_player=1,
        max_plays_per_target=0,
        max_total_plays=max_total_plays,
        max_target_plays={"PTS": 10, "TRB": 4, "AST": 4},
        max_plays_per_game=0,
        max_plays_per_script_cluster=3,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.52,
        full_bet_win_rate=0.56,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
        append_agreement_min=append_agreement_min,
        append_edge_percentile_min=append_edge_percentile_min,
        append_max_extra_plays=append_max_extra_plays,
    )


def build_price_defense_shadow_boards(
    *,
    output_dir: Path,
    candidate_pool_csv: Path | None = None,
    production_board_csv: Path | None = None,
    append_max_extra_plays: int = 1,
) -> dict[str, Any]:
    output_dir.resolve().mkdir(parents=True, exist_ok=True)

    candidates = _normalize_frame(_read_csv(candidate_pool_csv))
    production_board = _normalize_frame(_read_csv(production_board_csv))
    shadow_board = _build_shadow_board(candidates, len(production_board), append_max_extra_plays=append_max_extra_plays)

    shadow_path = output_dir / "price_defense_shadow_board.csv"
    shadow_board.to_csv(shadow_path, index=False)

    source_report = _build_source_consistency_report(production_board, candidates)
    source_report_path = output_dir / "source_consistency_report.json"
    _write_json(source_report_path, source_report)
    _write_markdown(output_dir / "source_consistency_report.md", _build_source_consistency_markdown(source_report))

    report = {
        "input_paths": {
            "candidate_pool_csv": str(candidate_pool_csv) if candidate_pool_csv else "",
            "production_board_csv": str(production_board_csv) if production_board_csv else "",
        },
        "output_paths": {
            "shadow_board_csv": str(shadow_path),
            "source_consistency_report_json": str(source_report_path),
            "source_consistency_report_md": str(output_dir / "source_consistency_report.md"),
        },
        "total_candidate_rows": int(len(candidates)),
        "total_production_rows": int(len(production_board)),
        "shadow_board_rows": int(len(shadow_board)),
        "shadow_added_rows": int(shadow_board.get("append_shadow_added", pd.Series(False, index=shadow_board.index)).fillna(False).astype(bool).sum()),
        "production_behavior_changed": False,
    }
    report_path = output_dir / "price_defense_shadow_report.json"
    _write_json(report_path, report)
    _write_markdown(
        output_dir / "price_defense_shadow_report.md",
        [
            "# Price Defense Shadow Board Report",
            "",
            f"- Candidate rows: {report['total_candidate_rows']}",
            f"- Production rows: {report['total_production_rows']}",
            f"- Shadow board rows: {report['shadow_board_rows']}",
            f"- Shadow added rows: {report['shadow_added_rows']}",
            "",
            "This report is audit-only and does not modify production behavior.",
        ],
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shadow price-defense boards from a production candidate pool.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool-csv", type=Path, required=True)
    parser.add_argument("--production-board-csv", type=Path, required=True)
    parser.add_argument("--append-max-extra-plays", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_price_defense_shadow_boards(
        output_dir=args.output_dir,
        candidate_pool_csv=args.candidate_pool_csv,
        production_board_csv=args.production_board_csv,
        append_max_extra_plays=int(args.append_max_extra_plays),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
