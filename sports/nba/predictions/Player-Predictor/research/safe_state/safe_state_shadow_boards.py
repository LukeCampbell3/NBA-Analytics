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
from research.safe_state.safe_state_classifier import annotate_safe_state_stack


VARIANTS = [
    "production_board_as_is",
    "price_defense_only_board",
    "forecastable_price_board",
    "structural_misprice_board",
    "safe_state_core_board",
    "safe_state_near_core_board",
    "safe_state_expanded_board",
]


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _identity(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    frame = candidate_identity_columns(frame.copy())
    return frame["candidate_id"].fillna("").astype(str)


def _sort_board(frame: pd.DataFrame, board_size: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sort_cols = [c for c in ["safe_state_score", "lcb_edge", "stress_edge", "overall_structural_mispricing_score", "overall_player_forecastability_score"] if c in frame.columns]
    if not sort_cols:
        return frame.head(board_size).copy()
    out = frame.copy()
    for col in sort_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(board_size).copy()


def _tier_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].fillna("").astype(str).value_counts().to_dict().items()}


def _numeric_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def _known_failure_exposure(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    text = (
        frame.get("known_failure_modes", pd.Series("", index=frame.index)).fillna("").astype(str)
        + ";"
        + frame.get("forecastability_failure_modes", pd.Series("", index=frame.index)).fillna("").astype(str)
    )
    counts: dict[str, int] = {}
    for item in text.tolist():
        for mode in [part.strip() for part in item.split(";") if part.strip()]:
            counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def _variant_summary(name: str, board: pd.DataFrame, production: pd.DataFrame) -> dict[str, Any]:
    board_ids = set(_identity(board).tolist())
    production_ids = set(_identity(production).tolist())
    overlap = len(board_ids & production_ids)
    return {
        "variant": name,
        "board_size": int(len(board)),
        "overlap_with_production": int(overlap),
        "shadow_only_rows": int(max(0, len(board_ids - production_ids))),
        "production_only_rows": int(max(0, len(production_ids - board_ids))),
        "edge_defendable_count": int(board.get("edge_defendability_tier", pd.Series("", index=board.index)).astype(str).eq("EDGE_DEFENDABLE").sum()) if not board.empty else 0,
        "forecastability_tier_counts": _tier_counts(board, "forecastability_tier"),
        "structural_mispricing_tier_counts": _tier_counts(board, "structural_mispricing_tier"),
        "similar_state_reliability_tier_counts": _tier_counts(board, "similar_state_reliability_tier"),
        "safe_state_tier_counts": _tier_counts(board, "safe_state_tier"),
        "avg_stress_edge": _numeric_mean(board, "stress_edge"),
        "avg_lcb_edge": _numeric_mean(board, "lcb_edge"),
        "avg_forecastability": _numeric_mean(board, "overall_player_forecastability_score"),
        "avg_structural_mispricing": _numeric_mean(board, "overall_structural_mispricing_score"),
        "avg_similar_state_tightness": _numeric_mean(board, "similar_state_tightness_score"),
        "avg_board_readiness_risk": _numeric_mean(board, "board_readiness_risk_score"),
        "known_failure_mode_exposure": _known_failure_exposure(board),
    }


def write_safe_state_shadow_boards_from_annotated(
    *,
    output_dir: Path,
    annotated: pd.DataFrame,
    production: pd.DataFrame,
    board_size: int | None = None,
    input_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = candidate_identity_columns(annotated)
    production = candidate_identity_columns(production)

    annotated_path = output_dir / "safe_state_annotated_candidates.csv"
    annotated.to_csv(annotated_path, index=False)

    target_size = int(board_size if board_size is not None else (len(production) if len(production) > 0 else min(12, len(annotated))))
    price_defense = annotated.loc[
        annotated.get("edge_defendability_tier", pd.Series("", index=annotated.index)).astype(str).eq("EDGE_DEFENDABLE")
        & annotated.get("price_validity_status", pd.Series("", index=annotated.index)).astype(str).eq("PRICE_VALID")
        & pd.to_numeric(annotated.get("stress_edge", pd.Series(0.0, index=annotated.index)), errors="coerce").gt(0.0)
        & pd.to_numeric(annotated.get("lcb_edge", pd.Series(0.0, index=annotated.index)), errors="coerce").gt(0.0)
    ].copy()
    forecastable = price_defense.loc[price_defense.get("forecastability_tier", pd.Series("", index=price_defense.index)).astype(str).isin({"HIGH_FORECASTABILITY", "MEDIUM_FORECASTABILITY"})].copy()
    structural = price_defense.loc[price_defense.get("structural_mispricing_tier", pd.Series("", index=price_defense.index)).astype(str).isin({"STRUCTURAL_MISPRICE_STRONG", "STRUCTURAL_MISPRICE_ACCEPTABLE"})].copy()
    safe_core = annotated.loc[annotated.get("safe_state_tier", pd.Series("", index=annotated.index)).astype(str).eq("SAFE_STATE_CORE")].copy()
    safe_near_core = annotated.loc[annotated.get("safe_state_tier", pd.Series("", index=annotated.index)).astype(str).eq("SAFE_STATE_NEAR_CORE")].copy()
    safe_expanded = annotated.loc[
        annotated.get("safe_state_tier", pd.Series("", index=annotated.index)).astype(str).eq("SAFE_STATE_CORE")
        | (
            annotated.get("safe_state_tier", pd.Series("", index=annotated.index)).astype(str).eq("SAFE_STATE_PRICE_ONLY")
            & annotated.get("similar_state_reliability_tier", pd.Series("", index=annotated.index)).astype(str).eq("TIGHT")
        )
    ].copy()

    boards = {
        "production_board_as_is": production.copy(),
        "price_defense_only_board": _sort_board(price_defense, target_size),
        "forecastable_price_board": _sort_board(forecastable, target_size),
        "structural_misprice_board": _sort_board(structural, target_size),
        "safe_state_core_board": _sort_board(safe_core, target_size),
        "safe_state_near_core_board": _sort_board(safe_near_core, target_size),
        "safe_state_expanded_board": _sort_board(safe_expanded, target_size),
    }

    output_paths: dict[str, str] = {"annotated_candidates": str(annotated_path)}
    summaries = []
    for name, board in boards.items():
        path = output_dir / f"{name}.csv"
        board.to_csv(path, index=False)
        output_paths[name] = str(path)
        summaries.append(_variant_summary(name, board, production))

    shadow_alias_path = output_dir / "safe_state_shadow_board.csv"
    boards["safe_state_core_board"].to_csv(shadow_alias_path, index=False)
    output_paths["safe_state_shadow_board"] = str(shadow_alias_path)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "safe_state_shadow_variant_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report = {
        "input_paths": input_paths or {},
        "output_paths": output_paths | {"variant_summary_csv": str(summary_path)},
        "total_candidate_rows": int(len(annotated)),
        "total_production_rows": int(len(production)),
        "variant_summaries": summaries,
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    _write_json(output_dir / "safe_state_shadow_report.json", report)
    (output_dir / "safe_state_shadow_report.md").write_text(
        "\n".join(
            [
                "# Safe-State Shadow Board Report",
                "",
                f"- Candidate rows: {len(annotated)}",
                f"- Production rows: {len(production)}",
                "- Production behavior changed: false",
                "- Promotion claim: false",
                "",
                "This report compares price-only, forecastable, structural, NEAR_CORE, and SAFE_STATE_CORE shadow boards.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def build_safe_state_shadow_boards(
    *,
    output_dir: Path,
    candidate_pool_csv: Path | None = None,
    production_board_csv: Path | None = None,
    historical_csv: Path | None = None,
    board_size: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_identity_columns(_read_csv(candidate_pool_csv))
    production = candidate_identity_columns(_read_csv(production_board_csv))
    history = _read_csv(historical_csv)
    annotated = annotate_safe_state_stack(candidates, history)
    return write_safe_state_shadow_boards_from_annotated(
        output_dir=output_dir,
        annotated=annotated,
        production=production,
        board_size=board_size,
        input_paths={
            "candidate_pool_csv": str(candidate_pool_csv) if candidate_pool_csv else "",
            "production_board_csv": str(production_board_csv) if production_board_csv else "",
            "historical_csv": str(historical_csv) if historical_csv else "",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shadow-only safe-state board variants.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool-csv", type=Path, required=True)
    parser.add_argument("--production-board-csv", type=Path, required=True)
    parser.add_argument("--historical-csv", type=Path)
    parser.add_argument("--board-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_safe_state_shadow_boards(
        output_dir=args.output_dir,
        candidate_pool_csv=args.candidate_pool_csv,
        production_board_csv=args.production_board_csv,
        historical_csv=args.historical_csv,
        board_size=args.board_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
