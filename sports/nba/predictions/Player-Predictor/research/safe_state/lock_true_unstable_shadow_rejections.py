from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.common import candidate_identity_columns


TRUE_UNSTABLE_CAUSES = {"REAL_MINUTES_VOLATILITY", "REAL_USAGE_VOLATILITY"}


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _join_key(row: pd.Series) -> str:
    parts = [
        row.get("game_id", ""),
        row.get("market_date", row.get("game_date", "")),
        row.get("player", row.get("player_name", "")),
        row.get("target", ""),
        row.get("side", row.get("direction", "")),
        row.get("line", row.get("market_line", "")),
    ]
    return "::".join("" if pd.isna(part) else str(part) for part in parts)


def _merge_optional(base: pd.DataFrame, extra: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if extra.empty:
        return base
    extra = candidate_identity_columns(extra)
    keep = ["candidate_id"] + [col for col in columns if col in extra.columns]
    merged = base.merge(extra[keep], on="candidate_id", how="left", suffixes=("", "_extra"))
    for col in [c for c in merged.columns if c.endswith("_extra")]:
        original = col.removesuffix("_extra")
        if original in merged.columns:
            merged[original] = merged[original].where(merged[original].notna() & merged[original].astype(str).str.strip().ne(""), merged[col])
            merged = merged.drop(columns=[col])
    return merged


def lock_true_unstable_shadow_rejections(
    *,
    output_dir: Path,
    annotated_candidates_csv: Path,
    blocker_resolution_rows_csv: Path,
    root_cause_rows_csv: Path,
    candidate_blockers_csv: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = candidate_identity_columns(_read_csv(annotated_candidates_csv))
    resolution = _read_csv(blocker_resolution_rows_csv)
    root = _read_csv(root_cause_rows_csv)
    blockers = _read_csv(candidate_blockers_csv)

    working = annotated.copy()
    working = _merge_optional(
        working,
        resolution,
        ["gap_family", "gap_subtype", "gap_fixability", "gap_severity", "gap_reason", "recommended_next_action", "gap_recommended_fix"],
    )
    working = _merge_optional(
        working,
        root,
        ["root_cause_primary", "root_cause_secondary", "recommended_repair", "evidence_found", "evidence_missing"],
    )
    working = _merge_optional(working, blockers, ["primary_blocker", "secondary_blockers", "evidence_gap_type", "missing_features"])

    edge_defendable = working.get("edge_defendability_tier", pd.Series("", index=working.index)).fillna("").astype(str).str.upper().eq("EDGE_DEFENDABLE")
    root_cause = working.get("root_cause_primary", pd.Series("", index=working.index)).fillna("").astype(str).str.upper()
    action = working.get("recommended_next_action", working.get("recommended_repair", pd.Series("", index=working.index))).fillna("").astype(str).str.upper()
    fixability = working.get("gap_fixability", working.get("forecastability_gap_fixability", pd.Series("", index=working.index))).fillna("").astype(str).str.upper()
    mask = edge_defendable & root_cause.isin(TRUE_UNSTABLE_CAUSES) & action.eq("KEEP_UNSAFE_TRUE_VOLATILITY") & fixability.eq("TRUE_UNSTABLE_STATE")
    locked = working.loc[mask].copy()
    if locked.empty:
        locked = pd.DataFrame(
            columns=[
                "candidate_id",
                "player",
                "game_id",
                "market_date",
                "target",
                "market_type",
                "side",
                "line",
                "market_side_price",
                "market_side_break_even",
                "stress_probability",
                "lcb_probability",
                "stress_edge",
                "lcb_edge",
                "forecastability_gap_primary",
                "gap_subtype",
                "root_cause_primary",
                "instability_reason",
                "safe_state_tier",
                "recommended_action",
                "settlement_join_key",
            ]
        )
    else:
        locked["player"] = locked.get("player", locked.get("player_name", pd.Series("", index=locked.index)))
        locked["side"] = locked.get("side", locked.get("direction", pd.Series("", index=locked.index)))
        locked["line"] = locked.get("line", locked.get("market_line", pd.Series(np.nan, index=locked.index)))
        locked["instability_reason"] = locked.get("gap_reason", pd.Series("", index=locked.index)).fillna("").astype(str)
        locked["recommended_action"] = "KEEP_UNSAFE_TRUE_VOLATILITY"
        locked["settlement_join_key"] = locked.apply(_join_key, axis=1)
        keep = [
            "candidate_id",
            "player",
            "game_id",
            "market_date",
            "target",
            "market_type",
            "side",
            "line",
            "market_side_price",
            "market_side_break_even",
            "stress_probability",
            "lcb_probability",
            "stress_edge",
            "lcb_edge",
            "forecastability_gap_primary",
            "gap_subtype",
            "root_cause_primary",
            "instability_reason",
            "safe_state_tier",
            "recommended_action",
            "settlement_join_key",
        ]
        locked = locked[[col for col in keep if col in locked.columns]]

    csv_path = output_dir / "true_unstable_shadow_rejections.csv"
    json_path = output_dir / "true_unstable_shadow_rejections.json"
    md_path = output_dir / "true_unstable_shadow_rejections.md"
    locked.to_csv(csv_path, index=False)

    report = {
        "input_paths": {
            "annotated_candidates_csv": str(annotated_candidates_csv),
            "blocker_resolution_rows_csv": str(blocker_resolution_rows_csv),
            "root_cause_rows_csv": str(root_cause_rows_csv),
            "candidate_blockers_csv": str(candidate_blockers_csv) if candidate_blockers_csv else "",
        },
        "output_paths": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
        "locked_true_unstable_count": int(len(locked)),
        "root_cause_counts": locked.get("root_cause_primary", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict(),
        "rules": [
            "shadow_rejected_only",
            "unsafe_despite_price_edge",
            "not_a_production_veto",
            "settlement_tracked_later",
        ],
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, report, locked)
    return report


def _write_markdown(path: Path, report: dict[str, Any], rows: pd.DataFrame) -> None:
    lines = [
        "# True-Unstable Shadow Rejections",
        "",
        f"- Locked rows: {report['locked_true_unstable_count']}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "These candidates are price-defendable but shadow-rejected because pre-event forecastability evidence shows real volatility.",
        "",
        "## Rows",
    ]
    if rows.empty:
        lines.append("- None")
    else:
        for _, row in rows.iterrows():
            lines.append(
                f"- {row.get('player', '')} {row.get('market_type', '')} {row.get('side', '')} {row.get('line', '')}: "
                f"{row.get('gap_subtype', '')} / {row.get('root_cause_primary', '')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock true-unstable price-defendable rows as shadow rejections.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotated-candidates-csv", type=Path, required=True)
    parser.add_argument("--blocker-resolution-rows-csv", type=Path, required=True)
    parser.add_argument("--root-cause-rows-csv", type=Path, required=True)
    parser.add_argument("--candidate-blockers-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = lock_true_unstable_shadow_rejections(
        output_dir=args.output_dir,
        annotated_candidates_csv=args.annotated_candidates_csv,
        blocker_resolution_rows_csv=args.blocker_resolution_rows_csv,
        root_cause_rows_csv=args.root_cause_rows_csv,
        candidate_blockers_csv=args.candidate_blockers_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
