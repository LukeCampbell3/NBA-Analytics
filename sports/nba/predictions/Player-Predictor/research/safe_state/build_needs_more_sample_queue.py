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


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _ensure_direction(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "direction" not in out.columns and "side" in out.columns:
        out["direction"] = out["side"]
    return out


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


def _next_recheck_date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return (parsed + pd.Timedelta(days=7)).strftime("%Y-%m-%d")


def build_needs_more_sample_queue(
    *,
    output_dir: Path,
    blocker_resolution_rows_csv: Path,
    root_cause_rows_csv: Path,
    annotated_candidates_csv: Path,
    required_similar_state_count: int = 5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = candidate_identity_columns(_ensure_direction(_read_csv(blocker_resolution_rows_csv)))
    root = _read_csv(root_cause_rows_csv)
    annotated = _read_csv(annotated_candidates_csv)

    working = resolution.copy()
    working = _merge_optional(
        working,
        root,
        ["root_cause_primary", "recommended_repair", "evidence_found", "evidence_missing"],
    )
    working = _merge_optional(
        working,
        annotated,
        [
            "similar_state_count",
            "similar_state_reliability_tier",
            "similar_state_tightness_score",
            "market_side_price",
            "market_side_break_even",
            "stress_probability",
            "lcb_probability",
            "safe_state_tier",
        ],
    )

    root_cause = working.get("root_cause_primary", pd.Series("", index=working.index)).fillna("").astype(str).str.upper()
    action = working.get("recommended_next_action", pd.Series("", index=working.index)).fillna("").astype(str).str.upper()
    edge = working.get("edge_defendability_tier", pd.Series("", index=working.index)).fillna("").astype(str).str.upper()
    mask = root_cause.eq("INSUFFICIENT_COMPARABLE_STATES") & action.eq("NEEDS_MORE_SAMPLE") & edge.eq("EDGE_DEFENDABLE")
    queue = working.loc[mask].copy()
    if queue.empty:
        queue = pd.DataFrame(
            columns=[
                "candidate_id",
                "player",
                "game_id",
                "market_date",
                "target",
                "market_type",
                "side",
                "line",
                "current_similar_state_count",
                "required_similar_state_count",
                "similar_state_reliability_tier",
                "similar_state_tightness_score",
                "fallback_level_needed",
                "recommended_comparable_state_expansion",
                "next_recheck_date",
                "settlement_join_key",
                "queue_status",
            ]
        )
    else:
        queue["player"] = queue.get("player", queue.get("player_name", pd.Series("", index=queue.index)))
        queue["side"] = queue.get("side", queue.get("direction", pd.Series("", index=queue.index)))
        queue["line"] = queue.get("line", queue.get("market_line", pd.Series(np.nan, index=queue.index)))
        current = pd.to_numeric(queue.get("similar_state_count", pd.Series(0, index=queue.index)), errors="coerce").fillna(0).astype(int)
        queue["current_similar_state_count"] = current
        queue["required_similar_state_count"] = int(required_similar_state_count)
        queue["fallback_level_needed"] = np.where(current < required_similar_state_count, "LEVEL_2_OR_HIGHER", "LEVEL_1_RECHECK")
        queue["recommended_comparable_state_expansion"] = "same_player_target_then_archetype_line_zone_fallback"
        queue["next_recheck_date"] = queue.get("market_date", pd.Series("", index=queue.index)).map(_next_recheck_date)
        queue["direction"] = queue.get("direction", queue.get("side", pd.Series("", index=queue.index)))
        queue["settlement_join_key"] = queue.apply(_join_key, axis=1)
        queue["queue_status"] = "NEEDS_MORE_SAMPLE"
        keep = [
            "candidate_id",
            "player",
            "game_id",
            "market_date",
            "target",
            "market_type",
            "side",
            "direction",
            "line",
            "current_similar_state_count",
            "required_similar_state_count",
            "similar_state_reliability_tier",
            "similar_state_tightness_score",
            "fallback_level_needed",
            "recommended_comparable_state_expansion",
            "next_recheck_date",
            "settlement_join_key",
            "queue_status",
        ]
        queue = queue[[col for col in keep if col in queue.columns]]

    csv_path = output_dir / "needs_more_sample_queue.csv"
    json_path = output_dir / "needs_more_sample_queue.json"
    md_path = output_dir / "needs_more_sample_queue.md"
    queue.to_csv(csv_path, index=False)
    report = {
        "input_paths": {
            "blocker_resolution_rows_csv": str(blocker_resolution_rows_csv),
            "root_cause_rows_csv": str(root_cause_rows_csv),
            "annotated_candidates_csv": str(annotated_candidates_csv),
        },
        "output_paths": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
        "needs_more_sample_count": int(len(queue)),
        "queue_status_counts": queue.get("queue_status", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict(),
        "rules": [
            "not_safe",
            "not_production_candidate",
            "evidence_building_only",
            "future_recheck_required",
        ],
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, report, queue)
    return report


def _write_markdown(path: Path, report: dict[str, Any], rows: pd.DataFrame) -> None:
    lines = [
        "# Needs-More-Sample Queue",
        "",
        f"- Queue rows: {report['needs_more_sample_count']}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "Rows here are evidence-building candidates only. They are not safe-state selections.",
        "",
        "## Rows",
    ]
    if rows.empty:
        lines.append("- None")
    else:
        for _, row in rows.iterrows():
            lines.append(
                f"- {row.get('player', '')} {row.get('market_type', '')} {row.get('side', '')} {row.get('line', '')}: "
                f"{row.get('current_similar_state_count', '')}/{row.get('required_similar_state_count', '')} comparable states"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shadow queue for safe-state candidates needing more comparable samples.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocker-resolution-rows-csv", type=Path, required=True)
    parser.add_argument("--root-cause-rows-csv", type=Path, required=True)
    parser.add_argument("--annotated-candidates-csv", type=Path, required=True)
    parser.add_argument("--required-similar-state-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_needs_more_sample_queue(
        output_dir=args.output_dir,
        blocker_resolution_rows_csv=args.blocker_resolution_rows_csv,
        root_cause_rows_csv=args.root_cause_rows_csv,
        annotated_candidates_csv=args.annotated_candidates_csv,
        required_similar_state_count=args.required_similar_state_count,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
