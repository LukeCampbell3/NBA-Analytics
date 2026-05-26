from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.common import build_candidate_id, safe_bool, safe_float


ADJUSTMENT_COLUMNS = [
    "candidate_id",
    "failure_mode_id",
    "penalty",
    "downgrade_tier",
    "veto_flag",
    "opposite_side_candidate_flag",
    "alt_line_candidate_flag",
    "explanation",
]

DOWNGRADE_TIER_PRIORITY = {
    "": 0,
    "keep": 0,
    "none": 0,
    "consider": 1,
    "boundary_shadow": 1,
    "price_dependent": 1,
    "price_dependent_tier": 1,
    "pass": 2,
    "veto": 3,
}

DOWNGRADE_TIER_TO_RECOMMENDATION = {
    "consider": "consider",
    "boundary_shadow": "consider",
    "price_dependent": "consider",
    "price_dependent_tier": "consider",
    "pass": "pass",
    "veto": "pass",
}


def _identity_adjustment_frame(index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["failure_mode_total_penalty"] = 0.0
    out["failure_mode_adjustment_count"] = 0
    out["failure_mode_ids"] = ""
    out["failure_mode_explanation"] = ""
    out["failure_mode_downgrade_tier"] = ""
    out["failure_mode_veto_flag"] = False
    out["failure_mode_opposite_side_candidate_flag"] = False
    out["failure_mode_alt_line_candidate_flag"] = False
    return out


def load_failure_mode_adjustments(adjustments: pd.DataFrame | Path | str | None) -> pd.DataFrame:
    if adjustments is None:
        return pd.DataFrame(columns=ADJUSTMENT_COLUMNS)
    if isinstance(adjustments, pd.DataFrame):
        frame = adjustments.copy()
    else:
        path = Path(adjustments)
        if not path.exists():
            return pd.DataFrame(columns=ADJUSTMENT_COLUMNS)
        frame = pd.read_csv(path)
    for column in ADJUSTMENT_COLUMNS:
        if column not in frame.columns:
            if column in {"penalty"}:
                frame[column] = 0.0
            elif column.endswith("_flag"):
                frame[column] = False
            else:
                frame[column] = ""
    return frame.loc[:, ADJUSTMENT_COLUMNS].copy()


def aggregate_failure_mode_adjustments(adjustments: pd.DataFrame) -> pd.DataFrame:
    if adjustments.empty:
        return pd.DataFrame(columns=["candidate_id"])
    work = adjustments.copy()
    work["candidate_id"] = work["candidate_id"].fillna("").astype(str)
    work = work.loc[work["candidate_id"] != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=["candidate_id"])
    work["penalty"] = pd.to_numeric(work["penalty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    work["failure_mode_id"] = work["failure_mode_id"].fillna("").astype(str)
    work["downgrade_tier"] = work["downgrade_tier"].fillna("").astype(str).str.strip().str.lower()
    work["veto_flag"] = work["veto_flag"].map(lambda value: safe_bool(value, default=False))
    work["opposite_side_candidate_flag"] = work["opposite_side_candidate_flag"].map(lambda value: safe_bool(value, default=False))
    work["alt_line_candidate_flag"] = work["alt_line_candidate_flag"].map(lambda value: safe_bool(value, default=False))
    work["explanation"] = work["explanation"].fillna("").astype(str)

    def _join_unique(values: pd.Series) -> str:
        tokens = [str(item).strip() for item in values.tolist() if str(item).strip()]
        seen: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
        return "|".join(seen)

    def _worst_tier(values: pd.Series) -> str:
        best = ""
        best_rank = -1
        for value in values.tolist():
            token = str(value).strip().lower()
            rank = DOWNGRADE_TIER_PRIORITY.get(token, 0)
            if rank > best_rank:
                best = token
                best_rank = rank
        return best

    aggregated = (
        work.groupby("candidate_id", dropna=False)
        .agg(
            failure_mode_total_penalty=("penalty", "sum"),
            failure_mode_adjustment_count=("candidate_id", "size"),
            failure_mode_ids=("failure_mode_id", _join_unique),
            failure_mode_explanation=("explanation", _join_unique),
            failure_mode_downgrade_tier=("downgrade_tier", _worst_tier),
            failure_mode_veto_flag=("veto_flag", "max"),
            failure_mode_opposite_side_candidate_flag=("opposite_side_candidate_flag", "max"),
            failure_mode_alt_line_candidate_flag=("alt_line_candidate_flag", "max"),
        )
        .reset_index()
    )
    aggregated["failure_mode_total_penalty"] = pd.to_numeric(aggregated["failure_mode_total_penalty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    aggregated["failure_mode_adjustment_count"] = pd.to_numeric(aggregated["failure_mode_adjustment_count"], errors="coerce").fillna(0).astype(int)
    aggregated["failure_mode_veto_flag"] = aggregated["failure_mode_veto_flag"].astype(bool)
    aggregated["failure_mode_opposite_side_candidate_flag"] = aggregated["failure_mode_opposite_side_candidate_flag"].astype(bool)
    aggregated["failure_mode_alt_line_candidate_flag"] = aggregated["failure_mode_alt_line_candidate_flag"].astype(bool)
    return aggregated


def apply_failure_mode_adjustments(
    frame: pd.DataFrame,
    adjustments: pd.DataFrame | Path | str | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "candidate_id" not in out.columns:
        out["candidate_id"] = build_candidate_id(out)
    def _apply_identity_defaults(target: pd.DataFrame) -> pd.DataFrame:
        defaults = _identity_adjustment_frame(target.index)
        for column in defaults.columns:
            if column not in target.columns:
                target[column] = defaults[column]
            elif defaults[column].dtype == bool:
                target[column] = target[column].fillna(False).astype(bool)
            elif str(defaults[column].dtype).startswith("int"):
                target[column] = pd.to_numeric(target[column], errors="coerce").fillna(0).astype(int)
            elif str(defaults[column].dtype).startswith("float"):
                target[column] = pd.to_numeric(target[column], errors="coerce").fillna(0.0)
            else:
                target[column] = target[column].fillna("").astype(str)
        return target
    loaded = load_failure_mode_adjustments(adjustments)
    if loaded.empty:
        return _apply_identity_defaults(out)
    aggregated = aggregate_failure_mode_adjustments(loaded)
    if aggregated.empty:
        return _apply_identity_defaults(out)
    merged = out.merge(aggregated, on="candidate_id", how="left")
    merged = _apply_identity_defaults(merged)

    penalty = pd.to_numeric(merged["failure_mode_total_penalty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    for column in ["expected_win_rate", "board_play_win_prob", "selected_board_prob_raw"]:
        if column in merged.columns:
            pre_column = f"{column}_pre_failure_mode"
            if pre_column not in merged.columns:
                merged[pre_column] = pd.to_numeric(merged[column], errors="coerce")
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).sub(penalty, fill_value=0.0).clip(lower=0.0, upper=1.0)
    for column in ["ev", "ev_adjusted", "final_confidence", "board_objective_base_score"]:
        if column in merged.columns:
            pre_column = f"{column}_pre_failure_mode"
            if pre_column not in merged.columns:
                merged[pre_column] = pd.to_numeric(merged[column], errors="coerce")
            scale = 0.5 if column == "final_confidence" else 1.0
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0) - float(scale) * penalty

    if "recommendation" in merged.columns:
        merged["recommendation_pre_failure_mode"] = merged["recommendation"].fillna("").astype(str)
        mapped_reco = merged["failure_mode_downgrade_tier"].fillna("").astype(str).str.lower().map(DOWNGRADE_TIER_TO_RECOMMENDATION).fillna("")
        merged["recommendation"] = np.where(
            mapped_reco != "",
            mapped_reco,
            merged["recommendation_pre_failure_mode"],
        )
        merged.loc[merged["failure_mode_veto_flag"], "recommendation"] = "pass"

    merged["failure_mode_veto_reason"] = np.where(
        merged["failure_mode_veto_flag"],
        merged["failure_mode_explanation"].replace("", "failure_mode_adjustment_veto"),
        "",
    )
    return merged
