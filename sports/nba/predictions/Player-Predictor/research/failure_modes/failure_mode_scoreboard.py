from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import (
    as_string_list,
    brier_score,
    calibration_gap,
    expected_calibration_error,
    safe_bool,
    safe_float,
    series_numeric,
    series_text,
    write_json,
)
from research.failure_modes.attribute_pick_failures import _detect_pre_event_failure_modes
from research.failure_modes.failure_mode_registry import get_failure_mode, load_failure_mode_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a failure-mode scoreboard from attributed pick rows.")
    parser.add_argument("--failure-attribution-csv", type=Path, required=True)
    parser.add_argument("--scoreboard-csv-out", type=Path, required=True)
    parser.add_argument("--scoreboard-summary-json-out", type=Path, required=True)
    return parser.parse_args()


def _explode_failure_modes(attributed_rows: pd.DataFrame) -> pd.DataFrame:
    if attributed_rows.empty:
        return attributed_rows.iloc[0:0].copy()
    expanded = attributed_rows.copy()
    expanded["failure_modes"] = expanded.get("failure_modes", pd.Series([[]] * len(expanded), index=expanded.index)).apply(as_string_list)
    expanded = expanded.explode("failure_modes")
    expanded["failure_mode_id"] = expanded["failure_modes"].fillna("").astype(str).str.strip()
    expanded = expanded.loc[expanded["failure_mode_id"] != ""].copy()
    return expanded


def build_failure_mode_scoreboard(
    attributed_rows: pd.DataFrame,
    *,
    registry: dict[str, Any] | None = None,
    candidate_pool_rows: pd.DataFrame | None = None,
    target_failure_modes: list[str] | set[str] | None = None,
    excluded_failure_modes: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    active_registry = registry or load_failure_mode_registry()
    target_set = {str(item).strip() for item in (target_failure_modes or []) if str(item).strip()}
    excluded_set = {str(item).strip() for item in (excluded_failure_modes or []) if str(item).strip()}
    base_columns = [
        "failure_mode_id",
        "candidate_count",
        "selected_count",
        "resolved_count",
        "losses",
        "wins",
        "hit_rate",
        "ROI",
        "Brier",
        "ECE",
        "calibration_gap",
        "avg_miss_distance",
        "loss_concentration",
        "recurrence_rate",
        "pre_event_detectability_rate",
        "intervention_available",
        "estimated_loss_removal_rate",
        "estimated_win_removal_rate",
        "estimated_coverage_cost",
        "non_target_damage_risk",
        "sample_reliability_weight",
        "priority_score",
    ]
    if attributed_rows.empty:
        out = pd.DataFrame(columns=base_columns)
        if target_set:
            zero_rows = pd.DataFrame({"failure_mode_id": sorted(target_set)})
            for column in base_columns:
                if column == "failure_mode_id":
                    continue
                zero_rows[column] = 0.0
            zero_rows["intervention_available"] = False
            zero_rows["hit_rate"] = np.nan
            zero_rows["ROI"] = np.nan
            zero_rows["Brier"] = np.nan
            zero_rows["ECE"] = np.nan
            zero_rows["calibration_gap"] = np.nan
            zero_rows["avg_miss_distance"] = np.nan
            return zero_rows.loc[:, base_columns]
        return out

    selected = attributed_rows.copy()
    selected["result"] = series_text(selected, "result").str.lower()
    selected["resolved_label"] = series_numeric(selected, "resolved_label", default=np.nan)
    missing_label = selected["resolved_label"].isna()
    if bool(missing_label.any()):
        selected.loc[missing_label, "resolved_label"] = np.where(
            selected.loc[missing_label, "result"].eq("win"),
            1.0,
            np.where(selected.loc[missing_label, "result"].eq("loss"), 0.0, np.nan),
        )
    selected["predicted_probability"] = series_numeric(
        selected,
        "predicted_probability",
        default=np.nan,
    ).fillna(series_numeric(selected, "stress_probability", default=np.nan)).fillna(series_numeric(selected, "expected_win_rate", default=np.nan))
    selected["profit_units"] = series_numeric(
        selected,
        "units",
        default=np.nan,
    ).fillna(series_numeric(selected, "profit_units", default=np.nan))
    pre_event_source = selected["pre_event_failure_modes"] if "pre_event_failure_modes" in selected.columns else selected.get(
        "failure_modes",
        pd.Series([[]] * len(selected), index=selected.index),
    )
    selected["pre_event_failure_modes"] = pre_event_source.apply(as_string_list)

    if candidate_pool_rows is not None and not candidate_pool_rows.empty:
        candidate_pool = candidate_pool_rows.copy()
        candidate_pool["pre_event_failure_modes"] = candidate_pool.apply(
            lambda row: _detect_pre_event_failure_modes(
                row,
                allowed_failure_modes=target_set or None,
                excluded_failure_modes=excluded_set or None,
            ),
            axis=1,
        )
    else:
        candidate_pool = pd.DataFrame(columns=list(selected.columns) + ["pre_event_failure_modes"])

    total_selected_rows = max(int(len(selected)), 1)
    total_losses = max(int(series_text(selected, "result").str.lower().eq("loss").sum()), 1)
    failure_modes = set(target_set)
    for modes in selected["pre_event_failure_modes"].tolist():
        failure_modes.update(mode for mode in as_string_list(modes) if mode and mode not in excluded_set)
    if not candidate_pool.empty:
        for modes in candidate_pool["pre_event_failure_modes"].tolist():
            failure_modes.update(mode for mode in as_string_list(modes) if mode and mode not in excluded_set)
    failure_modes = {mode for mode in failure_modes if mode and mode not in excluded_set and (not target_set or mode in target_set)}
    summaries: list[dict[str, Any]] = []
    for failure_mode_id in sorted(failure_modes):
        exposure_mask = selected["pre_event_failure_modes"].map(lambda modes: failure_mode_id in set(as_string_list(modes)))
        group = selected.loc[exposure_mask].copy()
        resolved_group = group.loc[group["result"].isin(["win", "loss"])].copy()
        wins = int(resolved_group["result"].eq("win").sum())
        losses = int(resolved_group["result"].eq("loss").sum())
        resolved_count = int(len(resolved_group))
        definition = get_failure_mode(str(failure_mode_id), active_registry)
        intervention_available = bool(definition and definition.candidate_interventions)
        hit_rate = float(wins / max(1, wins + losses)) if (wins + losses) > 0 else np.nan
        profit_units = float(pd.to_numeric(resolved_group["profit_units"], errors="coerce").fillna(0.0).sum())
        roi = float(profit_units / max(1, resolved_count)) if resolved_count > 0 else np.nan
        avg_miss_distance = float(
            pd.to_numeric(resolved_group.loc[resolved_group["result"].eq("loss"), "miss_distance"], errors="coerce").abs().mean()
        ) if losses > 0 else np.nan
        loss_group = resolved_group.loc[resolved_group["result"].eq("loss")].copy()
        detectability_rate = float(
            loss_group.get("was_pre_event_detectable", loss_group.get("was_failure_pre_event_detectable", pd.Series(False, index=loss_group.index))).astype(bool).mean()
        ) if losses > 0 else 0.0
        recurrence_rate = float(len(group) / total_selected_rows)
        loss_concentration = float(losses / total_losses)
        estimated_loss_removal_rate = float(detectability_rate * (losses / max(1, resolved_count))) if resolved_count > 0 else 0.0
        estimated_win_removal_rate = float(detectability_rate * (wins / max(1, resolved_count))) if resolved_count > 0 else 0.0
        sample_reliability_weight = float(np.clip(resolved_count / 12.0, 0.0, 1.0))
        candidate_count = 0
        if not candidate_pool.empty:
            candidate_count = int(
                candidate_pool["pre_event_failure_modes"].map(lambda modes: failure_mode_id in set(as_string_list(modes))).sum()
            )
        estimated_coverage_cost = float(len(group) / total_selected_rows)
        non_target_damage_risk = float(np.clip(estimated_win_removal_rate * estimated_coverage_cost, 0.0, 1.0))
        priority_score = float(
            recurrence_rate
            * loss_concentration
            * max(detectability_rate, 0.0)
            * estimated_loss_removal_rate
            * max(0.0, 1.0 - estimated_win_removal_rate)
            * max(0.0, 1.0 - non_target_damage_risk)
            * sample_reliability_weight
        )
        summaries.append(
            {
                "failure_mode_id": str(failure_mode_id),
                "candidate_count": int(candidate_count if candidate_count > 0 else len(group)),
                "selected_count": int(len(group)),
                "resolved_count": resolved_count,
                "losses": losses,
                "wins": wins,
                "hit_rate": hit_rate,
                "profit_units": profit_units,
                "ROI": roi,
                "Brier": brier_score(resolved_group["predicted_probability"], resolved_group["resolved_label"]),
                "ECE": expected_calibration_error(resolved_group["predicted_probability"], resolved_group["resolved_label"]),
                "calibration_gap": calibration_gap(resolved_group["predicted_probability"], resolved_group["resolved_label"]),
                "avg_miss_distance": avg_miss_distance,
                "loss_concentration": loss_concentration,
                "recurrence_rate": recurrence_rate,
                "pre_event_detectability_rate": detectability_rate,
                "intervention_available": intervention_available,
                "expected_improvement_if_gated": estimated_loss_removal_rate,
                "coverage_loss_if_gated": estimated_coverage_cost,
                "estimated_loss_removal_rate": estimated_loss_removal_rate,
                "estimated_win_removal_rate": estimated_win_removal_rate,
                "estimated_coverage_cost": estimated_coverage_cost,
                "non_target_damage_risk": non_target_damage_risk,
                "sample_reliability_weight": sample_reliability_weight,
                "priority_score": priority_score,
                "market_families": "|".join(definition.market_families) if definition else "",
            }
        )
    if not summaries:
        out = pd.DataFrame(columns=base_columns + ["profit_units", "expected_improvement_if_gated", "coverage_loss_if_gated", "market_families"])
    else:
        out = pd.DataFrame(summaries).sort_values(
            ["priority_score", "losses", "pre_event_detectability_rate", "candidate_count"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    if target_set:
        existing = set(out["failure_mode_id"].astype(str).tolist())
        missing = sorted(target_set - existing)
        if missing:
            fill = pd.DataFrame({"failure_mode_id": missing})
            for column in out.columns:
                if column == "failure_mode_id":
                    continue
                fill[column] = 0.0
            fill["intervention_available"] = False
            fill["hit_rate"] = np.nan
            fill["ROI"] = np.nan
            fill["Brier"] = np.nan
            fill["ECE"] = np.nan
            fill["calibration_gap"] = np.nan
            fill["avg_miss_distance"] = np.nan
            fill["market_families"] = ""
            out = pd.concat([out, fill], ignore_index=True)
            out = out.sort_values(
                ["priority_score", "losses", "pre_event_detectability_rate", "candidate_count"],
                ascending=[False, False, False, False],
            ).reset_index(drop=True)
    return out


def summarize_failure_mode_scoreboard(scoreboard: pd.DataFrame) -> dict[str, Any]:
    if scoreboard.empty:
        return {
            "failure_mode_count": 0,
            "top_priority_failure_modes": [],
            "intervention_available_count": 0,
        }
    top_modes = scoreboard.head(5)[["failure_mode_id", "priority_score", "losses", "wins"]].to_dict(orient="records")
    return {
        "failure_mode_count": int(len(scoreboard)),
        "top_priority_failure_modes": top_modes,
        "intervention_available_count": int(scoreboard.get("intervention_available", pd.Series(dtype=bool)).astype(bool).sum()),
    }


def main() -> None:
    args = parse_args()
    attributed = pd.read_csv(args.failure_attribution_csv)
    scoreboard = build_failure_mode_scoreboard(attributed)
    args.scoreboard_csv_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    scoreboard.to_csv(args.scoreboard_csv_out, index=False)
    write_json(args.scoreboard_summary_json_out, summarize_failure_mode_scoreboard(scoreboard))


if __name__ == "__main__":
    main()
