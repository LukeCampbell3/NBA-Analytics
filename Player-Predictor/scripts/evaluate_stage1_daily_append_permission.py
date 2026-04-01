#!/usr/bin/env python3
"""
Stage-1 daily append-permission evaluator (shadow-only).

This model predicts whether a slate/day should permit append activity.
It does not generate append candidates, replace baseline board rows, or
create fallback append sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LEAKAGE_SAFE_FEATURES = [
    "board_cutoff_gap",
    "board_top_to_cutoff_gap",
    "near_miss_local_slope",
    "board_concentration",
    "board_target_entropy",
    "pool_size",
    "pool_player_unique_count",
    "pool_target_unique_count",
    "pool_game_unique_count",
    "pool_script_cluster_unique_count",
    "pool_game_max_count",
    "pool_target_dir_max_count",
    "pool_edge_mean",
    "pool_edge_std",
    "pool_conf_mean",
    "pool_conf_std",
    "pool_corr_mean",
    "pool_corr_std",
    "pool_edge_gap_1_2",
    "pool_conf_gap_1_2",
    "pool_agreement_ge1_count",
    "pool_agreement_ge2_count",
    "pool_agreement_ge3_count",
    "shadow_candidate_exists",
    "shadow_candidate_edge",
    "shadow_candidate_confidence",
    "shadow_candidate_agreement",
    "shadow_candidate_corr_score",
    "shadow_candidate_edge_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 daily append-permission model.")
    parser.add_argument("--daily-context-csv", type=Path, required=True, help="Daily context CSV from evaluate_cutoff_meta_append.")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="top1_feasible_positive",
        choices=[
            "best_pool_positive",
            "shadow_improves_edge",
            "best_feasible_append_value",
            "top1_feasible_positive",
            "top1_feasible_improves_edge",
            "top1_feasible_value",
        ],
        help=(
            "Target definition. "
            "'best_pool_positive' => best eligible append candidate on the day is positive. "
            "'shadow_improves_edge' => shadow candidate improved edge on resolved days. "
            "'best_feasible_append_value' => continuous best-available append value delta for the day. "
            "'top1_feasible_positive' => decision-time top1 feasible append is positive. "
            "'top1_feasible_improves_edge' => decision-time top1 feasible append improves edge EV/resolved. "
            "'top1_feasible_value' => decision-time top1 feasible append value delta."
        ),
    )
    parser.add_argument(
        "--fallback-target-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "best_pool_positive",
            "shadow_improves_edge",
            "top1_feasible_positive",
            "top1_feasible_improves_edge",
        ],
        help="Optional fallback target when primary target has only one class.",
    )
    parser.add_argument(
        "--candidate-mode",
        type=str,
        default="top1_feasible",
        choices=["shadow", "top1_feasible", "none"],
        help=(
            "Candidate availability gate used for permit decisions. "
            "'shadow' uses shadow_candidate_exists, "
            "'top1_feasible' uses top1_feasible_exists, "
            "'none' ignores candidate-availability gating."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Permit threshold on predicted probability.")
    parser.add_argument("--min-train-days", type=int, default=12, help="Minimum labeled prior days required before fitting walk-forward model.")
    parser.add_argument(
        "--out-scored-csv",
        type=Path,
        required=True,
        help="Output CSV with per-day walk-forward scores and decisions.",
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        required=True,
        help="Output JSON summary with gate metrics and diagnostics.",
    )
    parser.add_argument(
        "--exclude-snapshot-modes",
        nargs="*",
        default=[],
        help="Optional snapshot_mode values to exclude from training/evaluation (for example: stale_fallback).",
    )
    return parser.parse_args()


def _safe_bool_as_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(float)


def _target_series(df: pd.DataFrame, target_mode: str) -> pd.Series:
    if target_mode == "best_pool_positive":
        base = pd.to_numeric(df.get("best_pool_ev_label"), errors="coerce")
        return pd.Series(np.where(base.notna(), (base > 0).astype(float), np.nan), index=df.index)
    if target_mode == "shadow_improves_edge":
        base = pd.to_numeric(df.get("label_shadow_day_improves_edge"), errors="coerce")
        return base
    if target_mode == "top1_feasible_positive":
        base = pd.to_numeric(df.get("label_top1_feasible_positive"), errors="coerce")
        if base.notna().sum() == 0:
            fallback = pd.to_numeric(df.get("top1_feasible_ev_label"), errors="coerce")
            return pd.Series(np.where(fallback.notna(), (fallback > 0).astype(float), np.nan), index=df.index)
        return base
    if target_mode == "top1_feasible_improves_edge":
        base = pd.to_numeric(df.get("label_top1_feasible_improves_edge"), errors="coerce")
        if base.notna().sum() == 0:
            fallback = pd.to_numeric(df.get("top1_feasible_value_delta"), errors="coerce")
            return pd.Series(np.where(fallback.notna(), (fallback > 0).astype(float), np.nan), index=df.index)
        return base
    if target_mode == "top1_feasible_value":
        base = pd.to_numeric(df.get("top1_feasible_value_delta"), errors="coerce")
        if base.notna().sum() == 0:
            base = pd.to_numeric(df.get("top1_feasible_uplift_vs_pool_mean"), errors="coerce")
        return base
    if target_mode == "best_feasible_append_value":
        base = pd.to_numeric(df.get("best_feasible_append_delta_vs_edge_mean"), errors="coerce")
        if base.notna().sum() == 0:
            base = pd.to_numeric(df.get("best_feasible_append_uplift_vs_pool_mean"), errors="coerce")
        return base
    raise ValueError(f"Unsupported target mode: {target_mode}")


def _is_regression_target(target_mode: str) -> bool:
    return str(target_mode) in {"best_feasible_append_value", "top1_feasible_value"}


def _candidate_exists_series(df: pd.DataFrame, candidate_mode: str) -> pd.Series:
    mode = str(candidate_mode or "shadow").strip().lower()
    if mode == "shadow":
        return _safe_bool_as_int(df.get("shadow_candidate_exists", pd.Series(0, index=df.index))).astype(int)
    if mode == "top1_feasible":
        return _safe_bool_as_int(df.get("top1_feasible_exists", pd.Series(0, index=df.index))).astype(int)
    if mode == "none":
        return pd.Series(1, index=df.index, dtype="int64")
    raise ValueError(f"Unsupported candidate mode: {candidate_mode}")


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in LEAKAGE_SAFE_FEATURES:
        if col == "shadow_candidate_exists":
            out[col] = _safe_bool_as_int(df.get(col, pd.Series(np.nan, index=df.index)))
        else:
            out[col] = pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")
        # Keep feature dimensionality stable on small windows.
        if pd.to_numeric(out[col], errors="coerce").notna().sum() == 0:
            out[col] = 0.0
    return out


def _build_model() -> Pipeline:
    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover
        imputer = SimpleImputer(strategy="median")
    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.50,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=17,
                ),
            ),
        ]
    )


def _build_regression_model() -> Pipeline:
    try:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover
        imputer = SimpleImputer(strategy="median")
    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=17)),
        ]
    )


def _binary_metrics(target: pd.Series, permit: pd.Series, value_delta: pd.Series) -> dict[str, Any]:
    mask = target.notna()
    target = pd.to_numeric(target.loc[mask], errors="coerce")
    permit = pd.to_numeric(permit.loc[mask], errors="coerce").fillna(0).astype(int)
    value_delta = pd.to_numeric(value_delta.loc[mask], errors="coerce")

    total = int(len(target))
    if total == 0:
        return {
            "labeled_days": 0,
            "positive_days": 0,
            "negative_days": 0,
            "permit_days": 0,
            "abstain_days": 0,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
            "precision_on_permit": np.nan,
            "recall_on_positive": np.nan,
            "false_permit_rate": np.nan,
            "veto_regret_rate": np.nan,
            "permit_mean_value_delta": np.nan,
            "abstain_mean_value_delta": np.nan,
            "all_mean_value_delta": np.nan,
        }

    y = target.astype(int)
    p = permit.astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    permit_days = int((p == 1).sum())
    abstain_days = int((p == 0).sum())
    positive_days = int((y == 1).sum())
    negative_days = int((y == 0).sum())

    precision = (tp / permit_days) if permit_days > 0 else np.nan
    recall = (tp / positive_days) if positive_days > 0 else np.nan
    false_permit_rate = (fp / permit_days) if permit_days > 0 else np.nan
    veto_regret_rate = (fn / abstain_days) if abstain_days > 0 else np.nan

    permit_mean_delta = float(value_delta.loc[p == 1].mean()) if permit_days > 0 else np.nan
    abstain_mean_delta = float(value_delta.loc[p == 0].mean()) if abstain_days > 0 else np.nan
    all_mean_delta = float(value_delta.mean()) if not value_delta.empty else np.nan

    return {
        "labeled_days": total,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "permit_days": permit_days,
        "abstain_days": abstain_days,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision_on_permit": float(precision) if not pd.isna(precision) else np.nan,
        "recall_on_positive": float(recall) if not pd.isna(recall) else np.nan,
        "false_permit_rate": float(false_permit_rate) if not pd.isna(false_permit_rate) else np.nan,
        "veto_regret_rate": float(veto_regret_rate) if not pd.isna(veto_regret_rate) else np.nan,
        "permit_mean_value_delta": permit_mean_delta,
        "abstain_mean_value_delta": abstain_mean_delta,
        "all_mean_value_delta": all_mean_delta,
    }


def main() -> None:
    args = parse_args()
    if not args.daily_context_csv.exists():
        raise FileNotFoundError(f"Daily context CSV not found: {args.daily_context_csv}")

    df = pd.read_csv(args.daily_context_csv)
    if df.empty:
        raise RuntimeError("Daily context CSV is empty.")
    if "run_date" not in df.columns:
        raise RuntimeError("Daily context CSV must contain run_date.")

    df = df.copy()
    rows_before_snapshot_filter = int(len(df))
    snapshot_excluded_count = 0
    snapshot_mode_counts_retained: dict[str, int] = {}
    if args.exclude_snapshot_modes and "snapshot_mode" in df.columns:
        exclude_modes = {str(x).strip().lower() for x in args.exclude_snapshot_modes if str(x).strip()}
        mode_series = df["snapshot_mode"].astype(str).str.strip()
        keep_mask = ~mode_series.str.lower().isin(exclude_modes)
        snapshot_excluded_count = int((~keep_mask).sum())
        df = df.loc[keep_mask].copy()
    if "snapshot_mode" in df.columns:
        snapshot_mode_counts_retained = (
            df["snapshot_mode"].fillna("").astype(str).value_counts(dropna=False).to_dict()
        )
    df["run_date"] = pd.to_datetime(df["run_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.loc[df["run_date"].notna()].sort_values("run_date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid run_date rows in daily context CSV.")

    target_primary = _target_series(df, args.target_mode)
    is_regression_target = _is_regression_target(str(args.target_mode))
    primary_class_count = int(pd.to_numeric(target_primary, errors="coerce").dropna().nunique())
    target = target_primary.copy()
    effective_target_mode = str(args.target_mode)
    fallback_used = False
    if (not is_regression_target) and primary_class_count < 2 and str(args.fallback_target_mode) != "none":
        fallback_target = _target_series(df, str(args.fallback_target_mode))
        fallback_class_count = int(pd.to_numeric(fallback_target, errors="coerce").dropna().nunique())
        if fallback_class_count >= 2:
            target = fallback_target.copy()
            effective_target_mode = str(args.fallback_target_mode)
            fallback_used = True
    value_delta = pd.to_numeric(df.get("top1_feasible_value_delta"), errors="coerce")
    value_delta_fallback = pd.to_numeric(df.get("shadow_minus_edge_ev_per_resolved"), errors="coerce")
    value_delta = value_delta.where(value_delta.notna(), value_delta_fallback)
    shadow_exists = _safe_bool_as_int(df.get("shadow_candidate_exists", pd.Series(0, index=df.index))).astype(int)
    candidate_exists = _candidate_exists_series(df, args.candidate_mode).astype(int)

    x_all = _build_feature_frame(df)
    run_dates = sorted(df["run_date"].dropna().astype(str).unique().tolist())

    scored_rows: list[dict[str, Any]] = []
    for run_date in run_dates:
        current_mask = df["run_date"].astype(str) == str(run_date)
        if int(current_mask.sum()) != 1:
            current_idx = df.loc[current_mask].index.min()
        else:
            current_idx = df.loc[current_mask].index[0]

        train_mask = (df["run_date"].astype(str) < str(run_date)) & target.notna()
        train_idx = df.loc[train_mask].index
        train_target = pd.to_numeric(target.loc[train_idx], errors="coerce").dropna()

        model_ready = False
        pred_score = np.nan
        pred_proba = np.nan
        pred_value = np.nan
        gate_reason = "abstain_no_model"
        train_rows = int(len(train_target))
        train_positive_rate = float((train_target > 0.0).mean()) if not train_target.empty else np.nan

        if train_rows >= int(args.min_train_days):
            if is_regression_target:
                if float(train_target.std(ddof=0)) > 1e-9:
                    model = _build_regression_model()
                    model.fit(x_all.loc[train_target.index], train_target.astype(float))
                    pred_value = float(model.predict(x_all.loc[[current_idx]])[0])
                    pred_score = pred_value
                    model_ready = True
                    if int(candidate_exists.loc[current_idx]) <= 0:
                        gate_reason = "abstain_no_candidate"
                    elif pred_score >= float(args.threshold):
                        gate_reason = "permit_value_above_threshold"
                    else:
                        gate_reason = "abstain_value_below_threshold"
                else:
                    gate_reason = "abstain_constant_train_target"
            else:
                class_count = int(train_target.nunique(dropna=True))
                if class_count >= 2:
                    model = _build_model()
                    model.fit(x_all.loc[train_target.index], train_target.astype(int))
                    proba = model.predict_proba(x_all.loc[[current_idx]])[:, 1]
                    pred_proba = float(proba[0])
                    pred_score = pred_proba
                    model_ready = True
                    if int(candidate_exists.loc[current_idx]) <= 0:
                        gate_reason = "abstain_no_candidate"
                    elif pred_score >= float(args.threshold):
                        gate_reason = "permit_probability_above_threshold"
                    else:
                        gate_reason = "abstain_probability_below_threshold"
                else:
                    gate_reason = "abstain_single_class_train"
        else:
            gate_reason = "abstain_min_train_not_met"

        permit = int(
            model_ready
            and (int(candidate_exists.loc[current_idx]) == 1)
            and (not pd.isna(pred_score))
            and (pred_score >= float(args.threshold))
        )
        target_value = pd.to_numeric(target.loc[current_idx], errors="coerce")
        target_binary = int(target_value > 0.0) if not pd.isna(target_value) else np.nan
        scored_rows.append(
            {
                "run_date": str(run_date),
                "task_type": ("regression" if is_regression_target else "classification"),
                "target_mode": str(args.target_mode),
                "effective_target_mode": effective_target_mode,
                "target_value_day": target_value,
                "target_append_positive_day": target_binary,
                "shadow_candidate_exists": int(shadow_exists.loc[current_idx]),
                "candidate_mode": str(args.candidate_mode),
                "candidate_exists": int(candidate_exists.loc[current_idx]),
                "shadow_minus_edge_ev_per_resolved": _safe_num(value_delta.loc[current_idx], default=np.nan),
                "model_ready": bool(model_ready),
                "train_rows": train_rows,
                "train_positive_rate": train_positive_rate,
                "predicted_permission_score": pred_score,
                "predicted_permission_probability": pred_proba,
                "predicted_append_value": pred_value,
                "threshold": float(args.threshold),
                "permit_append_day": int(permit),
                "gate_reason": gate_reason,
            }
        )

    scored_df = pd.DataFrame.from_records(scored_rows)
    scored_df["target_value_day"] = pd.to_numeric(scored_df["target_value_day"], errors="coerce")
    scored_df["target_append_positive_day"] = pd.to_numeric(scored_df["target_append_positive_day"], errors="coerce")
    scored_df["permit_append_day"] = pd.to_numeric(scored_df["permit_append_day"], errors="coerce").fillna(0).astype(int)
    scored_df["shadow_candidate_exists"] = pd.to_numeric(scored_df["shadow_candidate_exists"], errors="coerce").fillna(0).astype(int)
    scored_df["candidate_exists"] = pd.to_numeric(scored_df["candidate_exists"], errors="coerce").fillna(0).astype(int)
    scored_df["shadow_minus_edge_ev_per_resolved"] = pd.to_numeric(scored_df["shadow_minus_edge_ev_per_resolved"], errors="coerce")
    scored_df["predicted_permission_score"] = pd.to_numeric(scored_df["predicted_permission_score"], errors="coerce")
    scored_df["predicted_permission_probability"] = pd.to_numeric(scored_df["predicted_permission_probability"], errors="coerce")
    scored_df["predicted_append_value"] = pd.to_numeric(scored_df["predicted_append_value"], errors="coerce")

    actionable_mask = scored_df["candidate_exists"] == 1
    all_metrics = _binary_metrics(
        scored_df["target_append_positive_day"],
        scored_df["permit_append_day"],
        scored_df["shadow_minus_edge_ev_per_resolved"],
    )
    actionable_metrics = _binary_metrics(
        scored_df.loc[actionable_mask, "target_append_positive_day"],
        scored_df.loc[actionable_mask, "permit_append_day"],
        scored_df.loc[actionable_mask, "shadow_minus_edge_ev_per_resolved"],
    )

    summary: dict[str, Any] = {
        "task_type": ("regression" if is_regression_target else "classification"),
        "target_mode": str(args.target_mode),
        "effective_target_mode": effective_target_mode,
        "candidate_mode": str(args.candidate_mode),
        "fallback_target_mode": str(args.fallback_target_mode),
        "fallback_used": bool(fallback_used),
        "target_class_count_primary": int(primary_class_count),
        "target_positive_rate_effective": float((pd.to_numeric(target, errors="coerce") > 0.0).mean())
        if pd.to_numeric(target, errors="coerce").notna().any()
        else np.nan,
        "target_value_mean_effective": float(pd.to_numeric(target, errors="coerce").mean())
        if pd.to_numeric(target, errors="coerce").notna().any()
        else np.nan,
        "target_value_std_effective": float(pd.to_numeric(target, errors="coerce").std(ddof=0))
        if pd.to_numeric(target, errors="coerce").notna().any()
        else np.nan,
        "threshold": float(args.threshold),
        "min_train_days": int(args.min_train_days),
        "daily_context_csv": str(args.daily_context_csv),
        "rows_before_snapshot_filter": int(rows_before_snapshot_filter),
        "rows_excluded_snapshot_mode": int(snapshot_excluded_count),
        "exclude_snapshot_modes": [str(x) for x in args.exclude_snapshot_modes],
        "snapshot_mode_counts_retained": snapshot_mode_counts_retained,
        "rows_total": int(len(scored_df)),
        "rows_model_ready": int(pd.to_numeric(scored_df["model_ready"], errors="coerce").fillna(0).astype(bool).sum()),
        "rows_candidate_exists": int(actionable_mask.sum()),
        "rows_shadow_candidate_exists": int((scored_df["shadow_candidate_exists"] == 1).sum()),
        "window": {
            "start": str(scored_df["run_date"].min()) if not scored_df.empty else None,
            "end": str(scored_df["run_date"].max()) if not scored_df.empty else None,
        },
        "feature_columns": LEAKAGE_SAFE_FEATURES,
        "metrics_all_days": all_metrics,
        "metrics_actionable_days": actionable_metrics,
        "gate_reason_counts": scored_df["gate_reason"].value_counts(dropna=False).to_dict(),
    }

    args.out_scored_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(args.out_scored_csv, index=False)
    args.out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 96)
    print("STAGE1 DAILY APPEND PERMISSION SUMMARY")
    print("=" * 96)
    print(f"task_type: {summary['task_type']}")
    print(f"target_mode: {summary['target_mode']}")
    print(f"rows_total: {summary['rows_total']}")
    print(f"rows_model_ready: {summary['rows_model_ready']}")
    print(f"rows_shadow_candidate_exists: {summary['rows_shadow_candidate_exists']}")
    print(f"metrics_all_days: {summary['metrics_all_days']}")
    print(f"metrics_actionable_days: {summary['metrics_actionable_days']}")
    print(f"gate_reason_counts: {summary['gate_reason_counts']}")
    print("\nSaved:")
    print(f"  Scored:  {args.out_scored_csv}")
    print(f"  Summary: {args.out_summary_json}")


def _safe_num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


if __name__ == "__main__":
    main()
