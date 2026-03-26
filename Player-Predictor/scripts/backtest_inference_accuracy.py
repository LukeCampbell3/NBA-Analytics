#!/usr/bin/env python3
"""
Walk-forward inference backtest for the structured stack.

For each eligible row:
- use all games before that row as history
- predict the next game using production inference
- compare prediction to the real next-game outcome

This gives a practical view of how the model behaves in real inference.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from structured_stack_inference import StructuredStackInference  # noqa: E402


MODEL_DIR = REPO_ROOT / "model"
DATA_DIR = REPO_ROOT / "Data-Proc"
TARGETS = ["PTS", "TRB", "AST"]
WINDOWS = {
    "PTS": [3, 5, 7, 10],
    "TRB": [1, 2, 3],
    "AST": [1, 2, 3],
}


def resolve_manifest_path(run_id: str | None, latest: bool) -> Path | None:
    if run_id:
        return MODEL_DIR / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return MODEL_DIR / "latest_structured_lstm_stack.json"
    return MODEL_DIR / "production_structured_lstm_stack.json"


def gather_csvs(csvs: list[str] | None, player_dir: str | None, limit_players: int | None) -> list[Path]:
    if csvs:
        return [Path(item).resolve() for item in csvs]
    if player_dir:
        root = Path(player_dir)
        return sorted(root.glob("*.csv"))
    player_dirs = sorted([path for path in DATA_DIR.iterdir() if path.is_dir()])
    if limit_players:
        player_dirs = player_dirs[:limit_players]
    paths = []
    for directory in player_dirs:
        paths.extend(sorted(directory.glob("*.csv")))
    return paths


def choose_test_csvs(csv_paths: list[Path], seq_len: int, per_player_limit: int, count: int = 2) -> list[Path]:
    scored = []
    min_rows = seq_len + max(1, per_player_limit)
    for path in csv_paths:
        try:
            rows = len(pd.read_csv(path))
        except Exception:
            continue
        if rows >= min_rows:
            scored.append((rows, path))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    chosen = [path for _, path in scored[:count]]
    if chosen:
        return chosen
    return csv_paths[:count]


def infer_player_name(path: Path) -> str:
    return path.parent.name


def normalize_input_frame(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    out = df.copy()
    if "Player" not in out.columns:
        out["Player"] = player_name
    if "Game_Index" not in out.columns:
        out["Game_Index"] = np.arange(len(out), dtype=np.int32)
    return out


def is_prepared_frame(df: pd.DataFrame, required_columns: list[str]) -> bool:
    return all(column in df.columns for column in required_columns)


def parse_dates(df: pd.DataFrame) -> pd.Series | None:
    if "Date" not in df.columns:
        return None
    parsed = pd.to_datetime(df["Date"], errors="coerce")
    if parsed.notna().sum() == 0:
        return None
    return parsed


def compute_summary(df: pd.DataFrame, pred_prefix: str = "pred") -> dict:
    summary = {
        "n_predictions": int(len(df)),
        "targets": {},
    }
    for target in TARGETS:
        pred_col = f"{pred_prefix}_{target}"
        actual_col = f"actual_{target}"
        base_col = f"baseline_{target}"
        errors = df[pred_col] - df[actual_col]
        abs_errors = errors.abs()
        base_abs_errors = (df[base_col] - df[actual_col]).abs()
        payload = {
            "mae": float(mean_absolute_error(df[actual_col], df[pred_col])),
            "rmse": float(np.sqrt(mean_squared_error(df[actual_col], df[pred_col]))),
            "r2": float(r2_score(df[actual_col], df[pred_col])),
            "bias": float(errors.mean()),
            "median_abs_error": float(abs_errors.median()),
            "p80_abs_error": float(abs_errors.quantile(0.80)),
            "baseline_mae": float(base_abs_errors.mean()),
            "improvement_vs_baseline": float(base_abs_errors.mean() - abs_errors.mean()),
            "hit_rates": {},
        }
        for window in WINDOWS[target]:
            payload["hit_rates"][f"within_{window}"] = float((abs_errors <= window).mean())
        sigma_col = f"{target}_uncertainty_sigma"
        if sigma_col in df.columns and len(df) >= 10:
            high_sigma_cut = float(df[sigma_col].quantile(0.80))
            low_sigma_cut = float(df[sigma_col].quantile(0.20))
            high_sigma_mask = df[sigma_col] >= high_sigma_cut
            low_sigma_mask = df[sigma_col] <= low_sigma_cut
            payload["confidence"] = {
                "corr_abs_err_sigma": float(df[sigma_col].corr(abs_errors, method="spearman")) if df[sigma_col].nunique() > 1 else 0.0,
                "corr_abs_err_feasibility": float(df["feasibility"].corr(abs_errors, method="spearman")) if "feasibility" in df.columns and df["feasibility"].nunique() > 1 else 0.0,
                "corr_abs_err_belief_uncertainty": float(df["belief_uncertainty"].corr(abs_errors, method="spearman")) if "belief_uncertainty" in df.columns and df["belief_uncertainty"].nunique() > 1 else 0.0,
                "mae_high_sigma_20pct": float(abs_errors[high_sigma_mask].mean()) if high_sigma_mask.any() else None,
                "mae_low_sigma_20pct": float(abs_errors[low_sigma_mask].mean()) if low_sigma_mask.any() else None,
                "mae_low_feasibility_20pct": float(abs_errors[df["feasibility"] <= float(df["feasibility"].quantile(0.20))].mean()) if "feasibility" in df.columns else None,
                "mae_high_feasibility_20pct": float(abs_errors[df["feasibility"] >= float(df["feasibility"].quantile(0.80))].mean()) if "feasibility" in df.columns else None,
            }
        summary["targets"][target] = payload
    summary["overall_avg_mae"] = float(np.mean([summary["targets"][target]["mae"] for target in TARGETS]))
    return summary


def deterministic_setup(seed: int = 42):
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def print_summary(summary: dict, title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"Predictions evaluated: {summary['n_predictions']}")
    print(f"Overall avg MAE: {summary['overall_avg_mae']:.4f}")
    for target in TARGETS:
        payload = summary["targets"][target]
        hit_parts = ", ".join(
            f"{name}={value * 100:.1f}%"
            for name, value in payload["hit_rates"].items()
        )
        print(f"\n{target}:")
        print(f"  MAE: {payload['mae']:.4f}")
        print(f"  RMSE: {payload['rmse']:.4f}")
        print(f"  R2: {payload['r2']:.4f}")
        print(f"  Bias: {payload['bias']:+.4f}")
        print(f"  Median abs error: {payload['median_abs_error']:.4f}")
        print(f"  P80 abs error: {payload['p80_abs_error']:.4f}")
        print(f"  Baseline MAE: {payload['baseline_mae']:.4f}")
        print(f"  Improvement vs baseline: {payload['improvement_vs_baseline']:+.4f}")
        print(f"  Accuracy windows: {hit_parts}")
        conf = payload.get("confidence")
        if conf:
            print(
                "  Confidence diagnostics: "
                f"corr(abs_err,sigma)={conf['corr_abs_err_sigma']:+.3f}, "
                f"corr(abs_err,feasibility)={conf['corr_abs_err_feasibility']:+.3f}, "
                f"corr(abs_err,belief_uncertainty)={conf['corr_abs_err_belief_uncertainty']:+.3f}"
            )
            print(
                "  Confidence buckets: "
                f"high_sigma_20%={conf['mae_high_sigma_20pct']:.4f} vs low_sigma_20%={conf['mae_low_sigma_20pct']:.4f}, "
                f"low_feas_20%={conf['mae_low_feasibility_20pct']:.4f} vs high_feas_20%={conf['mae_high_feasibility_20pct']:.4f}"
            )


def print_examples(df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("EXAMPLE MISSES")
    print("=" * 100)
    worst_pts = df.sort_values("abs_err_PTS", ascending=False).head(10)
    for _, row in worst_pts.iterrows():
        print(
            f"  {row['player']} | idx={int(row['target_index'])} | "
            f"PTS pred={row['pred_PTS']:.2f} actual={row['actual_PTS']:.2f} "
            f"TRB pred={row['pred_TRB']:.2f} actual={row['actual_TRB']:.2f} "
            f"AST pred={row['pred_AST']:.2f} actual={row['actual_AST']:.2f} | "
            f"feas={row.get('feasibility', float('nan')):.3f} "
            f"belief_unc={row.get('belief_uncertainty', float('nan')):.3f} "
            f"pts_sigma={row.get('PTS_uncertainty_sigma', float('nan')):.2f}"
        )


def print_skip_summary(skipped: list[dict]):
    if not skipped:
        return
    print("\n" + "=" * 100)
    print("SKIP SUMMARY")
    print("=" * 100)
    skipped_df = pd.DataFrame.from_records(skipped)
    reason_counts = skipped_df["error"].value_counts().head(10)
    for reason, count in reason_counts.items():
        print(f"  {count:4d} | {reason}")


def print_data_quality_summary(df: pd.DataFrame):
    if df.empty or "schema_repaired" not in df.columns:
        return
    print("\n" + "=" * 100)
    print("DATA QUALITY / FALLBACK SUMMARY")
    print("=" * 100)
    repaired_rate = float(df["schema_repaired"].mean()) * 100.0
    default_id_rate = float(df["used_default_ids"].mean()) * 100.0
    nan_repair_rate = float(df["nan_feature_repaired"].mean()) * 100.0 if "nan_feature_repaired" in df.columns else 0.0
    mean_blend = float(df["fallback_blend"].mean())
    fallback_rate = float((df["fallback_blend"] > 0.0).mean()) * 100.0 if "fallback_blend" in df.columns else 0.0
    floor_guard_rate = float(df["floor_guard_applied"].mean()) * 100.0 if "floor_guard_applied" in df.columns else 0.0
    split_rate = float(df["pts_residual_split_applied"].mean()) * 100.0 if "pts_residual_split_applied" in df.columns else 0.0
    print(f"Schema repaired rows: {repaired_rate:.1f}%")
    print(f"Rows using default IDs: {default_id_rate:.1f}%")
    print(f"Rows with NaN feature repair: {nan_repair_rate:.1f}%")
    print(f"Mean fallback blend: {mean_blend:.3f}")
    print(f"Rows with fallback: {fallback_rate:.1f}%")
    print(f"Rows with floor guard: {floor_guard_rate:.1f}%")
    print(f"Rows with PTS residual split: {split_rate:.1f}%")
    if "fallback_reasons" in df.columns:
        counts = (
            df["fallback_reasons"]
            .fillna("")
            .astype(str)
            .value_counts()
            .head(10)
        )
        print("Fallback patterns:")
        for reason, count in counts.items():
            label = reason if reason else "<none>"
            print(f"  {count:4d} | {label}")


def print_nan_repair_summary(df: pd.DataFrame):
    if df.empty or "nan_feature_repaired" not in df.columns:
        return
    nan_df = df.loc[df["nan_feature_repaired"]].copy()
    if nan_df.empty:
        return
    print("\n" + "=" * 100)
    print("NAN FEATURE REPAIR ROWS")
    print("=" * 100)
    print(f"Rows repaired: {len(nan_df)}")
    print(f"Mean repaired feature count: {nan_df['nan_feature_count'].mean():.1f}")
    if "nan_feature_columns" in nan_df.columns:
        exploded = (
            nan_df["nan_feature_columns"]
            .fillna("")
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )
        exploded = exploded[exploded != ""]
        if not exploded.empty:
            print("Top repaired columns:")
            for column, count in exploded.value_counts().head(15).items():
                print(f"  {count:4d} | {column}")
    worst = nan_df.sort_values("abs_err_PTS", ascending=False).head(10)
    for _, row in worst.iterrows():
        print(
            f"  {row['player']} | idx={int(row['target_index'])} | nan_features={int(row['nan_feature_count'])} | "
            f"PTS pred={row['pred_PTS']:.2f} actual={row['actual_PTS']:.2f} "
            f"TRB pred={row['pred_TRB']:.2f} actual={row['actual_TRB']:.2f} "
            f"AST pred={row['pred_AST']:.2f} actual={row['actual_AST']:.2f}"
        )


def compute_market_summary(df: pd.DataFrame, pred_prefix: str = "pred") -> dict | None:
    if df.empty:
        return None
    summary = {"targets": {}, "available_targets": 0}
    for target in TARGETS:
        market_col = f"market_{target}"
        pred_col = f"{pred_prefix}_{target}"
        actual_col = f"actual_{target}"
        if market_col not in df.columns:
            continue
        market_df = df.loc[df[market_col].notna()].copy()
        if market_df.empty:
            continue
        model_abs_err = (market_df[pred_col] - market_df[actual_col]).abs()
        market_abs_err = (market_df[market_col] - market_df[actual_col]).abs()
        disagreement = (market_df[pred_col] - market_df[market_col]).abs()
        payload = {
            "n_rows": int(len(market_df)),
            "model_mae": float(model_abs_err.mean()),
            "market_mae": float(market_abs_err.mean()),
            "improvement_vs_market": float(market_abs_err.mean() - model_abs_err.mean()),
            "model_vs_market_mae": float(disagreement.mean()),
            "model_beats_market_rate": float((model_abs_err < market_abs_err).mean()),
            "tie_rate": float((model_abs_err == market_abs_err).mean()),
            "corr_abs_err_market_disagreement": float(disagreement.corr(model_abs_err, method="spearman")) if disagreement.nunique() > 1 else 0.0,
        }
        high_cut = float(disagreement.quantile(0.80))
        low_cut = float(disagreement.quantile(0.20))
        high_mask = disagreement >= high_cut
        low_mask = disagreement <= low_cut
        payload["high_disagreement_model_mae"] = float(model_abs_err[high_mask].mean()) if high_mask.any() else None
        payload["high_disagreement_market_mae"] = float(market_abs_err[high_mask].mean()) if high_mask.any() else None
        payload["low_disagreement_model_mae"] = float(model_abs_err[low_mask].mean()) if low_mask.any() else None
        payload["low_disagreement_market_mae"] = float(market_abs_err[low_mask].mean()) if low_mask.any() else None
        summary["targets"][target] = payload
    summary["available_targets"] = int(len(summary["targets"]))
    return summary if summary["available_targets"] > 0 else None


def print_market_summary(summary: dict | None, title: str):
    if not summary:
        return
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for target in TARGETS:
        payload = summary["targets"].get(target)
        if not payload:
            continue
        print(f"\n{target}:")
        print(f"  Market rows: {payload['n_rows']}")
        print(f"  Model MAE: {payload['model_mae']:.4f}")
        print(f"  Vegas MAE: {payload['market_mae']:.4f}")
        print(f"  Improvement vs Vegas: {payload['improvement_vs_market']:+.4f}")
        print(f"  Abs(model-vegas): {payload['model_vs_market_mae']:.4f}")
        print(f"  Model beats Vegas rate: {payload['model_beats_market_rate'] * 100:.1f}%")
        print(f"  Tie rate: {payload['tie_rate'] * 100:.1f}%")
        print(f"  Corr(abs_err, disagreement): {payload['corr_abs_err_market_disagreement']:+.3f}")
        print(
            "  Disagreement buckets: "
            f"high model={payload['high_disagreement_model_mae']:.4f} vs vegas={payload['high_disagreement_market_mae']:.4f}, "
            f"low model={payload['low_disagreement_model_mae']:.4f} vs vegas={payload['low_disagreement_market_mae']:.4f}"
        )


def print_subset_delta(
    label: str,
    guarded_summary: dict | None,
    raw_summary: dict | None,
    baseline_reference: dict | None,
    split_summary: dict | None = None,
):
    if guarded_summary is None or raw_summary is None or baseline_reference is None:
        return
    print("\n" + "=" * 100)
    print(f"MODEL COMPARISON: {label}")
    print("=" * 100)
    for target in TARGETS:
        g = guarded_summary["targets"][target]
        r = raw_summary["targets"][target]
        if split_summary is not None:
            s = split_summary["targets"][target]
            print(
                f"  {target}: guarded={g['mae']:.4f} split={s['mae']:.4f} raw={r['mae']:.4f} "
                f"baseline={g['baseline_mae']:.4f} guard_vs_raw={r['mae'] - g['mae']:+.4f} "
                f"split_vs_raw={r['mae'] - s['mae']:+.4f}"
            )
        else:
            print(
                f"  {target}: guarded={g['mae']:.4f} raw={r['mae']:.4f} "
                f"baseline={g['baseline_mae']:.4f} guard_vs_raw={r['mae'] - g['mae']:+.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest for production inference accuracy.")
    parser.add_argument("--csv", nargs="+", help="One or more CSVs to backtest")
    parser.add_argument("--player-dir", help="Single player directory to backtest")
    parser.add_argument("--limit-players", type=int, default=None, help="Limit player directories when using default dataset scan")
    parser.add_argument("--per-player-limit", type=int, default=25, help="Max predictions per player/csv, taken from the most recent eligible rows")
    parser.add_argument("--max-predictions", type=int, default=None, help="Global cap on evaluated predictions")
    parser.add_argument("--days-back", type=int, default=None, help="Only score target rows within the last N calendar days of each CSV")
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON summary path")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV path for row-level predictions")
    parser.add_argument("--test", action="store_true", help="Run in test mode with just 2 players and 5 predictions each")
    parser.add_argument("--strict-schema", action="store_true", help="Only evaluate rows that required no schema repair or default IDs")
    args = parser.parse_args()
    deterministic_setup()

    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    predictor = StructuredStackInference(
        model_dir=str(MODEL_DIR),
        manifest_path=manifest_path,
        allow_schema_repair=not args.strict_schema,
    )
    csv_paths = gather_csvs(args.csv, args.player_dir, args.limit_players)
    if not csv_paths:
        raise FileNotFoundError("No CSV files found to backtest")

    # Test mode: just 2 players, 5 predictions each
    if args.test:
        csv_paths = choose_test_csvs(csv_paths, predictor.seq_len, 5, count=2)
        args.per_player_limit = 5
        print(f"TEST MODE: Using {len(csv_paths)} players, {args.per_player_limit} predictions each")
        print(f"CSV paths: {[str(p) for p in csv_paths]}")

    records = []
    skipped = []
    for csv_path in csv_paths:
        player_name = infer_player_name(csv_path)
        raw_df = normalize_input_frame(pd.read_csv(csv_path), player_name)
        assume_prepared = is_prepared_frame(raw_df, predictor.feature_columns)
        if len(raw_df) <= predictor.seq_len:
            continue
        parsed_dates = parse_dates(raw_df)
        eligible_indices = list(range(predictor.seq_len, len(raw_df)))
        if parsed_dates is not None and args.days_back is not None:
            cutoff = parsed_dates.max() - pd.Timedelta(days=int(args.days_back))
            eligible_indices = [idx for idx in eligible_indices if pd.notna(parsed_dates.iloc[idx]) and parsed_dates.iloc[idx] >= cutoff]
        if args.per_player_limit is not None and len(eligible_indices) > args.per_player_limit:
            eligible_indices = eligible_indices[-args.per_player_limit:]

        for idx in eligible_indices:
            history_df = raw_df.iloc[:idx].copy()
            actual_row = raw_df.iloc[idx]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    explanation = predictor.predict(history_df, assume_prepared=assume_prepared)
            except Exception as exc:
                skipped.append({"csv": str(csv_path), "player": player_name, "target_index": int(idx), "error": str(exc)})
                continue
            record = {
                "player": player_name,
                "csv": str(csv_path),
                "target_index": int(idx),
                "target_date": str(actual_row["Date"]) if "Date" in actual_row else None,
                "did_not_play": float(actual_row["Did_Not_Play"]) if "Did_Not_Play" in actual_row else 0.0,
                "minutes": float(actual_row["MP"]) if "MP" in actual_row else None,
                "belief_uncertainty": float(explanation["latent_environment"].get("belief_uncertainty", 0.0)),
                "feasibility": float(explanation["latent_environment"].get("feasibility", 0.0)),
                "role_shift_risk": float(explanation["latent_environment"].get("role_shift_risk", 0.0)),
                "volatility_regime_risk": float(explanation["latent_environment"].get("volatility_regime_risk", 0.0)),
                "context_pressure_risk": float(explanation["latent_environment"].get("context_pressure_risk", 0.0)),
                "pts_trend_trust": float(explanation["latent_environment"].get("pts_trend_trust", 0.0)),
                "pts_baseline_trust": float(explanation["latent_environment"].get("pts_baseline_trust", 0.0)),
                "pts_elasticity": float(explanation["latent_environment"].get("pts_elasticity", 0.0)),
                "pts_opportunity_jump": float(explanation["latent_environment"].get("pts_opportunity_jump", 0.0)),
                "schema_repaired": bool(explanation.get("data_quality", {}).get("schema_repaired", False)),
                "used_default_ids": bool(explanation.get("data_quality", {}).get("used_default_ids", False)),
                "active_like": bool(explanation.get("data_quality", {}).get("active_like", False)),
                "floor_guard_applied": bool(explanation.get("data_quality", {}).get("floor_guard_applied", False)),
                "nan_feature_repaired": bool(explanation.get("data_quality", {}).get("nan_feature_repaired", False)),
                "nan_feature_count": int(explanation.get("data_quality", {}).get("nan_feature_count", 0)),
                "nan_feature_columns": ",".join(explanation.get("data_quality", {}).get("nan_feature_columns", [])),
                "fallback_blend": float(explanation.get("data_quality", {}).get("fallback_blend", 0.0)),
                "fallback_reasons": ",".join(explanation.get("data_quality", {}).get("fallback_reasons", [])),
                "pts_residual_split_applied": bool(explanation.get("data_quality", {}).get("pts_residual_split_applied", False)),
            }
            for target in TARGETS:
                record[f"pred_{target}"] = float(explanation["predicted"][target])
                record[f"split_pred_{target}"] = float(explanation.get("predicted_split_model", {}).get(target, explanation["predicted"][target]))
                record[f"raw_pred_{target}"] = float(explanation.get("predicted_raw_model", {}).get(target, explanation["predicted"][target]))
                record[f"baseline_{target}"] = float(explanation["baseline"][target])
                record[f"actual_{target}"] = float(actual_row[target])
                record[f"abs_err_{target}"] = abs(record[f"pred_{target}"] - record[f"actual_{target}"])
                record[f"{target}_spike_probability"] = float(explanation["target_factors"][target].get("spike_probability", 0.0))
                record[f"{target}_uncertainty_sigma"] = float(explanation["target_factors"][target].get("uncertainty_sigma", 0.0))
                market_value = actual_row.get(f"Market_{target}", np.nan)
                market_gap = actual_row.get(f"{target}_market_gap", np.nan)
                record[f"market_{target}"] = float(market_value) if pd.notna(market_value) else np.nan
                record[f"market_gap_{target}"] = float(market_gap) if pd.notna(market_gap) else np.nan
                record[f"model_minus_market_{target}"] = (
                    float(record[f"pred_{target}"] - record[f"market_{target}"])
                    if pd.notna(record[f"market_{target}"])
                    else np.nan
                )
                record[f"market_abs_err_{target}"] = (
                    abs(float(record[f"market_{target}"] - record[f"actual_{target}"]))
                    if pd.notna(record[f"market_{target}"])
                    else np.nan
                )
                record[f"market_books_{target}"] = float(actual_row.get(f"Market_{target}_books", np.nan)) if pd.notna(actual_row.get(f"Market_{target}_books", np.nan)) else np.nan
            records.append(record)
            if args.max_predictions is not None and len(records) >= args.max_predictions:
                break
        if args.max_predictions is not None and len(records) >= args.max_predictions:
            break

    if not records:
        sample_errors = "; ".join(
            f"{item['player']} idx={item['target_index']}: {item['error']}"
            for item in skipped[:5]
        )
        detail = f" Skipped {len(skipped)} rows."
        if sample_errors:
            detail += f" Sample errors: {sample_errors}"
        raise RuntimeError("No eligible predictions were generated." + detail)

    results_df = pd.DataFrame.from_records(records)
    active_mask = (
        (results_df["did_not_play"] < 0.5)
        & ~(
            (results_df["actual_PTS"] == 0.0)
            & (results_df["actual_TRB"] == 0.0)
            & (results_df["actual_AST"] == 0.0)
            & (results_df["minutes"].fillna(0.0) <= 0.0)
        )
    )
    active_df = results_df.loc[active_mask].copy()
    clean_df = results_df.loc[(~results_df["schema_repaired"]) & (~results_df["used_default_ids"])].copy()
    repaired_df = results_df.loc[(results_df["schema_repaired"]) | (results_df["used_default_ids"])].copy()
    nan_clean_df = results_df.loc[~results_df["nan_feature_repaired"]].copy()
    nan_repaired_df = results_df.loc[results_df["nan_feature_repaired"]].copy()
    all_summary = compute_summary(results_df, pred_prefix="pred")
    active_summary = compute_summary(active_df, pred_prefix="pred") if len(active_df) else None
    split_all_summary = compute_summary(results_df, pred_prefix="split_pred")
    split_active_summary = compute_summary(active_df, pred_prefix="split_pred") if len(active_df) else None
    raw_all_summary = compute_summary(results_df, pred_prefix="raw_pred")
    raw_active_summary = compute_summary(active_df, pred_prefix="raw_pred") if len(active_df) else None
    clean_summary = compute_summary(clean_df, pred_prefix="pred") if len(clean_df) else None
    repaired_summary = compute_summary(repaired_df, pred_prefix="pred") if len(repaired_df) else None
    nan_clean_summary = compute_summary(nan_clean_df, pred_prefix="pred") if len(nan_clean_df) else None
    nan_repaired_summary = compute_summary(nan_repaired_df, pred_prefix="pred") if len(nan_repaired_df) else None
    market_all_summary = compute_market_summary(results_df, pred_prefix="pred")
    market_active_summary = compute_market_summary(active_df, pred_prefix="pred") if len(active_df) else None
    market_clean_summary = compute_market_summary(clean_df, pred_prefix="pred") if len(clean_df) else None

    print("\n" + "=" * 100)
    print("STRUCTURED STACK INFERENCE BACKTEST")
    print("=" * 100)
    print(f"Manifest target: {manifest_path}")
    print(f"Run id: {predictor.metadata.get('run_id')}")
    print(f"Avg MAE from metadata: {predictor.metadata.get('avg_mae')}")
    if skipped:
        print(f"Skipped rows due to input issues: {len(skipped)}")
        print_skip_summary(skipped)
    print_summary(all_summary, "INFERENCE BACKTEST SUMMARY: ALL TARGET ROWS")
    print_summary(split_all_summary, "RESIDUAL-SPLIT MODEL SUMMARY: ALL TARGET ROWS")
    print_summary(raw_all_summary, "RAW MODEL SUMMARY: ALL TARGET ROWS")
    print_subset_delta("ALL TARGET ROWS", all_summary, raw_all_summary, all_summary, split_summary=split_all_summary)
    if active_summary is not None and len(active_df) != len(results_df):
        print_summary(active_summary, "INFERENCE BACKTEST SUMMARY: ACTIVE-ONLY ROWS")
        if split_active_summary is not None:
            print_summary(split_active_summary, "RESIDUAL-SPLIT MODEL SUMMARY: ACTIVE-ONLY ROWS")
        if raw_active_summary is not None:
            print_summary(raw_active_summary, "RAW MODEL SUMMARY: ACTIVE-ONLY ROWS")
            print_subset_delta("ACTIVE-ONLY ROWS", active_summary, raw_active_summary, active_summary, split_summary=split_active_summary)
    if clean_summary is not None and len(clean_df):
        print_summary(clean_summary, "INFERENCE BACKTEST SUMMARY: CLEAN-CONTRACT ROWS")
    if repaired_summary is not None and len(repaired_df):
        print_summary(repaired_summary, "INFERENCE BACKTEST SUMMARY: REPAIRED-CONTRACT ROWS")
    if nan_clean_summary is not None and len(nan_clean_df):
        print_summary(nan_clean_summary, "INFERENCE BACKTEST SUMMARY: NO-NAN-REPAIR ROWS")
    if nan_repaired_summary is not None and len(nan_repaired_df):
        print_summary(nan_repaired_summary, "INFERENCE BACKTEST SUMMARY: NAN-REPAIRED ROWS")
    print_market_summary(market_all_summary, "MARKET COMPARISON: ALL TARGET ROWS")
    if market_active_summary is not None and len(active_df):
        print_market_summary(market_active_summary, "MARKET COMPARISON: ACTIVE-ONLY ROWS")
    if market_clean_summary is not None and len(clean_df):
        print_market_summary(market_clean_summary, "MARKET COMPARISON: CLEAN-CONTRACT ROWS")
    print_data_quality_summary(results_df)
    print_nan_repair_summary(results_df)
    print_examples(active_df if len(active_df) else results_df)

    if args.csv_out:
        out_csv = Path(args.csv_out)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        print(f"\nSaved row-level CSV: {out_csv}")

    if args.json_out:
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_path": str(manifest_path),
            "run_id": predictor.metadata.get("run_id"),
            "summary_all_rows": all_summary,
            "summary_active_only": active_summary,
            "summary_split_all_rows": split_all_summary,
            "summary_split_active_only": split_active_summary,
            "summary_raw_all_rows": raw_all_summary,
            "summary_raw_active_only": raw_active_summary,
            "summary_clean_rows": clean_summary,
            "summary_repaired_rows": repaired_summary,
            "summary_no_nan_repair_rows": nan_clean_summary,
            "summary_nan_repaired_rows": nan_repaired_summary,
            "market_summary_all_rows": market_all_summary,
            "market_summary_active_only": market_active_summary,
            "market_summary_clean_rows": market_clean_summary,
            "skipped": skipped,
        }
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Saved summary JSON: {out_json}")


if __name__ == "__main__":
    main()
