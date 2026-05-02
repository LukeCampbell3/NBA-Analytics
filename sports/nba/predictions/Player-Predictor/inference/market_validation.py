from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["PTS", "TRB", "AST"]


def safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def active_only_mask(df: pd.DataFrame) -> pd.Series:
    minutes = pd.to_numeric(df.get("minutes", df.get("MP")), errors="coerce").fillna(0.0)
    did_not_play = pd.to_numeric(df.get("did_not_play", df.get("Did_Not_Play")), errors="coerce").fillna(0.0)
    actual_pts = pd.to_numeric(df.get("actual_PTS", df.get("PTS")), errors="coerce").fillna(0.0)
    actual_trb = pd.to_numeric(df.get("actual_TRB", df.get("TRB")), errors="coerce").fillna(0.0)
    actual_ast = pd.to_numeric(df.get("actual_AST", df.get("AST")), errors="coerce").fillna(0.0)
    return (did_not_play < 0.5) & ~(
        (actual_pts == 0.0)
        & (actual_trb == 0.0)
        & (actual_ast == 0.0)
        & (minutes <= 0.0)
    )


def _direction_label(edge: float) -> str:
    if np.isnan(edge):
        return "NA"
    if edge > 0.0:
        return "OVER"
    if edge < 0.0:
        return "UNDER"
    return "PUSH"


def _directional_correct(pred_edge: float, actual_edge: float) -> float:
    if np.isnan(pred_edge) or np.isnan(actual_edge):
        return np.nan
    if pred_edge == 0.0 or actual_edge == 0.0:
        return np.nan
    return float(((pred_edge > 0.0) and (actual_edge > 0.0)) or ((pred_edge < 0.0) and (actual_edge < 0.0)))


def backtest_history(
    predictor,
    history_df: pd.DataFrame,
    csv_path: str | Path | None = None,
    player_name: str | None = None,
    min_history_rows: int | None = None,
    max_predictions: int | None = None,
) -> tuple[list[dict], list[dict]]:
    df = history_df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.loc[df["Date"].notna()].sort_values("Date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if player_name is None:
        player_name = str(df["Player"].iloc[0]) if "Player" in df.columns and not df.empty else "Unknown_Player"
    if min_history_rows is None:
        min_history_rows = int(getattr(predictor, "seq_len", 10))

    rows: list[dict] = []
    failures: list[dict] = []
    start_idx = max(int(min_history_rows), 1)
    stop_idx = len(df)
    if max_predictions is not None:
        start_idx = max(start_idx, stop_idx - int(max_predictions))

    for idx in range(start_idx, stop_idx):
        prior_games = df.iloc[:idx].copy()
        actual_row = df.iloc[idx]
        try:
            explanation = predictor.predict(prior_games, assume_prepared=True)
        except Exception as exc:
            failures.append(
                {
                    "player": player_name,
                    "row_index": int(idx),
                    "date": str(pd.to_datetime(actual_row["Date"]).date()) if "Date" in actual_row.index and pd.notna(actual_row["Date"]) else None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        record = {
            "player": player_name,
            "date": str(pd.to_datetime(actual_row["Date"]).date()) if "Date" in actual_row.index and pd.notna(actual_row["Date"]) else None,
            "csv": str(csv_path) if csv_path is not None else None,
            "history_rows": int(len(prior_games)),
            "did_not_play": safe_float(actual_row.get("Did_Not_Play"), default=0.0),
            "minutes": safe_float(actual_row.get("MP"), default=np.nan),
            "fallback_blend": safe_float(explanation.get("data_quality", {}).get("fallback_blend"), default=0.0),
            "fallback_reasons": ",".join(explanation.get("data_quality", {}).get("fallback_reasons", [])),
            "belief_uncertainty": safe_float(explanation.get("latent_environment", {}).get("belief_uncertainty"), default=np.nan),
            "feasibility": safe_float(explanation.get("latent_environment", {}).get("feasibility"), default=np.nan),
        }
        for target in TARGETS:
            actual_value = safe_float(actual_row.get(target))
            pred_value = safe_float(explanation.get("predicted", {}).get(target))
            baseline_value = safe_float(explanation.get("baseline", {}).get(target))
            market_value = safe_float(actual_row.get(f"Market_{target}"))
            books_value = safe_float(actual_row.get(f"Market_{target}_books"))
            synthetic_market = safe_float(actual_row.get(f"Synthetic_Market_{target}"))

            pred_edge = pred_value - market_value if not np.isnan(market_value) else np.nan
            baseline_edge = baseline_value - market_value if not np.isnan(market_value) else np.nan
            actual_edge = actual_value - market_value if not np.isnan(market_value) else np.nan

            record[f"actual_{target}"] = actual_value
            record[f"pred_{target}"] = pred_value
            record[f"baseline_{target}"] = baseline_value
            record[f"market_{target}"] = market_value
            record[f"synthetic_market_{target}"] = synthetic_market
            record[f"market_books_{target}"] = books_value
            record[f"abs_error_{target}"] = abs(pred_value - actual_value) if not np.isnan(pred_value) and not np.isnan(actual_value) else np.nan
            record[f"baseline_abs_error_{target}"] = abs(baseline_value - actual_value) if not np.isnan(baseline_value) and not np.isnan(actual_value) else np.nan
            record[f"market_abs_error_{target}"] = abs(market_value - actual_value) if not np.isnan(market_value) and not np.isnan(actual_value) else np.nan
            record[f"pred_minus_market_{target}"] = pred_edge
            record[f"baseline_minus_market_{target}"] = baseline_edge
            record[f"actual_minus_market_{target}"] = actual_edge
            record[f"pick_{target}"] = _direction_label(pred_edge)
            record[f"baseline_pick_{target}"] = _direction_label(baseline_edge)
            record[f"actual_result_{target}"] = _direction_label(actual_edge)
            record[f"directional_correct_{target}"] = _directional_correct(pred_edge, actual_edge)
            record[f"baseline_directional_correct_{target}"] = _directional_correct(baseline_edge, actual_edge)
        rows.append(record)
    return rows, failures


def _target_summary(df: pd.DataFrame, target: str) -> dict:
    actual = pd.to_numeric(df[f"actual_{target}"], errors="coerce")
    pred = pd.to_numeric(df[f"pred_{target}"], errors="coerce")
    baseline = pd.to_numeric(df[f"baseline_{target}"], errors="coerce")
    market = pd.to_numeric(df[f"market_{target}"], errors="coerce")
    pred_edge = pd.to_numeric(df[f"pred_minus_market_{target}"], errors="coerce")
    actual_edge = pd.to_numeric(df[f"actual_minus_market_{target}"], errors="coerce")
    baseline_edge = pd.to_numeric(df[f"baseline_minus_market_{target}"], errors="coerce")
    directional_correct = pd.to_numeric(df[f"directional_correct_{target}"], errors="coerce")
    baseline_directional_correct = pd.to_numeric(df[f"baseline_directional_correct_{target}"], errors="coerce")

    pred_valid = actual.notna() & pred.notna()
    baseline_valid = actual.notna() & baseline.notna()
    market_valid = actual.notna() & market.notna()
    called_mask = market_valid & pred_edge.notna() & actual_edge.notna() & (pred_edge != 0.0) & (actual_edge != 0.0)
    baseline_called_mask = market_valid & baseline_edge.notna() & actual_edge.notna() & (baseline_edge != 0.0) & (actual_edge != 0.0)

    summary = {
        "rows": int(len(df)),
        "actual_rows": int(pred_valid.sum()),
        "market_rows": int(market_valid.sum()),
        "mae": float((pred[pred_valid] - actual[pred_valid]).abs().mean()) if pred_valid.any() else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(pred[pred_valid] - actual[pred_valid])))) if pred_valid.any() else np.nan,
        "baseline_mae": float((baseline[baseline_valid] - actual[baseline_valid]).abs().mean()) if baseline_valid.any() else np.nan,
        "market_mae": float((market[market_valid] - actual[market_valid]).abs().mean()) if market_valid.any() else np.nan,
        "mae_vs_baseline_delta": (
            float((baseline[baseline_valid] - actual[baseline_valid]).abs().mean() - (pred[pred_valid] - actual[pred_valid]).abs().mean())
            if pred_valid.any() and baseline_valid.any()
            else np.nan
        ),
        "mae_vs_market_delta": (
            float((market[market_valid] - actual[market_valid]).abs().mean() - (pred[market_valid] - actual[market_valid]).abs().mean())
            if market_valid.any()
            else np.nan
        ),
        "called_lines": int(called_mask.sum()),
        "baseline_called_lines": int(baseline_called_mask.sum()),
        "market_pushes": int((market_valid & actual_edge.notna() & (actual_edge == 0.0)).sum()),
        "win_rate": float(directional_correct[called_mask].mean()) if called_mask.any() else np.nan,
        "baseline_win_rate": float(baseline_directional_correct[baseline_called_mask].mean()) if baseline_called_mask.any() else np.nan,
        "over_rate": float((pred_edge[market_valid] > 0.0).mean()) if market_valid.any() else np.nan,
        "under_rate": float((pred_edge[market_valid] < 0.0).mean()) if market_valid.any() else np.nan,
        "avg_abs_edge": float(pred_edge[market_valid].abs().mean()) if market_valid.any() else np.nan,
        "avg_market_books": float(pd.to_numeric(df[f"market_books_{target}"], errors="coerce").mean()) if f"market_books_{target}" in df.columns else np.nan,
    }
    return summary


def summarize_validation_records(rows_df: pd.DataFrame, failures: list[dict] | None = None) -> dict:
    failures = failures or []
    if rows_df.empty:
        return {
            "rows": 0,
            "strict_rows": 0,
            "players": 0,
            "failures": {"count": int(len(failures)), "sample": failures[:10]},
            "targets": {target: {} for target in TARGETS},
        }

    strict_df = rows_df.loc[active_only_mask(rows_df)].copy()
    targets_summary = {target: _target_summary(strict_df, target) for target in TARGETS}

    mae_values = [targets_summary[target].get("mae") for target in TARGETS if not np.isnan(targets_summary[target].get("mae", np.nan))]
    market_wr_values = [targets_summary[target].get("win_rate") for target in TARGETS if not np.isnan(targets_summary[target].get("win_rate", np.nan))]

    return {
        "rows": int(len(rows_df)),
        "strict_rows": int(len(strict_df)),
        "players": int(rows_df["player"].nunique()) if "player" in rows_df.columns else 0,
        "date_min": str(pd.to_datetime(rows_df["date"], errors="coerce").min().date()) if "date" in rows_df.columns else None,
        "date_max": str(pd.to_datetime(rows_df["date"], errors="coerce").max().date()) if "date" in rows_df.columns else None,
        "overall_avg_mae": float(np.mean(mae_values)) if mae_values else np.nan,
        "overall_avg_win_rate": float(np.mean(market_wr_values)) if market_wr_values else np.nan,
        "targets": targets_summary,
        "failures": {
            "count": int(len(failures)),
            "sample": failures[:10],
        },
    }


def print_validation_summary(summary: dict) -> None:
    print("\n" + "=" * 88)
    print("INFERENCE HISTORICAL VALIDATION")
    print("=" * 88)
    print(f"Rows scored:        {summary.get('rows', 0)}")
    print(f"Strict active rows: {summary.get('strict_rows', 0)}")
    print(f"Players:            {summary.get('players', 0)}")
    if summary.get("date_min") and summary.get("date_max"):
        print(f"Date range:         {summary['date_min']} -> {summary['date_max']}")
    if not np.isnan(safe_float(summary.get("overall_avg_mae"))):
        print(f"Overall avg MAE:    {safe_float(summary.get('overall_avg_mae')):.4f}")
    if not np.isnan(safe_float(summary.get("overall_avg_win_rate"))):
        print(f"Overall win rate:   {safe_float(summary.get('overall_avg_win_rate')):.4f}")
    print(f"Failures:           {summary.get('failures', {}).get('count', 0)}")

    for target in TARGETS:
        stats = summary.get("targets", {}).get(target, {})
        if not stats:
            continue
        print("\n" + target)
        print(f"  MAE:              {safe_float(stats.get('mae')):.4f}")
        print(f"  Baseline MAE:     {safe_float(stats.get('baseline_mae')):.4f}")
        print(f"  Market MAE:       {safe_float(stats.get('market_mae')):.4f}")
        print(f"  MAE delta vs mkt: {safe_float(stats.get('mae_vs_market_delta')):+.4f}")
        print(f"  Win rate:         {safe_float(stats.get('win_rate')):.4f}")
        print(f"  Baseline win rt:  {safe_float(stats.get('baseline_win_rate')):.4f}")
        print(f"  Called lines:     {int(stats.get('called_lines', 0))}")
        print(f"  Pushes:           {int(stats.get('market_pushes', 0))}")
        print(f"  Avg abs edge:     {safe_float(stats.get('avg_abs_edge')):.4f}")


def save_validation_outputs(rows_df: pd.DataFrame, summary: dict, csv_out: str | Path | None, json_out: str | Path | None) -> None:
    if csv_out is not None:
        csv_path = Path(csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows_df.to_csv(csv_path, index=False)
    if json_out is not None:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
