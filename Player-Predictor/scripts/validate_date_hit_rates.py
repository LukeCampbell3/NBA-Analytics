#!/usr/bin/env python3
"""
Validate market-play hit rates on specific dates using baseline vs
volatility-aware selector logic.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decision_engine.policy_tuning import build_default_shadow_strategies
from post_process_market_plays import compute_final_board
from select_market_plays import build_history_lookup, build_play_rows


POLICY_PROFILES = {config.name: config for config in build_default_shadow_strategies()}
DEFAULT_POLICY = POLICY_PROFILES["production_calibrated"]
TARGETS = ["PTS", "TRB", "AST"]
DATA_DIR = REPO_ROOT / "Data-Proc"
MODEL_DIR = REPO_ROOT / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate baseline vs volatility-aware hit rates on specific dates.")
    parser.add_argument("--season", type=int, default=2026, help="Season end year, e.g. 2026.")
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["2026-03-27", "2026-03-27", "2026-03-28"],
        help="Evaluation dates (duplicates allowed to validate multiple top-N boards on one slate).",
    )
    parser.add_argument(
        "--top-n",
        nargs="+",
        type=int,
        default=[9, 10, 10],
        help="Top-N board size for each date entry. If one value is provided, it is reused for every date.",
    )
    parser.add_argument(
        "--baseline-rates",
        nargs="+",
        default=["6/9", "5/10", "5/10"],
        help="Reference rates to beat, one per date entry (format wins/total).",
    )
    parser.add_argument(
        "--policy-profile",
        type=str,
        default="production_calibrated",
        choices=sorted(POLICY_PROFILES.keys()),
        help="Policy profile for final board construction.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "latest_market_comparison_strict_rows.csv",
        help="Optional historical calibration CSV. If missing, heuristic calibration is used.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production.")
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Allow heuristic-only predictions if model artifacts are unavailable.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "date_hit_rate_validation.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "date_hit_rate_validation_rows.csv",
        help="Output CSV row-level picks path.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def resolve_manifest_path(run_id: str | None, latest: bool) -> Path:
    if run_id:
        return MODEL_DIR / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return MODEL_DIR / "latest_structured_lstm_stack.json"
    return MODEL_DIR / "production_structured_lstm_stack.json"


def build_heuristic_explanation(history_df: pd.DataFrame, failure_reason: str | None = None) -> dict:
    active = history_df.copy()
    if "Did_Not_Play" in active.columns:
        active = active.loc[pd.to_numeric(active["Did_Not_Play"], errors="coerce").fillna(0.0) < 0.5].copy()
    if active.empty:
        active = history_df.copy()

    predicted: dict[str, float] = {}
    baseline: dict[str, float] = {}
    target_factors: dict[str, dict] = {}
    sigma_values: list[float] = []

    for target in TARGETS:
        values = pd.to_numeric(active.get(target), errors="coerce").dropna()
        if values.empty:
            values = pd.to_numeric(history_df.get(target), errors="coerce").dropna()

        base_col = f"{target}_rolling_avg"
        baseline_series = pd.to_numeric(history_df.get(base_col), errors="coerce").dropna()
        baseline_value = float(baseline_series.iloc[-1]) if not baseline_series.empty else float(values.mean()) if not values.empty else 0.0

        if values.empty:
            pred_value = max(0.0, baseline_value)
            sigma = 0.0
            spike_prob = 0.10
        else:
            recent = values.tail(12)
            weights = np.linspace(1.0, 2.2, len(recent))
            recency_mean = float(np.average(recent.to_numpy(dtype=float), weights=weights))
            season_mean = float(values.mean())
            trend = float(recent.tail(min(3, len(recent))).mean() - recent.head(min(3, len(recent))).mean())

            pred_value = 0.55 * recency_mean + 0.30 * season_mean + 0.15 * (baseline_value + 0.35 * trend)
            pred_value = float(max(0.0, pred_value))
            sigma = float(np.std(recent.to_numpy(dtype=float), ddof=0)) if len(recent) > 1 else 0.0
            if len(recent) > 1:
                recent_std = float(np.std(recent.to_numpy(dtype=float), ddof=0)) + 1e-6
                z_score = float((recent.iloc[-1] - recent.mean()) / recent_std)
                spike_prob = float(np.clip(0.50 + 0.20 * z_score, 0.05, 0.95))
            else:
                spike_prob = 0.10

        predicted[target] = pred_value
        baseline[target] = float(max(0.0, baseline_value))
        sigma_values.append(sigma)
        target_factors[target] = {
            "uncertainty_sigma": sigma,
            "spike_probability": spike_prob,
        }

    avg_prediction = float(np.mean(list(predicted.values()))) if predicted else 0.0
    avg_sigma = float(np.mean(sigma_values)) if sigma_values else 0.0
    sigma_ratio = avg_sigma / max(1.0, avg_prediction)
    belief_uncertainty = float(np.clip(0.20 + 0.80 * sigma_ratio, 0.05, 0.95))
    mp_series = pd.to_numeric(active.get("MP"), errors="coerce").dropna()
    if mp_series.empty:
        feasibility = 0.70
    else:
        feasibility = float(np.clip(mp_series.tail(10).mean() / 34.0, 0.25, 0.98))

    fallback_reasons = ["heuristic_player_history"]
    if failure_reason:
        fallback_reasons.append(f"model_error:{failure_reason}")

    return {
        "predicted": predicted,
        "baseline": baseline,
        "data_quality": {
            "fallback_blend": 1.0,
            "fallback_reasons": fallback_reasons,
        },
        "latent_environment": {
            "belief_uncertainty": belief_uncertainty,
            "feasibility": feasibility,
            "role_shift_risk": 0.35,
            "volatility_regime_risk": float(np.clip(sigma_ratio, 0.05, 0.95)),
            "context_pressure_risk": 0.30,
        },
        "target_factors": target_factors,
    }


def build_slate_records(
    predictor,
    market_df: pd.DataFrame,
    season: int,
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    skipped: list[dict] = []

    for _, market_row in market_df.iterrows():
        player = normalize_name(str(market_row["Player"]))
        csv_path = DATA_DIR / player / f"{season}_processed_processed.csv"
        if not csv_path.exists():
            skipped.append({"player": player, "reason": f"missing processed csv for season {season}"})
            continue

        history_df = pd.read_csv(csv_path)
        if history_df.empty:
            skipped.append({"player": player, "reason": "empty processed csv"})
            continue
        if "Date" in history_df.columns:
            history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
            history_df = history_df.loc[history_df["Date"].notna()].copy()
            history_df = history_df.loc[history_df["Date"] < pd.to_datetime(market_row["Market_Date"], errors="coerce")].copy()
        if len(history_df) < 10:
            skipped.append({"player": player, "reason": f"insufficient history rows ({len(history_df)})"})
            continue

        explanation = None
        if predictor is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    explanation = predictor.predict(history_df, assume_prepared=True)
            except Exception as exc:
                explanation = build_heuristic_explanation(history_df, failure_reason=f"{type(exc).__name__}")
        if explanation is None:
            explanation = build_heuristic_explanation(history_df)

        latest_row = history_df.iloc[-1]
        record = {
            "player": player,
            "market_date": str(pd.to_datetime(market_row["Market_Date"]).date()),
            "history_rows": int(len(history_df)),
            "last_history_date": str(pd.to_datetime(latest_row["Date"]).date()) if "Date" in latest_row.index and pd.notna(latest_row["Date"]) else None,
            "csv": str(csv_path),
            "belief_uncertainty": float(explanation["latent_environment"].get("belief_uncertainty", 0.0)),
            "feasibility": float(explanation["latent_environment"].get("feasibility", 0.0)),
            "role_shift_risk": float(explanation["latent_environment"].get("role_shift_risk", 0.0)),
            "volatility_regime_risk": float(explanation["latent_environment"].get("volatility_regime_risk", 0.0)),
            "context_pressure_risk": float(explanation["latent_environment"].get("context_pressure_risk", 0.0)),
            "fallback_blend": float(explanation.get("data_quality", {}).get("fallback_blend", 0.0)),
            "fallback_reasons": ",".join(explanation.get("data_quality", {}).get("fallback_reasons", [])),
        }
        for target in TARGETS:
            pred_value = float(explanation["predicted"][target])
            baseline_value = float(explanation["baseline"][target])
            market_value = pd.to_numeric(pd.Series([market_row.get(f"Market_{target}", np.nan)]), errors="coerce").iloc[0]
            market_value = float(market_value) if pd.notna(market_value) else np.nan
            record[f"pred_{target}"] = pred_value
            record[f"baseline_{target}"] = baseline_value
            record[f"market_{target}"] = market_value
            record[f"edge_{target}"] = pred_value - market_value if pd.notna(market_value) else np.nan
            record[f"baseline_edge_{target}"] = baseline_value - market_value if pd.notna(market_value) else np.nan
            record[f"{target}_uncertainty_sigma"] = float(explanation["target_factors"][target].get("uncertainty_sigma", 0.0))
            record[f"{target}_spike_probability"] = float(explanation["target_factors"][target].get("spike_probability", 0.0))
            record[f"market_books_{target}"] = pd.to_numeric(pd.Series([market_row.get(f"Market_{target}_books", np.nan)]), errors="coerce").iloc[0]
        records.append(record)

    return records, skipped


def parse_ratio(token: str) -> tuple[int, int]:
    value = str(token).strip()
    if "/" not in value:
        raise ValueError(f"Invalid ratio '{token}'. Expected format wins/total.")
    wins_text, total_text = value.split("/", 1)
    wins = int(wins_text.strip())
    total = int(total_text.strip())
    if total <= 0:
        raise ValueError(f"Invalid denominator in ratio '{token}'.")
    if wins < 0 or wins > total:
        raise ValueError(f"Invalid numerator in ratio '{token}'.")
    return wins, total


def normalize_date_list(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        parsed = pd.Timestamp(value).date()
        out.append(str(parsed))
    return out


def align_run_lengths(args: argparse.Namespace, dates: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
    if len(args.top_n) == 1:
        top_n = [int(args.top_n[0])] * len(dates)
    elif len(args.top_n) == len(dates):
        top_n = [int(value) for value in args.top_n]
    else:
        raise ValueError("--top-n must be length 1 or match --dates length.")

    parsed_rates = [parse_ratio(token) for token in args.baseline_rates]
    if len(parsed_rates) == 1:
        parsed_rates = parsed_rates * len(dates)
    elif len(parsed_rates) != len(dates):
        raise ValueError("--baseline-rates must be length 1 or match --dates length.")
    return top_n, parsed_rates


def load_predictor_or_fallback(args: argparse.Namespace) -> tuple[object | None, str, str | None]:
    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    try:
        sys.path.insert(0, str(REPO_ROOT / "inference"))
        from structured_stack_inference import StructuredStackInference  # type: ignore

        predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
        run_id = str(predictor.metadata.get("run_id")) if predictor.metadata.get("run_id") is not None else None
        mode = "artifact_free_heuristic_model" if bool(getattr(predictor, "artifact_free", False)) else "structured_model"
        return predictor, mode, run_id
    except Exception as exc:
        if not args.allow_heuristic_fallback:
            raise RuntimeError(
                "Model inference failed and --allow-heuristic-fallback was not provided."
            ) from exc
        print(f"Warning: model inference unavailable; using heuristic fallback only ({type(exc).__name__}: {exc})")
        return None, "heuristic_fallback", None


def shrink_expected_win_rate(raw_rate: pd.Series, shrink_factor: float) -> pd.Series:
    raw = pd.to_numeric(raw_rate, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    shrink = float(np.clip(shrink_factor, 0.0, 1.0))
    return (0.5 + shrink * (raw - 0.5)).clip(lower=0.0, upper=1.0)


def apply_live_policy_calibration(selector_df: pd.DataFrame, policy_payload: dict) -> pd.DataFrame:
    if selector_df.empty:
        return selector_df.copy()
    out = selector_df.copy()
    out["raw_expected_win_rate"] = pd.to_numeric(out.get("raw_expected_win_rate", out["expected_win_rate"]), errors="coerce").fillna(0.5)
    out["expected_win_rate"] = shrink_expected_win_rate(out["expected_win_rate"], policy_payload["probability_shrink_factor"])
    elite_pct = float(policy_payload["elite_pct"])
    out["recommendation"] = np.where(
        pd.to_numeric(out["gap_percentile"], errors="coerce").fillna(0.0) >= elite_pct,
        "elite",
        out["recommendation"],
    )
    return out


def resolve_policy_payload(profile_name: str) -> dict:
    base = POLICY_PROFILES[profile_name].to_dict()
    return {
        "american_odds": int(base["american_odds"]),
        "probability_shrink_factor": float(base["probability_shrink_factor"]),
        "elite_pct": float(base["elite_pct"]),
        "min_ev": float(base["min_ev"]),
        "min_final_confidence": float(base["min_final_confidence"]),
        "min_recommendation": str(base["min_recommendation"]),
        "max_plays_per_player": int(base["max_plays_per_player"]),
        "max_plays_per_target": int(base["max_plays_per_target"]),
        "max_pts_plays": int(base["max_pts_plays"]),
        "max_trb_plays": int(base["max_trb_plays"]),
        "max_ast_plays": int(base["max_ast_plays"]),
        "max_total_plays": int(base["max_total_plays"]),
        "non_pts_min_gap_percentile": float(base["non_pts_min_gap_percentile"]),
        "edge_adjust_k": float(base["edge_adjust_k"]),
    }


def collect_market_snapshots_and_actuals(season: int, target_dates: list[str]) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str, str], dict]]:
    data_root = REPO_ROOT / "Data-Proc"
    unique_dates = set(target_dates)
    market_rows: dict[str, list[dict]] = {date: [] for date in unique_dates}
    outcomes: dict[tuple[str, str, str], dict] = {}

    use_cols = [
        "Date",
        "Player",
        "PTS",
        "TRB",
        "AST",
        "Market_PTS",
        "Market_TRB",
        "Market_AST",
        "Market_PTS_books",
        "Market_TRB_books",
        "Market_AST_books",
        "Market_PTS_over_price",
        "Market_TRB_over_price",
        "Market_AST_over_price",
        "Market_PTS_under_price",
        "Market_TRB_under_price",
        "Market_AST_under_price",
        "Market_PTS_line_std",
        "Market_TRB_line_std",
        "Market_AST_line_std",
        "Market_Fetched_At_UTC",
    ]

    for csv_path in sorted(data_root.glob(f"*/*{season}_processed_processed.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=lambda col: col in use_cols)
        except Exception:
            continue
        if df.empty or "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.loc[df["Date"].isin(unique_dates)].copy()
        if df.empty:
            continue

        for _, row in df.iterrows():
            date_key = str(row["Date"])
            player_name = normalize_name(str(row.get("Player", csv_path.parent.name)))
            market_values = {target: pd.to_numeric(pd.Series([row.get(f"Market_{target}")]), errors="coerce").iloc[0] for target in TARGETS}
            if all(pd.isna(value) for value in market_values.values()):
                continue

            market_rows[date_key].append(
                {
                    "Market_Date": date_key,
                    "Player": player_name,
                    "Market_Player_Raw": player_name.replace("_", " "),
                    "Market_PTS": market_values["PTS"],
                    "Market_TRB": market_values["TRB"],
                    "Market_AST": market_values["AST"],
                    "Market_PTS_books": row.get("Market_PTS_books"),
                    "Market_TRB_books": row.get("Market_TRB_books"),
                    "Market_AST_books": row.get("Market_AST_books"),
                    "Market_PTS_over_price": row.get("Market_PTS_over_price"),
                    "Market_TRB_over_price": row.get("Market_TRB_over_price"),
                    "Market_AST_over_price": row.get("Market_AST_over_price"),
                    "Market_PTS_under_price": row.get("Market_PTS_under_price"),
                    "Market_TRB_under_price": row.get("Market_TRB_under_price"),
                    "Market_AST_under_price": row.get("Market_AST_under_price"),
                    "Market_PTS_line_std": row.get("Market_PTS_line_std"),
                    "Market_TRB_line_std": row.get("Market_TRB_line_std"),
                    "Market_AST_line_std": row.get("Market_AST_line_std"),
                    "Market_Fetched_At_UTC": row.get("Market_Fetched_At_UTC"),
                }
            )

            for target in TARGETS:
                actual_value = pd.to_numeric(pd.Series([row.get(target)]), errors="coerce").iloc[0]
                market_line = market_values[target]
                outcomes[(date_key, player_name, target)] = {
                    "actual": float(actual_value) if pd.notna(actual_value) else np.nan,
                    "market_line": float(market_line) if pd.notna(market_line) else np.nan,
                }

    snapshots: dict[str, pd.DataFrame] = {}
    for date_key, rows in market_rows.items():
        frame = pd.DataFrame.from_records(rows)
        if frame.empty:
            snapshots[date_key] = frame
            continue
        frame = frame.sort_values("Player").drop_duplicates(subset=["Market_Date", "Player"], keep="last").reset_index(drop=True)
        snapshots[date_key] = frame
    return snapshots, outcomes


def build_selector_and_board(
    slate_df: pd.DataFrame,
    history_lookup: dict[str, dict],
    policy_payload: dict,
    volatility_adjustment: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selector = build_play_rows(slate_df, history_lookup, volatility_adjustment=volatility_adjustment)
    selector = apply_live_policy_calibration(selector, policy_payload)
    final_board = compute_final_board(
        selector,
        american_odds=policy_payload["american_odds"],
        min_ev=policy_payload["min_ev"],
        min_final_confidence=policy_payload["min_final_confidence"],
        min_recommendation=policy_payload["min_recommendation"],
        max_plays_per_player=policy_payload["max_plays_per_player"],
        max_plays_per_target=policy_payload["max_plays_per_target"],
        max_total_plays=policy_payload["max_total_plays"],
        max_target_plays={
            "PTS": policy_payload["max_pts_plays"],
            "TRB": policy_payload["max_trb_plays"],
            "AST": policy_payload["max_ast_plays"],
        },
        non_pts_min_gap_percentile=policy_payload["non_pts_min_gap_percentile"],
        edge_adjust_k=policy_payload["edge_adjust_k"],
    )
    return selector, final_board


def classify_result(direction: str, actual: float, market_line: float) -> str:
    if not np.isfinite(actual) or not np.isfinite(market_line):
        return "missing"
    if direction == "OVER":
        if actual > market_line:
            return "win"
        if actual < market_line:
            return "loss"
        return "push"
    if direction == "UNDER":
        if actual < market_line:
            return "win"
        if actual > market_line:
            return "loss"
        return "push"
    return "push"


def evaluate_top_n(
    board: pd.DataFrame,
    date_key: str,
    top_n: int,
    outcomes: dict[tuple[str, str, str], dict],
    mode: str,
) -> tuple[dict, pd.DataFrame]:
    picks = board.head(int(top_n)).copy().reset_index(drop=True)
    if picks.empty:
        return {
            "mode": mode,
            "date": date_key,
            "top_n_requested": int(top_n),
            "top_n_available": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "missing": 0,
            "hit_rate": None,
        }, picks

    resolved_rows: list[dict] = []
    wins = losses = pushes = missing = 0
    for _, row in picks.iterrows():
        player = str(row["player"])
        target = str(row["target"]).upper()
        key = (date_key, player, target)
        outcome = outcomes.get(key, {"actual": np.nan, "market_line": np.nan})
        actual = float(outcome["actual"]) if np.isfinite(outcome["actual"]) else np.nan
        market_line = float(row.get("market_line", np.nan))
        direction = str(row.get("direction", "PUSH")).upper()
        result = classify_result(direction, actual, market_line)
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        elif result == "push":
            pushes += 1
        else:
            missing += 1

        resolved = row.to_dict()
        resolved.update(
            {
                "date": date_key,
                "mode": mode,
                "actual": actual,
                "result": result,
            }
        )
        resolved_rows.append(resolved)

    resolved_df = pd.DataFrame.from_records(resolved_rows)
    total = int(len(resolved_df))
    hit_rate = float(wins / total) if total > 0 else None
    summary = {
        "mode": mode,
        "date": date_key,
        "top_n_requested": int(top_n),
        "top_n_available": int(total),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "missing": int(missing),
        "hit_rate": hit_rate,
    }
    return summary, resolved_df


def main() -> None:
    args = parse_args()
    dates = normalize_date_list(args.dates)
    top_n_list, baseline_rates = align_run_lengths(args, dates)

    predictor, model_mode, run_id = load_predictor_or_fallback(args)
    policy_payload = resolve_policy_payload(args.policy_profile)

    history_lookup: dict[str, dict] = {}
    history_mode = "heuristic_fallback"
    history_csv = args.history_csv.resolve()
    if history_csv.exists():
        history_df = pd.read_csv(history_csv)
        history_lookup = build_history_lookup(history_df)
        history_mode = "historical_backtest"

    snapshots, outcomes = collect_market_snapshots_and_actuals(args.season, dates)

    unique_dates = sorted(set(dates))
    per_date_payload: dict[str, dict] = {}
    for date_key in unique_dates:
        market_df = snapshots.get(date_key, pd.DataFrame())
        if market_df.empty:
            per_date_payload[date_key] = {
                "slate_rows": 0,
                "selector_rows_baseline": 0,
                "selector_rows_volatility": 0,
                "final_rows_baseline": 0,
                "final_rows_volatility": 0,
                "error": "no_market_rows_for_date",
            }
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            slate_records, slate_skipped = build_slate_records(predictor, market_df, args.season)
        slate_df = pd.DataFrame.from_records(slate_records).sort_values(["market_date", "player"]).reset_index(drop=True)
        if slate_df.empty:
            per_date_payload[date_key] = {
                "slate_rows": 0,
                "selector_rows_baseline": 0,
                "selector_rows_volatility": 0,
                "final_rows_baseline": 0,
                "final_rows_volatility": 0,
                "skipped_rows": int(len(slate_skipped)),
                "error": "no_slate_rows_built",
            }
            continue

        baseline_selector, baseline_board = build_selector_and_board(
            slate_df=slate_df,
            history_lookup=history_lookup,
            policy_payload=policy_payload,
            volatility_adjustment=False,
        )
        volatility_selector, volatility_board = build_selector_and_board(
            slate_df=slate_df,
            history_lookup=history_lookup,
            policy_payload=policy_payload,
            volatility_adjustment=True,
        )

        per_date_payload[date_key] = {
            "slate_rows": int(len(slate_df)),
            "skipped_rows": int(len(slate_skipped)),
            "selector_rows_baseline": int(len(baseline_selector)),
            "selector_rows_volatility": int(len(volatility_selector)),
            "final_rows_baseline": int(len(baseline_board)),
            "final_rows_volatility": int(len(volatility_board)),
            "baseline_board": baseline_board,
            "volatility_board": volatility_board,
        }

    run_rows: list[dict] = []
    all_pick_rows: list[pd.DataFrame] = []
    for idx, date_key in enumerate(dates):
        top_n = int(top_n_list[idx])
        target_wins, target_total = baseline_rates[idx]
        target_rate = target_wins / target_total
        payload = per_date_payload.get(date_key, {})
        baseline_board = payload.get("baseline_board", pd.DataFrame())
        volatility_board = payload.get("volatility_board", pd.DataFrame())

        baseline_summary, baseline_rows = evaluate_top_n(baseline_board, date_key, top_n, outcomes, mode="baseline")
        volatility_summary, volatility_rows = evaluate_top_n(volatility_board, date_key, top_n, outcomes, mode="volatility_adjusted")
        all_pick_rows.extend([baseline_rows, volatility_rows])

        baseline_hit = baseline_summary["hit_rate"] if baseline_summary["hit_rate"] is not None else 0.0
        volatility_hit = volatility_summary["hit_rate"] if volatility_summary["hit_rate"] is not None else 0.0

        run_rows.append(
            {
                "run_index": idx + 1,
                "date": date_key,
                "top_n": top_n,
                "target_wins": target_wins,
                "target_total": target_total,
                "target_rate": target_rate,
                "baseline_top_n_available": baseline_summary["top_n_available"],
                "baseline_wins": baseline_summary["wins"],
                "baseline_losses": baseline_summary["losses"],
                "baseline_pushes": baseline_summary["pushes"],
                "baseline_missing": baseline_summary["missing"],
                "baseline_hit_rate": baseline_hit,
                "volatility_top_n_available": volatility_summary["top_n_available"],
                "volatility_wins": volatility_summary["wins"],
                "volatility_losses": volatility_summary["losses"],
                "volatility_pushes": volatility_summary["pushes"],
                "volatility_missing": volatility_summary["missing"],
                "volatility_hit_rate": volatility_hit,
                "beats_target_rate": bool(volatility_hit > target_rate),
                "beats_baseline_mode": bool(volatility_hit > baseline_hit),
                "delta_vs_target": float(volatility_hit - target_rate),
                "delta_vs_baseline_mode": float(volatility_hit - baseline_hit),
            }
        )

    runs_df = pd.DataFrame.from_records(run_rows)
    picks_df = pd.concat([frame for frame in all_pick_rows if not frame.empty], ignore_index=True) if all_pick_rows else pd.DataFrame()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    if not picks_df.empty:
        picks_df.to_csv(args.csv_out, index=False)
    else:
        pd.DataFrame().to_csv(args.csv_out, index=False)

    payload = {
        "season": int(args.season),
        "dates": dates,
        "top_n": top_n_list,
        "baseline_rates": [f"{wins}/{total}" for wins, total in baseline_rates],
        "model_mode": model_mode,
        "run_id": run_id,
        "history_mode": history_mode,
        "history_csv": str(history_csv),
        "policy_profile": args.policy_profile,
        "policy": policy_payload,
        "per_date": {
            date_key: {
                key: value
                for key, value in payload_item.items()
                if key not in {"baseline_board", "volatility_board"}
            }
            for date_key, payload_item in per_date_payload.items()
        },
        "runs": run_rows,
        "beats_target_count": int(runs_df["beats_target_rate"].sum()) if not runs_df.empty else 0,
        "beats_baseline_mode_count": int(runs_df["beats_baseline_mode"].sum()) if not runs_df.empty else 0,
        "rows_csv": str(args.csv_out),
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DATE HIT-RATE VALIDATION")
    print("=" * 100)
    print(f"Model mode:      {model_mode}")
    print(f"History mode:    {history_mode}")
    print(f"Policy profile:  {args.policy_profile}")
    print(f"Runs evaluated:  {len(run_rows)}")
    if not runs_df.empty:
        display_cols = [
            "run_index",
            "date",
            "top_n",
            "target_wins",
            "target_total",
            "baseline_wins",
            "baseline_hit_rate",
            "volatility_wins",
            "volatility_hit_rate",
            "beats_target_rate",
            "beats_baseline_mode",
        ]
        print(runs_df[display_cols].to_string(index=False))
        print(f"\nBeats target count:       {int(runs_df['beats_target_rate'].sum())}/{len(runs_df)}")
        print(f"Beats baseline mode count:{int(runs_df['beats_baseline_mode'].sum())}/{len(runs_df)}")
    print(f"JSON: {args.json_out}")
    print(f"CSV:  {args.csv_out}")


if __name__ == "__main__":
    main()
