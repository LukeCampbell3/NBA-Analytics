#!/usr/bin/env python3
"""
Build an upcoming NBA market slate by pairing future market lines with the current
production predictor.

This intentionally does not merge future market rows into historical training data.
Instead it:
- loads a normalized market snapshot (wide format)
- finds each player's processed history
- runs inference on the history only
- writes a slate table with prediction, market, and model-vs-market edge columns
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from structured_stack_inference import StructuredStackInference  # noqa: E402


DATA_DIR = REPO_ROOT / "Data-Proc"
MODEL_DIR = REPO_ROOT / "model"
DEFAULT_MARKET_WIDE = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba" / "latest_player_props_wide.parquet"
DEFAULT_TARGET_PREDICTION_CALIBRATOR = REPO_ROOT / "model" / "analysis" / "calibration" / "short_term_target_prediction_calibrator.json"
TARGETS = ["PTS", "TRB", "AST"]

# Covers feed uses first-initial names (e.g., J_Brunson); these disambiguate
# rare collisions where multiple players share the same initial + last name.
AMBIGUOUS_PLAYER_OVERRIDES = {
    "M_Bridges": "Mikal_Bridges",
}


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


def team_abbr_from_matchup(value: str | None) -> str | None:
    if not value:
        return None
    match = re.match(r"\s*([A-Z]{2,3})\b", str(value))
    return match.group(1) if match else None


def resolve_manifest_path(run_id: str | None, latest: bool) -> Path:
    if run_id:
        return MODEL_DIR / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return MODEL_DIR / "latest_structured_lstm_stack.json"
    return MODEL_DIR / "production_structured_lstm_stack.json"


def load_market_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Market snapshot not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "Player" not in df.columns:
        raise ValueError("Market snapshot must include a Player column")
    if "Market_Date" not in df.columns:
        raise ValueError("Market snapshot must include a Market_Date column")
    df = df.copy()
    df["Player"] = df["Player"].astype(str).map(normalize_name)
    df["Market_Date"] = pd.to_datetime(df["Market_Date"], errors="coerce")
    return df


def load_target_prediction_calibrator(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    target_block = payload.get("targets", payload) if isinstance(payload, dict) else {}
    if not isinstance(target_block, dict):
        return None
    out: dict[str, dict] = {}
    for key, value in target_block.items():
        if str(key).upper().strip() not in TARGETS or not isinstance(value, dict):
            continue
        out[str(key).upper().strip()] = dict(value)
    return out or None


def apply_target_prediction_calibration(
    target: str,
    raw_prediction: float,
    market_line: float,
    calibrator: dict[str, dict] | None,
) -> tuple[float, dict[str, Any]]:
    meta = {
        "applied": False,
        "source": "identity",
        "edge_multiplier": 1.0,
        "edge_bias": 0.0,
        "support_rows": 0,
    }
    if calibrator is None:
        return float(raw_prediction), meta
    payload = calibrator.get(str(target).upper().strip())
    if not payload or not bool(payload.get("enabled", True)):
        meta["source"] = "disabled"
        return float(raw_prediction), meta
    raw_prediction = float(raw_prediction)
    market_line = float(market_line)
    if not np.isfinite(raw_prediction) or not np.isfinite(market_line):
        meta["source"] = "non_finite_input"
        return float(raw_prediction), meta

    edge_multiplier = float(np.clip(float(payload.get("edge_multiplier", 1.0)), 0.0, 1.25))
    edge_bias = float(payload.get("edge_bias", 0.0))
    tuned_prediction = float(market_line + edge_bias + edge_multiplier * (raw_prediction - market_line))
    max_adjustment_abs = float(payload.get("max_adjustment_abs", {"PTS": 3.0, "TRB": 2.0, "AST": 1.5}.get(str(target).upper().strip(), 2.0)))
    if np.isfinite(max_adjustment_abs) and max_adjustment_abs > 0.0:
        tuned_prediction = float(raw_prediction + np.clip(tuned_prediction - raw_prediction, -max_adjustment_abs, max_adjustment_abs))
    meta.update(
        {
            "applied": True,
            "source": str(payload.get("source", "target_prediction_calibrator")),
            "edge_multiplier": edge_multiplier,
            "edge_bias": edge_bias,
            "support_rows": int(payload.get("support_rows", 0)),
        }
    )
    return tuned_prediction, meta


def _player_aliases(player_dir_name: str) -> set[str]:
    normalized = normalize_name(player_dir_name)
    aliases = {normalized}
    parts = [part for part in normalized.split("_") if part]
    if len(parts) >= 2:
        aliases.add(f"{parts[0][0]}_{'_'.join(parts[1:])}")
        aliases.add(f"{parts[0][0]}_{parts[-1]}")
    return aliases


@lru_cache(maxsize=16)
def build_player_csv_index(season: int) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for player_dir in DATA_DIR.iterdir():
        if not player_dir.is_dir():
            continue
        csv_path = player_dir / f"{season}_processed_processed.csv"
        if not csv_path.exists():
            continue
        for alias in _player_aliases(player_dir.name):
            index.setdefault(alias, []).append(csv_path)
    return index


def infer_player_csv(player_name: str, season: int, player_csv_index: dict[str, list[Path]] | None = None) -> Path | None:
    candidate = DATA_DIR / player_name / f"{season}_processed_processed.csv"
    if candidate.exists():
        return candidate

    lookup = player_csv_index if player_csv_index is not None else build_player_csv_index(season)
    key = normalize_name(player_name)
    matches = lookup.get(key, [])
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    override_name = AMBIGUOUS_PLAYER_OVERRIDES.get(player_name) or AMBIGUOUS_PLAYER_OVERRIDES.get(key)
    if override_name:
        override_path = DATA_DIR / override_name / f"{season}_processed_processed.csv"
        if override_path.exists():
            return override_path

    # Skip unresolved ambiguous aliases instead of forcing a potentially wrong
    # mapping that can contaminate predictions and downstream validation.
    return None


def build_records(
    predictor: StructuredStackInference | None,
    market_df: pd.DataFrame,
    season: int,
    target_prediction_calibrator_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    skipped: list[dict] = []
    player_csv_index = build_player_csv_index(season)
    target_prediction_calibrator = load_target_prediction_calibrator(target_prediction_calibrator_path)

    for _, market_row in market_df.iterrows():
        player = str(market_row["Player"])
        csv_path = infer_player_csv(player, season, player_csv_index=player_csv_index)
        if csv_path is None:
            skipped.append({"player": player, "reason": f"missing processed csv for season {season}"})
            continue

        history_df = pd.read_csv(csv_path)
        if history_df.empty:
            skipped.append({"player": player, "reason": "empty processed csv"})
            continue

        if "Date" in history_df.columns:
            history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
            history_df = history_df.loc[history_df["Date"].notna()].copy()
            history_df = history_df.loc[history_df["Date"] < market_row["Market_Date"]].copy()
        min_history_rows = max(5, int(getattr(predictor, "seq_len", 5)))
        if len(history_df) < min_history_rows:
            skipped.append({"player": player, "reason": f"insufficient history rows ({len(history_df)})"})
            continue

        explanation = None
        if predictor is not None:
            try:
                market_context = {
                    "player": player,
                    "market_date": str(pd.to_datetime(market_row["Market_Date"], errors="coerce").date()) if pd.notna(market_row["Market_Date"]) else None,
                    "market_home_team": market_row.get("Market_Home_Team"),
                    "market_away_team": market_row.get("Market_Away_Team"),
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    explanation = predictor.predict(history_df, assume_prepared=True, market_context=market_context)
            except Exception as exc:
                explanation = build_heuristic_explanation(history_df, failure_reason=f"{type(exc).__name__}")
        if explanation is None:
            explanation = build_heuristic_explanation(history_df)

        latest_row = history_df.iloc[-1]
        player_team = team_abbr_from_matchup(latest_row.get("MATCHUP")) if "MATCHUP" in latest_row.index else None
        market_home_team = market_row.get("Market_Home_Team")
        market_away_team = market_row.get("Market_Away_Team")
        market_teams = {str(item) for item in [market_home_team, market_away_team] if pd.notna(item) and str(item)}
        if market_teams and player_team is not None and player_team not in market_teams:
            skipped.append(
                {
                    "player": player,
                    "reason": f"market_team_mismatch:{player_team} not in {sorted(market_teams)}",
                }
            )
            continue
        record = {
            "player": player,
            "market_date": str(market_row["Market_Date"].date()) if pd.notna(market_row["Market_Date"]) else None,
            "market_player_raw": market_row.get("Market_Player_Raw"),
            "market_event_id": market_row.get("Market_Event_ID"),
            "market_commence_time_utc": market_row.get("Market_Commence_Time_UTC"),
            "market_home_team": market_home_team if pd.notna(market_home_team) else None,
            "market_away_team": market_away_team if pd.notna(market_away_team) else None,
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
            raw_pred_value = float(explanation["predicted"][target])
            baseline_value = float(explanation["baseline"][target])
            market_value = market_row.get(f"Market_{target}", np.nan)
            market_value = float(market_value) if pd.notna(market_value) else np.nan
            pred_value = raw_pred_value
            calibration_meta = {
                "applied": False,
                "source": "identity",
                "edge_multiplier": 1.0,
                "edge_bias": 0.0,
                "support_rows": 0,
            }
            if pd.notna(market_value):
                pred_value, calibration_meta = apply_target_prediction_calibration(
                    target,
                    raw_prediction=raw_pred_value,
                    market_line=market_value,
                    calibrator=target_prediction_calibrator,
                )
            record[f"pred_{target}"] = pred_value
            record[f"raw_pred_{target}"] = raw_pred_value
            record[f"baseline_{target}"] = baseline_value
            record[f"market_{target}"] = market_value
            record[f"edge_{target}"] = pred_value - market_value if pd.notna(market_value) else np.nan
            record[f"raw_edge_{target}"] = raw_pred_value - market_value if pd.notna(market_value) else np.nan
            record[f"baseline_edge_{target}"] = baseline_value - market_value if pd.notna(market_value) else np.nan
            record[f"{target}_uncertainty_sigma"] = float(explanation["target_factors"][target].get("uncertainty_sigma", 0.0))
            record[f"{target}_spike_probability"] = float(explanation["target_factors"][target].get("spike_probability", 0.0))
            record[f"market_books_{target}"] = float(market_row.get(f"Market_{target}_books", np.nan)) if pd.notna(market_row.get(f"Market_{target}_books", np.nan)) else np.nan
            record[f"prediction_calibration_applied_{target}"] = bool(calibration_meta["applied"])
            record[f"prediction_calibration_source_{target}"] = str(calibration_meta["source"])
            record[f"prediction_calibration_multiplier_{target}"] = float(calibration_meta["edge_multiplier"])
            record[f"prediction_calibration_bias_{target}"] = float(calibration_meta["edge_bias"])
            record[f"prediction_calibration_support_rows_{target}"] = int(calibration_meta["support_rows"])
        records.append(record)

    return records, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an upcoming market slate with model-vs-market edges.")
    parser.add_argument("--season", type=int, required=True, help="Season end year, e.g. 2026 for 2025-26.")
    parser.add_argument("--market-wide-path", type=Path, default=DEFAULT_MARKET_WIDE, help="Normalized wide market snapshot.")
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production.")
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Allow slate build to continue with heuristic-only predictions when model load fails.",
    )
    parser.add_argument(
        "--target-prediction-calibrator-json",
        type=Path,
        default=DEFAULT_TARGET_PREDICTION_CALIBRATOR,
        help="Optional target-level short-term prediction calibrator JSON.",
    )
    parser.add_argument(
        "--disable-target-prediction-calibration",
        action="store_true",
        help="Disable target-level prediction calibration and keep raw model outputs.",
    )
    parser.add_argument("--csv-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "upcoming_market_slate.csv", help="Output CSV path.")
    parser.add_argument("--json-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "upcoming_market_slate.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    predictor: StructuredStackInference | None = None
    predictor_error = None
    try:
        predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    except Exception as exc:
        predictor_error = f"{type(exc).__name__}: {exc}"
        if not args.allow_heuristic_fallback:
            raise RuntimeError(
                "Model inference failed while heuristic fallback is disabled. "
                "Pass --allow-heuristic-fallback to continue anyway. "
                f"Root cause: {predictor_error}"
            ) from exc
        print(f"Warning: model inference unavailable, using heuristic fallback only ({predictor_error})")
    market_df = load_market_wide(args.market_wide_path)
    calibrator_path = None if args.disable_target_prediction_calibration else args.target_prediction_calibrator_json
    records, skipped = build_records(
        predictor,
        market_df,
        args.season,
        target_prediction_calibrator_path=calibrator_path,
    )

    if not records:
        raise RuntimeError(f"No upcoming slate rows built. Skipped={len(skipped)} sample={skipped[:5]}")

    results_df = pd.DataFrame.from_records(records).sort_values(["market_date", "player"]).reset_index(drop=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.csv_out, index=False)
    payload = {
        "manifest_path": str(manifest_path),
        "run_id": predictor.metadata.get("run_id") if predictor is not None else None,
        "predictor_error": predictor_error,
        "market_snapshot": str(args.market_wide_path),
        "season": args.season,
        "rows": int(len(results_df)),
        "skipped": skipped,
        "target_prediction_calibrator_json": str(calibrator_path) if calibrator_path is not None else None,
        "target_prediction_calibration_enabled": bool(not args.disable_target_prediction_calibration),
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("UPCOMING MARKET SLATE BUILT")
    print("=" * 80)
    print(f"Rows:     {len(results_df)}")
    print(f"Skipped:  {len(skipped)}")
    print(f"CSV:      {args.csv_out}")
    print(f"JSON:     {args.json_out}")
    print("\nSample:")
    sample_cols = [
        "player",
        "market_date",
        "pred_PTS",
        "market_PTS",
        "edge_PTS",
        "pred_TRB",
        "market_TRB",
        "edge_TRB",
        "pred_AST",
        "market_AST",
        "edge_AST",
    ]
    print(results_df[sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
