#!/usr/bin/env python3
"""
Train the v9 hierarchical player-prop pricing artifacts.

This script turns the research modules in Player-Predictor/training into a
repeatable production training run:
  - builds player-game-market rows from historical comparison data
  - fits walk-forward style calibration artifacts by market and direction
  - fits residual chaos diagnostics
  - fits copula dependency artifacts from player stat histories
  - fits a regime model when enough state features are available
  - writes a manifest and validation report beside the model artifacts
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"
sys.path.insert(0, str(TRAINING_DIR))

from nba_v9_adaptive_calibration import AdaptiveCalibrator
from nba_v9_copula_dependency import GaussianCopula
from nba_v9_lineup_impact import LineupImpactModel
from nba_v9_prop_engine import no_vig_probs
from nba_v9_regime_hmm import PlayerRegimeHMM
from nba_v9_residual_chaos import ResidualChaosAnalyzer


DEFAULT_INPUTS = [
    ROOT / "model" / "analysis" / "refreshed_market_comparison_rows.csv",
    ROOT / "model" / "analysis" / "refreshed_market_comparison_strict_rows.csv",
]
DEFAULT_MARKETS = ["PTS", "TRB", "AST"]


def _read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"historical input not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns or "player" not in df.columns:
        raise ValueError("historical input must contain player and date columns")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "player"]).copy()
    return df.sort_values(["date", "player"]).reset_index(drop=True)


def _first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of the default historical comparison files exist")


def _safe_float_series(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def build_prop_rows(history: pd.DataFrame, markets: list[str]) -> pd.DataFrame:
    """Convert the repository's wide comparison rows to player-game-market rows."""
    records: list[dict] = []
    for market in markets:
        required = [f"actual_{market}", f"pred_{market}", f"market_{market}"]
        if any(col not in history.columns for col in required):
            continue

        frame = history[
            ["player", "date", *required]
            + [c for c in [f"actual_result_{market}", f"pick_{market}", "minutes", "belief_uncertainty"] if c in history.columns]
        ].copy()
        frame = frame.rename(
            columns={
                f"actual_{market}": "actual_value",
                f"pred_{market}": "model_mean",
                f"market_{market}": "line",
                f"actual_result_{market}": "actual_result",
                f"pick_{market}": "pick",
            }
        )
        for col in ["actual_value", "model_mean", "line", "minutes", "belief_uncertainty"]:
            if col in frame.columns:
                frame[col] = _safe_float_series(frame[col])

        frame = frame.dropna(subset=["actual_value", "model_mean", "line"])
        frame = frame[frame["line"] > 0].copy()
        if frame.empty:
            continue

        frame["market"] = market
        frame["result_over"] = (frame["actual_value"] > frame["line"]).astype(float)
        frame["push"] = (frame["actual_value"] == frame["line"]).astype(float)
        frame["over_odds"] = -110
        frame["under_odds"] = -110
        frame["snapshot_time"] = frame["date"].dt.strftime("%Y-%m-%dT10:00:00-05:00")
        frame["residual"] = frame["actual_value"] - frame["model_mean"]
        frame["abs_residual"] = frame["residual"].abs()
        frame["player_id"] = frame["player"].astype(str)
        frame["game_id"] = frame["date"].dt.strftime("%Y-%m-%d") + "_" + frame["player_id"]
        records.extend(frame.to_dict("records"))

    rows = pd.DataFrame(records)
    if rows.empty:
        raise ValueError("no trainable player-game-market rows were produced")

    rows = rows.sort_values(["date", "player", "market"]).reset_index(drop=True)
    return rows


def add_distribution_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Add honest baseline distribution columns for calibration training."""
    rows = rows.copy()
    global_sigma = rows.groupby("market")["residual"].std().replace(0, np.nan).fillna(5.0)
    rows["sigma"] = rows["market"].map(global_sigma).astype(float)

    player_market_sigma = (
        rows.groupby(["player", "market"])["residual"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    )
    rows["sigma"] = player_market_sigma.fillna(rows["sigma"]).clip(lower=0.75)

    z = (rows["line"] - rows["model_mean"]) / rows["sigma"]
    rows["p_over_raw"] = 0.5 * (1.0 - np.vectorize(math.erf)(z / math.sqrt(2.0)))
    rows["p_over_raw"] = rows["p_over_raw"].clip(0.01, 0.99)
    rows["market_no_vig_over"], rows["market_no_vig_under"] = zip(
        *rows.apply(lambda r: no_vig_probs(r["over_odds"], r["under_odds"]), axis=1)
    )
    rows["edge_over_raw"] = rows["p_over_raw"] - rows["market_no_vig_over"]
    rows["edge_under_raw"] = (1.0 - rows["p_over_raw"]) - rows["market_no_vig_under"]
    return rows


def fit_calibrators(rows: pd.DataFrame, output_dir: Path, min_samples: int) -> dict:
    calibrator_dir = output_dir / "calibration"
    calibrator_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    global_cal = AdaptiveCalibrator(target="GLOBAL", min_recalibration_samples=min_samples)
    global_cal.fit_initial(rows["p_over_raw"].to_numpy(), rows["result_over"].to_numpy())
    global_cal.save(calibrator_dir / "global_adaptive_calibrator.pkl")
    summary["GLOBAL"] = asdict(global_cal.get_health())

    for market, market_rows in rows.groupby("market"):
        if len(market_rows) < max(20, min_samples):
            continue
        cal = AdaptiveCalibrator(target=market, min_recalibration_samples=min_samples)
        cal.fit_initial(market_rows["p_over_raw"].to_numpy(), market_rows["result_over"].to_numpy())
        cal.save(calibrator_dir / f"{market}_adaptive_calibrator.pkl")
        summary[market] = asdict(cal.get_health())

    (calibrator_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def build_player_stat_history(history: pd.DataFrame) -> pd.DataFrame:
    cols = ["player", "date"]
    available = [c for c in ["minutes", "actual_PTS", "actual_TRB", "actual_AST"] if c in history.columns]
    stat_history = history[cols + available].copy()
    rename = {
        "minutes": "MP",
        "actual_PTS": "PTS",
        "actual_TRB": "TRB",
        "actual_AST": "AST",
    }
    stat_history = stat_history.rename(columns=rename)
    for col in ["MP", "PTS", "TRB", "AST"]:
        if col in stat_history.columns:
            stat_history[col] = _safe_float_series(stat_history[col])
    if "MP" in stat_history.columns and "PTS" in stat_history.columns:
        stat_history["USG%"] = (stat_history["PTS"] / stat_history["MP"].clip(lower=1) / 1.2).clip(0.05, 0.45)
    if "PTS" in stat_history.columns:
        stat_history["FGA"] = (stat_history["PTS"] / 1.25).clip(lower=1)
        stat_history["TS%"] = (stat_history["PTS"] / (2.0 * stat_history["FGA"]).clip(lower=1)).clip(0.35, 0.75)
    stat_history["PLUS_MINUS"] = 0.0
    return stat_history.drop_duplicates(["player", "date"]).sort_values(["player", "date"])


def fit_regime_model(stat_history: pd.DataFrame, output_dir: Path) -> dict:
    regime_dir = output_dir / "regime"
    regime_dir.mkdir(parents=True, exist_ok=True)
    hmm = PlayerRegimeHMM()
    pooled = stat_history.dropna(subset=["MP", "USG%", "TS%", "FGA"])
    hmm.fit(pooled, stat_cols=[c for c in ["PTS", "TRB", "AST"] if c in pooled.columns])
    hmm.save(regime_dir / "regime_hmm.pkl")
    metadata = {
        "is_fitted": hmm.is_fitted,
        "hmm_backend_available": hmm.model is not None,
        "cluster_fallback_available": hmm._fallback_centers is not None,
        "rows": int(len(pooled)),
        "note": "USG%, FGA, TS%, and PLUS_MINUS are proxy state features when only comparison rows are available.",
    }
    (regime_dir / "regime_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def fit_copulas(stat_history: pd.DataFrame, output_dir: Path, max_players: int) -> dict:
    copula_dir = output_dir / "copula"
    copula_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    saved = 0

    global_copula = GaussianCopula(stat_cols=["PTS", "TRB", "AST"])
    global_copula.fit(stat_history)
    global_copula.save(copula_dir / "global_copula.pkl")
    summary["GLOBAL"] = {"rows": int(len(stat_history)), "stats": global_copula.stat_cols}

    counts = stat_history.groupby("player").size().sort_values(ascending=False)
    for player in counts.index[:max_players]:
        player_rows = stat_history[stat_history["player"] == player]
        if len(player_rows) < 20:
            continue
        copula = GaussianCopula(stat_cols=["PTS", "TRB", "AST"])
        copula.fit(player_rows)
        safe_name = str(player).replace("/", "_").replace("\\", "_").replace(" ", "_")
        copula.save(copula_dir / f"{safe_name}_copula.pkl")
        summary[str(player)] = {"rows": int(len(player_rows)), "stats": copula.stat_cols}
        saved += 1

    summary["saved_player_copulas"] = saved
    (copula_dir / "copula_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fit_lineup_baseline(stat_history: pd.DataFrame, output_dir: Path) -> dict:
    lineup_dir = output_dir / "lineup"
    lineup_dir.mkdir(parents=True, exist_ok=True)
    logs = stat_history.rename(columns={"player": "Player", "date": "Date"})
    model = LineupImpactModel(min_games=5)
    model.fit(logs, player_col="Player", date_col="Date")
    model.save(lineup_dir / "lineup_impact.json")
    summary = {
        "players_with_baselines": len(model._player_baselines),
        "players_with_teammate_impacts": len(model._teammate_impacts),
        "note": "No teammate availability column was present, so this artifact contains player baselines and will learn teammate deltas when split data is supplied.",
    }
    (lineup_dir / "lineup_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def compute_validation(rows: pd.DataFrame, output_dir: Path) -> dict:
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    analyzer = ResidualChaosAnalyzer(target="ALL")
    analyzer.add_observations(
        rows["p_over_raw"].to_numpy(),
        rows["result_over"].to_numpy(),
        rows["market_no_vig_over"].to_numpy(),
        timestamps=rows["date"].dt.strftime("%Y-%m-%d").tolist(),
    )
    analyzer.save_report(validation_dir / "residual_chaos")
    metrics = analyzer.compute_chaos_metrics()

    by_market = {}
    for market, market_rows in rows.groupby("market"):
        if len(market_rows) < 30:
            continue
        m = ResidualChaosAnalyzer(target=market)
        m.add_observations(
            market_rows["p_over_raw"].to_numpy(),
            market_rows["result_over"].to_numpy(),
            market_rows["market_no_vig_over"].to_numpy(),
        )
        by_market[market] = asdict(m.compute_chaos_metrics())

    report = {
        "resolved": int(len(rows)),
        "date_range": f"{rows['date'].min().date()}_to_{rows['date'].max().date()}",
        "markets": sorted(rows["market"].unique().tolist()),
        "overall": asdict(metrics),
        "by_market": by_market,
    }
    (validation_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def write_training_rows(rows: pd.DataFrame, output_dir: Path) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(data_dir / "prop_training_rows.csv", index=False)
    ledger_cols = [
        "snapshot_time", "game_id", "player_id", "market", "line", "over_odds",
        "under_odds", "p_over_raw", "market_no_vig_over", "edge_over_raw",
        "actual_value", "result_over", "push",
    ]
    rows[[c for c in ledger_cols if c in rows.columns]].to_csv(
        data_dir / "append_only_training_ledger_seed.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NBA v9 prop pricing artifacts")
    parser.add_argument("--input", type=Path, default=None, help="Historical comparison CSV")
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9")
    parser.add_argument("--min-calibration-samples", type=int, default=50)
    parser.add_argument("--max-player-copulas", type=int, default=75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or _first_existing(DEFAULT_INPUTS)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _read_history(input_path)
    if args.train_start:
        history = history[history["date"] >= pd.Timestamp(args.train_start)]
    if args.train_end:
        history = history[history["date"] <= pd.Timestamp(args.train_end)]
    if history.empty:
        raise ValueError("no rows remain after train-start/train-end filters")

    rows = add_distribution_columns(build_prop_rows(history, args.markets))
    stat_history = build_player_stat_history(history)

    write_training_rows(rows, output_dir)
    calibration_summary = fit_calibrators(rows, output_dir, args.min_calibration_samples)
    regime_summary = fit_regime_model(stat_history, output_dir)
    copula_summary = fit_copulas(stat_history, output_dir, args.max_player_copulas)
    lineup_summary = fit_lineup_baseline(stat_history, output_dir)
    validation_report = compute_validation(rows, output_dir)

    manifest = {
        "model_version": "prop_engine_v9",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_dir),
        "markets": args.markets,
        "rows": int(len(rows)),
        "players": int(rows["player"].nunique()),
        "date_min": str(rows["date"].min().date()),
        "date_max": str(rows["date"].max().date()),
        "artifacts": {
            "calibration": "calibration/",
            "regime": "regime/regime_hmm.pkl",
            "copula": "copula/",
            "lineup": "lineup/lineup_impact.json",
            "validation": "validation/validation_report.json",
        },
        "summaries": {
            "calibration_targets": list(calibration_summary.keys()),
            "regime": regime_summary,
            "copula": {k: v for k, v in copula_summary.items() if k in ["GLOBAL", "saved_player_copulas"]},
            "lineup": lineup_summary,
            "validation": {
                "resolved": validation_report["resolved"],
                "brier": validation_report["overall"]["brier_score"],
                "ece": validation_report["overall"]["ece"],
                "chaos_level": validation_report["overall"]["chaos_level"],
            },
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(json.dumps(manifest["summaries"], indent=2, default=str))
    print(f"\nWrote v9 prop artifacts to {output_dir}")


if __name__ == "__main__":
    main()
