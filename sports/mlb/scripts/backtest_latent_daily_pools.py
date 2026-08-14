#!/usr/bin/env python3
"""Replay the frozen MLB latent policy on complete post-training slate snapshots."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    from .latent_parlay_model import (
        DEFAULT_ARTIFACT_PATH,
        LatentParlayBundle,
        candidate_features,
        market_residual_probability,
    )
except ImportError:
    from latent_parlay_model import (
        DEFAULT_ARTIFACT_PATH,
        LatentParlayBundle,
        candidate_features,
        market_residual_probability,
    )


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAILY_RUNS_ROOT = SCRIPT_ROOT / "data" / "predictions" / "daily_runs"
DEFAULT_PROCESSED_ROOT = SCRIPT_ROOT.parents[1] / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_OUTPUT = SCRIPT_ROOT / "data" / "predictions" / "backtests" / "latent_daily_pool_replay_2026.json"
MAX_CANDIDATES_PER_BOOK = 12
MIN_DECIMAL_PRICE = 1.40
MAX_DECIMAL_PRICE = 2.25
MIN_TICKET_DECIMAL_PRICE = 2.0
MAX_TICKET_DECIMAL_PRICE = {2: 6.0, 3: 10.0, 4: 18.0}
STRATEGIES = ("latent_independent", "latent_joint", "market", "hybrid")


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def game_key(value: Any) -> str:
    text = str(value or "").strip()
    return text[:-2] if text.endswith(".0") else text


def implied_probability(american_price: Any) -> float | None:
    try:
        price = float(american_price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or abs(price) < 100.0:
        return None
    return (-price / (-price + 100.0)) if price < 0.0 else (100.0 / (price + 100.0))


def decimal_price(american_price: Any) -> float | None:
    probability = implied_probability(american_price)
    return (1.0 / probability) if probability else None


def wilson_interval(wins: int, rows: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if rows <= 0:
        return None, None
    probability = wins / rows
    denominator = 1.0 + z * z / rows
    center = (probability + z * z / (2.0 * rows)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / rows + z * z / (4.0 * rows * rows)
    ) / denominator
    return center - margin, center + margin


def latest_complete_snapshots(daily_runs_root: Path, *, start_date: date) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for run_dir in sorted(daily_runs_root.glob("20*")):
        try:
            run_date = datetime.strptime(run_dir.name, "%Y%m%d").date()
        except ValueError:
            continue
        if run_date < start_date:
            continue
        manifests = sorted(run_dir.glob("governance/slates/*/*/manifest.json"))
        candidates: list[tuple[datetime, Path, dict[str, Any]]] = []
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                observed = datetime.fromisoformat(str(manifest["observed_at_utc"]).replace("Z", "+00:00"))
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue
            feature_path = manifest_path.parent / "feature_pool.csv.gz"
            universe_path = manifest_path.parent / "candidate_universe.csv.gz"
            if feature_path.exists() and universe_path.exists():
                candidates.append((observed, manifest_path, manifest))
        if not candidates:
            continue
        _, manifest_path, manifest = max(candidates, key=lambda item: item[0])
        snapshots.append(
            {
                "run_date": run_date.isoformat(),
                "snapshot_id": str(manifest.get("snapshot_id", manifest_path.parent.name)),
                "observed_at_utc": str(manifest.get("observed_at_utc", "")),
                "feature_path": manifest_path.parent / "feature_pool.csv.gz",
                "universe_path": manifest_path.parent / "candidate_universe.csv.gz",
            }
        )
    return snapshots


def load_outcomes(processed_root: Path, slate_dates: set[str]) -> dict[tuple[str, str, str], dict[str, float]]:
    outcomes: dict[tuple[str, str, str], dict[str, float]] = {}
    columns = {"Date", "Player", "Player_Type", "Game_ID", "H", "Batting_Order"}
    for path in sorted(processed_root.glob("*/20*_processed_processed.csv")):
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in columns, low_memory=False)
        except Exception:
            continue
        if frame.empty or "H" not in frame or not frame.get("Player_Type", pd.Series()).eq("hitter").any():
            continue
        frame = frame.loc[frame["Player_Type"].eq("hitter")].copy()
        frame["_date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.sort_values(["_date", "Game_ID"], kind="stable").reset_index(drop=True)
        for index, row in frame.iterrows():
            if pd.isna(row["_date"]):
                continue
            row_date = row["_date"].date().isoformat()
            if row_date not in slate_dates or pd.isna(row.get("H")):
                continue
            prior = frame.iloc[index - 1] if index > 0 else None
            outcomes[(row_date, normalize(row.get("Player")), game_key(row.get("Game_ID")))] = {
                "actual": float(row["H"]),
                "last_hits": (
                    float(prior["H"]) if prior is not None and pd.notna(prior.get("H")) else 0.0
                ),
                "batting_order": (
                    float(prior["Batting_Order"])
                    if prior is not None and pd.notna(prior.get("Batting_Order"))
                    else 6.0
                ),
            }
    return outcomes


def no_vig_quotes(universe: pd.DataFrame) -> pd.DataFrame:
    quotes = universe.copy()
    quotes["market"] = quotes["market"].astype(str).str.upper()
    quotes["side"] = quotes["side"].astype(str).str.upper()
    quotes["line"] = pd.to_numeric(quotes["line"], errors="coerce")
    quotes["price"] = pd.to_numeric(quotes["price"], errors="coerce")
    quotes = quotes.loc[quotes["market"].eq("H") & quotes["line"].sub(0.5).abs().le(1e-9)]
    quotes["player_key"] = quotes["player_name"].map(normalize)
    quotes["implied"] = quotes["price"].map(implied_probability)
    keys = ["book", "player_key", "line"]
    pivot = quotes.pivot_table(index=keys, columns="side", values="implied", aggfunc="last")
    if "OVER" not in pivot or "UNDER" not in pivot:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["OVER", "UNDER"]).reset_index()
    pivot["market_probability"] = pivot["OVER"] / (pivot["OVER"] + pivot["UNDER"])
    over = quotes.loc[quotes["side"].eq("OVER"), [*keys, "price"]].drop_duplicates(keys, keep="last")
    output = over.merge(pivot[[*keys, "market_probability"]], on=keys, how="inner")
    output["decimal_price"] = output["price"].map(decimal_price)
    return output.loc[output["decimal_price"].between(MIN_DECIMAL_PRICE, MAX_DECIMAL_PRICE)]


def score_snapshot(
    snapshot: dict[str, Any],
    outcomes: dict[tuple[str, str, str], dict[str, float]],
    bundle: LatentParlayBundle,
) -> list[dict[str, Any]]:
    features = pd.read_csv(snapshot["feature_path"], low_memory=False)
    universe = pd.read_csv(snapshot["universe_path"], low_memory=False)
    features = features.loc[
        features["Target"].astype(str).str.upper().eq("H")
        & features["Player_Type"].astype(str).str.lower().eq("hitter")
        & pd.to_numeric(features["History_Rows"], errors="coerce").ge(35)
        & features["Model_Selected"].astype(str).str.lower().ne("baseline")
    ].copy()
    quotes = no_vig_quotes(universe)
    if features.empty or quotes.empty:
        return []
    run_date = date.fromisoformat(snapshot["run_date"])
    records: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        key = (snapshot["run_date"], normalize(row.get("Player")), game_key(row.get("Game_ID")))
        outcome = outcomes.get(key)
        if outcome is None:
            continue
        last_history = pd.to_datetime(row.get("Last_History_Date"), errors="coerce")
        if pd.isna(last_history) or (run_date - last_history.date()).days > 4:
            continue
        candidate = SimpleNamespace(
            raw=row.to_dict(),
            run_date=run_date,
            prediction=float(row.get("Prediction", 0.0)),
            history_rows=int(row.get("History_Rows", 0)),
            player_id=str(row.get("Player_ID", "")),
            player=str(row.get("Player", "")),
            team=str(row.get("Team", "")),
        )
        numeric, categorical = candidate_features(
            candidate,
            last_hits=outcome["last_hits"],
            batting_order=outcome["batting_order"],
        )
        prediction = bundle.predict_leg(numeric, categorical)
        records.append(
            {
                "run_date": snapshot["run_date"],
                "snapshot_id": snapshot["snapshot_id"],
                "player": str(row.get("Player", "")),
                "player_key": normalize(row.get("Player")),
                "game_id": game_key(row.get("Game_ID")),
                "win": int(outcome["actual"] > 0.5),
                "actual": outcome["actual"],
                "latent_probability": prediction.probability,
                "latent_raw_probability": prediction.raw_probability,
                "ensemble_std": prediction.ensemble_std,
                "support_fraction": prediction.support_fraction,
                "in_support": prediction.in_support,
                "_numeric": numeric,
                "_categorical": categorical,
            }
        )
    if not records:
        return []
    scored = pd.DataFrame(records)
    joined = scored.merge(quotes, on="player_key", how="inner")
    joined["hybrid_probability"] = joined.apply(
        lambda row: market_residual_probability(
            row["latent_probability"], row["market_probability"], row["ensemble_std"]
        ),
        axis=1,
    )
    return joined.to_dict("records")


def strategy_leg_probability(row: dict[str, Any], strategy: str) -> float:
    if strategy == "market":
        return float(row["market_probability"])
    if strategy == "hybrid":
        return float(row["hybrid_probability"])
    return float(row["latent_probability"])


def choose_ticket(
    rows: list[dict[str, Any]],
    *,
    leg_count: int,
    strategy: str,
    bundle: LatentParlayBundle,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for book, book_rows in pd.DataFrame(rows).groupby("book", sort=True):
        candidates = sorted(
            book_rows.to_dict("records"),
            key=lambda row: strategy_leg_probability(row, strategy),
            reverse=True,
        )[:MAX_CANDIDATES_PER_BOOK]
        for legs in itertools.combinations(candidates, leg_count):
            if len({str(leg["game_id"]) for leg in legs}) != leg_count:
                continue
            if len({str(leg["player_key"]) for leg in legs}) != leg_count:
                continue
            combined_price = math.prod(float(leg["decimal_price"]) for leg in legs)
            if not MIN_TICKET_DECIMAL_PRICE <= combined_price <= MAX_TICKET_DECIMAL_PRICE[leg_count]:
                continue
            independent = math.prod(strategy_leg_probability(leg, strategy) for leg in legs)
            if strategy == "latent_joint":
                joint = bundle.predict_ticket(
                    [(leg["_numeric"], leg["_categorical"]) for leg in legs]
                )
                probability = joint.probability
                uncertainty = joint.ensemble_std
            else:
                probability = independent
                uncertainty = max(float(leg["ensemble_std"]) for leg in legs)
            expected_return = probability * combined_price - 1.0
            rank = (probability, expected_return, -uncertainty)
            if best is None or rank > best["_rank"]:
                hit = int(all(int(leg["win"]) == 1 for leg in legs))
                best = {
                    "run_date": str(legs[0]["run_date"]),
                    "snapshot_id": str(legs[0]["snapshot_id"]),
                    "strategy": strategy,
                    "leg_count": leg_count,
                    "book": str(book),
                    "players": [str(leg["player"]) for leg in legs],
                    "games": [str(leg["game_id"]) for leg in legs],
                    "leg_probabilities": [strategy_leg_probability(leg, strategy) for leg in legs],
                    "estimated_probability": probability,
                    "combined_decimal_price": combined_price,
                    "expected_return": expected_return,
                    "ensemble_uncertainty": uncertainty,
                    "leg_wins": int(sum(int(leg["win"]) for leg in legs)),
                    "ticket_hit": hit,
                    "realized_unit_return": combined_price - 1.0 if hit else -1.0,
                    "_rank": rank,
                }
    if best is not None:
        best.pop("_rank", None)
    return best


def aggregate(tickets: Iterable[dict[str, Any]], strategy: str, leg_count: int) -> dict[str, Any]:
    selected = [row for row in tickets if row["strategy"] == strategy and row["leg_count"] == leg_count]
    if not selected:
        return {"slates": 0, "ticket_hits": 0, "ticket_hit_rate": None, "roi": None}
    hits = int(sum(row["ticket_hit"] for row in selected))
    total_legs = sum(row["leg_count"] for row in selected)
    leg_wins = int(sum(row["leg_wins"] for row in selected))
    low, high = wilson_interval(hits, len(selected))
    probabilities = np.asarray([row["estimated_probability"] for row in selected], dtype=float)
    actual = np.asarray([row["ticket_hit"] for row in selected], dtype=int)
    return {
        "slates": len(selected),
        "ticket_hits": hits,
        "ticket_hit_rate": hits / len(selected),
        "ticket_hit_rate_wilson_95": [low, high],
        "leg_wins": leg_wins,
        "legs": total_legs,
        "leg_hit_rate": leg_wins / total_legs,
        "mean_estimated_probability": float(probabilities.mean()),
        "calibration_gap": float(probabilities.mean() - actual.mean()),
        "brier_score": float(brier_score_loss(actual, probabilities)),
        "roi": float(sum(row["realized_unit_return"] for row in selected) / len(selected)),
    }


def build_report(
    daily_runs_root: Path,
    processed_root: Path,
    bundle: LatentParlayBundle,
) -> dict[str, Any]:
    start_date = date.fromisoformat(str(bundle.artifact["trained_before_date"]))
    snapshots = latest_complete_snapshots(daily_runs_root, start_date=start_date)
    outcomes = load_outcomes(processed_root, {row["run_date"] for row in snapshots})
    scored_by_date: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        scored = score_snapshot(snapshot, outcomes, bundle)
        if scored:
            scored_by_date[snapshot["run_date"]] = scored
    tickets: list[dict[str, Any]] = []
    for run_date, rows in sorted(scored_by_date.items()):
        supported = [row for row in rows if bool(row["in_support"])]
        for strategy in STRATEGIES:
            for leg_count in range(2, 5):
                ticket = choose_ticket(
                    supported,
                    leg_count=leg_count,
                    strategy=strategy,
                    bundle=bundle,
                )
                if ticket is not None:
                    tickets.append(ticket)
    quote_rows = [row for rows in scored_by_date.values() for row in rows]
    if quote_rows:
        candidate_frame = pd.DataFrame(quote_rows).groupby(
            ["run_date", "game_id", "player_key"], as_index=False, sort=True
        ).agg(
            win=("win", "first"),
            latent_probability=("latent_probability", "first"),
            market_probability=("market_probability", "mean"),
            ensemble_std=("ensemble_std", "first"),
            support_fraction=("support_fraction", "first"),
            in_support=("in_support", "first"),
        )
        candidate_frame["hybrid_probability"] = candidate_frame.apply(
            lambda row: market_residual_probability(
                row["latent_probability"], row["market_probability"], row["ensemble_std"]
            ),
            axis=1,
        )
        candidate_rows = candidate_frame.to_dict("records")
    else:
        candidate_rows = []
    candidate_probabilities = np.asarray(
        [row["latent_probability"] for row in candidate_rows], dtype=float
    )
    candidate_actual = np.asarray([row["win"] for row in candidate_rows], dtype=int)
    metrics = {
        strategy: {
            str(leg_count): aggregate(tickets, strategy, leg_count)
            for leg_count in range(2, 5)
        }
        for strategy in STRATEGIES
    }
    signal_metrics: dict[str, dict[str, float | None]] = {}
    for signal in ("latent_probability", "market_probability", "hybrid_probability"):
        probabilities = np.asarray([row[signal] for row in candidate_rows], dtype=float)
        signal_metrics[signal] = {
            "mean_probability": float(probabilities.mean()) if len(probabilities) else None,
            "calibration_gap": (
                float(probabilities.mean() - candidate_actual.mean()) if len(probabilities) else None
            ),
            "brier_score": (
                float(brier_score_loss(candidate_actual, probabilities)) if len(probabilities) else None
            ),
            "roc_auc": (
                float(roc_auc_score(candidate_actual, probabilities))
                if len(np.unique(candidate_actual)) > 1
                else None
            ),
        }
    public_tickets = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in tickets
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": bundle.model_version,
        "artifact_trained_before_date": start_date.isoformat(),
        "evidence_label": "FROZEN_COMPLETE_SLATE_POST_TRAINING_REPLAY",
        "claim_scope": "exact same-book H over 0.5 replay; small-sample research evidence, not certification",
        "snapshot_count": len(snapshots),
        "settled_snapshot_count": len(scored_by_date),
        "settled_dates": sorted(scored_by_date),
        "candidate_rows": len(candidate_rows),
        "executable_quote_rows": len(quote_rows),
        "candidate_support_rate": (
            float(np.mean([row["in_support"] for row in candidate_rows])) if candidate_rows else None
        ),
        "candidate_hit_rate": float(candidate_actual.mean()) if len(candidate_actual) else None,
        "candidate_mean_probability": (
            float(candidate_probabilities.mean()) if len(candidate_probabilities) else None
        ),
        "candidate_brier_score": (
            float(brier_score_loss(candidate_actual, candidate_probabilities))
            if len(candidate_actual)
            else None
        ),
        "candidate_roc_auc": (
            float(roc_auc_score(candidate_actual, candidate_probabilities))
            if len(np.unique(candidate_actual)) > 1
            else None
        ),
        "candidate_signal_metrics": signal_metrics,
        "metrics": metrics,
        "tickets": public_tickets,
        "limitations": [
            "The archive begins after model training and currently contains only a few settled slates.",
            "Lineup state was UNKNOWN_AT_CAPTURE for some candidates; prior batting order is used without hindsight.",
            "Results may authorize continued shadow operation only; they do not establish future expected return.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-runs-root", type=Path, default=DEFAULT_DAILY_RUNS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--artifact-json", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = LatentParlayBundle.load(args.artifact_json.resolve())
    if bundle is None:
        raise SystemExit(f"No compatible latent artifact: {args.artifact_json}")
    report = build_report(args.daily_runs_root.resolve(), args.processed_root.resolve(), bundle)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Settled snapshots: {report['settled_snapshot_count']}")
    print(f"Candidate rows: {report['candidate_rows']}")
    for strategy, by_legs in report["metrics"].items():
        summary = ", ".join(
            f"{legs}-leg={values['ticket_hit_rate']} ({values['ticket_hits']}/{values['slates']})"
            for legs, values in by_legs.items()
        )
        print(f"{strategy}: {summary}")


if __name__ == "__main__":
    main()
