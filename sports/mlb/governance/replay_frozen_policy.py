#!/usr/bin/env python3
"""Deterministically replay frozen single-wager policies on an immutable MLB slate."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.governance.policy_governance import load_policy_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay frozen MLB policies against a complete-slate snapshot.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _reason(row: pd.Series, policy: dict[str, Any]) -> str:
    scope = policy["scope"]
    decision = policy["decision_rule"]
    if not bool(row.get("eligible_by_input_rules")):
        return "INPUT_INELIGIBLE"
    if str(row.get("market")).upper() not in {str(value).upper() for value in scope["markets"]}:
        return "MARKET_OUT_OF_SCOPE"
    if str(row.get("side")).upper() not in {str(value).upper() for value in scope["sides"]}:
        return "SIDE_OUT_OF_SCOPE"
    if str(row.get("book")).lower() not in {str(value).lower() for value in scope["books"]}:
        return "BOOK_OUT_OF_SCOPE"
    if float(row.get("line")) not in {float(value) for value in scope["lines"]}:
        return "LINE_OUT_OF_SCOPE"
    price = float(row.get("price_decimal"))
    if not float(scope["minimum_decimal_odds"]) <= price <= float(scope["maximum_decimal_odds"]):
        return "ODDS_OUT_OF_SCOPE"
    if pd.isna(row.get("model_score")) or float(row.get("model_score")) < float(decision["minimum_model_score"]):
        return "MODEL_SCORE_BELOW_THRESHOLD"
    return "ELIGIBLE_FOR_RANKING"


def replay_single_policy(universe: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    decision = policy["decision_rule"]
    if not bool(decision.get("family_is_frozen")):
        raise ValueError("Policy family is not frozen; deterministic replay is prohibited.")
    if "minimum_model_score" not in decision:
        raise ValueError("Frozen policy does not declare minimum_model_score.")
    result = universe.copy()
    result["policy_version"] = policy["policy_version"]
    result["policy_digest"] = policy["policy_digest"]
    result["replay_rejection_reason"] = result.apply(lambda row: _reason(row, policy), axis=1)
    result["selected_by_policy"] = False
    eligible = result.loc[result["replay_rejection_reason"] == "ELIGIBLE_FOR_RANKING"].copy()
    eligible = eligible.sort_values(
        ["model_score", "price_decimal", "event_id", "player_id", "market", "book"],
        ascending=[False, False, True, True, True, True],
        kind="stable",
    )
    chosen: list[int] = []
    used_players: set[str] = set()
    used_games: dict[str, int] = {}
    maximum = int(decision["maximum_daily_selections"])
    maximum_per_game = int(policy["exposure_controls"]["maximum_per_game"])
    for index, row in eligible.iterrows():
        player = str(row["player_id"])
        game = str(row["event_id"])
        if player in used_players:
            result.at[index, "replay_rejection_reason"] = "PLAYER_EXPOSURE_LIMIT"
            continue
        if used_games.get(game, 0) >= maximum_per_game:
            result.at[index, "replay_rejection_reason"] = "GAME_EXPOSURE_LIMIT"
            continue
        chosen.append(index)
        used_players.add(player)
        used_games[game] = used_games.get(game, 0) + 1
        if len(chosen) >= maximum:
            break
    if chosen:
        result.loc[chosen, "selected_by_policy"] = True
        result.loc[chosen, "replay_rejection_reason"] = "SELECTED"
    remaining = (result["replay_rejection_reason"] == "ELIGIBLE_FOR_RANKING") & ~result["selected_by_policy"]
    result.loc[remaining, "replay_rejection_reason"] = "DAILY_SELECTION_LIMIT"
    return result


def replay_snapshot(manifest_path: Path, registry_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("capture_label") != "FULL_SLATE_SNAPSHOT" or not bool(manifest.get("immutable")):
        raise ValueError("Policy replay requires an immutable full-slate snapshot.")
    with gzip.open(manifest_path.parent / "candidate_universe.csv.gz", "rt", encoding="utf-8") as handle:
        universe = pd.read_csv(handle)
    registry = load_policy_registry(registry_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for policy in registry["policies"]:
        version = str(policy["policy_version"])
        if policy["policy_kind"] != "SINGLE_WAGER_BOARD":
            reports.append({"policy_version": version, "status": "REPLAY_BLOCKED_SEPARATE_CONSTRUCTOR_REQUIRED"})
            continue
        try:
            replay = replay_single_policy(universe, policy)
        except ValueError as exc:
            reports.append({"policy_version": version, "status": "REPLAY_BLOCKED", "reason": str(exc)})
            continue
        replay_path = output_dir / f"{version}_replay.csv.gz"
        replay.to_csv(replay_path, index=False, compression={"method": "gzip", "mtime": 0})
        reports.append(
            {
                "policy_version": version,
                "policy_digest": policy["policy_digest"],
                "status": "REPLAY_COMPLETE",
                "candidate_rows": int(len(replay)),
                "selected_rows": int(replay["selected_by_policy"].sum()),
                "output": str(replay_path),
            }
        )
    report = {
        "schema_version": "MLB_POLICY_REPLAY_V1",
        "slate_id": manifest["slate_id"],
        "snapshot_id": manifest["snapshot_id"],
        "candidate_universe_sha256": manifest["candidate_universe_sha256"],
        "policies": reports,
    }
    (output_dir / "replay_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = replay_snapshot(args.manifest.resolve(), args.registry.resolve(), args.output_dir.resolve())
    for policy in report["policies"]:
        print(f"{policy['policy_version']}: {policy['status']}")


if __name__ == "__main__":
    main()
