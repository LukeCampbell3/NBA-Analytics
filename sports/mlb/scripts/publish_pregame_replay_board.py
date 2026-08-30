#!/usr/bin/env python3
"""Publish a clearly labelled point-in-time MLB singles replay.

The replay is allowed only when the immutable candidate snapshot predates the
first event in the retained odds snapshot. It never promotes shadow candidates
or reconstructs team markets whose decision-time quotes were not retained.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_replay(board: dict, snapshot: dict, report: dict, *, cutoff_utc: str) -> dict:
    if snapshot.get("slate_date") != board.get("run_date") or report.get("slate_date") != board.get("run_date"):
        raise ValueError("replay inputs do not match board run_date")
    observed = str(snapshot.get("observed_at_utc") or "")
    if not observed or observed >= cutoff_utc:
        raise ValueError("candidate snapshot is not strictly pregame")
    replay = dict(board)
    replay["policy_profile"] = "PREGAME_REPLAY__premium_confidence_value_frontier_v19_shadow"
    replay["publication_status"] = "review"
    replay["publication_state"] = "PREGAME_REPLAY_SHADOW_ONLY"
    replay["publication_message"] = (
        "Pregame replay: the updated evidence/value selector searched every retained real-priced "
        "player market and published its qualifying shadow candidates. No stake or execution authority."
    )
    replay["pregame_replay"] = {
        "enabled": True,
        "scope": "PLAYER_PROPS_ONLY",
        "snapshot_observed_at_utc": observed,
        "first_slate_start_utc": cutoff_utc,
        "candidate_count": int(snapshot.get("candidate_count", 0)),
        "snapshot_identity_sha256": snapshot.get("identity_sha256"),
        "team_market_reconstruction": "UNAVAILABLE_NO_RETAINED_PREGAME_QUOTES",
        "outcomes_used": False,
    }
    replay["v4_singles_shadow"] = {
        "version": report.get("version"),
        "status": report.get("status"),
        "candidate_count": report.get("candidate_count", 0),
        "eligible_count": report.get("eligible_count", 0),
        "plays": report.get("frontend_plays", []),
        "dynamic_gate": report.get("dynamic_gate"),
        "publication_authority": False,
        "evidence_status": "PREGAME_REPLAY_SHADOW_ONLY",
        "strictly_prior_settled_slates": report.get("strictly_prior_settled_slates", 0),
    }
    return replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--first-start-utc", required=True)
    parser.add_argument("--selected-csv", type=Path)
    args = parser.parse_args()
    root = REPO_ROOT / "sports/mlb/data/predictions/balanced_ranking_v3_prospective" / args.run_date
    board_path = REPO_ROOT / "sports/mlb/web/data/daily_predictions.json"
    replay = build_replay(
        json.loads(board_path.read_text()),
        json.loads((root / "snapshot.json").read_text()),
        json.loads((root / "v4_optimized_singles_shadow_report.json").read_text()),
        cutoff_utc=args.first_start_utc,
    )
    if args.selected_csv:
        plays = []
        with args.selected_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                probability = float(row["Final_Hit_Probability"])
                price = float(row["Selected_Side_Price"])
                decimal_price = 1.0 + (100.0 / abs(price) if price < 0 else price / 100.0)
                break_even = 1.0 / decimal_price
                conservative_ev = probability * decimal_price - 1.0
                # Market-agnostic dynamic gate: the final displayed probability,
                # not an earlier intermediate estimate, must clear price by 1 pp.
                if probability < break_even + 0.01 or conservative_ev <= 0.0:
                    continue
                rank = len(plays) + 1
                plays.append({
                    "rank": rank,
                    "sport": "mlb",
                    "player": row["Player"],
                    "player_display_name": row["Player"],
                    "team": row.get("Team", ""),
                    "opponent": row.get("Opponent", ""),
                    "game_id": row.get("Game_ID", ""),
                    "target": row["Target"],
                    "direction": row["Direction"],
                    "market_line": float(row["Market_Line"]),
                    "final_hit_probability": probability,
                    "estimated_hit_probability": float(row["Estimated_Hit_Probability"]),
                    "expected_value_per_unit": conservative_ev,
                    "selector_ev_diagnostic": float(row["Expected_Value_Per_Unit"]),
                    "market_break_even_probability": break_even,
                    "probability_edge": probability - break_even,
                    "american_price": price,
                    "selected_side_price": price,
                    "sportsbook": row.get("Selected_Sportsbook", "fanduel") or "fanduel",
                    "authorization_status": "SHADOW_ONLY",
                    "candidate_authorized": False,
                    "execution_status": "HISTORICAL_MARKET_UNAVAILABLE",
                    "execution_reason": "pregame_replay_quote_no_longer_executable",
                    "confidence_tier": row.get("Confidence_Tier", "shadow"),
                    "selection_profile": row.get("Selection_Profile", "evidence_value"),
                    "history_rows": int(float(row.get("History_Rows") or 0)),
                    "price_confirmed": True,
                })
        replay["plays"] = plays
        replay["pregame_replay"]["selected_count"] = len(plays)
        replay["pregame_replay"]["dynamic_gate"] = (
            "final_hit_probability >= exact_price_break_even + 0.01 and conservative_EV > 0"
        )
    encoded = json.dumps(replay, indent=2, sort_keys=True) + "\n"
    for target in (
        board_path,
        REPO_ROOT / "dist/mlb/data/daily_predictions.json",
        REPO_ROOT / "paywall/private-content/app/mlb/data/daily_predictions.json",
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded)
    print(json.dumps({"status": "ok", "candidate_count": replay["pregame_replay"]["candidate_count"], "selected_count": len(replay.get("plays", [])), "v4_plays": len(replay["v4_singles_shadow"]["plays"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
