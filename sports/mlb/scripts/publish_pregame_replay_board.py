#!/usr/bin/env python3
"""Publish a clearly labelled point-in-time MLB singles replay.

The replay is allowed only when the immutable candidate snapshot predates the
first event in the retained odds snapshot. It never promotes shadow candidates
or reconstructs team markets whose decision-time quotes were not retained.
"""
from __future__ import annotations

import argparse
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
    replay["plays"] = []
    replay["policy_profile"] = "PREGAME_REPLAY__premium_confidence_value_frontier_v19_shadow"
    replay["publication_status"] = "review"
    replay["publication_state"] = "PREGAME_REPLAY_SHADOW_ONLY"
    replay["publication_message"] = (
        "Pregame replay: V19 certified no singles; V4 shadow displays the two candidates selected "
        "from the immutable pre-first-pitch pool. No stake or execution authority."
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
    args = parser.parse_args()
    root = REPO_ROOT / "sports/mlb/data/predictions/balanced_ranking_v3_prospective" / args.run_date
    board_path = REPO_ROOT / "sports/mlb/web/data/daily_predictions.json"
    replay = build_replay(
        json.loads(board_path.read_text()),
        json.loads((root / "snapshot.json").read_text()),
        json.loads((root / "v4_optimized_singles_shadow_report.json").read_text()),
        cutoff_utc=args.first_start_utc,
    )
    encoded = json.dumps(replay, indent=2, sort_keys=True) + "\n"
    for target in (
        board_path,
        REPO_ROOT / "dist/mlb/data/daily_predictions.json",
        REPO_ROOT / "paywall/private-content/app/mlb/data/daily_predictions.json",
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded)
    print(json.dumps({"status": "ok", "candidate_count": replay["pregame_replay"]["candidate_count"], "v4_plays": len(replay["v4_singles_shadow"]["plays"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
