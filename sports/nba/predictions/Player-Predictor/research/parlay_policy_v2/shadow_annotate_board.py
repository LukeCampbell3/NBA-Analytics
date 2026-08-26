"""Shadow-annotate a real NBA board with parlay_policy_v2 eligibility.

Continuation of the "what would be needed" list in REPORT.md: this repo has
no settled NBA two-leg parlay dataset carrying the full candidate schema the
policy needs, so a real hit-rate backtest isn't possible yet (see
real_data_summary_nba.py). What *is* possible today is building real
candidate-level records from a real, already-published NBA board and running
the gate on them prospectively -- unsettled, but real. Doing this now, and
saving the output, is what turns a future settled-results file into an
actual backtestable dataset: once `won` is known for these plays, these
records already carry every other required field.

Input: a real daily export JSON from sports/nba/web/data/history/*.json
(the `plays` array -- the production board for one date, unmodified,
un-graded). Output: every real cross-game 2-leg pair CONTROL's own gates
(sports/parlay_analysis.py, sport="nba") would consider, with
parlay_policy_v2's eligibility decision attached. No leg here has a
settled result -- `won` is deliberately absent from the output schema.

Field provenance (read before trusting any column):
  - min_leg_probability, joint_probability (naive product): real, from each
    play's `expected_win_rate`.
  - min_leg_sigma, joint_sigma: NOT the model's true predictive sigma (raw
    `uncertainty_sigma` in this export is in raw stat units -- points/
    rebounds/assists -- not probability units, so it is never used as a
    probability-scale penalty here, silently or otherwise). Instead this
    uses `max(0, expected_win_rate - lcb_probability)` per leg, i.e. the
    real, already-computed gap between the point probability and
    production's own lower-confidence-bound estimate, as a probability-scale
    uncertainty proxy. Documented proxy, not the model's real sigma.
  - joint_lcb: naive product of each leg's own real `lcb_probability` --
    conservative, not a real joint-distribution LCB (no joint model exists
    for these pairs).
  - actual_quote_decimal: product of each leg's own real `market_side_price`
    (converted to decimal) -- the same real, standard cross-game-parlay
    pricing convention documented in
    sports/mlb/research/joint_position_builder_v2/REPORT.md, not a
    synthetic substitute for a same-game SGP quote (no real SGP quote
    exists in this export).
  - shared_failure_risk, compatible_state_score, shift_risk,
    lineup_confirmed, role_stable, material_injury_uncertainty,
    all_legs_in_support, joint_model_reliable: NOT sourced from real state
    in this export -- left at permissive pass-through defaults
    (0.0 / 1.0 / 0.0 / True / True / False / True / True) and explicitly
    excluded from every claim this script makes. Those gates are simply not
    exercised here.

Run: python3 sports/nba/predictions/Player-Predictor/research/parlay_policy_v2/shadow_annotate_board.py [board.json]
(defaults to the most recent sports/nba/web/data/history/*.json snapshot that has a non-empty "plays" list)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT))

from sports.parlay_analysis import score_candidate_parlays  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"))
from research.parlay_policy_v2.policy import (  # noqa: E402
    ParlayPolicy,
    american_to_decimal,
    evaluate_candidate,
)

HISTORY_DIR = REPO_ROOT / "sports" / "nba" / "web" / "data" / "history"

PASS_THROUGH_STATE_DEFAULTS = {
    "shared_failure_risk": 0.0,
    "compatible_state_score": 1.0,
    "shift_risk": 0.0,
    "lineup_confirmed": True,
    "role_stable": True,
    "material_injury_uncertainty": False,
    "all_legs_in_support": True,
    "joint_model_reliable": True,
}


def _default_board_path() -> Path:
    for path in sorted(HISTORY_DIR.glob("2026-*.json"), reverse=True):
        data = json.loads(path.read_text())
        if data.get("plays"):
            return path
    raise FileNotFoundError(f"no history snapshot under {HISTORY_DIR} has a non-empty 'plays' list")


def build_shadow_candidates(plays: list[dict[str, Any]], market_date: str) -> pd.DataFrame:
    rows = [dict(p) for p in plays]
    for i, r in enumerate(rows):
        r["play_key"] = r.get("play_key") or f"{market_date}-{i}"
        r.setdefault("probability", r.get("expected_win_rate"))

    parlays = score_candidate_parlays(
        rows, sport="nba", probability_field="expected_win_rate", min_legs_per_parlay=2, max_legs_per_parlay=2
    )

    policy = ParlayPolicy()
    out = []
    for p in parlays:
        idx = [int(x) for x in p["leg_indices"]]
        legs = [rows[i] for i in idx]
        try:
            p1, p2 = float(legs[0]["expected_win_rate"]), float(legs[1]["expected_win_rate"])
            lcb1, lcb2 = float(legs[0]["lcb_probability"]), float(legs[1]["lcb_probability"])
            sigma1 = max(0.0, p1 - lcb1)
            sigma2 = max(0.0, p2 - lcb2)
            d1 = american_to_decimal(float(legs[0]["market_side_price"]))
            d2 = american_to_decimal(float(legs[1]["market_side_price"]))
        except (KeyError, TypeError, ValueError):
            continue  # a leg is missing a required real field -- skip rather than fabricate one

        candidate = {
            "leg_count": 2,
            "min_leg_probability": min(p1, p2),
            "min_leg_sigma": max(sigma1, sigma2),
            "joint_probability": p1 * p2,
            "joint_sigma": 0.0,  # no joint model exists for these pairs -- see docstring
            "joint_lcb": lcb1 * lcb2,
            "dependency_penalty": 0.0,  # cross-game legs; no fitted dependency model for NBA exists yet
            "actual_quote_decimal": d1 * d2,
            **PASS_THROUGH_STATE_DEFAULTS,
        }
        decision = evaluate_candidate(candidate, policy)
        out.append(
            {
                "market_date": market_date,
                "leg_a": f"{legs[0]['player']}|{legs[0]['target']}|{legs[0]['direction']}",
                "leg_b": f"{legs[1]['player']}|{legs[1]['target']}|{legs[1]['direction']}",
                "leg_a_game": legs[0].get("game_key"),
                "leg_b_game": legs[1].get("game_key"),
                **candidate,
                **decision,
            }
        )
    return pd.DataFrame(out)


def main(board_path: Path | None = None) -> dict:
    path = board_path or _default_board_path()
    data = json.loads(path.read_text())
    plays = data.get("plays", [])
    market_date = data.get("run_date") or data.get("through_date") or path.stem

    candidates = build_shadow_candidates(plays, market_date)
    eligible = candidates[candidates["eligible"]] if not candidates.empty else candidates

    return {
        "board_snapshot": str(path.relative_to(REPO_ROOT)),
        "market_date": market_date,
        "settled": False,
        "note": "prospective/unsettled -- no 'won' field exists yet; see docstring for exact field provenance",
        "real_plays_on_board": len(plays),
        "real_candidate_pairs": len(candidates),
        "eligible_pairs": len(eligible),
        "candidates": candidates.to_dict(orient="records"),
    }


if __name__ == "__main__":
    board_arg = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    print(json.dumps(main(board_arg), indent=2, default=str))
