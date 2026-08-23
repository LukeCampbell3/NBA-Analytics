"""Build one player's advantage-routing JSON artifact (spec section 22
data contract). Orchestrates every module in this package:

    sources (real box scores + real Basketball-Reference data)
        -> routing (drive/post/recipients)
        -> gravity
        -> stats (shrinkage, pass value)
        -> simulation (usage, saturation, monte carlo)
        -> archetype (research summary)
        -> this module's JSON assembly

CLI:
    python -m sports.nba.analytics.advantage_routing.build.build_player \
        --player "Derik Queen" --season 2025-26
    python -m sports.nba.analytics.advantage_routing.build.build_player \
        --player "Donovan Clingan" --season 2025-26 --mode post
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..gravity.gravity_model import build_gravity_profile
from ..models.schemas import Metric
from ..routing import drive as drive_module
from ..routing import post as post_module
from ..routing.recipients import build_recipient_network
from ..simulation.monte_carlo import MonteCarloInputs, RateObservation, run_monte_carlo
from ..simulation.usage import ScenarioParameters, SimulationBaseline, standard_scenarios
from ..sources import bball_ref
from ..sources.boxscore import load_player_boxscores
from ..sources.collect import GAMES_SAMPLED_PER_PLAYER, PlayerRealDataBundle, collect_player_real_data
from ..stats.pass_value import build_pass_value_model
from .archetype import build_research_summary

REPO_ROOT = Path(__file__).resolve().parents[5]
OUTPUT_ROOT = REPO_ROOT / "sports" / "nba" / "web" / "data" / "advantage-routing"
PACKAGE_VERSION = "ADVANTAGE_ROUTING_V1"


def _season_end_year(season: str) -> str:
    return str(int(season.split("-")[0]) + 1) if "-" in season else season


def _slugify(player_name: str) -> str:
    return player_name.strip().lower().replace(" ", "-").replace(".", "")


def _flatten_gravity_values(gravity_dict: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for mech, metrics in gravity_dict.get("components", {}).items():
        for name, metric in metrics.items():
            value = metric.get("value")
            if value is not None:
                flat[f"{mech}_{name}"] = value
    return flat


def build_player_artifact(player_name: str, season: str = "2025-26", *, games_sampled: int = GAMES_SAMPLED_PER_PLAYER) -> dict[str, Any]:
    season_end_year = _season_end_year(season)
    generated_at = datetime.now(timezone.utc).isoformat()

    boxscore_table = load_player_boxscores(player_name, season)
    real_data = collect_player_real_data(player_name, season, games_sampled=games_sampled)
    league_baseline = bball_ref.fetch_league_shooting_baseline(season_end_year)

    provenance_notes: list[str] = []

    # ---------------- baseline ----------------
    if boxscore_table is not None and boxscore_table.games_played > 0:
        games = boxscore_table.games
        mean_ast = float(games["AST"].mean())
        mean_tov = float(games["TOV"].mean())
        mean_fga = float(games["FGA"].mean())
        mean_fta = float(games["FTA"].mean()) if "FTA" in games.columns else 0.0
        mean_usg = float(games["USG%"].mean())
        mean_mp = float(games["MP"].mean()) if "MP" in games.columns else None
        games_played = boxscore_table.games_played
        decision_touches = mean_fga + mean_ast + mean_tov
        ast_per_touch = (mean_ast / decision_touches) if decision_touches else 0.0
        tov_per_touch = (mean_tov / decision_touches) if decision_touches else 0.0
    else:
        provenance_notes.append(f"No real Player-Predictor box-score file found for {player_name!r}/{season!r}.")
        mean_ast = mean_tov = mean_fga = mean_fta = mean_usg = decision_touches = ast_per_touch = tov_per_touch = None
        mean_mp = None
        games_played = 0

    # Real sampled-assist-implied made-shot rate per decision touch --
    # computed here (before baseline/simulation both need it) from the
    # same real sampled games used for the recipient network.
    sampled_touch_trials = (len(real_data.games_sampled) * decision_touches) if (real_data.games_sampled and decision_touches) else None
    makes_per_touch_baseline = (len(real_data.assists_as_passer) / sampled_touch_trials) if sampled_touch_trials else ast_per_touch

    baseline: dict[str, Any] = {
        "usage_pct": Metric.observed("usage_pct", mean_usg, source="Player-Predictor real box scores (season mean of real per-game USG%)", season=season, sample_size=games_played).as_dict(),
        "minutes_per_game": Metric.observed("minutes_per_game", mean_mp, source="Player-Predictor real box scores", season=season, sample_size=games_played).as_dict(),
        "games_played": Metric.observed("games_played", games_played, source="Player-Predictor real box scores", season=season).as_dict(),
        "decision_touches_per_game": Metric.derived(
            "decision_touches_per_game", decision_touches,
            method="mean(FGA + AST + TOV) per real game -- a scoring-decision-volume proxy, NOT a real touch count (touches are unreachable; see sources/bball_ref.py)",
            season=season, sample_size=games_played,
        ).as_dict(),
        "ast_per_game": Metric.observed("ast_per_game", mean_ast, source="Player-Predictor real box scores", season=season, sample_size=games_played).as_dict(),
        "tov_per_game": Metric.observed("tov_per_game", mean_tov, source="Player-Predictor real box scores", season=season, sample_size=games_played).as_dict(),
        "ast_per_decision_touch": Metric.derived("ast_per_decision_touch", ast_per_touch, method="mean(AST) / decision_touches_per_game", season=season, sample_size=games_played).as_dict(),
        "tov_per_decision_touch": Metric.derived("tov_per_decision_touch", tov_per_touch, method="mean(TOV) / decision_touches_per_game", season=season, sample_size=games_played).as_dict(),
        "makes_per_decision_touch": Metric.derived(
            "makes_per_decision_touch", makes_per_touch_baseline,
            method="real sampled assists / (sampled games * decision_touches_per_game) -- the simulator's baseline receiver-makes-per-touch rate; undercounts true receiver shot generation since missed/non-assisted passes are invisible to this source",
            season=season, sample_size=(len(real_data.games_sampled) if real_data.games_sampled else 0),
        ).as_dict(),
        "advantage_pass_pct": Metric.unavailable("advantage_pass_pct", reason="Requires origin-touch/routing-state classification; unavailable in this data environment -- see routing/states.py.").as_dict(),
        "recipient_shot_pct": Metric.unavailable("recipient_shot_pct", reason="Requires total (not just assisted) recipient FGA after a pass; unavailable -- see routing/recipients.py.").as_dict(),
    }

    # ---------------- recipients ----------------
    recipient_network = build_recipient_network(
        player_name, real_data.assists_as_passer,
        games_sampled=len(real_data.games_sampled), games_available_total=real_data.games_available_total, season=season,
    )
    recipient_dict = recipient_network.as_dict()

    # ---------------- gravity ----------------
    gravity_profile = build_gravity_profile(
        player_name, season, shooting_table=real_data.shooting_table,
        mean_fga_per_game=mean_fga, mean_fta_per_game=mean_fta, games_played=games_played,
    )
    gravity_dict = gravity_profile.as_dict()

    # ---------------- shot outcomes / pass value ----------------
    pass_value_model = build_pass_value_model(league_baseline, season)

    # ---------------- drive / post / interior hub ----------------
    drive_profile = drive_module.build_drive_profile().as_dict()
    strict_post_profile = post_module.build_post_profile("STRICT_POST_HUB").as_dict()
    interior_hub_profile = post_module.build_post_profile("INTERIOR_HUB").as_dict()

    # ---------------- simulation ----------------
    sim_baseline = None
    scenarios_dict: dict[str, Any] = {}
    monte_carlo_dict: dict[str, Any] = {}
    default_params_note = None
    if decision_touches:
        sim_baseline = SimulationBaseline(
            baseline_decision_touches_per_game=decision_touches, baseline_ast_per_game=mean_ast, baseline_tov_per_game=mean_tov,
            baseline_ast_per_touch=ast_per_touch, baseline_tov_per_touch=tov_per_touch,
            baseline_makes_per_touch=makes_per_touch_baseline, current_usage_pct=mean_usg,
        )
        target_usage = round(mean_usg * 1.3, 1) if mean_usg else 20.0
        params = ScenarioParameters(target_usage_pct=target_usage, pass_tendency_change=0.10)
        default_params_note = asdict(params)
        scenarios = standard_scenarios(sim_baseline, target_usage, 0.10)
        scenarios_dict = {name: s.as_dict() for name, s in scenarios.items()}

        trials = int(round(sampled_touch_trials)) if sampled_touch_trials else 1
        mc_inputs = MonteCarloInputs(
            decision_touches=RateObservation(successes=0, trials=0, prior_mean=0.5),
            ast_per_touch=RateObservation(successes=len(real_data.assists_as_passer), trials=max(trials, 1), prior_mean=max(ast_per_touch, 0.01)),
            makes_per_touch=RateObservation(successes=len(real_data.assists_as_passer), trials=max(trials, 1), prior_mean=max(makes_per_touch_baseline, 0.01)),
            tov_per_touch=RateObservation(successes=len(real_data.turnovers), trials=max(trials, 1), prior_mean=max(tov_per_touch, 0.01)),
            baseline_decision_touches_per_game=decision_touches, current_usage_pct=mean_usg,
        )
        monte_carlo_dict = run_monte_carlo(mc_inputs, params, scenario_name="NEUTRAL").as_dict()

    # ---------------- research summary / archetype ----------------
    neutral_delta = 0.0
    if scenarios_dict and mean_ast is not None:
        neutral_assists = scenarios_dict.get("NEUTRAL", {}).get("simulated_assists", {}).get("value")
        if neutral_assists is not None:
            neutral_delta = neutral_assists - mean_ast

    research_summary = build_research_summary(
        player_name=player_name,
        gravity_mechanisms_present=gravity_profile.mechanisms_present,
        gravity_values=_flatten_gravity_values(gravity_dict),
        recipient_network_as_dict=recipient_dict,
        sampled_assists=len(real_data.assists_as_passer),
        baseline_decision_touches_per_game=decision_touches or 0.0,
        baseline_usage_pct=mean_usg or 0.0,
        scenario_neutral_assists_delta=neutral_delta,
    )

    # ---------------- provenance ----------------
    provenance = {
        "package_version": PACKAGE_VERSION,
        "season": season,
        "generated_at_utc": generated_at,
        "box_score_source": str(boxscore_table.source_path) if boxscore_table else None,
        "bball_ref_player_slug": real_data.player_slug,
        "bball_ref_games_sampled": real_data.games_sampled,
        "bball_ref_games_available_total": real_data.games_available_total,
        "bball_ref_sampling_method": "most recent N real games from the player's real season game log, chronological order preserved (see sources/collect.py)",
        "league_shooting_baseline_url": league_baseline.url if league_baseline else None,
        "notes": provenance_notes,
        "stats_nba_com_reachable": False,
        "stats_nba_com_note": "Verified unreachable from this environment (see sources/bball_ref.py module docstring) -- touch/post/drive/passing-network tracking and exact shot coordinates are UNAVAILABLE for this reason, not omitted by oversight.",
    }

    return {
        "player": {"name": player_name, "season": season, "bball_ref_slug": real_data.player_slug},
        "baseline": baseline,
        "drive": drive_profile,
        "post": strict_post_profile,
        "interior_hub": interior_hub_profile,
        "gravity": gravity_dict,
        "recipients": recipient_dict,
        "origin_target_states": [],  # requires real x/y pass geometry -- unavailable, kept as an explicit empty list (see docs)
        "shot_outcomes": pass_value_model.as_dict(),
        "simulation_parameters": {
            "default_scenario_params": default_params_note,
            "scenarios": scenarios_dict,
            "monte_carlo": monte_carlo_dict,
        },
        "research_summary": research_summary.as_dict(),
        "provenance": provenance,
        "updated_at": generated_at,
    }


def write_player_artifact(player_name: str, season: str = "2025-26", *, output_root: Path = OUTPUT_ROOT, games_sampled: int = GAMES_SAMPLED_PER_PLAYER) -> Path:
    artifact = build_player_artifact(player_name, season, games_sampled=games_sampled)
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"{_slugify(player_name)}.json"
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", required=True)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--mode", choices=["drive", "post", "both"], default="both", help="Reserved for future mode-scoped builds; both are always computed in the current pipeline.")
    parser.add_argument("--games-sampled", type=int, default=GAMES_SAMPLED_PER_PLAYER)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    out_path = write_player_artifact(args.player, args.season, output_root=args.output_root, games_sampled=args.games_sampled)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
