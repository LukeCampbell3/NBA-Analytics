from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def build_team_run_environment(
    pool: pd.DataFrame,
    *,
    same_game_json: Path | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return `(game_id, team)` -> expected-runs context for PA opportunity.

    Preferred source is the repo's existing pitching-enriched game simulation.
    When that current-run artifact cannot expose per-side means, the fallback is
    the sum of the pool's own leakage-safe player R projections for the team.
    The fallback affects PA opportunity only; it never inflates per-PA contact
    quality.
    """
    result: dict[tuple[str, str], dict[str, Any]] = {}
    if same_game_json is not None and same_game_json.exists():
        try:
            payload = json.loads(same_game_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            for game in payload.get("games") or []:
                if not isinstance(game, dict):
                    continue
                game_id = str(game.get("game_id") or "")
                home = str(game.get("home_team") or "").upper()
                away = str(game.get("away_team") or "").upper()
                home_expected = _finite(game.get("home_expected_runs"))
                away_expected = _finite(game.get("away_expected_runs"))
                if game_id and home and home_expected is not None:
                    result[(game_id, home)] = {
                        "expected_runs": home_expected,
                        "source": "PITCHING_ENRICHED_GAME_SIMULATION",
                    }
                if game_id and away and away_expected is not None:
                    result[(game_id, away)] = {
                        "expected_runs": away_expected,
                        "source": "PITCHING_ENRICHED_GAME_SIMULATION",
                    }

    if pool.empty:
        return result
    target = pool.get("Target", pd.Series("", index=pool.index)).astype(str).str.upper()
    player_type = pool.get("Player_Type", pd.Series("", index=pool.index)).astype(str).str.lower()
    runs = pool.loc[target.eq("R") & player_type.eq("hitter")].copy()
    if runs.empty:
        return result
    runs["Prediction_num"] = pd.to_numeric(runs.get("Prediction"), errors="coerce")
    runs = runs.loc[runs["Prediction_num"].notna()].copy()
    if runs.empty:
        return result

    # One row per player/game/target; sum individual expected runs. Requiring at
    # least seven hitters prevents a partial-market snapshot from masquerading
    # as a complete team expectation.
    runs["Game_ID_key"] = runs.get("Game_ID", "").astype(str)
    runs["Team_key"] = runs.get("Team", "").astype(str).str.upper()
    runs["Player_key"] = runs.get("Player", "").astype(str).str.lower()
    runs = runs.drop_duplicates(subset=["Game_ID_key", "Team_key", "Player_key"], keep="last")
    for (game_id, team), group in runs.groupby(["Game_ID_key", "Team_key"]):
        key = (str(game_id), str(team))
        if key in result or len(group) < 7:
            continue
        expected = float(group["Prediction_num"].sum())
        if not math.isfinite(expected):
            continue
        result[key] = {
            "expected_runs": max(1.5, min(8.5, expected)),
            "source": "SUM_PLAYER_RUN_PROJECTIONS_FALLBACK",
            "hitter_count": int(len(group)),
        }
    return result
