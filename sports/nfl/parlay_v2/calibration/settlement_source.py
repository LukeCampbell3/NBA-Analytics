from __future__ import annotations

"""Real settlement source for NFL calibration/evidence admission -- the
NFL-specific replacement for MLB's
joint_position_builder_v2.multi_target_universe.build_multi_target_universe
(which MLB's ingest.py/pair_ingest.py/settle_evidence.py all reuse as a
shared, already-existing settled-outcomes grader). NFL has no equivalent
"grade this stamp against known outcomes" utility, because its predictive
pipeline never needed one before this system: sports.nfl.predictions.pbp_stats
.load_aggregated_season(season) is real, already-used-elsewhere play-by-play
aggregation (nflverse), but it returns per-player-per-week STATS, not a
graded win/loss against a specific market line -- this module is the
(new, but not fabricated) bridge between the two, shared by every NFL
PARLAY_V2 module that needs a real graded outcome for one play.

A player-week with no real aggregated row yet (bye week, not yet updated,
injury/DNP, or the season/week simply hasn't happened yet) grades as None
-- "not yet settled" -- exactly like MLB's ingest.py silently skips
ungraded rows rather than fabricating an outcome. A push (actual value
exactly equal to the line) is likewise graded None -- push/void handling
is out of scope here for the same reason it is already out of scope for
MLB's identical ingestion paths (see settle_evidence.py's own docstring).
"""

from pathlib import Path
from typing import Any

import pandas as pd

from sports.nfl.predictions.market_selector import TARGET_SCALES
from sports.nfl.predictions.pbp_stats import load_aggregated_season

SETTLEMENT_SOURCE_VERSION = "NFL_PBP_SETTLEMENT_SOURCE_V1"


def load_season_actuals(season: int, *, cache_path: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Real per-player-week stats for `season`, via the same nflverse
    play-by-play aggregation the rest of this pipeline already uses
    (pbp_stats.load_aggregated_season). Cached to `cache_path` when given,
    exactly like every other NFL script that calls this function."""
    return load_aggregated_season(season, cache_path=cache_path, refresh=refresh)


def _stat_column_for_target(target: str | None) -> str | None:
    if not target:
        return None
    key = str(target).strip().lower()
    if key not in TARGET_SCALES:
        return None
    return f"{key}_yards"


def grade_play(play: dict[str, Any], actuals: pd.DataFrame, *, season: int, week: int) -> bool | None:
    """Returns True (leg won), False (leg lost), or None (not yet
    settled/ungraded -- push, missing target-stat mapping, or no real
    aggregated row for this player-week). Callers MUST treat None as "skip
    this row for now", never as a loss."""
    column = _stat_column_for_target(play.get("target") or play.get("market"))
    if column is None or column not in actuals.columns:
        return None
    player_id = play.get("player_id")
    line = play.get("line")
    direction = play.get("direction")
    if player_id is None or line is None or not direction:
        return None
    match = actuals[
        (actuals["player_id"].astype(str) == str(player_id))
        & (actuals["season"].astype(int) == int(season))
        & (actuals["week"].astype(int) == int(week))
    ]
    if match.empty:
        return None
    actual_value = float(match.iloc[0][column])
    line_value = float(line)
    if actual_value == line_value:
        return None  # push -- out of scope, see module docstring
    return actual_value > line_value if str(direction).upper() == "OVER" else actual_value < line_value
