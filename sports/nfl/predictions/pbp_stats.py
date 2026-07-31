"""Aggregate nflverse play-by-play into the weekly player-stat model contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.parquet"

KEYS = ["player_id", "season", "week", "recent_team", "opponent_team"]
OUTPUT_STATS = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "attempts",
    "completions",
    "carries",
    "targets",
    "receptions",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "interceptions",
    "passing_epa",
    "rushing_epa",
    "receiving_epa",
    "target_share",
    "air_yards_share",
    "wopr",
]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _role_table(
    pbp: pd.DataFrame,
    *,
    role: str,
    aggregations: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    player_column = f"{role}_player_id"
    name_column = f"{role}_player_name"
    frame = pbp.loc[pbp[player_column].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=KEYS)
    frame = frame.rename(
        columns={
            player_column: "player_id",
            name_column: "pbp_player_name",
            "posteam": "recent_team",
            "defteam": "opponent_team",
        }
    )
    for source, _ in aggregations.values():
        frame[source] = _numeric(frame, source)
    named = {output: pd.NamedAgg(column=source, aggfunc=operation) for output, (source, operation) in aggregations.items()}
    grouped = frame.groupby(KEYS, dropna=False).agg(
        pbp_player_name=pd.NamedAgg(column="pbp_player_name", aggfunc="last"),
        **named,
    )
    return grouped.reset_index()


def aggregate_player_stats_from_pbp(pbp: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Build regular-season player-week rows using only public play fields."""

    required = {
        "season",
        "week",
        "season_type",
        "posteam",
        "defteam",
        "passer_player_id",
        "rusher_player_id",
        "receiver_player_id",
    }
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"Play-by-play source is missing: {', '.join(missing)}")
    plays = pbp.loc[pbp["season_type"].eq("REG")].copy()
    plays["official_pass_attempt"] = (
        _numeric(plays, "complete_pass")
        + _numeric(plays, "incomplete_pass")
        + _numeric(plays, "interception")
    ).clip(upper=1.0)

    passing = _role_table(
        plays,
        role="passer",
        aggregations={
            "attempts": ("official_pass_attempt", "sum"),
            "completions": ("complete_pass", "sum"),
            "passing_yards": ("passing_yards", "sum"),
            "passing_tds": ("pass_touchdown", "sum"),
            "interceptions": ("interception", "sum"),
            "passing_epa": ("epa", "sum"),
        },
    )
    rushing = _role_table(
        plays,
        role="rusher",
        aggregations={
            "carries": ("rush_attempt", "sum"),
            "rushing_yards": ("rushing_yards", "sum"),
            "rushing_tds": ("rush_touchdown", "sum"),
            "rushing_epa": ("epa", "sum"),
        },
    )
    receiving = _role_table(
        plays,
        role="receiver",
        aggregations={
            "targets": ("official_pass_attempt", "sum"),
            "receptions": ("complete_pass", "sum"),
            "receiving_yards": ("receiving_yards", "sum"),
            "receiving_tds": ("pass_touchdown", "sum"),
            "receiving_epa": ("epa", "sum"),
            "player_air_yards": ("air_yards", "sum"),
        },
    )

    identity = pd.concat(
        [table[KEYS + ["pbp_player_name"]] for table in (passing, rushing, receiving) if not table.empty],
        ignore_index=True,
    ).drop_duplicates(KEYS, keep="last")
    output = identity.copy()
    for table in (passing, rushing, receiving):
        if table.empty:
            continue
        value_columns = [column for column in table.columns if column not in KEYS + ["pbp_player_name"]]
        output = output.merge(table[KEYS + value_columns], on=KEYS, how="left")

    for column in OUTPUT_STATS + ["player_air_yards"]:
        if column not in output.columns:
            output[column] = 0.0
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    team_keys = ["season", "week", "recent_team"]
    team_targets = output.groupby(team_keys)["targets"].transform("sum")
    team_air_yards = output.groupby(team_keys)["player_air_yards"].transform("sum")
    output["target_share"] = np.divide(
        output["targets"], team_targets, out=np.zeros(len(output)), where=team_targets.ne(0)
    )
    output["air_yards_share"] = np.divide(
        output["player_air_yards"],
        team_air_yards,
        out=np.zeros(len(output)),
        where=team_air_yards.ne(0),
    )
    output["wopr"] = 1.5 * output["target_share"] + 0.7 * output["air_yards_share"]

    roster_frame = roster.copy()
    roster_frame = roster_frame.rename(
        columns={"gsis_id": "player_id", "team": "recent_team", "full_name": "player_display_name"}
    )
    roster_columns = ["player_id", "season", "week", "recent_team", "player_display_name", "position"]
    roster_frame = roster_frame[[column for column in roster_columns if column in roster_frame.columns]].drop_duplicates(
        ["player_id", "season", "week", "recent_team"], keep="last"
    )
    output = output.merge(
        roster_frame,
        on=["player_id", "season", "week", "recent_team"],
        how="left",
    )
    output["player_display_name"] = output["player_display_name"].fillna(output["pbp_player_name"])
    output["position"] = output["position"].fillna("UNK")
    output["season_type"] = "REG"
    output = output.drop(columns=["pbp_player_name", "player_air_yards"])
    ordered = [
        "player_id",
        "player_display_name",
        "position",
        "recent_team",
        "opponent_team",
        "season",
        "week",
        "season_type",
        *OUTPUT_STATS,
    ]
    return output[ordered].sort_values(["season", "week", "player_id"]).reset_index(drop=True)


def load_aggregated_season(
    season: int,
    *,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    if cache_path is not None and cache_path.is_file():
        return pd.read_parquet(cache_path)
    pbp = pd.read_parquet(PBP_URL.format(season=season))
    roster = pd.read_parquet(ROSTER_URL.format(season=season))
    output = aggregate_player_stats_from_pbp(pbp, roster)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(cache_path, index=False)
    return output
