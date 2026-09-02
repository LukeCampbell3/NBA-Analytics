"""Join current NFL markets to lagged player features and frozen artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .latent_pipeline import predict_week_components_latent
from .market_backtest import normalize_player_name
from .market_selector import TARGET_SCALES
from .pipeline import HISTORY_COLUMNS, TARGET_SPECS, build_features


TEAM_ALIASES = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
TEAM_ALIASES.update({team: team for team in set(TEAM_ALIASES.values())})
TEAM_ALIASES.update({"LAR": "LA", "JAC": "JAX"})


def attach_schedule_identity(markets: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    frame = markets.copy()
    starts = pd.to_datetime(frame["commence_time_utc"], utc=True, errors="coerce")
    schedule_frame = schedule.copy()
    schedule_frame["commence_time_utc"] = pd.to_datetime(
        schedule_frame["commence_time_utc"], utc=True, errors="coerce"
    )
    identities: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        home = TEAM_ALIASES.get(str(row.get("home_team") or ""))
        away = TEAM_ALIASES.get(str(row.get("away_team") or ""))
        candidates = schedule_frame
        provider_aliases = {"LAR": "LA", "JAC": "JAX"}
        provider_team_raw = str(row.get("provider_team") or "")
        provider_opponent_raw = str(row.get("provider_opponent") or "")
        provider_team = provider_aliases.get(provider_team_raw, provider_team_raw)
        provider_opponent = provider_aliases.get(provider_opponent_raw, provider_opponent_raw)
        provider_season = pd.to_numeric(row.get("provider_season"), errors="coerce")
        provider_week = pd.to_numeric(row.get("rotowire_week"), errors="coerce")
        if provider_team and provider_opponent:
            pair = {provider_team, provider_opponent}
            candidates = candidates.loc[
                schedule_frame.apply(
                    lambda game: {str(game["home_team"]), str(game["away_team"])} == pair,
                    axis=1,
                )
            ]
            if pd.notna(provider_season):
                candidates = candidates.loc[candidates["season"].eq(int(provider_season))]
        if pd.notna(provider_week):
            candidates = candidates.loc[candidates["week"].eq(int(provider_week))]
        if home and away:
            candidates = candidates.loc[
                candidates["home_team"].astype(str).eq(home)
                & candidates["away_team"].astype(str).eq(away)
            ]
        if candidates.empty:
            identities.append({"season": None, "week": None, "home_abbr": home, "away_abbr": away})
            continue
        if pd.isna(starts.loc[index]):
            if len(candidates) != 1:
                identities.append({"season": None, "week": None, "home_abbr": home, "away_abbr": away})
                continue
            match = candidates.iloc[0]
            frame.loc[index, "commence_time_utc"] = match["commence_time_utc"]
        else:
            deltas = (candidates["commence_time_utc"] - starts.loc[index]).abs()
            match = candidates.loc[deltas.idxmin()]
            if deltas.min() > pd.Timedelta(hours=18):
                identities.append({"season": None, "week": None, "home_abbr": home, "away_abbr": away})
                continue
        if match is None:
            identities.append({"season": None, "week": None, "home_abbr": home, "away_abbr": away})
            continue
        identities.append(
            {
                "season": int(match["season"]),
                "week": int(match["week"]),
                "home_abbr": str(match["home_team"]),
                "away_abbr": str(match["away_team"]),
            }
        )
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(identities)], axis=1
    )


def add_market_placeholders(
    stats: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    current_roster: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Map offered players to known identities and append outcome-free current rows."""

    history = stats.copy()
    latest = (
        history.sort_values(["season", "week"])
        .drop_duplicates("player_id", keep="last")
        .copy()
    )
    latest["player_key"] = latest["player_display_name"].map(normalize_player_name)
    history_names = latest.drop_duplicates("player_key", keep=False).set_index("player_key")
    history_by_id = latest.drop_duplicates("player_id", keep="last").set_index(
        "player_id", drop=False
    )
    current_name_to_id: dict[str, str] = {}
    current_team_by_id: dict[str, str] = {}
    if current_roster is not None and not current_roster.empty:
        roster = current_roster.rename(
            columns={"gsis_id": "player_id", "full_name": "player_display_name", "team": "recent_team"}
        ).copy()
        required = {"player_id", "player_display_name", "recent_team"}
        if required.issubset(roster.columns):
            sort_columns = [column for column in ("season", "week") if column in roster.columns]
            if sort_columns:
                roster = roster.sort_values(sort_columns)
            roster = roster.dropna(subset=list(required)).drop_duplicates("player_id", keep="last")
            roster = roster.loc[roster["player_id"].astype(str).isin(history_by_id.index.astype(str))]
            roster["player_key"] = roster["player_display_name"].map(normalize_player_name)
            unique_names = roster.drop_duplicates("player_key", keep=False)
            current_name_to_id = dict(
                zip(unique_names["player_key"], unique_names["player_id"].astype(str))
            )
            current_team_by_id = dict(
                zip(roster["player_id"].astype(str), roster["recent_team"].astype(str))
            )

    market_frame = markets.copy()
    market_frame["player_key"] = market_frame["player"].map(normalize_player_name)
    historical_ids = market_frame["player_key"].map(history_names["player_id"])
    current_ids = market_frame["player_key"].map(current_name_to_id)
    market_frame["player_id"] = current_ids.fillna(historical_ids).astype("string")
    market_frame = market_frame.dropna(subset=["player_id", "season", "week"]).copy()
    placeholders: list[pd.Series] = []
    accepted_ids: set[tuple[str, int, int]] = set()
    team_mismatch = 0
    for row in market_frame.drop_duplicates(
        ["player_id", "season", "week", "event_id"]
    ).itertuples(index=False):
        source = history_by_id.loc[str(row.player_id)].copy()
        team = current_team_by_id.get(str(row.player_id), str(source.get("recent_team") or ""))
        event_teams = {str(row.home_abbr or ""), str(row.away_abbr or "")}
        if team not in event_teams:
            team_mismatch += 1
            continue
        opponent = str(row.away_abbr if team == row.home_abbr else row.home_abbr)
        source["recent_team"] = team
        source["season"] = int(row.season)
        source["week"] = int(row.week)
        source["season_type"] = "REG"
        source["opponent_team"] = opponent
        for column in HISTORY_COLUMNS:
            source[column] = 0.0
        placeholders.append(source[history.columns])
        accepted_ids.add((str(row.player_id), int(row.season), int(row.week)))
    if placeholders:
        history = pd.concat([history, pd.DataFrame(placeholders)], ignore_index=True)
    accepted = market_frame.loc[
        market_frame.apply(
            lambda row: (str(row.player_id), int(row.season), int(row.week)) in accepted_ids,
            axis=1,
        )
    ].copy()
    return history, accepted, {
        "input_market_rows": int(len(markets)),
        "identity_matched_rows": int(market_frame.shape[0]),
        "current_roster_players": len(current_team_by_id),
        "team_mismatch_players": int(team_mismatch),
        "placeholder_players": len(accepted_ids),
    }


def build_live_scoring_frame(
    stats_with_placeholders: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    yardage_artifact: dict[str, Any],
    selector_artifact: dict[str, Any],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (season, week), market_week in markets.groupby(["season", "week"], sort=True):
        projections = predict_week_components_latent(
            stats_with_placeholders,
            yardage_artifact,
            season=int(season),
            week=int(week),
        )
        if projections.empty:
            continue
        for spec in TARGET_SPECS:
            model_info = selector_artifact["models"].get(spec.key)
            if not model_info or model_info.get("promotion_status") != "passed":
                continue
            features, raw_columns = build_features(stats_with_placeholders, spec)
            current_features = features.loc[
                features["season"].eq(int(season)) & features["week"].eq(int(week))
            ][["player_id", "season", "week"] + raw_columns]
            current_projections = projections.loc[projections["target"].eq(spec.key)]
            offers = market_week.loc[market_week["target"].eq(spec.key)].copy()
            joined = offers.merge(
                current_projections,
                on=["player_id", "season", "week", "target"],
                how="inner",
                validate="many_to_one",
            ).merge(
                current_features,
                on=["player_id", "season", "week"],
                how="inner",
                validate="many_to_one",
            )
            if joined.empty:
                continue
            scale = TARGET_SCALES[spec.key]
            joined["line_scaled"] = joined["line"] / scale
            joined["current_edge_scaled"] = (
                joined["current_prediction"] - joined["line"]
            ) / scale
            joined["challenger_edge_scaled"] = (
                joined["challenger_prediction"] - joined["line"]
            ) / scale
            joined["baseline_edge_scaled"] = (joined["baseline"] - joined["line"]) / scale
            joined["model_disagreement_scaled"] = (
                joined["current_prediction"] - joined["challenger_prediction"]
            ).abs() / scale
            expected = list(model_info["features"])
            missing = sorted(set(expected).difference(joined.columns))
            if missing:
                raise ValueError(
                    f"NFL live scoring is missing selector features for {spec.key}: {missing}"
                )
            joined["over_probability"] = model_info["model"].predict_proba(
                joined[expected]
            )[:, 1]
            parts.append(joined)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
