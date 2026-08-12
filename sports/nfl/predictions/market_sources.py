"""Provider adapters for authentic historical NFL player-prop lines."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


SPORTSGAMEODDS_STAT_MARKETS = {
    "passing_yards": "player_pass_yds",
    "rushing_yards": "player_rush_yds",
    "receiving_yards": "player_reception_yds",
}

XSPORTSBOOK_STAT_MARKETS = {
    "passing yards": "player_pass_yds",
    "rushing yards": "player_rush_yds",
    "receiving yards": "player_reception_yds",
}

XSPORTSBOOK_TEAM_ALIASES = {
    "JAC": "JAX",
    "LAR": "LA",
    "LVS": "LV",
}


def _events(payload: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        values = payload.get("data", payload.get("events", []))
    else:
        values = payload
    return [value for value in values if isinstance(value, Mapping)]


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _week_from_event(event: Mapping[str, Any]) -> int | None:
    info = event.get("info") if isinstance(event.get("info"), Mapping) else {}
    label = str(info.get("seasonWeek") or info.get("week") or "")
    match = re.search(r"\bweek\s*(\d{1,2})\b", label, flags=re.IGNORECASE)
    if match is None and re.fullmatch(r"\s*\d{1,2}\s*", label):
        match = re.search(r"\d{1,2}", label)
    return int(match.group(1)) if match else None


def infer_schedule_week(commence_time: Any, schedule: pd.DataFrame) -> int | None:
    """Resolve a provider event to the nearest nflverse regular-season kickoff."""

    kickoff = pd.to_datetime(commence_time, utc=True, errors="coerce")
    if pd.isna(kickoff) or schedule.empty:
        return None
    deltas = (schedule["commence_time_utc"] - kickoff).abs()
    if deltas.empty or deltas.min() > pd.Timedelta(hours=18):
        return None
    return int(schedule.loc[deltas.idxmin(), "week"])


def flatten_sportsgameodds_closing_lines(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    season: int,
    schedule: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize per-book closing lines from a SportsGameOdds event response.

    Only the provider's explicit ``close*`` fields are accepted. Current/live
    ``odds`` and ``overUnder`` values are never used as historical lines.
    """

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    audit = {
        "events_received": 0,
        "events_not_finalized": 0,
        "events_without_week": 0,
        "target_odds_seen": 0,
        "book_sides_without_close": 0,
        "closing_sides_accepted": 0,
    }
    schedule_frame = schedule if schedule is not None else pd.DataFrame()

    for event in _events(payload):
        audit["events_received"] += 1
        status = event.get("status") if isinstance(event.get("status"), Mapping) else {}
        if not bool(status.get("finalized")):
            audit["events_not_finalized"] += 1
            continue
        commence = status.get("startsAt") or event.get("startsAt") or event.get("commence_time")
        week = _week_from_event(event)
        if week is None and not schedule_frame.empty:
            week = infer_schedule_week(commence, schedule_frame)
        if week is None:
            audit["events_without_week"] += 1
            continue

        players = event.get("players") if isinstance(event.get("players"), Mapping) else {}
        teams = event.get("teams") if isinstance(event.get("teams"), Mapping) else {}
        home = teams.get("home") if isinstance(teams.get("home"), Mapping) else {}
        away = teams.get("away") if isinstance(teams.get("away"), Mapping) else {}
        home_names = home.get("names") if isinstance(home.get("names"), Mapping) else {}
        away_names = away.get("names") if isinstance(away.get("names"), Mapping) else {}
        odds = event.get("odds") if isinstance(event.get("odds"), Mapping) else {}

        for odd in odds.values():
            if not isinstance(odd, Mapping):
                continue
            stat_id = str(odd.get("statID") or "")
            if stat_id not in SPORTSGAMEODDS_STAT_MARKETS:
                continue
            if str(odd.get("periodID") or "") != "game" or str(odd.get("betTypeID") or "") != "ou":
                continue
            side = str(odd.get("sideID") or "").lower()
            if side not in {"over", "under"}:
                continue
            audit["target_odds_seen"] += 1
            player_id = str(odd.get("playerID") or odd.get("statEntityID") or "")
            player = players.get(player_id) if isinstance(players.get(player_id), Mapping) else {}
            player_name = player.get("name") or player.get("display")
            if not player_name:
                market_name = str(odd.get("marketName") or "")
                suffix = re.search(r"\s+(?:Passing|Rushing|Receiving)\s+Yards", market_name, re.IGNORECASE)
                player_name = market_name[: suffix.start()].strip() if suffix else ""
            if not player_name:
                continue

            by_book = odd.get("byBookmaker") if isinstance(odd.get("byBookmaker"), Mapping) else {}
            for bookmaker, book_value in by_book.items():
                if not isinstance(book_value, Mapping):
                    continue
                line = _as_float(book_value.get("closeOverUnder"))
                price = _as_float(book_value.get("closeOdds"))
                if line is None or price is None:
                    audit["book_sides_without_close"] += 1
                    continue
                key = (event.get("eventID"), player_id, stat_id, str(bookmaker), line)
                row = grouped.setdefault(
                    key,
                    {
                        "season": int(season),
                        "week": int(week),
                        "player": str(player_name),
                        "player_id": player_id,
                        "market": SPORTSGAMEODDS_STAT_MARKETS[stat_id],
                        "line": line,
                        "over_price": pd.NA,
                        "under_price": pd.NA,
                        "bookmaker": str(bookmaker),
                        "source": "sportsgameodds_historical_close",
                        "event_id": event.get("eventID"),
                        "home_team": home_names.get("long") or home.get("name"),
                        "away_team": away_names.get("long") or away.get("name"),
                        "snapshot_time_utc": pd.NA,
                        "commence_time_utc": commence,
                        "line_phase": "closing_pregame",
                        "pregame_verified": True,
                        "verification_method": "provider_explicit_close_fields",
                    },
                )
                row[f"{side}_price"] = price
                audit["closing_sides_accepted"] += 1

    columns = [
        "season", "week", "player", "player_id", "market", "line",
        "over_price", "under_price", "bookmaker", "source", "event_id",
        "home_team", "away_team", "snapshot_time_utc", "commence_time_utc",
        "line_phase", "pregame_verified", "verification_method",
    ]
    frame = pd.DataFrame(grouped.values(), columns=columns)
    paired = frame["over_price"].notna() & frame["under_price"].notna() if not frame.empty else pd.Series(dtype=bool)
    audit["normalized_rows_before_price_pair_filter"] = int(len(frame))
    audit["dropped_one_sided_rows"] = int((~paired).sum()) if not frame.empty else 0
    frame = frame.loc[paired].reset_index(drop=True) if not frame.empty else frame
    audit["normalized_rows"] = int(len(frame))
    audit["two_sided_price_rows"] = int(len(frame))
    return frame, audit


def flatten_sportsgameodds_consensus_closing_lines(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    season: int,
    schedule: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize explicit provider consensus closes when per-book closes are unavailable.

    These rows are valid research lines and prices, but they are never represented as
    executable at DraftKings, FanDuel, or another named sportsbook.
    """

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    audit = {
        "events_received": 0,
        "events_not_finalized": 0,
        "events_without_week": 0,
        "target_odds_seen": 0,
        "consensus_sides_without_close": 0,
        "consensus_closing_sides_accepted": 0,
    }
    schedule_frame = schedule if schedule is not None else pd.DataFrame()
    for event in _events(payload):
        audit["events_received"] += 1
        status = event.get("status") if isinstance(event.get("status"), Mapping) else {}
        if not bool(status.get("finalized")):
            audit["events_not_finalized"] += 1
            continue
        commence = status.get("startsAt") or event.get("startsAt") or event.get("commence_time")
        week = _week_from_event(event)
        if week is None and not schedule_frame.empty:
            week = infer_schedule_week(commence, schedule_frame)
        if week is None:
            audit["events_without_week"] += 1
            continue
        players = event.get("players") if isinstance(event.get("players"), Mapping) else {}
        teams = event.get("teams") if isinstance(event.get("teams"), Mapping) else {}
        home = teams.get("home") if isinstance(teams.get("home"), Mapping) else {}
        away = teams.get("away") if isinstance(teams.get("away"), Mapping) else {}
        home_names = home.get("names") if isinstance(home.get("names"), Mapping) else {}
        away_names = away.get("names") if isinstance(away.get("names"), Mapping) else {}
        odds = event.get("odds") if isinstance(event.get("odds"), Mapping) else {}
        for odd in odds.values():
            if not isinstance(odd, Mapping):
                continue
            stat_id = str(odd.get("statID") or "")
            if stat_id not in SPORTSGAMEODDS_STAT_MARKETS:
                continue
            if str(odd.get("periodID") or "") != "game" or str(odd.get("betTypeID") or "") != "ou":
                continue
            side = str(odd.get("sideID") or "").lower()
            if side not in {"over", "under"}:
                continue
            audit["target_odds_seen"] += 1
            line = _as_float(odd.get("closeBookOverUnder"))
            price = _as_float(odd.get("closeBookOdds"))
            if line is None or price is None:
                audit["consensus_sides_without_close"] += 1
                continue
            player_id = str(odd.get("playerID") or odd.get("statEntityID") or "")
            player = players.get(player_id) if isinstance(players.get(player_id), Mapping) else {}
            player_name = player.get("name") or player.get("display")
            if not player_name:
                market_name = str(odd.get("marketName") or "")
                suffix = re.search(
                    r"\s+(?:Passing|Rushing|Receiving)\s+Yards", market_name, re.IGNORECASE
                )
                player_name = market_name[: suffix.start()].strip() if suffix else ""
            if not player_name:
                continue
            key = (event.get("eventID"), player_id, stat_id, line)
            row = grouped.setdefault(
                key,
                {
                    "season": int(season),
                    "week": int(week),
                    "player": str(player_name),
                    "player_id": player_id,
                    "market": SPORTSGAMEODDS_STAT_MARKETS[stat_id],
                    "line": line,
                    "over_price": pd.NA,
                    "under_price": pd.NA,
                    "bookmaker": "sportsgameodds_consensus",
                    "source": "sportsgameodds_consensus_close",
                    "event_id": event.get("eventID"),
                    "home_team": home_names.get("long") or home.get("name"),
                    "away_team": away_names.get("long") or away.get("name"),
                    "snapshot_time_utc": pd.NA,
                    "commence_time_utc": commence,
                    "line_phase": "provider_consensus_close",
                    "pregame_verified": True,
                    "executable_book_verified": False,
                    "verification_method": "provider_explicit_consensus_close_fields",
                },
            )
            row[f"{side}_price"] = price
            audit["consensus_closing_sides_accepted"] += 1

    columns = [
        "season", "week", "player", "player_id", "market", "line",
        "over_price", "under_price", "bookmaker", "source", "event_id",
        "home_team", "away_team", "snapshot_time_utc", "commence_time_utc",
        "line_phase", "pregame_verified", "executable_book_verified",
        "verification_method",
    ]
    frame = pd.DataFrame(grouped.values(), columns=columns)
    paired = (
        frame["over_price"].notna() & frame["under_price"].notna()
        if not frame.empty
        else pd.Series(dtype=bool)
    )
    audit["normalized_rows_before_price_pair_filter"] = int(len(frame))
    audit["dropped_one_sided_rows"] = int((~paired).sum()) if not frame.empty else 0
    frame = frame.loc[paired].reset_index(drop=True) if not frame.empty else frame
    audit["normalized_rows"] = int(len(frame))
    audit["two_sided_price_rows"] = int(len(frame))
    return frame, audit


def _xsportsbook_team(value: Any) -> str:
    team = str(value or "").strip().upper()
    return XSPORTSBOOK_TEAM_ALIASES.get(team, team)


def _schedule_kickoff_lookup(schedule: pd.DataFrame) -> dict[tuple[int, str, str], Any]:
    if schedule.empty:
        return {}
    frame = schedule.copy()
    if "commence_time_utc" not in frame.columns:
        if not {"gameday", "gametime"}.issubset(frame.columns):
            return {}
        local = pd.to_datetime(
            frame["gameday"].astype(str) + " " + frame["gametime"].astype(str),
            errors="coerce",
        )
        frame["commence_time_utc"] = local.dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
    lookup: dict[tuple[int, str, str], Any] = {}
    for row in frame.itertuples(index=False):
        week = int(getattr(row, "week"))
        home = _xsportsbook_team(getattr(row, "home_team"))
        away = _xsportsbook_team(getattr(row, "away_team"))
        kickoff = getattr(row, "commence_time_utc")
        lookup[(week, home, away)] = kickoff
        lookup[(week, away, home)] = kickoff
    return lookup


def flatten_xsportsbook_bovada_archive(
    raw: pd.DataFrame,
    *,
    season: int,
    schedule: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize XSportsbook's intentionally downloadable Bovada prop archive.

    The publisher identifies the rows as historical Bovada prop bets and
    provides both side prices, but it does not publish capture timestamps or
    explicitly call the observations closing lines.  Rows are therefore useful
    for hit-rate/ROI research but deliberately remain unverified for the strict
    deployment gate.
    """

    required = {
        "Game_Id", "Player", "Betting Event", "Team", "Opp", "Week",
        "O-Line", "O-Odds", "U-Line", "U-Odds",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"XSportsbook archive is missing columns: {sorted(missing)}")

    frame = raw.copy()
    frame["market"] = (
        frame["Betting Event"].astype(str).str.replace("\u00a0", " ").str.strip().str.lower()
        .map(XSPORTSBOOK_STAT_MARKETS)
    )
    audit = {
        "input_rows": int(len(frame)),
        "non_target_market_rows": int(frame["market"].isna().sum()),
    }
    frame = frame.loc[frame["market"].notna()].copy()
    frame["player"] = (
        frame["Player"].astype(str).str.replace("\u00a0", " ").str.strip()
    )
    frame["week"] = pd.to_numeric(
        frame["Week"].astype(str).str.extract(r"W(\d{1,2})", expand=False),
        errors="coerce",
    )
    for column in ("O-Line", "O-Odds", "U-Line", "U-Odds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    line_mismatch = frame["O-Line"].notna() & frame["U-Line"].notna() & ~frame[
        "O-Line"
    ].eq(frame["U-Line"])
    invalid = (
        frame["player"].eq("")
        | frame["week"].isna()
        | ~frame["week"].between(1, 18)
        | frame[["O-Line", "O-Odds", "U-Line", "U-Odds"]].isna().any(axis=1)
        | frame["O-Odds"].eq(0)
        | frame["U-Odds"].eq(0)
        | line_mismatch
    )
    audit["line_mismatch_rows"] = int(line_mismatch.sum())
    audit["invalid_contract_rows"] = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    frame["week"] = frame["week"].astype(int)
    frame["event_id"] = frame["Game_Id"].astype(str)
    frame["team_key"] = frame["Team"].map(_xsportsbook_team)
    frame["opponent_key"] = frame["Opp"].map(_xsportsbook_team)

    identity = ["event_id", "player", "market"]
    signature = ["O-Line", "O-Odds", "U-Line", "U-Odds"]
    signature_counts = frame.groupby(identity, dropna=False)[signature].apply(
        lambda group: len(group.drop_duplicates())
    )
    ambiguous_keys = set(signature_counts.loc[signature_counts.gt(1)].index)
    ambiguous = frame.set_index(identity).index.isin(ambiguous_keys)
    audit["ambiguous_duplicate_rows"] = int(ambiguous.sum())
    frame = frame.loc[~ambiguous].drop_duplicates(identity, keep="first").copy()

    kickoff_lookup = _schedule_kickoff_lookup(
        schedule if schedule is not None else pd.DataFrame()
    )
    frame["commence_time_utc"] = [
        kickoff_lookup.get((week, team, opponent), pd.NA)
        for week, team, opponent in zip(
            frame["week"], frame["team_key"], frame["opponent_key"]
        )
    ]
    output = pd.DataFrame(
        {
            "season": int(season),
            "week": frame["week"],
            "player": frame["player"],
            "player_id": frame.get("Player.id", pd.Series(pd.NA, index=frame.index)),
            "market": frame["market"],
            "line": frame["O-Line"],
            "over_price": frame["O-Odds"],
            "under_price": frame["U-Odds"],
            "bookmaker": "bovada",
            "source": "xsportsbook_bovada_archive",
            "event_id": frame["event_id"],
            "home_team": frame.get("Hteam", pd.Series(pd.NA, index=frame.index)),
            "away_team": frame.get("Ateam", pd.Series(pd.NA, index=frame.index)),
            "snapshot_time_utc": pd.NA,
            "commence_time_utc": frame["commence_time_utc"],
            "line_phase": "historical_posted_unstamped",
            "pregame_verified": False,
            "verification_method": "publisher_identified_bovada_prop_archive_no_timestamp",
        }
    ).reset_index(drop=True)
    audit["schedule_matched_rows"] = int(output["commence_time_utc"].notna().sum())
    audit["normalized_rows"] = int(len(output))
    audit["passing_rows"] = int(output["market"].eq("player_pass_yds").sum())
    audit["rushing_rows"] = int(output["market"].eq("player_rush_yds").sum())
    audit["receiving_rows"] = int(output["market"].eq("player_reception_yds").sum())
    return output, audit
