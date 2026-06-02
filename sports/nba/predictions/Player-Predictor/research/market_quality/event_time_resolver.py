from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))


EVENT_TIME_SOURCES = {
    "PROVIDER",
    "NBA_SCHEDULE",
    "LOCAL_SCHEDULE_CACHE",
    "TEAM_DATE_MATCH",
    "MISSING",
}

EVENT_TIME_CONFIDENCE = {"exact", "matched", "inferred", "missing"}


TEAM_ALIASES = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LA CLIPPERS": "LAC",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("games", "schedule", "rows", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return pd.DataFrame(value)
            return pd.DataFrame([payload])
    return pd.read_csv(path)


def _first_existing(frame: pd.DataFrame, columns: Iterable[str], default: Any = "") -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype("object")
        out = out.where(out.notna() & out.astype(str).str.strip().ne(""), values)
    return out


def _normalize_team(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return ""
    text = text.replace(".", "").replace("-", " ")
    if text in TEAM_ALIASES.values():
        return text
    return TEAM_ALIASES.get(" ".join(text.split()), text)


def _normalize_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").fillna("")


def _normalize_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _combine_date_and_time(date_values: pd.Series, time_values: pd.Series) -> pd.Series:
    date_text = date_values.fillna("").astype(str).str.slice(0, 10)
    time_text = time_values.fillna("").astype(str).str.strip()
    combined = date_text + " " + time_text
    return pd.to_datetime(combined.where(time_text.ne(""), pd.NA), errors="coerce", utc=True)


def normalize_schedule_frame(schedule_rows: pd.DataFrame, *, source: str = "LOCAL_SCHEDULE_CACHE") -> pd.DataFrame:
    if schedule_rows is None or schedule_rows.empty:
        return pd.DataFrame(
            columns=[
                "schedule_game_id",
                "schedule_game_date",
                "schedule_home_team",
                "schedule_away_team",
                "schedule_commence_time_utc",
                "schedule_event_time_source",
            ]
        )
    working = schedule_rows.copy()
    game_id = _first_existing(working, ["game_id", "GAME_ID", "id", "event_id", "Market_Event_ID"])
    game_date_raw = _first_existing(working, ["game_date", "GAME_DATE", "date", "GAME_DATE_EST", "market_date", "Market_Date"])
    home_team = _first_existing(
        working,
        ["home_team", "HOME_TEAM", "home_team_abbreviation", "HOME_TEAM_ABBREVIATION", "Market_Home_Team"],
    ).map(_normalize_team)
    away_team = _first_existing(
        working,
        ["away_team", "AWAY_TEAM", "away_team_abbreviation", "AWAY_TEAM_ABBREVIATION", "visitor_team", "Market_Away_Team"],
    ).map(_normalize_team)
    commence = _normalize_timestamp(
        _first_existing(
            working,
            [
                "market_commence_time_utc",
                "Market_Commence_Time_UTC",
                "commence_time_utc",
                "start_time_utc",
                "game_time_utc",
                "GAME_TIME_UTC",
                "GAME_DATE_TIME_UTC",
                "datetime_utc",
            ],
        )
    )
    if commence.isna().all():
        date_series = _first_existing(working, ["game_date", "GAME_DATE", "date", "GAME_DATE_EST", "market_date", "Market_Date"])
        time_series = _first_existing(working, ["game_time", "GAME_TIME", "GAME_STATUS_TEXT", "start_time", "time"])
        commence = _combine_date_and_time(date_series, time_series)
    out = pd.DataFrame(
        {
            "schedule_game_id": game_id.fillna("").astype(str),
            "schedule_game_date": _normalize_date(game_date_raw),
            "schedule_home_team": home_team.fillna("").astype(str),
            "schedule_away_team": away_team.fillna("").astype(str),
            "schedule_commence_time_utc": commence,
            "schedule_event_time_source": str(source or "LOCAL_SCHEDULE_CACHE"),
        }
    )
    out = out.loc[out["schedule_commence_time_utc"].notna()].copy()
    return out.drop_duplicates(
        subset=["schedule_game_id", "schedule_game_date", "schedule_home_team", "schedule_away_team"],
        keep="first",
    ).reset_index(drop=True)


def load_schedule_rows(schedule_paths: Iterable[Path] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in schedule_paths or []:
        try:
            rows = _read_table(Path(path))
        except Exception:
            continue
        normalized = normalize_schedule_frame(rows, source="LOCAL_SCHEDULE_CACHE")
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _market_identity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_event_time_game_id"] = _first_existing(out, ["market_event_id", "Market_Event_ID", "game_id"]).fillna("").astype(str)
    out["_event_time_game_date"] = _normalize_date(_first_existing(out, ["market_date", "Market_Date", "game_date"]))
    out["_event_time_home_team"] = _first_existing(out, ["market_home_team", "Market_Home_Team", "home_team"]).map(_normalize_team)
    out["_event_time_away_team"] = _first_existing(out, ["market_away_team", "Market_Away_Team", "away_team"]).map(_normalize_team)
    return out


def resolve_event_times(
    market_rows: pd.DataFrame,
    *,
    schedule_rows: pd.DataFrame | None = None,
    schedule_paths: Iterable[Path] | None = None,
) -> pd.DataFrame:
    if market_rows is None or market_rows.empty:
        return market_rows.copy() if market_rows is not None else pd.DataFrame()
    out = _market_identity(market_rows)
    provider_time = _normalize_timestamp(
        _first_existing(out, ["market_commence_time_utc", "Market_Commence_Time_UTC", "commence_time_utc"])
    )
    resolved_time = provider_time.copy()
    source = pd.Series("MISSING", index=out.index, dtype="object")
    confidence = pd.Series("missing", index=out.index, dtype="object")
    reason = pd.Series("event_time_missing_from_provider_and_schedule", index=out.index, dtype="object")
    warning = pd.Series("timestamp_safety_blocked_until_event_time_resolved", index=out.index, dtype="object")

    provider_mask = provider_time.notna()
    source = source.mask(provider_mask, "PROVIDER")
    confidence = confidence.mask(provider_mask, "exact")
    reason = reason.mask(provider_mask, "provider_supplied_market_commence_time_utc")
    warning = warning.mask(provider_mask, "")

    schedule = normalize_schedule_frame(schedule_rows, source="NBA_SCHEDULE") if schedule_rows is not None else load_schedule_rows(schedule_paths)
    unresolved = ~provider_mask
    if not schedule.empty and unresolved.any():
        schedule_by_id = schedule.loc[schedule["schedule_game_id"].astype(str).str.strip().ne("")].drop_duplicates(
            subset=["schedule_game_id"],
            keep="first",
        )
        if not schedule_by_id.empty:
            id_join = out.loc[unresolved].merge(
                schedule_by_id[["schedule_game_id", "schedule_commence_time_utc", "schedule_event_time_source"]],
                left_on="_event_time_game_id",
                right_on="schedule_game_id",
                how="left",
            )
            id_time = pd.Series(id_join["schedule_commence_time_utc"].to_numpy(), index=id_join.index)
            matched_index = out.loc[unresolved].index[id_time.notna().to_numpy()]
            if len(matched_index) > 0:
                resolved_time.loc[matched_index] = id_join.loc[id_time.notna(), "schedule_commence_time_utc"].to_numpy()
                source.loc[matched_index] = "NBA_SCHEDULE"
                confidence.loc[matched_index] = "exact"
                reason.loc[matched_index] = "schedule_exact_game_id_match"
                warning.loc[matched_index] = ""
        unresolved = resolved_time.isna()
        if unresolved.any():
            team_schedule = schedule.loc[
                schedule["schedule_game_date"].astype(str).str.strip().ne("")
                & schedule["schedule_home_team"].astype(str).str.strip().ne("")
                & schedule["schedule_away_team"].astype(str).str.strip().ne("")
            ].drop_duplicates(
                subset=["schedule_game_date", "schedule_home_team", "schedule_away_team"],
                keep="first",
            )
            if not team_schedule.empty:
                team_join = out.loc[unresolved].merge(
                    team_schedule[
                        [
                            "schedule_game_date",
                            "schedule_home_team",
                            "schedule_away_team",
                            "schedule_commence_time_utc",
                        ]
                    ],
                    left_on=["_event_time_game_date", "_event_time_home_team", "_event_time_away_team"],
                    right_on=["schedule_game_date", "schedule_home_team", "schedule_away_team"],
                    how="left",
                )
                team_time = pd.Series(team_join["schedule_commence_time_utc"].to_numpy(), index=team_join.index)
                matched_index = out.loc[unresolved].index[team_time.notna().to_numpy()]
                if len(matched_index) > 0:
                    resolved_time.loc[matched_index] = team_join.loc[team_time.notna(), "schedule_commence_time_utc"].to_numpy()
                    source.loc[matched_index] = "TEAM_DATE_MATCH"
                    confidence.loc[matched_index] = "matched"
                    reason.loc[matched_index] = "schedule_home_away_date_match"
                    warning.loc[matched_index] = ""

    out["market_commence_time_utc"] = resolved_time
    out["Market_Commence_Time_UTC"] = resolved_time
    out["event_time_source"] = source
    out["event_time_confidence"] = confidence
    out["event_time_resolution_reason"] = reason
    out["event_time_resolution_warning"] = warning
    return out.drop(
        columns=[
            "_event_time_game_id",
            "_event_time_game_date",
            "_event_time_home_team",
            "_event_time_away_team",
        ],
        errors="ignore",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve NBA market event commence times without postgame inference.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule-cache", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_table(args.input)
    resolved = resolve_event_times(rows, schedule_paths=list(args.schedule_cache))
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        resolved.to_parquet(args.output, index=False)
    else:
        resolved.to_csv(args.output, index=False)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(resolved)),
        "resolved_rows": int(pd.to_datetime(resolved["market_commence_time_utc"], errors="coerce", utc=True).notna().sum()),
        "event_time_source_counts": resolved["event_time_source"].value_counts(dropna=False).to_dict(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
