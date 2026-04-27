#!/usr/bin/env python3
"""
Fetch and normalize NFL player prop markets into a stable contract.

This mirrors the NBA market fetch workflow:
- optional Odds API pull or snapshot ingestion
- normalized long + wide tables
- rolling append-only history outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nfl"
SPORT_KEY = "americanfootball_nfl"
DEFAULT_MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds"]
DEFAULT_BOOKMAKERS = ["draftkings", "fanduel"]
EASTERN_TZ = "America/New_York"
MARKET_WIDE_COLUMNS = [
    "Market_Date",
    "Player",
    "Market_Player_Raw",
    "Market_Event_ID",
    "Market_Commence_Time_UTC",
    "Market_Home_Team",
    "Market_Away_Team",
    "Market_PASS_YDS",
    "Market_RUSH_YDS",
    "Market_REC_YDS",
    "Market_PASS_YDS_books",
    "Market_RUSH_YDS_books",
    "Market_REC_YDS_books",
    "Market_PASS_YDS_over_price",
    "Market_RUSH_YDS_over_price",
    "Market_REC_YDS_over_price",
    "Market_PASS_YDS_under_price",
    "Market_RUSH_YDS_under_price",
    "Market_REC_YDS_under_price",
    "Market_PASS_YDS_line_std",
    "Market_RUSH_YDS_line_std",
    "Market_REC_YDS_line_std",
    "Market_Fetched_At_UTC",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_name(value: str) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch normalized NFL player prop lines.")
    parser.add_argument(
        "--provider",
        type=str,
        default="odds_api",
        choices=["odds_api", "snapshot"],
        help="Market data provider. 'snapshot' ingests an already downloaded CSV/parquet.",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Odds API key. Defaults to THE_ODDS_API_KEY / ODDS_API_KEY.")
    parser.add_argument("--input-path", type=Path, default=None, help="Input CSV/parquet for --provider snapshot.")
    parser.add_argument("--regions", type=str, default="us", help="API regions parameter.")
    parser.add_argument("--markets", type=str, default=",".join(DEFAULT_MARKETS), help="Comma-separated market keys.")
    parser.add_argument("--bookmakers", type=str, default=",".join(DEFAULT_BOOKMAKERS), help="Comma-separated bookmakers.")
    parser.add_argument("--odds-format", type=str, default="american", choices=["american", "decimal"], help="Odds format.")
    parser.add_argument("--date-format", type=str, default="iso", choices=["iso", "unix"], help="Date format.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for raw + normalized snapshots.")
    parser.add_argument("--event-limit", type=int, default=None, help="Optional event limit for smoke tests.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Cooldown between event calls.")
    return parser.parse_args()


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        text = str(value or "").strip()
        lowered = text.lower()
        if text and "paste-your" not in lowered and "your_api_key" not in lowered:
            return text
    return None


def _load_api_key_from_yaml(path: Path) -> str | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    odds_api = payload.get("odds_api")
    secrets = payload.get("secrets")
    return _first_non_empty(
        payload.get("ODDS_API_KEY"),
        payload.get("THE_ODDS_API_KEY"),
        odds_api.get("api_key") if isinstance(odds_api, dict) else None,
        odds_api.get("odds_api_key") if isinstance(odds_api, dict) else None,
        secrets.get("ODDS_API_KEY") if isinstance(secrets, dict) else None,
        secrets.get("THE_ODDS_API_KEY") if isinstance(secrets, dict) else None,
        secrets.get("odds_api_key") if isinstance(secrets, dict) else None,
    )


def _load_api_key_from_dotenv(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in {"ODDS_API_KEY", "THE_ODDS_API_KEY"}:
            continue
        cleaned = value.strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return None


def _load_api_key_from_local_files(start_path: Path) -> str | None:
    candidate_names = ("config.local.yaml", ".env.local", ".env", "config.yaml")
    checked: set[Path] = set()
    for base in [start_path, *start_path.parents]:
        for name in candidate_names:
            candidate = (base / name).resolve()
            if candidate in checked or not candidate.exists():
                continue
            checked.add(candidate)
            if candidate.suffix.lower() in {".yaml", ".yml"}:
                value = _load_api_key_from_yaml(candidate)
            else:
                value = _load_api_key_from_dotenv(candidate)
            if value:
                return value
    return None


def resolve_api_key(explicit_key: str | None) -> str:
    if explicit_key:
        return explicit_key
    for key in ("THE_ODDS_API_KEY", "ODDS_API_KEY"):
        value = os.getenv(key)
        if value:
            return value
    local_value = _load_api_key_from_local_files(Path(__file__).resolve().parent)
    if local_value:
        return local_value
    raise RuntimeError("Missing Odds API key. Set THE_ODDS_API_KEY, create config.local.yaml, or pass --api-key.")


def request_json(base_url: str, params: dict[str, object]) -> tuple[object, dict[str, str]]:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{base_url}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Player-Predictor/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {key.lower(): value for key, value in response.headers.items()}
            return payload, headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Odds API request failed [{exc.code}] {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Odds API network failure for {url}: {exc}") from exc


def safe_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _event_date_et(commence_time_value: str) -> str | None:
    if not commence_time_value:
        return None
    parsed = pd.to_datetime(commence_time_value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return str(parsed.tz_convert(EASTERN_TZ).date())


def _collapse_market_outcomes(
    *,
    event: dict,
    bookmaker: dict,
    market: dict,
    fetched_at_utc: str,
) -> list[dict]:
    grouped: dict[tuple[str, float | None], dict] = {}
    for outcome in market.get("outcomes", []):
        player_name = outcome.get("description") or outcome.get("participant")
        if not player_name:
            continue
        point = outcome.get("point")
        point_value = float(point) if point is not None else None
        group_key = (str(player_name), point_value)
        row = grouped.setdefault(
            group_key,
            {
                "fetched_at_utc": fetched_at_utc,
                "event_id": event.get("id"),
                "commence_time_utc": event.get("commence_time"),
                "event_date_et": _event_date_et(event.get("commence_time")),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "bookmaker_key": bookmaker.get("key"),
                "bookmaker_title": bookmaker.get("title"),
                "market_key": market.get("key"),
                "player_name_raw": str(player_name),
                "player_name_norm": normalize_name(str(player_name)),
                "line": point_value,
                "over_price": np.nan,
                "under_price": np.nan,
            },
        )
        name = str(outcome.get("name", "")).strip().lower()
        price = outcome.get("price")
        if name == "over":
            row["over_price"] = float(price) if price is not None else np.nan
        elif name == "under":
            row["under_price"] = float(price) if price is not None else np.nan
    return list(grouped.values())


def normalize_event_odds(events: list[dict], event_odds: dict[str, dict], fetched_at_utc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    event_by_id = {str(event.get("id")): event for event in events}
    for event_id, payload in event_odds.items():
        event = event_by_id.get(str(event_id), payload)
        bookmakers = payload.get("bookmakers", [])
        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                rows.extend(
                    _collapse_market_outcomes(
                        event=event,
                        bookmaker=bookmaker,
                        market=market,
                        fetched_at_utc=fetched_at_utc,
                    )
                )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    for col in ["line", "over_price", "under_price"]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    consensus = (
        long_df.groupby(["event_date_et", "player_name_norm", "player_name_raw", "market_key"], dropna=False)
        .agg(
            market_line=("line", "median"),
            market_line_std=("line", "std"),
            over_price_avg=("over_price", "mean"),
            under_price_avg=("under_price", "mean"),
            book_count=("bookmaker_key", "nunique"),
            event_count=("event_id", "nunique"),
            first_event_id=("event_id", "min"),
            first_commence_time_utc=("commence_time_utc", "min"),
            first_home_team=("home_team", "min"),
            first_away_team=("away_team", "min"),
        )
        .reset_index()
    )
    consensus["market_line_std"] = consensus["market_line_std"].fillna(0.0)

    value_map = {
        "player_pass_yds": "Market_PASS_YDS",
        "player_rush_yds": "Market_RUSH_YDS",
        "player_reception_yds": "Market_REC_YDS",
    }
    books_map = {
        "player_pass_yds": "Market_PASS_YDS_books",
        "player_rush_yds": "Market_RUSH_YDS_books",
        "player_reception_yds": "Market_REC_YDS_books",
    }
    over_map = {
        "player_pass_yds": "Market_PASS_YDS_over_price",
        "player_rush_yds": "Market_RUSH_YDS_over_price",
        "player_reception_yds": "Market_REC_YDS_over_price",
    }
    under_map = {
        "player_pass_yds": "Market_PASS_YDS_under_price",
        "player_rush_yds": "Market_RUSH_YDS_under_price",
        "player_reception_yds": "Market_REC_YDS_under_price",
    }
    spread_map = {
        "player_pass_yds": "Market_PASS_YDS_line_std",
        "player_rush_yds": "Market_RUSH_YDS_line_std",
        "player_reception_yds": "Market_REC_YDS_line_std",
    }

    def _pivot(metric_col: str, rename_map: dict[str, str]) -> pd.DataFrame:
        wide = (
            consensus.pivot_table(
                index=["event_date_et", "player_name_norm", "player_name_raw"],
                columns="market_key",
                values=metric_col,
                aggfunc="first",
            )
            .rename(columns=rename_map)
            .reset_index()
        )
        wide.columns.name = None
        return wide

    metadata = consensus[
        [
            "event_date_et",
            "player_name_norm",
            "player_name_raw",
            "first_event_id",
            "first_commence_time_utc",
            "first_home_team",
            "first_away_team",
        ]
    ].drop_duplicates(subset=["event_date_et", "player_name_norm", "player_name_raw"], keep="last")

    wide = metadata.merge(
        _pivot("market_line", value_map),
        how="left",
        on=["event_date_et", "player_name_norm", "player_name_raw"],
    )
    for metric_col, rename_map in [
        ("book_count", books_map),
        ("over_price_avg", over_map),
        ("under_price_avg", under_map),
        ("market_line_std", spread_map),
    ]:
        wide = wide.merge(
            _pivot(metric_col, rename_map),
            how="left",
            on=["event_date_et", "player_name_norm", "player_name_raw"],
        )

    wide = wide.rename(
        columns={
            "event_date_et": "Market_Date",
            "player_name_norm": "Player",
            "player_name_raw": "Market_Player_Raw",
            "first_event_id": "Market_Event_ID",
            "first_commence_time_utc": "Market_Commence_Time_UTC",
            "first_home_team": "Market_Home_Team",
            "first_away_team": "Market_Away_Team",
        }
    )
    wide["Market_Fetched_At_UTC"] = fetched_at_utc
    return long_df, wide


def normalize_wide_snapshot(df: pd.DataFrame, fetched_at_utc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    out = df.copy()
    rename_map = {}
    if "event_date_et" in out.columns and "Market_Date" not in out.columns:
        rename_map["event_date_et"] = "Market_Date"
    if "player_name_norm" in out.columns and "Player" not in out.columns:
        rename_map["player_name_norm"] = "Player"
    if "player_name_raw" in out.columns and "Market_Player_Raw" not in out.columns:
        rename_map["player_name_raw"] = "Market_Player_Raw"
    if rename_map:
        out = out.rename(columns=rename_map)

    if "Player" not in out.columns:
        raise ValueError("Snapshot provider requires a Player column or player_name_norm.")
    if "Market_Date" not in out.columns:
        raise ValueError("Snapshot provider requires a Market_Date column or event_date_et.")

    out["Player"] = out["Player"].astype(str).map(normalize_name)
    out["Market_Date"] = pd.to_datetime(out["Market_Date"], errors="coerce").dt.date.astype(str)
    if "Market_Player_Raw" not in out.columns:
        out["Market_Player_Raw"] = out["Player"]
    for market_col in ["Market_Event_ID", "Market_Commence_Time_UTC", "Market_Home_Team", "Market_Away_Team"]:
        if market_col not in out.columns:
            out[market_col] = pd.NA
    if "Market_Fetched_At_UTC" not in out.columns:
        out["Market_Fetched_At_UTC"] = fetched_at_utc

    for col in MARKET_WIDE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    numeric_cols = [
        col
        for col in MARKET_WIDE_COLUMNS
        if col.startswith("Market_")
        and col
        not in {
            "Market_Date",
            "Market_Player_Raw",
            "Market_Event_ID",
            "Market_Commence_Time_UTC",
            "Market_Home_Team",
            "Market_Away_Team",
            "Market_Fetched_At_UTC",
        }
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    wide_df = out[MARKET_WIDE_COLUMNS].drop_duplicates(subset=["Market_Date", "Player"], keep="last").copy()

    long_rows = []
    target_map = {
        "PASS_YDS": "player_pass_yds",
        "RUSH_YDS": "player_rush_yds",
        "REC_YDS": "player_reception_yds",
    }
    for _, row in wide_df.iterrows():
        for short_target, market_key in target_map.items():
            line = row.get(f"Market_{short_target}")
            if pd.isna(line):
                continue
            long_rows.append(
                {
                    "fetched_at_utc": row.get("Market_Fetched_At_UTC", fetched_at_utc),
                    "event_id": row.get("Market_Event_ID", np.nan),
                    "commence_time_utc": row.get("Market_Commence_Time_UTC", np.nan),
                    "event_date_et": row["Market_Date"],
                    "home_team": row.get("Market_Home_Team", np.nan),
                    "away_team": row.get("Market_Away_Team", np.nan),
                    "bookmaker_key": np.nan,
                    "bookmaker_title": np.nan,
                    "market_key": market_key,
                    "player_name_raw": row.get("Market_Player_Raw", row["Player"]),
                    "player_name_norm": row["Player"],
                    "line": float(line),
                    "over_price": row.get(f"Market_{short_target}_over_price", np.nan),
                    "under_price": row.get(f"Market_{short_target}_under_price", np.nan),
                }
            )
    long_df = pd.DataFrame(long_rows)
    return long_df, wide_df


def fetch_from_odds_api(args: argparse.Namespace, fetched_at_utc: str) -> tuple[list[dict], dict[str, dict], pd.DataFrame, pd.DataFrame, dict]:
    api_key = resolve_api_key(args.api_key)
    markets = [item.strip() for item in args.markets.split(",") if item.strip()]
    bookmakers = [item.strip() for item in args.bookmakers.split(",") if item.strip()]
    events_url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events"
    odds_url_template = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{{event_id}}/odds"

    events, events_headers = request_json(
        events_url,
        {
            "apiKey": api_key,
            "dateFormat": args.date_format,
        },
    )
    if not isinstance(events, list):
        raise RuntimeError(f"Unexpected events payload: {type(events)!r}")
    if args.event_limit is not None:
        events = events[: args.event_limit]

    event_payloads: dict[str, dict] = {}
    errors: list[dict[str, str]] = []
    for idx, event in enumerate(events, start=1):
        event_id = str(event.get("id"))
        try:
            payload, _headers = request_json(
                odds_url_template.format(event_id=event_id),
                {
                    "apiKey": api_key,
                    "regions": args.regions,
                    "markets": ",".join(markets),
                    "bookmakers": ",".join(bookmakers),
                    "oddsFormat": args.odds_format,
                    "dateFormat": args.date_format,
                },
            )
            event_payloads[event_id] = payload
        except Exception as exc:
            errors.append({"event_id": event_id, "error": str(exc)})
        if idx < len(events) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    long_df, wide_df = normalize_event_odds(events, event_payloads, fetched_at_utc)
    manifest = {
        "provider": "odds_api",
        "fetched_at_utc": fetched_at_utc,
        "sport": SPORT_KEY,
        "markets": markets,
        "bookmakers": bookmakers,
        "regions": args.regions,
        "odds_format": args.odds_format,
        "event_count_requested": int(len(events)),
        "event_count_fetched": int(len(event_payloads)),
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "events_remaining_header": events_headers.get("x-requests-remaining"),
        "events_used_header": events_headers.get("x-requests-used"),
        "errors": errors,
    }
    return events, event_payloads, long_df, wide_df, manifest


def fetch_from_snapshot(args: argparse.Namespace, fetched_at_utc: str) -> tuple[list[dict], dict[str, dict], pd.DataFrame, pd.DataFrame, dict]:
    if args.input_path is None or not args.input_path.exists():
        raise RuntimeError("Snapshot provider requires --input-path pointing to an existing CSV/parquet file.")
    snapshot_df = load_table(args.input_path)
    long_df, wide_df = normalize_wide_snapshot(snapshot_df, fetched_at_utc)
    manifest = {
        "provider": "snapshot",
        "fetched_at_utc": fetched_at_utc,
        "input_path": str(args.input_path),
        "event_count_requested": 0,
        "event_count_fetched": 0,
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "errors": [],
    }
    return [], {}, long_df, wide_df, manifest


def write_snapshot(
    outdir: Path,
    stamp: str,
    events: list[dict],
    event_payloads: dict[str, dict],
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    manifest: dict,
) -> None:
    raw_dir = outdir / "raw" / stamp
    norm_dir = outdir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    safe_write_json(raw_dir / "events.json", events)
    safe_write_json(raw_dir / "event_odds.json", event_payloads)
    safe_write_json(raw_dir / "manifest.json", manifest)

    if not long_df.empty:
        long_df.to_parquet(norm_dir / f"player_props_long_{stamp}.parquet", index=False)
        long_df.to_csv(norm_dir / f"player_props_long_{stamp}.csv", index=False)
        long_df.to_parquet(outdir / "latest_player_props_long.parquet", index=False)
        long_df.to_csv(outdir / "latest_player_props_long.csv", index=False)

    if not wide_df.empty:
        wide_df.to_parquet(norm_dir / f"player_props_wide_{stamp}.parquet", index=False)
        wide_df.to_csv(norm_dir / f"player_props_wide_{stamp}.csv", index=False)
        wide_df.to_parquet(outdir / "latest_player_props_wide.parquet", index=False)
        wide_df.to_csv(outdir / "latest_player_props_wide.csv", index=False)

    safe_write_json(outdir / "latest_manifest.json", manifest)


def append_history(outdir: Path, long_df: pd.DataFrame, wide_df: pd.DataFrame) -> dict:
    summary = {
        "history_long_rows": 0,
        "history_wide_rows": 0,
    }
    if not long_df.empty:
        history_long_path = outdir / "history_player_props_long.parquet"
        if history_long_path.exists():
            history_long = pd.read_parquet(history_long_path)
            combined_long = pd.concat([history_long, long_df], ignore_index=True)
        else:
            combined_long = long_df.copy()
        long_dedupe_cols = [
            "event_date_et",
            "player_name_norm",
            "market_key",
            "bookmaker_key",
            "line",
            "over_price",
            "under_price",
            "fetched_at_utc",
        ]
        combined_long = combined_long.drop_duplicates(subset=[col for col in long_dedupe_cols if col in combined_long.columns], keep="last")
        write_table(combined_long, history_long_path)
        write_table(combined_long, outdir / "history_player_props_long.csv")
        summary["history_long_rows"] = int(len(combined_long))

    if not wide_df.empty:
        history_wide_path = outdir / "history_player_props_wide.parquet"
        if history_wide_path.exists():
            history_wide = pd.read_parquet(history_wide_path)
            combined_wide = pd.concat([history_wide, wide_df], ignore_index=True)
        else:
            combined_wide = wide_df.copy()
        wide_dedupe_cols = [
            "Market_Date",
            "Player",
            "Market_Fetched_At_UTC",
        ]
        combined_wide = combined_wide.drop_duplicates(subset=[col for col in wide_dedupe_cols if col in combined_wide.columns], keep="last")
        write_table(combined_wide, history_wide_path)
        write_table(combined_wide, outdir / "history_player_props_wide.csv")
        summary["history_wide_rows"] = int(len(combined_wide))
    return summary


def main() -> None:
    args = parse_args()
    stamp = utc_compact_timestamp()
    fetched_at_utc = utc_now_iso()

    if args.provider == "odds_api":
        events, event_payloads, long_df, wide_df, manifest = fetch_from_odds_api(args, fetched_at_utc)
    elif args.provider == "snapshot":
        events, event_payloads, long_df, wide_df, manifest = fetch_from_snapshot(args, fetched_at_utc)
    else:
        raise RuntimeError(f"Unsupported provider: {args.provider}")

    write_snapshot(args.outdir, stamp, events, event_payloads, long_df, wide_df, manifest)
    history_summary = append_history(args.outdir, long_df, wide_df)
    manifest.update(history_summary)
    safe_write_json(args.outdir / "latest_manifest.json", manifest)

    print("\n" + "=" * 80)
    print("NFL MARKET PROPS FETCH COMPLETE")
    print("=" * 80)
    print(f"Provider:         {manifest.get('provider')}")
    print(f"Events requested: {len(events)}")
    print(f"Events fetched:   {len(event_payloads)}")
    print(f"Long rows:        {len(long_df)}")
    print(f"Wide rows:        {len(wide_df)}")
    print(f"History wide:     {history_summary['history_wide_rows']}")
    print(f"Errors:           {len(manifest.get('errors', []))}")
    print(f"Output:           {args.outdir}")
    errors = manifest.get("errors", [])
    if errors:
        print("Sample error:")
        print(f"  {errors[0]['event_id']}: {errors[0]['error']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
