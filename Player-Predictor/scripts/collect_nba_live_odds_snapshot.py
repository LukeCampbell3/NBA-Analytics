#!/usr/bin/env python3
"""Collect live NBA player-prop odds snapshots from The Odds API free endpoints.

This replaces the paid historical CLV path with forward-looking collection:
  - Uses /v4/sports/{sport}/odds (free) instead of /v4/historical/... (paid)
  - Stores each snapshot append-only with snapshot_type labels
  - Builds CLV by comparing prelock vs close snapshots over time

Run on a schedule (T-6h, T-90m, T-15m, T-2m before game start) to build
a valid CLV dataset without requiring a paid plan.

Usage:
  python collect_nba_live_odds_snapshot.py --snapshot-type prelock
  python collect_nba_live_odds_snapshot.py --snapshot-type close --rebuild-sequence
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from collect_market_snapshots_v9_6 import append_collection, build_sequence
from market_odds_quality import add_american_odds_quality, odds_quality_report


ROOT = Path(__file__).resolve().parents[1]
SPORT_KEY = "basketball_nba"
DEFAULT_OUTDIR = ROOT / "data" / "market_odds" / "nba" / "the_odds_api_live"
DEFAULT_SEQUENCE_OUTDIR = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence"
DEFAULT_COLLECTION_FILE = DEFAULT_SEQUENCE_OUTDIR / "collected_book_snapshots.csv"
DEFAULT_MARKETS = [
    "player_points", "player_rebounds", "player_assists",
    "player_threes", "player_steals", "player_blocks", "player_turnovers",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
]
MARKET_MAP = {
    "player_points": "PTS",
    "player_rebounds": "TRB",
    "player_assists": "AST",
    "player_threes": "3PM",
    "player_steals": "STL",
    "player_blocks": "BLK",
    "player_turnovers": "TOV",
    "player_points_rebounds_assists": "PRA",
    "player_points_rebounds": "PR",
    "player_points_assists": "PA",
    "player_rebounds_assists": "RA",
    "player_double_double": "DD",
}
VALID_SNAPSHOT_TYPES = ("open_like", "intraday", "injury_sensitive", "prelock", "close")

# The Odds API free endpoints
ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
EVENTS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events"
EVENT_ODDS_URL_TEMPLATE = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{{event_id}}/odds"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_player_name(value: object) -> str:
    out = str(value or "").strip()
    for old, new in [(" ", "_"), (".", ""), ("'", ""), (",", ""), ("/", "-"), ("\\", "-"), (":", "")]:
        out = out.replace(old, new)
    return out


def resolve_api_key(explicit_key: str | None) -> str:
    if explicit_key:
        return explicit_key
    for key in ("THE_ODDS_API_KEY", "ODDS_API_KEY"):
        value = os.getenv(key)
        if value:
            return value
    for name in (".env.local", ".env", "config.local.yaml", "config.yaml"):
        candidate = ROOT.parent / name
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for key in ("THE_ODDS_API_KEY", "ODDS_API_KEY"):
                if stripped.startswith(f"{key}=") or stripped.startswith(f"{key}:"):
                    return stripped.split("=", 1)[-1].split(":", 1)[-1].strip().strip('"').strip("'")
    # Also check yaml-style api_key under odds_api section
    for name in ("config.local.yaml", "config.yaml"):
        candidate = ROOT.parent / name
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("api_key:"):
                val = stripped.split(":", 1)[-1].strip().strip('"').strip("'")
                if val and val != "paste-your-odds-api-key-here":
                    return val
    raise RuntimeError(
        "Missing Odds API key. Set THE_ODDS_API_KEY env var, pass --api-key, "
        "or add it to config.local.yaml."
    )


def request_json(url: str, params: dict[str, object]) -> tuple[object, dict[str, str]]:
    """Make a GET request and return (parsed_json, response_headers)."""
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}
    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}"
    request = urllib.request.Request(
        full_url,
        headers={"Accept": "application/json", "User-Agent": "NBA-Analytics/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {key.lower(): value for key, value in response.headers.items()}
            return payload, headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # Provide clear guidance for the historical-endpoint error
        if exc.code == 401 and "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN" in body:
            raise RuntimeError(
                "The Odds API historical endpoint is paid-only. "
                "This script uses live/current endpoints only. "
                "If you see this error, check that the URL does not contain '/historical/'."
            ) from exc
        raise RuntimeError(
            f"The Odds API request failed [{exc.code}] {full_url}\n{body}"
        ) from exc


def _parse_utc(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _event_date_et(commence_time: object) -> str | None:
    parsed = _parse_utc(commence_time)
    if parsed is None:
        return None
    return str(parsed.tz_convert("America/New_York").date())


def fetch_current_events(api_key: str, args: argparse.Namespace) -> tuple[list[dict], dict[str, str]]:
    """Fetch upcoming/live events using the FREE /v4/sports/{sport}/events endpoint."""
    params: dict[str, object] = {
        "apiKey": api_key,
        "dateFormat": "iso",
    }
    if args.commence_time_from:
        params["commenceTimeFrom"] = args.commence_time_from
    if args.commence_time_to:
        params["commenceTimeTo"] = args.commence_time_to
    payload, headers = request_json(EVENTS_URL, params)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected events payload type: {type(payload)!r}")
    if args.event_limit is not None:
        payload = payload[: args.event_limit]
    return payload, headers


def fetch_event_odds(
    api_key: str, event_id: str, args: argparse.Namespace
) -> tuple[object, dict[str, str]]:
    """Fetch odds for a single event using the FREE per-event odds endpoint."""
    url = EVENT_ODDS_URL_TEMPLATE.format(event_id=event_id)
    params: dict[str, object] = {
        "apiKey": api_key,
        "regions": args.regions,
        "markets": ",".join(args.markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if args.bookmakers:
        params["bookmakers"] = ",".join(args.bookmakers)
    return request_json(url, params)


def fetch_all_odds(api_key: str, args: argparse.Namespace) -> tuple[object, dict[str, str]]:
    """Fetch odds for all upcoming events using the FREE bulk odds endpoint.
    
    Note: The bulk /v4/sports/{sport}/odds endpoint doesn't support player props.
    This function is kept for compatibility but will likely return only h2h markets.
    For player props, use per-event mode instead.
    """
    params: dict[str, object] = {
        "apiKey": api_key,
        "regions": args.regions,
        "markets": ",".join(args.markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if args.bookmakers:
        params["bookmakers"] = ",".join(args.bookmakers)
    if args.commence_time_from:
        params["commenceTimeFrom"] = args.commence_time_from
    if args.commence_time_to:
        params["commenceTimeTo"] = args.commence_time_to
    return request_json(ODDS_URL, params)



def normalize_event_odds(
    event_payload: dict,
    collection_time: str,
    snapshot_type: str,
) -> pd.DataFrame:
    """Normalize a single event's odds payload into the standard book-level schema."""
    records: list[dict[str, object]] = []
    event_date = _event_date_et(event_payload.get("commence_time"))
    game_start = event_payload.get("commence_time")
    event_id = event_payload.get("id")

    for bookmaker in event_payload.get("bookmakers", []) or []:
        book_key = bookmaker.get("key")
        book_title = bookmaker.get("title") or book_key
        for market in bookmaker.get("markets", []) or []:
            market_key = market.get("key")
            market_code = MARKET_MAP.get(str(market_key))
            if not market_code:
                continue
            last_update = market.get("last_update")
            grouped: dict[tuple[str, float], dict[str, object]] = {}
            for outcome in market.get("outcomes", []) or []:
                side = str(outcome.get("name", "")).strip().lower()
                player_raw = outcome.get("description") or outcome.get("participant")
                point = outcome.get("point")
                if side not in {"over", "under"} or player_raw is None or point is None:
                    continue
                try:
                    line = float(point)
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                row = grouped.setdefault(
                    (str(player_raw), line),
                    {
                        "snapshot_time": collection_time,
                        "requested_snapshot_time": collection_time,
                        "snapshot_date": event_date,
                        "date": event_date,
                        "snapshot_type": snapshot_type,
                        "book": book_title,
                        "book_key": book_key,
                        "game_id": event_id,
                        "player_id": "",
                        "player": normalize_player_name(player_raw),
                        "player_raw": str(player_raw),
                        "market": market_code,
                        "line": line,
                        "over_odds": np.nan,
                        "under_odds": np.nan,
                        "game_start_time": game_start,
                        "team": "",
                        "opponent": "",
                        "source": "the_odds_api_live",
                        "provider_market_key": market_key,
                        "home_team": event_payload.get("home_team"),
                        "away_team": event_payload.get("away_team"),
                        "book_last_update": last_update,
                    },
                )
                if side == "over":
                    row["over_odds"] = price
                elif side == "under":
                    row["under_odds"] = price
            records.extend(grouped.values())

    if not records:
        return pd.DataFrame()
    rows = pd.DataFrame(records)
    rows = add_american_odds_quality(rows)
    return rows[rows["is_valid_american_odds"]].copy()


def collect_bulk_snapshot(api_key: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Collect odds for all matching events in one bulk call (uses 1 API request)."""
    collection_time = utc_now_iso()
    payload, headers = fetch_all_odds(api_key, args)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected bulk odds payload type: {type(payload)!r}")

    frames: list[pd.DataFrame] = []
    for event in payload:
        rows = normalize_event_odds(event, collection_time, args.snapshot_type)
        if not rows.empty:
            frames.append(rows)

    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    quota_remaining = headers.get("x-requests-remaining", "unknown")
    quota_used = headers.get("x-requests-used", "unknown")

    meta = {
        "mode": "bulk",
        "collection_time": collection_time,
        "events_in_response": len(payload),
        "rows_collected": int(len(all_rows)),
        "api_requests_remaining": quota_remaining,
        "api_requests_used": quota_used,
    }
    return all_rows, meta


def collect_per_event_snapshot(
    api_key: str, events: list[dict], args: argparse.Namespace
) -> tuple[pd.DataFrame, dict]:
    """Collect odds event-by-event (uses N API requests but allows filtering)."""
    collection_time = utc_now_iso()
    frames: list[pd.DataFrame] = []
    event_reports: list[dict] = []
    last_headers: dict[str, str] = {}

    for idx, event in enumerate(events, start=1):
        event_id = event.get("id")
        if not event_id:
            continue
        try:
            payload, headers = fetch_event_odds(api_key, event_id, args)
            last_headers = headers
            if isinstance(payload, dict):
                rows = normalize_event_odds(payload, collection_time, args.snapshot_type)
                if not rows.empty:
                    frames.append(rows)
                event_reports.append({"event_id": event_id, "rows": int(len(rows))})
            else:
                event_reports.append({"event_id": event_id, "error": f"unexpected_type_{type(payload)!r}"})
        except Exception as exc:
            event_reports.append({"event_id": event_id, "error": str(exc)})

        if args.sleep_seconds > 0 and idx < len(events):
            time.sleep(args.sleep_seconds)

    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta = {
        "mode": "per_event",
        "collection_time": collection_time,
        "events_queried": len(events),
        "events_with_rows": sum(1 for r in event_reports if r.get("rows", 0) > 0),
        "rows_collected": int(len(all_rows)),
        "api_requests_remaining": last_headers.get("x-requests-remaining", "unknown"),
        "api_requests_used": last_headers.get("x-requests-used", "unknown"),
        "event_reports": event_reports,
    }
    return all_rows, meta


def write_snapshot_outputs(outdir: Path, rows: pd.DataFrame, manifest: dict) -> Path:
    """Write snapshot CSV and manifest to disk."""
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = manifest["snapshot_stamp"]
    snapshot_type = manifest.get("snapshot_type", "unknown")

    # Timestamped file
    norm_dir = outdir / "snapshots"
    norm_dir.mkdir(parents=True, exist_ok=True)
    path = norm_dir / f"nba_live_odds_{snapshot_type}_{stamp}.csv"
    rows.to_csv(path, index=False)

    # Latest file (overwritten each run)
    rows.to_csv(outdir / f"latest_{snapshot_type}_snapshot.csv", index=False)
    (outdir / "latest_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect live NBA player-prop odds from The Odds API (free endpoints). "
            "Run repeatedly before game start to build CLV dataset forward."
        )
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--snapshot-type",
        type=str,
        default="prelock",
        choices=VALID_SNAPSHOT_TYPES,
        help="Label for this collection pass (open_like, intraday, injury_sensitive, prelock, close).",
    )
    parser.add_argument("--commence-time-from", type=str, default=None)
    parser.add_argument("--commence-time-to", type=str, default=None)
    parser.add_argument("--regions", type=str, default="us")
    parser.add_argument(
        "--markets",
        type=str,
        default=",".join(DEFAULT_MARKETS),
    )
    parser.add_argument(
        "--bookmakers",
        type=str,
        default="draftkings,fanduel,betmgm,caesars,betrivers,fanatics",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bulk", "per_event"],
        default="per_event",
        help="bulk = 1 API call for all events (h2h only, no player props); per_event = 1 call per event (supports player props).",
    )
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--collection-file", type=Path, default=DEFAULT_COLLECTION_FILE)
    parser.add_argument("--sequence-outdir", type=Path, default=DEFAULT_SEQUENCE_OUTDIR)
    parser.add_argument("--append-collection", action="store_true")
    parser.add_argument("--rebuild-sequence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.markets = [item.strip() for item in str(args.markets).split(",") if item.strip()]
    args.bookmakers = [item.strip() for item in str(args.bookmakers).split(",") if item.strip()] if args.bookmakers else []
    api_key = resolve_api_key(args.api_key)

    print(f"[collect_nba_live_odds_snapshot] snapshot_type={args.snapshot_type} mode={args.mode}")
    print(f"[collect_nba_live_odds_snapshot] markets={args.markets}")
    print(f"[collect_nba_live_odds_snapshot] Using FREE current-odds endpoints (no paid plan required)")

    # Collect snapshot
    if args.mode == "bulk":
        rows, collect_meta = collect_bulk_snapshot(api_key, args)
    else:
        events, _headers = fetch_current_events(api_key, args)
        rows, collect_meta = collect_per_event_snapshot(api_key, events, args)

    # Build manifest
    manifest = {
        "provider": "the_odds_api_live",
        "endpoint": "current_odds_free",
        "snapshot_stamp": utc_stamp(),
        "created_at": utc_now_iso(),
        "sport": SPORT_KEY,
        "snapshot_type": args.snapshot_type,
        "markets": args.markets,
        "bookmakers": args.bookmakers,
        "rows": int(len(rows)),
        "odds_quality": odds_quality_report(rows),
        "collection_meta": collect_meta,
        "clv_ready": False,  # Single snapshot is never CLV-ready alone
        "notes": [
            "This snapshot was collected from the FREE /v4/sports/{sport}/odds endpoint.",
            "CLV requires both prelock and close snapshots for the same events.",
            "Run this script on a schedule to build CLV dataset forward.",
        ],
    }

    # Write outputs
    output_path = write_snapshot_outputs(args.outdir, rows, manifest)
    print(f"[collect_nba_live_odds_snapshot] Wrote {len(rows)} rows -> {output_path}")

    # Append to collection
    collection_report: dict = {}
    if args.append_collection and not rows.empty:
        collection, appended = append_collection(args.collection_file, rows)
        collection_report = {
            "collection_file": str(args.collection_file),
            "appended_rows": int(appended),
            "collected_rows": int(len(collection)),
        }
        print(f"[collect_nba_live_odds_snapshot] Appended {appended} rows to collection ({len(collection)} total)")

    # Rebuild sequence
    sequence_report: dict = {}
    if args.rebuild_sequence:
        inputs = [args.collection_file] if args.append_collection else [output_path]
        sequence_report = build_sequence(inputs, args.sequence_outdir, min_valid_rate=0.98)
        print(f"[collect_nba_live_odds_snapshot] Rebuilt sequence -> {args.sequence_outdir}")

    # Final result
    result = {
        "manifest": manifest,
        "output_path": str(output_path),
        "collection_report": collection_report,
        "sequence_report": sequence_report,
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
