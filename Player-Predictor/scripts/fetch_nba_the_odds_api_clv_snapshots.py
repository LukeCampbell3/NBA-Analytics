#!/usr/bin/env python3
"""Fetch NBA player-prop prelock/close snapshots from The Odds API.

This is the production-valid CLV path for v9.6+: historical event odds are
queried at explicit timestamps around game start, then normalized into the same
book-level schema used by the market snapshot sequence builder.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from collect_market_snapshots_v9_6 import append_collection
from collect_market_snapshots_v9_6 import build_sequence
from market_odds_quality import add_american_odds_quality, odds_quality_report


ROOT = Path(__file__).resolve().parents[1]
SPORT_KEY = "basketball_nba"
DEFAULT_OUTDIR = ROOT / "data" / "market_odds" / "nba" / "the_odds_api"
DEFAULT_SEQUENCE_OUTDIR = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence"
DEFAULT_COLLECTION_FILE = DEFAULT_SEQUENCE_OUTDIR / "collected_book_snapshots.csv"
DEFAULT_MARKETS = ["player_points", "player_rebounds", "player_assists"]
MARKET_MAP = {
    "player_points": "PTS",
    "player_rebounds": "TRB",
    "player_assists": "AST",
}


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
    raise RuntimeError("Missing Odds API key. Set THE_ODDS_API_KEY / ODDS_API_KEY or pass --api-key.")


def request_json(url: str, params: dict[str, object]) -> tuple[object, dict[str, str]]:
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"Accept": "application/json", "User-Agent": "NBA-Analytics/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {key.lower(): value for key, value in response.headers.items()}
            return payload, headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 401 and "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN" in body:
            raise RuntimeError(
                "The Odds API historical endpoint is paid-only.\n"
                "Switch to live snapshot collection using the FREE current-odds endpoints:\n"
                "  python Player-Predictor/scripts/collect_nba_live_odds_snapshot.py \\\n"
                "    --snapshot-type prelock --append-collection --rebuild-sequence\n"
                "This uses /v4/sports/{sport}/odds and builds CLV forward without a paid plan."
            ) from exc
        raise RuntimeError(f"The Odds API request failed [{exc.code}] {request.full_url}\n{body}") from exc


def _parse_utc(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _phase_time(commence_time: object, offset_minutes: int) -> str | None:
    parsed = _parse_utc(commence_time)
    if parsed is None:
        return None
    target = parsed.to_pydatetime() - timedelta(minutes=offset_minutes)
    return target.isoformat().replace("+00:00", "Z")


def _event_date_et(commence_time: object) -> str | None:
    parsed = _parse_utc(commence_time)
    if parsed is None:
        return None
    return str(parsed.tz_convert("America/New_York").date())


def extract_data_envelope(payload: object) -> tuple[object, dict[str, object]]:
    if isinstance(payload, dict) and "data" in payload:
        meta = {key: payload.get(key) for key in ["timestamp", "previous_timestamp", "next_timestamp"]}
        return payload.get("data"), meta
    return payload, {}


def normalize_event_payload(payload: dict, requested_snapshot_time: str, snapshot_type: str, source_snapshot_time: str | None = None) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    event_date = _event_date_et(payload.get("commence_time"))
    game_start = payload.get("commence_time")
    for bookmaker in payload.get("bookmakers", []) or []:
        for market in bookmaker.get("markets", []) or []:
            market_key = market.get("key")
            market_code = MARKET_MAP.get(str(market_key))
            if not market_code:
                continue
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
                        "snapshot_time": source_snapshot_time or requested_snapshot_time,
                        "requested_snapshot_time": requested_snapshot_time,
                        "snapshot_date": event_date,
                        "date": event_date,
                        "snapshot_type": snapshot_type,
                        "book": bookmaker.get("title") or bookmaker.get("key"),
                        "book_key": bookmaker.get("key"),
                        "game_id": payload.get("id"),
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
                        "source": "the_odds_api_historical_event_odds",
                        "provider_market_key": market_key,
                        "home_team": payload.get("home_team"),
                        "away_team": payload.get("away_team"),
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


def fetch_historical_events(api_key: str, snapshot_time: str, args: argparse.Namespace) -> tuple[list[dict], dict[str, str]]:
    url = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events"
    params: dict[str, object] = {
        "apiKey": api_key,
        "date": snapshot_time,
        "dateFormat": "iso",
    }
    if args.commence_time_from:
        params["commenceTimeFrom"] = args.commence_time_from
    if args.commence_time_to:
        params["commenceTimeTo"] = args.commence_time_to
    payload, headers = request_json(url, params)
    data, _meta = extract_data_envelope(payload)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected historical events payload: {type(data)!r}")
    if args.event_limit is not None:
        data = data[: args.event_limit]
    return data, headers


def fetch_event_phase(api_key: str, event: dict, phase: str, offset_minutes: int, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    requested_time = _phase_time(event.get("commence_time"), offset_minutes)
    if requested_time is None:
        return pd.DataFrame(), {"event_id": event.get("id"), "phase": phase, "error": "missing_commence_time"}
    url = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/events/{event.get('id')}/odds"
    payload, headers = request_json(
        url,
        {
            "apiKey": api_key,
            "regions": args.regions,
            "markets": ",".join(args.markets),
            "bookmakers": ",".join(args.bookmakers) if args.bookmakers else None,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date": requested_time,
        },
    )
    data, meta = extract_data_envelope(payload)
    if not isinstance(data, dict):
        return pd.DataFrame(), {"event_id": event.get("id"), "phase": phase, "error": f"unexpected_payload_{type(data)!r}"}
    rows = normalize_event_payload(data, requested_time, phase, source_snapshot_time=meta.get("timestamp"))
    rows["api_requested_phase_time"] = requested_time
    rows["api_snapshot_time"] = meta.get("timestamp")
    rows["api_previous_snapshot_time"] = meta.get("previous_timestamp")
    rows["api_next_snapshot_time"] = meta.get("next_timestamp")
    return rows, {"event_id": event.get("id"), "phase": phase, "rows": int(len(rows)), "headers": headers}


def write_outputs(outdir: Path, rows: pd.DataFrame, manifest: dict) -> Path:
    stamp = manifest["snapshot_stamp"]
    norm_dir = outdir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    path = norm_dir / f"nba_player_props_clv_snapshots_{stamp}.csv"
    rows.to_csv(path, index=False)
    rows.to_csv(outdir / "latest_nba_player_props_clv_snapshots.csv", index=False)
    (outdir / "latest_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch true NBA player-prop CLV snapshots from The Odds API")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--snapshot-time", type=str, default=None, help="Historical events discovery timestamp. Defaults to now.")
    parser.add_argument("--commence-time-from", type=str, default=None)
    parser.add_argument("--commence-time-to", type=str, default=None)
    parser.add_argument("--regions", type=str, default="us")
    parser.add_argument("--markets", type=str, default=",".join(DEFAULT_MARKETS))
    parser.add_argument("--bookmakers", type=str, default="draftkings,fanduel,betmgm,caesars,betrivers,fanatics")
    parser.add_argument("--prelock-offset-minutes", type=int, default=15)
    parser.add_argument("--close-offset-minutes", type=int, default=0)
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
    args.bookmakers = [item.strip() for item in str(args.bookmakers).split(",") if item.strip()]
    api_key = resolve_api_key(args.api_key)
    snapshot_time = args.snapshot_time or utc_now_iso().replace("+00:00", "Z")
    events, events_headers = fetch_historical_events(api_key, snapshot_time, args)

    frames: list[pd.DataFrame] = []
    phase_reports: list[dict] = []
    phases = [
        ("prelock", args.prelock_offset_minutes),
        ("close", args.close_offset_minutes),
    ]
    for event_idx, event in enumerate(events, start=1):
        for phase, offset in phases:
            try:
                rows, report = fetch_event_phase(api_key, event, phase, offset, args)
                if not rows.empty:
                    frames.append(rows)
                phase_reports.append(report)
            except Exception as exc:
                phase_reports.append({"event_id": event.get("id"), "phase": phase, "error": str(exc)})
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest = {
        "provider": "the_odds_api",
        "snapshot_stamp": utc_stamp(),
        "created_at": utc_now_iso(),
        "sport": SPORT_KEY,
        "events_discovery_snapshot_time": snapshot_time,
        "event_count": int(len(events)),
        "markets": args.markets,
        "bookmakers": args.bookmakers,
        "prelock_offset_minutes": args.prelock_offset_minutes,
        "close_offset_minutes": args.close_offset_minutes,
        "rows": int(len(rows)),
        "odds_quality": odds_quality_report(rows),
        "events_headers": events_headers,
        "phase_reports": phase_reports,
        "clv_ready": bool(not rows.empty and set(rows.get("snapshot_type", [])) >= {"prelock", "close"}),
    }
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = write_outputs(outdir, rows, manifest)

    collection_report = {}
    if args.append_collection and not rows.empty:
        collection, appended = append_collection(args.collection_file, rows)
        collection_report = {"collection_file": str(args.collection_file), "appended_rows": int(appended), "collected_rows": int(len(collection))}
    sequence_report = {}
    if args.rebuild_sequence:
        inputs = [args.collection_file] if args.append_collection else [output_path]
        sequence_report = build_sequence(inputs, args.sequence_outdir, min_valid_rate=0.98)
    result = {"manifest": manifest, "output_path": str(output_path), "collection_report": collection_report, "sequence_report": sequence_report}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
