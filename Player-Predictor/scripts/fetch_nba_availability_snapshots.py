#!/usr/bin/env python3
"""Fetch or normalize NBA pregame availability snapshots for v9.5."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://www.rotowire.com/basketball/injury-report.php"
STATUS_PROBABILITY = {
    "out": 1.0,
    "doubtful": 0.8,
    "questionable": 0.45,
    "probable": 0.15,
    "available": 0.0,
    "active": 0.0,
}


def _normalize_name(value: object) -> str:
    return str(value).strip().replace(" ", "_")


def _normalize_status(value: object) -> str:
    text = str(value).strip().lower()
    for token in STATUS_PROBABILITY:
        if token in text:
            return token
    return text or "unknown"


def _read_source(path: Path | None, url: str, timeout: int) -> pd.DataFrame:
    if path:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(path, lines=suffix == ".jsonl")
        return pd.read_csv(path)

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "NBA-Analytics-v9.5/1.0"})
    response.raise_for_status()
    tables = pd.read_html(response.text)
    if not tables:
        raise ValueError(f"no tables found at {url}")
    return max(tables, key=len)


def normalize_availability(frame: pd.DataFrame, snapshot_time: str, game_start_time: str, source: str) -> pd.DataFrame:
    frame = frame.copy()
    lower_map = {str(c).strip().lower(): c for c in frame.columns}
    player_col = lower_map.get("player") or lower_map.get("name")
    team_col = lower_map.get("team")
    status_col = lower_map.get("status") or lower_map.get("injury status")
    if not player_col or not team_col or not status_col:
        raise ValueError(f"availability source needs player/team/status columns, found {list(frame.columns)}")

    out = pd.DataFrame()
    out["snapshot_time"] = snapshot_time
    out["game_start_time"] = game_start_time
    out["date"] = pd.to_datetime(game_start_time, errors="coerce", utc=True).date().isoformat()
    out["team"] = frame[team_col].astype(str).str.upper().str.strip()
    out["player"] = frame[player_col].map(_normalize_name)
    out["status"] = frame[status_col].map(_normalize_status)
    out["out_probability"] = out["status"].map(STATUS_PROBABILITY).fillna(0.35)
    out["availability_confidence"] = out["status"].map({
        "out": 1.0,
        "available": 1.0,
        "active": 1.0,
        "doubtful": 0.85,
        "questionable": 0.75,
        "probable": 0.75,
    }).fillna(0.45)
    out["source"] = source
    if "game_id" in frame.columns:
        out["game_id"] = frame["game_id"]
    if "opponent" in lower_map:
        out["opponent"] = frame[lower_map["opponent"]]
    return out.dropna(subset=["player", "team"]).drop_duplicates(["date", "team", "player"], keep="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch or normalize NBA availability snapshots")
    parser.add_argument("--input", type=Path, help="Optional local CSV/JSON/Parquet source with player/team/status columns")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--snapshot-time", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--game-start-time", required=True, help="UTC ISO lock/start time for this snapshot batch")
    parser.add_argument("--source", default="rotowire_injury_report")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "availability" / "nba" / "latest_nba_availability_snapshots.csv")
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = _read_source(args.input, args.url, args.timeout)
    normalized = normalize_availability(raw, args.snapshot_time, args.game_start_time, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False)
    manifest = {
        "schema": "availability_snapshot_schema_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "rows": int(len(normalized)),
        "output": str(args.output),
        "status_counts": normalized["status"].value_counts().to_dict(),
    }
    (args.output.parent / "latest_availability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
