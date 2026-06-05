#!/usr/bin/env python3
"""Build true no-vig/CLV-ready market snapshots from historical sportsbook rows."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_odds_quality import add_american_odds_quality, odds_quality_report


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
MARKET_MAP = {
    "player_points": "PTS",
    "player_rebounds": "TRB",
    "player_assists": "AST",
}
REQUIRED_FIELDS = [
    "snapshot_time",
    "book",
    "market",
    "player",
    "line",
    "over_odds",
    "under_odds",
    "no_vig_over",
    "no_vig_under",
    "open_line",
    "current_line",
    "close_line",
    "close_over_odds",
    "close_under_odds",
]


def _resolve(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / text).resolve()


def _american_to_implied(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _no_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    over = _american_to_implied(over_odds)
    under = _american_to_implied(under_odds)
    total = over + under
    if not np.isfinite(total) or total <= 0:
        return 0.5, 0.5
    return over / total, under / total


def _normalize_name(value: object) -> str:
    return str(value).strip().replace(" ", "_")


def _build_name_map(game_logs: Path | None, prop_rows: Path | None) -> dict[str, str]:
    names: set[str] = set()
    if game_logs and _resolve(game_logs).exists():
        logs = pd.read_parquet(_resolve(game_logs)) if _resolve(game_logs).suffix.lower() == ".parquet" else pd.read_csv(_resolve(game_logs))
        col = "PLAYER_NAME" if "PLAYER_NAME" in logs.columns else "player" if "player" in logs.columns else None
        if col:
            names.update(logs[col].dropna().map(_normalize_name).tolist())
    if prop_rows and _resolve(prop_rows).exists():
        rows = pd.read_csv(_resolve(prop_rows), usecols=lambda c: c == "player")
        names.update(rows["player"].dropna().map(_normalize_name).tolist())
    mapping = {}
    for name in names:
        parts = name.split("_")
        if len(parts) >= 2:
            mapping[f"{parts[0][0]}_{parts[-1]}"] = name
    return mapping


def _load_long_sources(source_dir: Path) -> pd.DataFrame:
    source_dir = _resolve(source_dir)
    files = sorted((source_dir / "normalized").glob("player_props_long_*.csv"))
    if files:
        frames = [pd.read_csv(path) for path in files]
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(source_dir / "history_player_props_long.csv")


def _normalize_long(rows: pd.DataFrame, name_map: dict[str, str], require_same_day: bool) -> pd.DataFrame:
    rows = rows.copy()
    rows["market"] = rows["market_key"].map(MARKET_MAP)
    rows = rows[rows["market"].notna()].copy()
    rows["date"] = pd.to_datetime(rows["event_date_et"], errors="coerce").dt.date.astype(str)
    rows["snapshot_time"] = pd.to_datetime(rows["fetched_at_utc"], errors="coerce", utc=True)
    rows["snapshot_date"] = rows["snapshot_time"].dt.date.astype(str)
    rows["player_short"] = rows["player_name_norm"].map(_normalize_name)
    rows["player"] = rows["player_short"].map(name_map).fillna(rows["player_short"])
    rows["line"] = pd.to_numeric(rows["line"], errors="coerce")
    rows["over_odds"] = pd.to_numeric(rows["over_price"], errors="coerce")
    rows["under_odds"] = pd.to_numeric(rows["under_price"], errors="coerce")
    rows = rows.dropna(subset=["date", "snapshot_time", "player", "market", "line", "over_odds", "under_odds"]).copy()
    quality_before = odds_quality_report(rows)
    rows = add_american_odds_quality(rows)
    rows = rows[rows["is_valid_american_odds"]].copy()
    rows["same_day_snapshot"] = rows["snapshot_date"] == rows["date"]
    if require_same_day:
        # Without exact commence time in this source, only same-day snapshots can be treated as pre-lock candidates.
        rows = rows[rows["same_day_snapshot"]].copy()
    rows.attrs["odds_quality_before_filter"] = quality_before
    rows.attrs["odds_quality_after_filter"] = odds_quality_report(rows)
    return rows


def build_snapshots(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if rows.empty:
        return pd.DataFrame(columns=REQUIRED_FIELDS), {"status": "empty_no_complete_same_day_pairs"}
    keys = ["date", "player", "market"]
    rows = rows.sort_values(keys + ["snapshot_time"]).copy()
    first = rows.drop_duplicates(keys, keep="first")
    close = rows.drop_duplicates(keys, keep="last")
    merged = first.merge(
        close[keys + ["line", "over_odds", "under_odds", "snapshot_time"]],
        on=keys,
        how="inner",
        suffixes=("", "_close"),
    )
    no_vig = merged.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    merged["no_vig_over"], merged["no_vig_under"] = zip(*no_vig)
    close_no_vig = merged.apply(lambda r: _no_vig(r["over_odds_close"], r["under_odds_close"]), axis=1)
    merged["close_no_vig_over"], merged["close_no_vig_under"] = zip(*close_no_vig)
    out = pd.DataFrame({
        "snapshot_time": merged["snapshot_time"].astype(str),
        "book": "CoversConsensus",
        "market": merged["market"],
        "player": merged["player"],
        "date": merged["date"],
        "line": merged["line"],
        "over_odds": merged["over_odds"],
        "under_odds": merged["under_odds"],
        "no_vig_over": merged["no_vig_over"],
        "no_vig_under": merged["no_vig_under"],
        "open_line": merged["line"],
        "current_line": merged["line"],
        "close_line": merged["line_close"],
        "close_over_odds": merged["over_odds_close"],
        "close_under_odds": merged["under_odds_close"],
        "close_no_vig_over": merged["close_no_vig_over"],
        "close_no_vig_under": merged["close_no_vig_under"],
        "close_status": np.where(
            merged["snapshot_time"].dt.date.astype(str).eq(merged["date"])
            & merged["snapshot_time_close"].dt.date.astype(str).eq(merged["date"]),
            "same_day_latest_snapshot_proxy_commence_missing",
            "archived_historical_market_not_clv",
        ),
        "source": "covers_historical_consensus",
    })
    report = {
        "status": "built_true_no_vig_same_day_clv_proxy",
        "complete_same_day_rows": int(len(rows)),
        "odds_quality": odds_quality_report(rows),
        "snapshot_rows": int(len(out)),
        "players": int(out["player"].nunique()),
        "markets": sorted(out["market"].unique().tolist()),
        "date_min": str(out["date"].min()),
        "date_max": str(out["date"].max()),
        "clv_limit": "close fields use latest same-day snapshot because exact commence_time_utc is missing in source rows",
        "close_status_counts": out["close_status"].value_counts().to_dict(),
    }
    return out, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical true no-vig/CLV market snapshots for v9.5")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data copy" / "raw" / "market_odds" / "nba")
    parser.add_argument("--game-logs", type=Path, default=ROOT / "data copy" / "raw" / "nba_enrichment" / "season=2026" / "player_game_logs.parquet")
    parser.add_argument("--prop-rows", type=Path, default=ROOT / "model" / "props" / "v9_5_prelock_availability_w050" / "data" / "prop_training_rows.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "market_odds" / "nba" / "historical_true_no_vig_clv_snapshots_v9_5.csv")
    parser.add_argument("--require-same-day", action="store_true", help="Keep only odds snapshots fetched on the event date. Lower coverage but closer to CLV-safe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name_map = _build_name_map(args.game_logs, args.prop_rows)
    raw = _load_long_sources(args.source_dir)
    normalized = _normalize_long(raw, name_map, args.require_same_day)
    snapshots, report = build_snapshots(normalized)
    args.output = _resolve(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(args.output, index=False)
    report = {
        **report,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "raw_rows": int(len(raw)),
        "normalized_complete_rows": int(len(normalized)),
        "source_odds_quality_before_filter": normalized.attrs.get("odds_quality_before_filter", {}),
        "source_odds_quality_after_filter": normalized.attrs.get("odds_quality_after_filter", {}),
        "name_map_size": int(len(name_map)),
    }
    (args.output.parent / "historical_true_no_vig_clv_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
