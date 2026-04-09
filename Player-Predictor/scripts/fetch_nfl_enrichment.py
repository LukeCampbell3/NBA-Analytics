#!/usr/bin/env python3
"""
Fetch NFL enrichment tables for modeling from nflverse release assets.

Outputs are written under:
    data copy/raw/nfl_enrichment/season=<YEAR>/

The script mirrors the NBA enrichment fetch pattern:
- season-scoped parquet outputs
- retry-aware remote fetches
- manifest files with row counts and provenance
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "data copy" / "raw" / "nfl_enrichment"
SCRAPED_AT_COL = "scraped_at_utc"
SOURCE_TABLE_COL = "source_table"
SOURCE_URL_COL = "source_url"
POST_GAME_TYPES = {"WC", "DIV", "CON", "SB"}
SEASON_TYPES = {"REG", "POST", "ALL"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_provenance(df: pd.DataFrame, table_name: str, source_url: str) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out[SOURCE_TABLE_COL] = table_name
    out[SOURCE_URL_COL] = source_url
    out[SCRAPED_AT_COL] = utc_now_iso()
    return out


def normalize_season_type(value: str) -> str:
    out = str(value or "REG").strip().upper()
    if out not in SEASON_TYPES:
        raise ValueError(f"Invalid season type: {value!r}. Choose one of {sorted(SEASON_TYPES)}")
    return out


def to_season_type_from_game_type(game_type: str) -> str:
    token = str(game_type or "").strip().upper()
    return "REG" if token == "REG" else "POST"


class NFLEnrichmentScraper:
    def __init__(
        self,
        outdir: Path,
        *,
        season_type: str = "REG",
        sleep_seconds: float = 0.35,
        retries: int = 3,
        overwrite: bool = False,
        include_nextgen: bool = True,
        include_rosters: bool = True,
        include_depth_charts: bool = True,
    ) -> None:
        self.outdir = outdir
        self.season_type = normalize_season_type(season_type)
        self.sleep_seconds = float(max(0.0, sleep_seconds))
        self.retries = int(max(1, retries))
        self.overwrite = bool(overwrite)
        self.include_nextgen = bool(include_nextgen)
        self.include_rosters = bool(include_rosters)
        self.include_depth_charts = bool(include_depth_charts)
        self.source_urls = {
            "player_weekly_all": "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet",
            "player_weekly_by_season": "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.parquet",
            "games": "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet",
            "ngs_passing": "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_passing.parquet",
            "ngs_rushing": "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_rushing.parquet",
            "ngs_receiving": "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_receiving.parquet",
            "roster_by_season": "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.parquet",
            "depth_by_season": "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.parquet",
        }

    def _season_dir(self, season: int) -> Path:
        return self.outdir / f"season={season}"

    def _rate_limit(self) -> None:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

    def _call_with_retries(self, func: Callable[[], pd.DataFrame], label: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                df = func()
                self._rate_limit()
                return df
            except Exception as exc:  # pragma: no cover
                last_error = exc
                wait = min(20.0, max(0.5, self.sleep_seconds) * (2 ** attempt))
                print(f"  retry {attempt}/{self.retries} for {label}: {exc}")
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {label}") from last_error

    def _read_parquet_url(self, url: str, label: str) -> pd.DataFrame:
        return self._call_with_retries(lambda: pd.read_parquet(url), label)

    def _read_parquet_with_fallback(self, urls: list[str], label: str) -> tuple[pd.DataFrame, str]:
        last_error: Exception | None = None
        for url in urls:
            try:
                return self._read_parquet_url(url, f"{label}:{Path(url).name}"), url
            except Exception as exc:
                last_error = exc
                print(f"  fallback for {label}: {exc}")
        raise RuntimeError(f"Unable to load {label} from any known source") from last_error

    def _apply_season_type_filter(self, df: pd.DataFrame, col: str = "season_type") -> pd.DataFrame:
        if self.season_type == "ALL" or col not in df.columns:
            return df
        token = self.season_type.upper()
        return df.loc[df[col].astype(str).str.upper() == token].copy()

    def fetch_player_weekly(self, season: int) -> pd.DataFrame:
        urls = [
            self.source_urls["player_weekly_by_season"].format(season=season),
            self.source_urls["player_weekly_all"],
        ]
        df, source_url = self._read_parquet_with_fallback(urls, f"player_weekly_{season}")
        if "season" in df.columns:
            df = df.loc[pd.to_numeric(df["season"], errors="coerce") == int(season)].copy()
        if "season_type" in df.columns:
            df["season_type"] = df["season_type"].astype(str).str.upper()
            df = self._apply_season_type_filter(df, col="season_type")
        return append_provenance(df, "player_weekly", source_url)

    def fetch_games(self, season: int) -> pd.DataFrame:
        source_url = self.source_urls["games"]
        df = self._read_parquet_url(source_url, f"games_{season}")
        if "season" in df.columns:
            df = df.loc[pd.to_numeric(df["season"], errors="coerce") == int(season)].copy()
        if "game_type" in df.columns:
            game_type = df["game_type"].astype(str).str.upper()
            if self.season_type == "REG":
                df = df.loc[game_type == "REG"].copy()
            elif self.season_type == "POST":
                df = df.loc[game_type.isin(POST_GAME_TYPES)].copy()
            df["season_type"] = game_type.map(to_season_type_from_game_type)
        else:
            df["season_type"] = self.season_type if self.season_type != "ALL" else "REG"
        return append_provenance(df, "games", source_url)

    def fetch_ngs_table(self, season: int, kind: str) -> pd.DataFrame:
        key = f"ngs_{kind}"
        source_url = self.source_urls[key]
        df = self._read_parquet_url(source_url, f"{key}_{season}")
        if "season" in df.columns:
            df = df.loc[pd.to_numeric(df["season"], errors="coerce") == int(season)].copy()
        if "season_type" in df.columns:
            df["season_type"] = df["season_type"].astype(str).str.upper()
            df = self._apply_season_type_filter(df, col="season_type")
        return append_provenance(df, key, source_url)

    def fetch_rosters(self, season: int) -> pd.DataFrame:
        source_url = self.source_urls["roster_by_season"].format(season=season)
        df = self._read_parquet_url(source_url, f"rosters_{season}")
        if "season" in df.columns:
            df = df.loc[pd.to_numeric(df["season"], errors="coerce") == int(season)].copy()
        if "game_type" in df.columns:
            game_type = df["game_type"].astype(str).str.upper()
            if self.season_type == "REG":
                df = df.loc[game_type == "REG"].copy()
            elif self.season_type == "POST":
                df = df.loc[game_type.isin(POST_GAME_TYPES)].copy()
            df["season_type"] = game_type.map(to_season_type_from_game_type)
        return append_provenance(df, "rosters", source_url)

    def fetch_depth_charts(self, season: int) -> pd.DataFrame:
        source_url = self.source_urls["depth_by_season"].format(season=season)
        df = self._read_parquet_url(source_url, f"depth_charts_{season}")
        if "season" in df.columns:
            df = df.loc[pd.to_numeric(df["season"], errors="coerce") == int(season)].copy()
        if "game_type" in df.columns:
            game_type = df["game_type"].astype(str).str.upper()
            if self.season_type == "REG":
                df = df.loc[game_type == "REG"].copy()
            elif self.season_type == "POST":
                df = df.loc[game_type.isin(POST_GAME_TYPES)].copy()
            df["season_type"] = game_type.map(to_season_type_from_game_type)
        return append_provenance(df, "depth_charts", source_url)

    def scrape_season(
        self,
        season: int,
        *,
        max_weeks: int | None = None,
        player_ids: set[str] | None = None,
    ) -> dict:
        season_dir = self._season_dir(season)
        manifest_path = season_dir / "manifest.json"
        if self.overwrite and season_dir.exists():
            shutil.rmtree(season_dir)
        season_dir.mkdir(parents=True, exist_ok=True)

        if manifest_path.exists() and not self.overwrite:
            print(f"Skipping season {season}: existing manifest found at {manifest_path}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        failures: list[dict[str, str]] = []
        tables: dict[str, pd.DataFrame] = {}

        weekly = self.fetch_player_weekly(season)
        if player_ids:
            weekly = weekly.loc[weekly["player_id"].astype(str).isin(player_ids)].copy()
        if max_weeks is not None and "week" in weekly.columns:
            weekly = weekly.loc[pd.to_numeric(weekly["week"], errors="coerce") <= int(max_weeks)].copy()
        tables["player_weekly"] = weekly

        games = self.fetch_games(season)
        if max_weeks is not None and "week" in games.columns:
            games = games.loc[pd.to_numeric(games["week"], errors="coerce") <= int(max_weeks)].copy()
        tables["games"] = games

        if self.include_nextgen:
            for kind in ("passing", "rushing", "receiving"):
                try:
                    ngs = self.fetch_ngs_table(season, kind)
                    if max_weeks is not None and "week" in ngs.columns:
                        ngs = ngs.loc[pd.to_numeric(ngs["week"], errors="coerce") <= int(max_weeks)].copy()
                    tables[f"ngs_{kind}"] = ngs
                except Exception as exc:
                    failures.append({"table": f"ngs_{kind}", "error": str(exc)})
                    print(f"  failed ngs_{kind}: {exc}")

        if self.include_rosters:
            try:
                rosters = self.fetch_rosters(season)
                if max_weeks is not None and "week" in rosters.columns:
                    rosters = rosters.loc[pd.to_numeric(rosters["week"], errors="coerce") <= int(max_weeks)].copy()
                tables["rosters"] = rosters
            except Exception as exc:
                failures.append({"table": "rosters", "error": str(exc)})
                print(f"  failed rosters: {exc}")

        if self.include_depth_charts:
            try:
                depth = self.fetch_depth_charts(season)
                if max_weeks is not None and "week" in depth.columns:
                    depth = depth.loc[pd.to_numeric(depth["week"], errors="coerce") <= int(max_weeks)].copy()
                tables["depth_charts"] = depth
            except Exception as exc:
                failures.append({"table": "depth_charts", "error": str(exc)})
                print(f"  failed depth_charts: {exc}")

        manifest = {
            "season": int(season),
            "season_type": self.season_type,
            "max_weeks": int(max_weeks) if max_weeks is not None else None,
            "player_filter_count": int(len(player_ids)) if player_ids else None,
            "scraped_at_utc": utc_now_iso(),
            "tables_written": {},
            "failures": failures,
        }

        for table_name, df in tables.items():
            path = season_dir / f"{table_name}.parquet"
            df.to_parquet(path, index=False)
            manifest["tables_written"][table_name] = {
                "rows": int(len(df)),
                "path": str(path),
            }

        safe_write_json(manifest_path, manifest)
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NFL enrichment tables for training.")
    parser.add_argument("--seasons", type=int, nargs="+", required=True, help="Season years, e.g. 2023 2024")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--season-type", type=str, default="REG", choices=sorted(SEASON_TYPES), help="REG, POST, or ALL")
    parser.add_argument("--max-weeks", type=int, default=None, help="Optional max week cutoff")
    parser.add_argument("--player-ids", type=str, nargs="*", default=None, help="Optional player_id filters")
    parser.add_argument("--sleep-seconds", type=float, default=0.35, help="Delay between remote requests")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per remote fetch")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing season outputs")
    parser.add_argument("--exclude-nextgen", action="store_true", help="Skip Next Gen Stats tables")
    parser.add_argument("--exclude-rosters", action="store_true", help="Skip roster table")
    parser.add_argument("--exclude-depth-charts", action="store_true", help="Skip depth chart table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scraper = NFLEnrichmentScraper(
        outdir=args.outdir,
        season_type=args.season_type,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        overwrite=args.overwrite,
        include_nextgen=not args.exclude_nextgen,
        include_rosters=not args.exclude_rosters,
        include_depth_charts=not args.exclude_depth_charts,
    )

    player_ids = {str(value) for value in args.player_ids} if args.player_ids else None
    runs = []
    for season in args.seasons:
        print(f"\nFetching NFL enrichment season={season} type={args.season_type} ...")
        manifest = scraper.scrape_season(season, max_weeks=args.max_weeks, player_ids=player_ids)
        runs.append(manifest)

    safe_write_json(
        args.outdir / "manifest_all.json",
        {
            "scraped_at_utc": utc_now_iso(),
            "runs": runs,
        },
    )
    print("\nFinished NFL enrichment fetch.")


if __name__ == "__main__":
    main()
