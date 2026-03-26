#!/usr/bin/env python3
"""
Fetch richer NBA data for model enrichment.

This scraper uses official NBA endpoints exposed by ``nba_api`` to collect:
- player game logs to enumerate game_ids
- per-game advanced box score splits
- player tracking splits
- scoring, misc, hustle, and four-factor splits
- offensive/defensive matchups
- live play-by-play actions

Outputs are written under ``data copy/raw/nba_enrichment`` by season.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from nba_api.live.nba.endpoints import playbyplay
from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscorefourfactorsv3,
    boxscorehustlev2,
    boxscorematchupsv3,
    boxscoremiscv3,
    boxscoreplayertrackv3,
    boxscorescoringv3,
    playergamelogs,
)


DEFAULT_OUTDIR = Path("data copy/raw/nba_enrichment")
SCRAPED_AT_COL = "scraped_at_utc"
SOURCE_TABLE_COL = "source_table"
SOURCE_URL_COL = "source_url"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def season_to_label(season: int) -> str:
    start_year = season - 1
    return f"{start_year}-{str(season)[-2:]}"


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


def series_or_default(df: pd.DataFrame, column: str, default: int | str = 0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    factory: Callable
    source_url: str
    split_names: tuple[str, ...]
    timeout_multiplier: float = 1.0
    cooldown_seconds: float = 0.0


class NBAEnrichmentScraper:
    def __init__(
        self,
        outdir: Path,
        *,
        season_type: str = "Regular Season",
        per_mode: str = "PerGame",
        sleep_seconds: float = 0.75,
        timeout: int = 30,
        retries: int = 3,
        overwrite: bool = False,
        enabled_endpoints: set[str] | None = None,
        include_playbyplay: bool = True,
    ) -> None:
        self.outdir = outdir
        self.season_type = season_type
        self.per_mode = per_mode
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.retries = retries
        self.overwrite = overwrite
        self.include_playbyplay = include_playbyplay
        self.endpoint_specs = [
            EndpointSpec(
                name="boxscore_advanced",
                factory=boxscoreadvancedv3.BoxScoreAdvancedV3,
                source_url="https://nba-api-sbang.readthedocs.io/en/latest/nba_api/stats/endpoints/boxscoreadvancedv3/",
                split_names=("player", "team"),
                timeout_multiplier=1.5,
            ),
            EndpointSpec(
                name="boxscore_playertrack",
                factory=boxscoreplayertrackv3.BoxScorePlayerTrackV3,
                source_url="https://nba-api-sbang.readthedocs.io/en/latest/nba_api/stats/endpoints/boxscoreplayertrackv3/",
                split_names=("player", "team"),
                timeout_multiplier=1.5,
            ),
            EndpointSpec(
                name="boxscore_matchups",
                factory=boxscorematchupsv3.BoxScoreMatchupsV3,
                source_url="https://nba-api-sbang.readthedocs.io/en/latest/nba_api/stats/endpoints/boxscorematchupsv3/",
                split_names=("matchups",),
                timeout_multiplier=2.0,
                cooldown_seconds=0.5,
            ),
            EndpointSpec(
                name="boxscore_hustle",
                factory=boxscorehustlev2.BoxScoreHustleV2,
                source_url="https://nba-api-sbang.readthedocs.io/en/latest/nba_api/stats/endpoints/boxscorehustlev2/",
                split_names=("player", "team"),
                timeout_multiplier=1.5,
            ),
            EndpointSpec(
                name="boxscore_scoring",
                factory=boxscorescoringv3.BoxScoreScoringV3,
                source_url="https://nba-api-sbang.readthedocs.io/en/latest/nba_api/stats/endpoints/boxscorescoringv3/",
                split_names=("player", "team"),
            ),
            EndpointSpec(
                name="boxscore_misc",
                factory=boxscoremiscv3.BoxScoreMiscV3,
                source_url="https://nba-api-sbang.readthedocs.io/en/latest/nba_api/stats/endpoints/boxscoremiscv3/",
                split_names=("player", "team"),
            ),
            EndpointSpec(
                name="boxscore_fourfactors",
                factory=boxscorefourfactorsv3.BoxScoreFourFactorsV3,
                source_url="https://nba-api-sbang.readthedocs.io/en/latest/nba_api/stats/endpoints/boxscorefourfactorsv3/",
                split_names=("player", "team"),
                timeout_multiplier=2.0,
                cooldown_seconds=0.5,
            ),
        ]
        if enabled_endpoints is not None:
            self.endpoint_specs = [spec for spec in self.endpoint_specs if spec.name in enabled_endpoints]

    def _rate_limit(self) -> None:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

    def _call_with_retries(self, func: Callable[[], object], label: str, *, base_wait: float | None = None) -> object:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                result = func()
                self._rate_limit()
                return result
            except Exception as exc:  # pragma: no cover
                last_error = exc
                root_wait = self.sleep_seconds if base_wait is None else base_wait
                wait = min(20.0, max(0.5, root_wait) * (2 ** attempt))
                print(f"  retry {attempt}/{self.retries} for {label}: {exc}")
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {label}") from last_error

    def _season_dir(self, season: int) -> Path:
        return self.outdir / f"season={season}"

    def _cache_dir(self, season: int) -> Path:
        return self._season_dir(season) / "cache"

    def _cache_path(self, season: int, table_name: str, game_id: str) -> Path:
        return self._cache_dir(season) / table_name / f"{game_id}.parquet"

    def _load_cached_frame(self, season: int, table_name: str, game_id: str) -> pd.DataFrame | None:
        path = self._cache_path(season, table_name, game_id)
        if path.exists() and not self.overwrite:
            return pd.read_parquet(path)
        return None

    def _store_cached_frame(self, season: int, table_name: str, game_id: str, df: pd.DataFrame) -> None:
        path = self._cache_path(season, table_name, game_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def _collect_cached_tables(self, season: int, table_names: list[str]) -> dict[str, pd.DataFrame]:
        collected: dict[str, pd.DataFrame] = {}
        for table_name in table_names:
            table_dir = self._cache_dir(season) / table_name
            if not table_dir.exists():
                continue
            frames = [pd.read_parquet(path) for path in sorted(table_dir.glob("*.parquet"))]
            if frames:
                collected[table_name] = pd.concat(frames, ignore_index=True)
        return collected

    def fetch_player_game_logs(self, season: int) -> pd.DataFrame:
        season_label = season_to_label(season)
        print(f"\nFetching player game logs for {season_label}...")
        logs = self._call_with_retries(
            lambda: playergamelogs.PlayerGameLogs(
                season_nullable=season_label,
                season_type_nullable=self.season_type,
                per_mode_simple_nullable=self.per_mode,
                timeout=self.timeout,
            ).get_data_frames()[0],
            f"playergamelogs:{season_label}",
        )
        logs = append_provenance(
            logs,
            "player_game_logs",
            "https://nba-api-sbang.readthedocs.io/en/latest/nba_api/stats/endpoints/playergamelogs/",
        )
        logs["season"] = season
        return logs

    def _flatten_playbyplay(self, game_id: str) -> pd.DataFrame:
        payload = self._call_with_retries(
            lambda: playbyplay.PlayByPlay(game_id=game_id, timeout=self.timeout).get_dict(),
            f"playbyplay:{game_id}",
            base_wait=max(self.sleep_seconds, 1.0),
        )
        actions = payload.get("game", {}).get("actions", [])
        df = pd.json_normalize(actions)
        if df.empty:
            df = pd.DataFrame({"game_id": [game_id]})
        else:
            df["game_id"] = game_id
        df = append_provenance(
            df,
            "playbyplay_actions",
            "https://nba-api-sbang.readthedocs.io/en/latest/nba_api/live/endpoints/playbyplay/",
        )
        return df

    def _extract_pbp_player_summary(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        if pbp_df.empty or "personId" not in pbp_df.columns:
            return pd.DataFrame()
        df = pbp_df.copy()
        df["personId"] = pd.to_numeric(df["personId"], errors="coerce").fillna(0).astype("int64")
        df = df[df["personId"] > 0]
        if df.empty:
            return pd.DataFrame()
        action_type = series_or_default(df, "actionType", "")
        df["is_turnover"] = (action_type == "turnover").astype(int)
        df["is_foul"] = (action_type == "foul").astype(int)
        df["is_rebound"] = (action_type == "rebound").astype(int)
        if "isFieldGoal" in df.columns:
            df["is_field_goal"] = pd.to_numeric(df["isFieldGoal"], errors="coerce").fillna(0).astype(int)
        else:
            df["is_field_goal"] = 0
        summary = (
            df.groupby(["game_id", "personId"], as_index=False)
            .agg(
                pbp_event_count=("personId", "size"),
                pbp_turnovers=("is_turnover", "sum"),
                pbp_fouls=("is_foul", "sum"),
                pbp_rebounds=("is_rebound", "sum"),
                pbp_field_goal_events=("is_field_goal", "sum"),
                pbp_unique_possessions=("possession", "nunique"),
            )
        )
        return append_provenance(
            summary,
            "playbyplay_player_summary",
            "https://nba-api-sbang.readthedocs.io/en/latest/nba_api/live/endpoints/playbyplay/",
        )

    def _fetch_boxscore_endpoint(self, spec: EndpointSpec, game_id: str) -> list[pd.DataFrame]:
        timeout = max(int(self.timeout * spec.timeout_multiplier), self.timeout)
        obj = self._call_with_retries(
            lambda: spec.factory(game_id=game_id, timeout=timeout),
            f"{spec.name}:{game_id}",
            base_wait=max(self.sleep_seconds, spec.cooldown_seconds, 1.0),
        )
        dfs = obj.get_data_frames()
        frames: list[pd.DataFrame] = []
        for split_name, df in zip(spec.split_names, dfs):
            out = append_provenance(df, f"{spec.name}_{split_name}", spec.source_url)
            out["game_id"] = game_id
            frames.append(out)
        if spec.cooldown_seconds > 0:
            time.sleep(spec.cooldown_seconds)
        return frames

    def scrape_season(
        self,
        season: int,
        *,
        max_games: int | None = None,
        player_ids: set[int] | None = None,
    ) -> dict:
        season_dir = self._season_dir(season)
        manifest_path = season_dir / "manifest.json"
        if self.overwrite and season_dir.exists():
            shutil.rmtree(season_dir)
        season_dir.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() and not self.overwrite:
            print(f"Skipping season {season}: existing manifest found at {manifest_path}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        logs = self.fetch_player_game_logs(season)
        if player_ids:
            logs = logs[logs["PLAYER_ID"].astype(int).isin(player_ids)].copy()
        logs_path = season_dir / "player_game_logs.parquet"
        logs.to_parquet(logs_path, index=False)

        game_ids = sorted(logs["GAME_ID"].astype(str).unique().tolist())
        if max_games is not None:
            game_ids = game_ids[:max_games]
        print(f"Found {len(game_ids)} unique games for season {season}.")

        failures: list[dict[str, str]] = []
        written_tables: list[str] = []
        if self.include_playbyplay:
            written_tables.extend(["playbyplay_actions", "playbyplay_player_summary"])
        for spec in self.endpoint_specs:
            written_tables.extend([f"{spec.name}_{split}" for split in spec.split_names])

        for idx, game_id in enumerate(game_ids, start=1):
            print(f"  [{idx}/{len(game_ids)}] game {game_id}")
            for spec in self.endpoint_specs:
                try:
                    cached_frames = []
                    missing = []
                    for split_name in spec.split_names:
                        table_name = f"{spec.name}_{split_name}"
                        cached = self._load_cached_frame(season, table_name, game_id)
                        if cached is not None:
                            cached_frames.append(cached)
                        else:
                            missing.append(split_name)
                    if len(missing) == 0 and len(cached_frames) == len(spec.split_names):
                        continue
                    frames = self._fetch_boxscore_endpoint(spec, game_id)
                    for split_name, frame in zip(spec.split_names, frames):
                        self._store_cached_frame(season, f"{spec.name}_{split_name}", game_id, frame)
                except Exception as exc:
                    failures.append({"game_id": game_id, "endpoint": spec.name, "error": str(exc)})
                    print(f"    failed {spec.name}: {exc}")
            if self.include_playbyplay:
                try:
                    pbp_cached = self._load_cached_frame(season, "playbyplay_actions", game_id)
                    if pbp_cached is None:
                        pbp_df = self._flatten_playbyplay(game_id)
                        self._store_cached_frame(season, "playbyplay_actions", game_id, pbp_df)
                    else:
                        pbp_df = pbp_cached
                    pbp_summary_cached = self._load_cached_frame(season, "playbyplay_player_summary", game_id)
                    if pbp_summary_cached is None:
                        pbp_summary = self._extract_pbp_player_summary(pbp_df)
                        if not pbp_summary.empty:
                            self._store_cached_frame(season, "playbyplay_player_summary", game_id, pbp_summary)
                except Exception as exc:
                    failures.append({"game_id": game_id, "endpoint": "playbyplay", "error": str(exc)})
                    print(f"    failed playbyplay: {exc}")
            if idx % 25 == 0:
                safe_write_json(
                    season_dir / "progress.json",
                    {
                        "season": season,
                        "completed_games": idx,
                        "total_games": len(game_ids),
                        "last_game_id": game_id,
                        "updated_at_utc": utc_now_iso(),
                        "failure_count": len(failures),
                    },
                )

        manifest = {
            "season": season,
            "season_label": season_to_label(season),
            "scraped_at_utc": utc_now_iso(),
            "season_type": self.season_type,
            "per_mode": self.per_mode,
            "games_requested": len(game_ids),
            "player_logs_rows": int(len(logs)),
            "tables_written": {},
            "failures": failures,
        }

        for table_name, combined in self._collect_cached_tables(season, written_tables).items():
            path = season_dir / f"{table_name}.parquet"
            combined.to_parquet(path, index=False)
            manifest["tables_written"][table_name] = {
                "rows": int(len(combined)),
                "path": str(path),
            }

        safe_write_json(manifest_path, manifest)
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch official NBA enrichment data for training.")
    parser.add_argument("--seasons", type=int, nargs="+", required=True, help="Season end years, e.g. 2025 2026")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--max-games", type=int, default=None, help="Limit games per season for smoke tests")
    parser.add_argument("--player-ids", type=int, nargs="*", default=None, help="Optional player ids to restrict logs")
    parser.add_argument("--season-type", type=str, default="Regular Season", help="NBA season type")
    parser.add_argument("--per-mode", type=str, default="PerGame", help="Per-mode for game logs")
    parser.add_argument("--sleep-seconds", type=float, default=0.75, help="Delay between API calls")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per endpoint")
    parser.add_argument("--overwrite", action="store_true", help="Reserved for future selective overwrite logic")
    parser.add_argument(
        "--endpoints",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of endpoints: advanced playertrack matchups hustle scoring misc fourfactors playbyplay",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint_aliases = {
        "advanced": "boxscore_advanced",
        "playertrack": "boxscore_playertrack",
        "matchups": "boxscore_matchups",
        "hustle": "boxscore_hustle",
        "scoring": "boxscore_scoring",
        "misc": "boxscore_misc",
        "fourfactors": "boxscore_fourfactors",
    }
    enabled_endpoints = None
    if args.endpoints:
        enabled_endpoints = {endpoint_aliases.get(name, name) for name in args.endpoints if name != "playbyplay"}
    include_playbyplay = args.endpoints is None or "playbyplay" in args.endpoints
    scraper = NBAEnrichmentScraper(
        args.outdir,
        season_type=args.season_type,
        per_mode=args.per_mode,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        retries=args.retries,
        overwrite=args.overwrite,
        enabled_endpoints=enabled_endpoints,
        include_playbyplay=include_playbyplay,
    )
    player_ids = set(args.player_ids) if args.player_ids else None
    overall = []
    for season in args.seasons:
        manifest = scraper.scrape_season(season, max_games=args.max_games, player_ids=player_ids)
        overall.append(manifest)
    safe_write_json(args.outdir / "manifest_all.json", {"runs": overall, "scraped_at_utc": utc_now_iso()})
    print("\nFinished scraping.")


if __name__ == "__main__":
    main()
