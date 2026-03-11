"""
build_college_player_data.py

Robots-aware backend ingestion for college player season data.

Phase-1 focus:
- Ingest Sports-Reference CBB player per-game and advanced season tables
- Save raw source tables with provenance
- Build a canonical player-season backend table for downstream features/cards

Compliance:
- Reads and enforces robots.txt rules before every request
- Respects crawl-delay directives (when published)
- Uses conservative fallback delays to avoid aggressive traffic
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


LOGGER = logging.getLogger("college_data_backend")


class RobotsDisallowedError(RuntimeError):
    """Raised when robots.txt disallows a URL."""


def utc_now_iso() -> str:
    """UTC timestamp in ISO8601 with timezone."""
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    """Convert free text into stable slug key."""
    value = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return value or "unknown"


def unique_column_names(columns: List[str]) -> List[str]:
    """Ensure DataFrame column names are unique and snake_case."""
    out: List[str] = []
    seen: Dict[str, int] = {}
    for raw in columns:
        name = re.sub(r"[^a-zA-Z0-9]+", "_", str(raw).strip().lower()).strip("_")
        name = name or "col"
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}_{count + 1}")
    return out


def safe_float(value: object, default: float = 0.0) -> float:
    """Best-effort float conversion."""
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def maybe_numeric(series: pd.Series) -> pd.Series:
    """
    Best-effort numeric coercion while keeping text columns intact.
    - Strips commas and percent signs
    - Returns original series if conversion would mostly fail
    """
    if series.dtype != object:
        return series
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    valid_rate = float(numeric.notna().mean()) if len(numeric) else 0.0
    return numeric if valid_rate >= 0.7 else series


@dataclass
class SourceConfig:
    """Definition for one source domain."""

    name: str
    base_url: str
    min_delay_seconds: float = 3.0


class RobotsPolicyManager:
    """
    Fetches and caches robots.txt policies, then answers allow/delay checks.

    Default behavior is fail-closed when robots cannot be retrieved:
    this keeps the pipeline compliant by skipping unknown policy domains.
    """

    def __init__(self, session: requests.Session, user_agent: str, timeout_seconds: int = 20, strict: bool = True):
        self.session = session
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.strict = strict
        self._parser_cache: Dict[str, RobotFileParser] = {}

    def _domain_key(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _load_parser(self, base_url: str) -> RobotFileParser:
        domain = self._domain_key(base_url)
        if domain in self._parser_cache:
            return self._parser_cache[domain]

        robots_url = f"{domain}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            response = self.session.get(robots_url, timeout=self.timeout_seconds)
            response.raise_for_status()
            parser.parse(response.text.splitlines())
            LOGGER.info("Loaded robots.txt: %s", robots_url)
        except Exception as exc:  # pylint: disable=broad-except
            msg = f"Could not load robots policy for {domain}: {exc}"
            if self.strict:
                raise RuntimeError(msg) from exc
            LOGGER.warning("%s. Falling back to permissive policy.", msg)
            parser.parse([])

        self._parser_cache[domain] = parser
        return parser

    def can_fetch(self, url: str) -> bool:
        """True if robots policy allows this URL for current user-agent."""
        parser = self._load_parser(url)
        return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str, default_delay: float) -> float:
        """Return policy crawl delay if present, otherwise default delay."""
        parser = self._load_parser(url)
        delay = parser.crawl_delay(self.user_agent)
        if delay is None:
            delay = parser.crawl_delay("*")
        return float(delay) if delay is not None else float(default_delay)


class RobotsAwareHttpClient:
    """
    HTTP client with:
    - retries/backoff for transient errors
    - robots permission checks
    - per-domain throttling with crawl-delay compliance
    """

    def __init__(
        self,
        user_agent: str,
        timeout_seconds: int = 25,
        max_retries: int = 4,
        backoff_factor: float = 1.0,
        max_rate_limit_retries: int = 3,
        strict_robots: bool = True,
    ):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_rate_limit_retries = max(0, int(max_rate_limit_retries))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
            respect_retry_after_header=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.robots = RobotsPolicyManager(
            session=self.session,
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
            strict=strict_robots,
        )
        self._last_hit_by_domain: Dict[str, float] = {}

    def _domain(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def get(self, url: str, min_delay_seconds: float) -> str:
        """Fetch URL with robots checks and throttling."""
        if not self.robots.can_fetch(url):
            raise RobotsDisallowedError(f"robots.txt disallows: {url}")

        domain = self._domain(url)
        required_delay = self.robots.crawl_delay(url, default_delay=min_delay_seconds)
        now = time.time()
        last = self._last_hit_by_domain.get(domain)
        if last is not None:
            wait_seconds = max(0.0, required_delay - (now - last))
            if wait_seconds > 0:
                time.sleep(wait_seconds)

        for attempt in range(self.max_rate_limit_retries + 1):
            response = self.session.get(url, timeout=self.timeout_seconds)
            self._last_hit_by_domain[domain] = time.time()

            if response.status_code != 429:
                response.raise_for_status()
                return response.text

            if attempt >= self.max_rate_limit_retries:
                raise RuntimeError(f"Rate limited (429) for URL: {url}")

            retry_after_header = response.headers.get("Retry-After", "").strip()
            retry_after_seconds = safe_float(retry_after_header, default=0.0)
            if retry_after_seconds <= 0:
                retry_after_seconds = required_delay * (2 ** (attempt + 1))
            retry_after_seconds = min(max(retry_after_seconds, required_delay), 120.0)
            LOGGER.warning(
                "Rate limited for %s (attempt %s/%s). Sleeping %.1fs before retry.",
                url,
                attempt + 1,
                self.max_rate_limit_retries,
                retry_after_seconds,
            )
            time.sleep(retry_after_seconds)

        raise RuntimeError(f"Rate limited (429) for URL: {url}")


class SportsReferenceCollegeCollector:
    """Collector for Sports-Reference CBB player season tables."""

    def __init__(self, client: RobotsAwareHttpClient):
        self.client = client
        self.source = SourceConfig(
            name="sports_reference_cbb",
            base_url="https://www.sports-reference.com",
            min_delay_seconds=3.0,
        )

    def _season_table_specs(self, season: int) -> List[Tuple[str, List[str], str]]:
        """
        Returns list of:
        (table_name, candidate_urls, html_table_id)
        """
        base = self.source.base_url.rstrip("/")
        return [
            (
                "players_per_game_raw",
                [
                    f"{base}/cbb/seasons/men/{season}-per-game.html",
                    f"{base}/cbb/seasons/{season}-per-game.html",
                ],
                "players_per_game",
            ),
            (
                "players_advanced_raw",
                [
                    f"{base}/cbb/seasons/men/{season}-advanced.html",
                    f"{base}/cbb/seasons/{season}-advanced.html",
                ],
                "players_advanced",
            ),
        ]

    def _find_table_html(self, html: str, table_id: str) -> str:
        """Find table by id, including tables embedded inside HTML comments."""
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id=table_id)
        if table is not None:
            return str(table)

        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if table_id not in str(comment):
                continue
            commented_soup = BeautifulSoup(str(comment), "html.parser")
            table = commented_soup.find("table", id=table_id)
            if table is not None:
                return str(table)

        raise ValueError(f"Table with id '{table_id}' not found.")

    def _table_to_dataframe(self, table_html: str) -> pd.DataFrame:
        """Parse HTML table into DataFrame."""
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")
        if table is None:
            return pd.DataFrame()

        thead = table.find("thead")
        if thead is None:
            return pd.DataFrame()

        header_rows = thead.find_all("tr")
        if not header_rows:
            return pd.DataFrame()

        header_cells = header_rows[-1].find_all(["th", "td"])
        columns = [cell.get_text(" ", strip=True) for cell in header_cells]

        body = table.find("tbody")
        if body is None:
            return pd.DataFrame(columns=columns)

        rows: List[List[str]] = []
        for tr in body.find_all("tr"):
            if "thead" in (tr.get("class") or []):
                continue
            cells = tr.find_all(["th", "td"])
            row = [cell.get_text(" ", strip=True) for cell in cells]
            if not any(row):
                continue

            if len(row) < len(columns):
                row.extend([""] * (len(columns) - len(row)))
            elif len(row) > len(columns):
                row = row[: len(columns)]
            rows.append(row)

        return pd.DataFrame(rows, columns=columns)

    def _clean_player_table(self, df: pd.DataFrame, season: int, source_url: str, source_table: str) -> pd.DataFrame:
        """Normalize table to canonical shape and attach provenance."""
        if df.empty:
            return df

        # Drop repeated header rows inside tbody.
        first_col = str(df.columns[0])
        df = df[df[first_col].str.lower() != first_col.lower()].copy()
        df.columns = unique_column_names([str(c) for c in df.columns])

        # Common name/team aliases used downstream.
        if "player" in df.columns:
            df["player_name"] = df["player"]
        if "school" in df.columns:
            df["team_name"] = df["school"]

        for col in df.columns:
            df[col] = maybe_numeric(df[col])

        # Canonical keys + provenance.
        df["season"] = season
        df["source_site"] = self.source.name
        df["source_table"] = source_table
        df["source_url"] = source_url
        df["scraped_at_utc"] = utc_now_iso()

        player_name = df.get("player_name", pd.Series(["unknown"] * len(df))).astype(str)
        team_name = df.get("team_name", pd.Series(["unknown"] * len(df))).astype(str)
        df["player_key"] = [f"{slugify(p)}_{season}_{slugify(t)}" for p, t in zip(player_name, team_name)]
        df["team_key"] = [f"{slugify(t)}_{season}" for t in team_name]

        return df.reset_index(drop=True)

    def collect_season(self, season: int) -> Dict[str, pd.DataFrame]:
        """Collect all configured tables for one season."""
        out: Dict[str, pd.DataFrame] = {}
        for table_name, candidate_urls, table_id in self._season_table_specs(season):
            html: Optional[str] = None
            resolved_url: Optional[str] = None
            last_error: Optional[Exception] = None
            for url in candidate_urls:
                LOGGER.info("Fetching %s (%s)", url, table_name)
                try:
                    html = self.client.get(url, min_delay_seconds=self.source.min_delay_seconds)
                    resolved_url = url
                    break
                except Exception as exc:  # pylint: disable=broad-except
                    last_error = exc
                    continue

            if html is None or resolved_url is None:
                if last_error is not None:
                    raise last_error
                raise RuntimeError(f"No URL candidates configured for {table_name} ({season}).")

            table_html = self._find_table_html(html, table_id=table_id)
            raw_df = self._table_to_dataframe(table_html)
            out[table_name] = self._clean_player_table(
                raw_df, season=season, source_url=resolved_url, source_table=table_name
            )
            LOGGER.info("Collected %s rows for %s %s", len(out[table_name]), season, table_name)
        return out

    def _player_directory_letters(self) -> List[str]:
        """Fetch /cbb/players/ and discover letter-directory pages."""
        base = self.source.base_url.rstrip("/")
        index_url = f"{base}/cbb/players/"
        html = self.client.get(index_url, min_delay_seconds=self.source.min_delay_seconds)
        soup = BeautifulSoup(html, "html.parser")
        letter_urls: Set[str] = set()
        for a in soup.select('a[href^="/cbb/players/"]'):
            href = str(a.get("href", "")).strip()
            if re.match(r"^/cbb/players/[a-z]/$", href):
                letter_urls.add(f"{base}{href}")
        return sorted(letter_urls)

    def _player_urls_from_directory_index(self) -> List[str]:
        """
        Fetch /cbb/players/ and extract direct player profile URLs.

        Sports-Reference currently surfaces player links directly on the index page.
        This path is preferred to avoid unnecessary letter-page redirects.
        """
        base = self.source.base_url.rstrip("/")
        index_url = f"{base}/cbb/players/"
        html = self.client.get(index_url, min_delay_seconds=self.source.min_delay_seconds)
        soup = BeautifulSoup(html, "html.parser")
        urls: Set[str] = set()
        for a in soup.select('a[href^="/cbb/players/"]'):
            href = str(a.get("href", "")).strip()
            if re.match(r"^/cbb/players/[a-z0-9\\-]+\\.html$", href) and not href.endswith("-index.html"):
                urls.add(f"{base}{href}")
        return sorted(urls)

    def _player_urls_from_letter_page(self, letter_url: str) -> List[str]:
        """Fetch one letter page and return all player profile URLs."""
        html = self.client.get(letter_url, min_delay_seconds=self.source.min_delay_seconds)
        soup = BeautifulSoup(html, "html.parser")
        base = self.source.base_url.rstrip("/")
        urls: Set[str] = set()
        for a in soup.select('a[href^="/cbb/players/"]'):
            href = str(a.get("href", "")).strip()
            if re.match(r"^/cbb/players/[a-z0-9\\-]+\\.html$", href) and not href.endswith("-index.html"):
                urls.add(f"{base}{href}")
        return sorted(urls)

    def _extract_player_name_from_page(self, html: str, fallback_url: str) -> str:
        """Read player name from page header with URL slug fallback."""
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        if h1:
            txt = h1.get_text(" ", strip=True)
            if txt:
                return txt
        slug = Path(urlparse(fallback_url).path).name.replace(".html", "")
        slug = re.sub(r"-\\d+$", "", slug)
        return " ".join(w.capitalize() for w in slug.split("-")) or "Unknown"

    def _clean_player_page_table(
        self,
        df: pd.DataFrame,
        player_name: str,
        source_url: str,
        source_table: str,
        seasons_filter: Set[int],
    ) -> pd.DataFrame:
        """Normalize per-player history table to player-season rows."""
        if df.empty:
            return df

        df.columns = unique_column_names([str(c) for c in df.columns])
        if "season" not in df.columns:
            return pd.DataFrame()

        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df = df[df["season"].notna()].copy()
        if df.empty:
            return df
        df["season"] = df["season"].astype(int)
        df = df[df["season"].isin(seasons_filter)].copy()
        if df.empty:
            return df

        if "school" in df.columns:
            df["team_name"] = df["school"]
        else:
            df["team_name"] = "UNKNOWN"
        df["player_name"] = player_name

        for col in df.columns:
            df[col] = maybe_numeric(df[col])

        df["source_site"] = self.source.name
        df["source_table"] = source_table
        df["source_url"] = source_url
        df["scraped_at_utc"] = utc_now_iso()
        df["player_key"] = [f"{slugify(player_name)}_{int(s)}_{slugify(t)}" for s, t in zip(df["season"], df["team_name"])]
        df["team_key"] = [f"{slugify(t)}_{int(s)}" for s, t in zip(df["season"], df["team_name"])]
        return df.reset_index(drop=True)

    def collect_from_player_pages(
        self,
        seasons: List[int],
        max_player_pages: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fallback collector when season-level player tables are unavailable.
        Crawls player profile pages and extracts rows for requested seasons.
        """
        season_set = set(seasons)
        per_game_rows: List[pd.DataFrame] = []
        advanced_rows: List[pd.DataFrame] = []

        player_urls = self._player_urls_from_directory_index()
        if not player_urls:
            letter_urls = self._player_directory_letters()
            for letter_url in letter_urls:
                player_urls.extend(self._player_urls_from_letter_page(letter_url))
                if max_player_pages is not None and len(player_urls) >= max_player_pages:
                    break

        if max_player_pages is not None and len(player_urls) > max_player_pages:
            player_urls = player_urls[:max_player_pages]

        LOGGER.info("Player-page fallback discovered %s profiles", len(player_urls))

        for idx, player_url in enumerate(player_urls, start=1):
            try:
                html = self.client.get(player_url, min_delay_seconds=self.source.min_delay_seconds)
                player_name = self._extract_player_name_from_page(html, fallback_url=player_url)

                pg_html = self._find_table_html(html, "players_per_game")
                pg_df = self._table_to_dataframe(pg_html)
                pg_df = self._clean_player_page_table(
                    pg_df,
                    player_name=player_name,
                    source_url=player_url,
                    source_table="players_per_game_raw",
                    seasons_filter=season_set,
                )
                if not pg_df.empty:
                    per_game_rows.append(pg_df)

                try:
                    adv_html = self._find_table_html(html, "players_advanced")
                    adv_df = self._table_to_dataframe(adv_html)
                    adv_df = self._clean_player_page_table(
                        adv_df,
                        player_name=player_name,
                        source_url=player_url,
                        source_table="players_advanced_raw",
                        seasons_filter=season_set,
                    )
                    if not adv_df.empty:
                        advanced_rows.append(adv_df)
                except Exception:
                    # Advanced table is optional for some historical pages.
                    pass

                if idx % 250 == 0:
                    LOGGER.info("Processed %s player pages...", idx)
            except RobotsDisallowedError:
                LOGGER.warning("Robots disallowed player page: %s", player_url)
                continue
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.debug("Failed player page %s: %s", player_url, exc)
                continue

        return {
            "players_per_game_raw": pd.concat(per_game_rows, ignore_index=True) if per_game_rows else pd.DataFrame(),
            "players_advanced_raw": pd.concat(advanced_rows, ignore_index=True) if advanced_rows else pd.DataFrame(),
        }


def merge_player_tables(
    per_game: pd.DataFrame,
    advanced: pd.DataFrame,
) -> pd.DataFrame:
    """Build canonical players_season table from per-game + advanced raw tables."""
    if per_game.empty and advanced.empty:
        return pd.DataFrame()

    merge_keys = ["player_key", "season", "player_name", "team_name", "team_key"]
    keep_keys = [k for k in merge_keys if (k in per_game.columns) or (k in advanced.columns)]

    # Keep source columns and all stats; avoid collisions with suffixes.
    pg = per_game.copy()
    adv = advanced.copy()

    canonical = pg.merge(
        adv,
        how="outer",
        on=keep_keys,
        suffixes=("_per_game", "_advanced"),
    )

    # Prefer explicit class/position names without suffix if only one exists.
    for base_col in ["class", "pos", "conf", "g", "mp"]:
        c1 = f"{base_col}_per_game"
        c2 = f"{base_col}_advanced"
        if c1 in canonical.columns and c2 in canonical.columns:
            canonical[base_col] = canonical[c1].combine_first(canonical[c2])
        elif c1 in canonical.columns:
            canonical[base_col] = canonical[c1]
        elif c2 in canonical.columns:
            canonical[base_col] = canonical[c2]

    canonical["player_key"] = canonical["player_key"].astype(str)
    canonical["season"] = pd.to_numeric(canonical["season"], errors="coerce").astype("Int64")
    canonical["source_tables"] = "players_per_game_raw,players_advanced_raw"
    canonical["built_at_utc"] = utc_now_iso()
    return canonical


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save table to CSV with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline(
    seasons: List[int],
    raw_dir: Path,
    processed_dir: Path,
    user_agent: str,
    strict_robots: bool,
    collection_mode: str,
    max_player_pages: Optional[int],
) -> Dict[str, object]:
    """Execute end-to-end collection and canonical player backend build."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    client = RobotsAwareHttpClient(
        user_agent=user_agent,
        strict_robots=strict_robots,
    )
    collector = SportsReferenceCollegeCollector(client)

    per_game_frames: List[pd.DataFrame] = []
    advanced_frames: List[pd.DataFrame] = []
    errors: List[Dict[str, str]] = []
    mode_used = collection_mode

    if collection_mode in ("season_pages", "auto"):
        season_success = False
        for season in seasons:
            try:
                season_tables = collector.collect_season(season)
            except RobotsDisallowedError as exc:
                msg = f"Season {season} skipped due to robots policy: {exc}"
                LOGGER.warning(msg)
                errors.append({"season": str(season), "error": msg})
                continue
            except Exception as exc:  # pylint: disable=broad-except
                msg = f"Season {season} collection failed: {exc}"
                LOGGER.exception(msg)
                errors.append({"season": str(season), "error": msg})
                continue

            season_success = True
            pg = season_tables.get("players_per_game_raw", pd.DataFrame())
            adv = season_tables.get("players_advanced_raw", pd.DataFrame())
            per_game_frames.append(pg)
            advanced_frames.append(adv)

            save_table(pg, raw_dir / "players_per_game_raw" / f"players_per_game_{season}.csv")
            save_table(adv, raw_dir / "players_advanced_raw" / f"players_advanced_{season}.csv")

        if collection_mode == "auto" and not season_success:
            LOGGER.warning("Season-page mode produced no data; falling back to player-page mode.")
            mode_used = "player_pages"
            try:
                fallback_tables = collector.collect_from_player_pages(seasons=seasons, max_player_pages=max_player_pages)
                per_game_frames.append(fallback_tables.get("players_per_game_raw", pd.DataFrame()))
                advanced_frames.append(fallback_tables.get("players_advanced_raw", pd.DataFrame()))
            except Exception as exc:  # pylint: disable=broad-except
                msg = f"Player-page fallback failed: {exc}"
                LOGGER.exception(msg)
                errors.append({"season": "all", "error": msg})

    elif collection_mode == "player_pages":
        mode_used = "player_pages"
        try:
            fallback_tables = collector.collect_from_player_pages(seasons=seasons, max_player_pages=max_player_pages)
            per_game_frames.append(fallback_tables.get("players_per_game_raw", pd.DataFrame()))
            advanced_frames.append(fallback_tables.get("players_advanced_raw", pd.DataFrame()))
        except Exception as exc:  # pylint: disable=broad-except
            msg = f"Player-page collection failed: {exc}"
            LOGGER.exception(msg)
            errors.append({"season": "all", "error": msg})
    else:
        raise ValueError(f"Unsupported collection_mode: {collection_mode}")

    per_game_all = pd.concat(per_game_frames, ignore_index=True) if per_game_frames else pd.DataFrame()
    advanced_all = pd.concat(advanced_frames, ignore_index=True) if advanced_frames else pd.DataFrame()
    canonical_players = merge_player_tables(per_game_all, advanced_all)

    save_table(per_game_all, raw_dir / "players_per_game_raw.csv")
    save_table(advanced_all, raw_dir / "players_advanced_raw.csv")
    save_table(canonical_players, processed_dir / "players_season.csv")

    summary = {
        "run_at_utc": utc_now_iso(),
        "seasons_requested": seasons,
        "rows_per_game": int(len(per_game_all)),
        "rows_advanced": int(len(advanced_all)),
        "rows_players_season": int(len(canonical_players)),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "errors": errors,
        "strict_robots": bool(strict_robots),
        "user_agent": user_agent,
        "collection_mode_requested": collection_mode,
        "collection_mode_used": mode_used,
        "max_player_pages": max_player_pages,
    }
    with open(processed_dir / "build_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_seasons(start: int, end: int) -> List[int]:
    """Inclusive season range."""
    if end < start:
        raise ValueError("--season-end must be >= --season-start")
    return list(range(start, end + 1))


def configure_logging(verbose: bool) -> None:
    """Set root logging level/format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build college player backend data (robots-aware, compliant)."
    )
    parser.add_argument("--season-start", type=int, default=2021, help="First season year (inclusive).")
    parser.add_argument("--season-end", type=int, default=2025, help="Last season year (inclusive).")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/college"),
        help="Output directory for raw source tables.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/college"),
        help="Output directory for canonical processed tables.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="NBAAnalyticsCollegePipeline/1.0 (+local research use)",
        help="HTTP user agent used for source requests.",
    )
    parser.add_argument(
        "--non-strict-robots",
        action="store_true",
        help="Allow permissive fallback if robots.txt cannot be retrieved.",
    )
    parser.add_argument(
        "--collection-mode",
        choices=["auto", "season_pages", "player_pages"],
        default="auto",
        help="Data collection strategy. 'auto' tries season pages then falls back to player pages.",
    )
    parser.add_argument(
        "--max-player-pages",
        type=int,
        default=None,
        help="Optional cap for player-page mode (useful for smoke tests).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    args = parser.parse_args()

    configure_logging(args.verbose)
    seasons = parse_seasons(args.season_start, args.season_end)

    try:
        summary = run_pipeline(
            seasons=seasons,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            user_agent=args.user_agent,
            strict_robots=not args.non_strict_robots,
            collection_mode=args.collection_mode,
            max_player_pages=args.max_player_pages,
        )
        LOGGER.info("Build complete: %s", summary)
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("College backend build failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
