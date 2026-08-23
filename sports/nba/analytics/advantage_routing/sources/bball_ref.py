"""Real Basketball-Reference source adapter.

WHY THIS SOURCE, AND ITS REAL LIMITS: stats.nba.com (the only source for
touches, post-ups, drives, tracking passing-network data, and shot x/y
coordinates) is unreachable from this environment -- verified directly
(general internet egress works; every live stats.nba.com call times out
completely with zero bytes across multiple endpoints and header
combinations). Basketball-Reference IS reachable and provides real,
OBSERVED data of a coarser kind:

  - a season shooting table broken down by shot distance zone and shot
    type (Dunk/Hook Shot/Jump Shot), including the real "% of makes that
    were assisted" per zone;
  - real per-game play-by-play text, which includes real assist
    attribution ("<shooter> makes <shot description> (assist by
    <passer>)") and real turnovers.

It does NOT have: touches, post-up/drive counts, exact shot x/y
coordinates, corner-3-vs-above-break-3 splits, or any defensive
rotation/response signal. Every value this module returns is labeled
OBSERVED (it is literally what the page reports); anything this system
computes FROM these values (a zone classification, a rate) is labeled
DERIVED or RECONSTRUCTED by the caller, never here.

Caching: every fetched page is written to data/raw/ alongside a
{source, url, retrieved_at, season, parameters, hash} manifest entry
(see cache_manifest.py) so re-running the pipeline never re-fetches the
same page, and so every real number in a player artifact can be traced
back to the exact page it came from.

Politeness: Basketball-Reference asks for <= ~20 requests/minute from a
single client. This module sleeps between requests and caches
aggressively so a re-run costs zero real requests.
"""

from __future__ import annotations

import html as html_module
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import cache_manifest

BASE_URL = "https://www.basketball-reference.com"
USER_AGENT = "Mozilla/5.0 (compatible; advantage-routing-research/1.0)"
REQUEST_DELAY_SECONDS = 2.2
SOURCE_LABEL = "Basketball-Reference (real HTML pages, scraped)"

REPO_ROOT = Path(__file__).resolve().parents[5]
RAW_DIR = REPO_ROOT / "sports" / "nba" / "analytics" / "advantage_routing" / "data" / "raw" / "bball_ref"


class BballRefUnavailable(RuntimeError):
    """Raised when a real page cannot be fetched or cached. Callers must
    treat this as 'no real data available', never fall back to a
    fabricated value."""


def _get(url: str, *, cache_key: str) -> str:
    """Fetches `url`, using the on-disk cache when present. Every real
    network call sleeps REQUEST_DELAY_SECONDS first (politeness); cache
    hits never sleep and never re-fetch."""
    cached = cache_manifest.read_cached_text(RAW_DIR, cache_key)
    if cached is not None:
        return cached

    time.sleep(REQUEST_DELAY_SECONDS)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BballRefUnavailable(f"could not fetch {url}: {exc}") from exc

    cache_manifest.write_cached_text(
        RAW_DIR, cache_key, body,
        source=SOURCE_LABEL, url=url, retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    return body


# ---------------------------------------------------------------------
# Player resolution
# ---------------------------------------------------------------------

def _ascii_fold(value: str) -> str:
    """Strips diacritics for name matching (e.g. "Şengün" -> "Sengun",
    "Dončić" -> "Doncic") -- real bball-ref search-result labels often
    carry the player's real native-spelling diacritics even when the
    caller's own player list uses the plain-ASCII form (matching this
    repo's own Player-Predictor box-score naming), so a literal
    substring match against the raw label silently fails for any such
    player. Mirrors build/build_player.py::_slugify's normalization."""
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def resolve_player_slug(player_name: str) -> Optional[str]:
    """Real search against basketball-reference.com/search -- returns
    the player's real page slug (e.g. "queende01") or None if no real
    match is found. Never guesses a slug algorithmically (bball-ref's
    disambiguation digits make that unreliable)."""
    query = urllib.parse.quote(player_name)
    url = f"{BASE_URL}/search/search.fcgi?search={query}"
    html = _get(url, cache_key=f"search_{player_name.replace(' ', '_')}")

    last_name_key = _ascii_fold(player_name.strip().split()[-1]).lower().replace("-", "")
    for match in re.finditer(r'href="(/players/([a-z])/([a-z0-9]+)\.html)"[^>]*>([^<]+)</a>', html):
        _, _, slug, label = match.groups()
        folded_label = _ascii_fold(html_module.unescape(label)).lower().replace("-", "").replace(".", "").replace(" ", "")
        if last_name_key in folded_label or last_name_key in slug:
            return slug
    return None


# ---------------------------------------------------------------------
# Season shooting-zone table (real, OBSERVED)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ZoneShootingRow:
    zone_label: str  # bball-ref's own label, e.g. "At Rim", "3 to <10 ft"
    fg: int
    fga: int
    fg_pct: Optional[float]
    fg_assisted: Optional[int]
    fg_assisted_pct: Optional[float]


@dataclass(frozen=True)
class ShotTypeRow:
    shot_type: str  # "DUNK" | "HOOK_SHOT" | "JUMP_SHOT"
    fg: int
    fga: int
    fg_pct: Optional[float]
    fg_assisted: Optional[int]
    fg_assisted_pct: Optional[float]


@dataclass(frozen=True)
class SeasonShootingTable:
    player_slug: str
    season_end_year: str
    zones: list[ZoneShootingRow]
    shot_types: list[ShotTypeRow]
    season_fga: int
    season_fg_assisted_pct: Optional[float]
    url: str


def _parse_row(row_html: str) -> Optional[dict]:
    def stat(name: str) -> Optional[str]:
        m = re.search(rf'data-stat="{name}"\s*>([^<]*)</td>', row_html)
        return m.group(1).strip() if m else None

    label_match = re.search(r'data-stat="split_value"\s*><a[^>]*>([^<]+)</a>', row_html) or re.search(r'data-stat="split_value"\s*>([^<]*)</td>', row_html)
    if not label_match:
        return None
    label = html_module.unescape(label_match.group(1).strip())

    def to_int(s: Optional[str]) -> Optional[int]:
        if s is None or s == "":
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def to_pct(s: Optional[str]) -> Optional[float]:
        if s is None or s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    return {
        "label": label,
        "fg": to_int(stat("fg")) or 0,
        "fga": to_int(stat("fga")) or 0,
        "fg_pct": to_pct(stat("fg_pct")),
        "fg_ast": to_int(stat("fg_ast")),
        "fg_ast_pct": to_pct(stat("fg_ast_pct")),
    }


def fetch_season_shooting_table(player_slug: str, season_end_year: str) -> Optional[SeasonShootingTable]:
    url = f"{BASE_URL}/players/{player_slug[0]}/{player_slug}/shooting/{season_end_year}"
    try:
        html = _get(url, cache_key=f"shooting_{player_slug}_{season_end_year}")
    except BballRefUnavailable:
        return None

    zone_labels = {
        "At Rim": "0-3", "3 to <10 ft": "3-10", "10 to <16 ft": "10-16", "16 ft to <3-pt": "16-3P", "3-pt": "3P",
    }
    shot_type_labels = {"Dunk": "DUNK", "Hook Shot": "HOOK_SHOT", "Jump Shot": "JUMP_SHOT"}

    rows = re.findall(r"<tr[^>]*>.*?</tr>", html, re.DOTALL)
    zones: list[ZoneShootingRow] = []
    shot_types: list[ShotTypeRow] = []
    season_fga = 0
    season_fg_assisted_pct: Optional[float] = None

    for row in rows:
        parsed = _parse_row(row)
        if parsed is None:
            continue
        if "Season" == parsed["label"] and "Regular Season" not in row:
            continue
        if parsed["label"] in zone_labels:
            zones.append(ZoneShootingRow(
                zone_label=parsed["label"], fg=parsed["fg"], fga=parsed["fga"],
                fg_pct=parsed["fg_pct"], fg_assisted=parsed["fg_ast"], fg_assisted_pct=parsed["fg_ast_pct"],
            ))
        elif parsed["label"] in shot_type_labels:
            shot_types.append(ShotTypeRow(
                shot_type=shot_type_labels[parsed["label"]], fg=parsed["fg"], fga=parsed["fga"],
                fg_pct=parsed["fg_pct"], fg_assisted=parsed["fg_ast"], fg_assisted_pct=parsed["fg_ast_pct"],
            ))
        elif "Regular Season" in row and parsed["fga"] > 0 and not zones and not shot_types:
            season_fga = parsed["fga"]
            season_fg_assisted_pct = parsed["fg_ast_pct"]

    if not zones and not shot_types:
        return None
    return SeasonShootingTable(
        player_slug=player_slug, season_end_year=season_end_year, zones=zones, shot_types=shot_types,
        season_fga=season_fga, season_fg_assisted_pct=season_fg_assisted_pct, url=url,
    )


# ---------------------------------------------------------------------
# Game log -> real box-score URLs for the season
# ---------------------------------------------------------------------

def fetch_season_game_ids(player_slug: str, season_end_year: str) -> list[str]:
    """Returns real bball-ref game_ids (e.g. "202510220MEM") for every
    game the player actually APPEARED IN this season, in chronological
    order.

    Bball-Reference's gamelog page includes one row per team game all
    season, including games the player was Inactive or Did Not Play for
    -- those rows still link to that game's real boxscore, so a naive
    "find every /boxscores/ link on the page" extraction silently pulls
    in games the player never set foot on the court for. For any player
    with a real recent injury absence, that corrupts the "most recent N
    games" sample this whole pipeline relies on (sources/collect.py):
    the sample can land entirely inside an inactive stretch, producing a
    real but misleadingly-empty recipient network -- not a fabrication,
    but a preventable one. Each table row is checked for the
    Inactive/Did Not Play placeholder cell (data-stat="is_starter" with
    a colspan, real bball-ref markup for a non-appearance) and skipped
    before its boxscore link is collected."""
    url = f"{BASE_URL}/players/{player_slug[0]}/{player_slug}/gamelog/{season_end_year}"
    try:
        html = _get(url, cache_key=f"gamelog_{player_slug}_{season_end_year}")
    except BballRefUnavailable:
        return []
    seen: list[str] = []
    for row in re.finditer(r"<tr[^>]*>.*?</tr>", html, re.DOTALL):
        row_html = row.group(0)
        if "Inactive" in row_html or "Did Not Play" in row_html or "Not With Team" in row_html:
            continue
        m = re.search(r'href="/boxscores/(\d{9}[A-Z]{3})\.html"', row_html)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


# ---------------------------------------------------------------------
# Play-by-play (real assist attribution + turnovers)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RealAssistEvent:
    game_id: str
    passer_slug: str
    passer_label: str
    recipient_slug: str
    recipient_label: str
    shot_description: str  # bball-ref's own text, e.g. "3-pt jump shot from 28 ft"
    shot_distance_ft: Optional[float]
    is_three: bool


@dataclass(frozen=True)
class RealTurnoverEvent:
    game_id: str
    player_slug: str
    player_label: str
    description: str


_ASSIST_RE = re.compile(
    r'href="/players/[a-z]/([a-z0-9]+)\.html">([^<]+)</a>\s*makes\s+([^(<]+?)\s*\(assist by <a[^>]*href="/players/[a-z]/([a-z0-9]+)\.html">([^<]+)</a>\)'
)
_DISTANCE_RE = re.compile(r"from (\d+) ft")
_TURNOVER_RE = re.compile(
    r'Turnover by <a[^>]*href="/players/[a-z]/([a-z0-9]+)\.html">([^<]+)</a>\s*(\([^)<]*\))?', re.IGNORECASE
)


# ---------------------------------------------------------------------
# League-average shooting baseline (real, for expected-pass-value section 13)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class LeagueShootingBaseline:
    season_end_year: str
    fg_pct_rim: Optional[float]
    fg_pct_short_paint: Optional[float]
    fg_pct_midrange: Optional[float]
    fg_pct_long_midrange: Optional[float]
    fg_pct_three: Optional[float]
    freq_rim: Optional[float]
    freq_short_paint: Optional[float]
    freq_midrange: Optional[float]
    freq_long_midrange: Optional[float]
    freq_three: Optional[float]
    url: str


def fetch_league_shooting_baseline(season_end_year: str) -> Optional[LeagueShootingBaseline]:
    """One real, cheap request for the whole league's real average FG%
    by real shot-distance zone this season -- used as the baseline
    "value of a state" in stats/pass_value.py's expected-pass-value
    model (section 13). Real, OBSERVED (a single reported league-average
    row), never estimated."""
    url = f"{BASE_URL}/leagues/NBA_{season_end_year}_shooting.html"
    try:
        html = _get(url, cache_key=f"league_shooting_{season_end_year}")
    except BballRefUnavailable:
        return None

    idx = html.find("League Average")
    if idx == -1:
        return None
    row = html[idx:idx + 2000]

    def stat(name: str) -> Optional[float]:
        m = re.search(rf'data-stat="{name}"[^>]*>([.\d]*)<', row)
        if not m or not m.group(1):
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    return LeagueShootingBaseline(
        season_end_year=season_end_year,
        fg_pct_rim=stat("fg_pct_00_03"),
        fg_pct_short_paint=stat("fg_pct_03_10"),
        fg_pct_midrange=stat("fg_pct_10_16"),
        fg_pct_long_midrange=stat("fg_pct_16_xx"),
        fg_pct_three=stat("fg_pct_fg3a"),
        freq_rim=stat("pct_fga_00_03"),
        freq_short_paint=stat("pct_fga_03_10"),
        freq_midrange=stat("pct_fga_10_16"),
        freq_long_midrange=stat("pct_fga_16_xx"),
        freq_three=stat("pct_fga_fg3a"),
        url=url,
    )


def fetch_game_events(player_slug: str, game_id: str) -> tuple[list[RealAssistEvent], list[RealTurnoverEvent]]:
    """Parses one real game's play-by-play page. Returns EVERY real
    assist/turnover event in the game (not just this player's), so a
    single fetch can serve multiple players' recipient-network
    reconstruction from a shared game."""
    url = f"{BASE_URL}/boxscores/pbp/{game_id}.html"
    try:
        html = _get(url, cache_key=f"pbp_{game_id}")
    except BballRefUnavailable:
        return [], []

    assists: list[RealAssistEvent] = []
    for m in _ASSIST_RE.finditer(html):
        recipient_slug, recipient_label, shot_desc, passer_slug, passer_label = m.groups()
        dist_match = _DISTANCE_RE.search(shot_desc)
        distance = float(dist_match.group(1)) if dist_match else None
        assists.append(RealAssistEvent(
            game_id=game_id, passer_slug=passer_slug, passer_label=passer_label,
            recipient_slug=recipient_slug, recipient_label=recipient_label,
            shot_description=shot_desc.strip(), shot_distance_ft=distance,
            is_three=("3-pt" in shot_desc),
        ))

    turnovers: list[RealTurnoverEvent] = []
    for m in _TURNOVER_RE.finditer(html):
        slug, label, desc = m.groups()
        turnovers.append(RealTurnoverEvent(game_id=game_id, player_slug=slug, player_label=label, description=(desc or "").strip()))

    return assists, turnovers
