#!/usr/bin/env python3
"""Real, deduplicated local headshot cache for MLB's real bettable players.

Every real headshot URL already produced anywhere in the MLB pipeline
(export_web_prediction_payload.py for the main board,
enrich_parlay_leg_headshots.py for parlay/legacy-ticket legs) still hot-
links img.mlbstatic.com directly -- a player who appears in both places
gets fetched twice by every visitor, with zero local control if the CDN
is ever slow or unavailable. This script:

  1. Scans one or more already-exported daily_predictions.json payloads
     for every real (person_id, headshot_url) pair already present (the
     real numeric MLB Stats API person_id is parsed straight out of the
     URL itself -- e.g. .../people/624413/headshot/... -- never guessed).
  2. Downloads exactly one real image per real person_id not already
     cached under sports/mlb/web/data/headshots/ (via the shared
     sports.shared.headshots.cache module), or every id when
     --force-refresh is passed (the weekly refresh sweep).
  3. Rewrites each payload's player_headshot_url to the local cached
     path, with the original real remote URL kept as
     player_headshot_fallback_url -- so a viewer whose local copy is
     somehow missing (a race, or a repo not yet caught up) still falls
     back to the real CDN, never a broken image.

Never touches any other key in the payload. A person_id whose real
fetch fails is left pointing at its original real remote URL --
nothing is ever faked.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.shared.headshots.cache import HeadshotEntry, cached_relative_path, sync_headshot_cache  # noqa: E402

DEFAULT_CACHE_DIR = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "headshots"
DEFAULT_MANIFEST_PATH = DEFAULT_CACHE_DIR / "manifest.json"
LOCAL_URL_PREFIX = "data/headshots"

PERSON_ID_PATTERN = re.compile(r"/people/(\d+)/")

# Every real leg/play shape that can carry a real headshot pair, across
# the main board and every parlay product this session's earlier work
# wired headshots into.
_LEG_PATHS = (
    ("parlays", "selected_parlay", "leg_1"),
    ("parlays", "selected_parlay", "leg_2"),
    ("parlays", "selected_parlay", "leg_3"),
    ("parlays", "selected_parlay", "leg_4"),
    ("parlays", "shadow_candidate", "leg_1"),
    ("parlays", "shadow_candidate", "leg_2"),
)


def _person_id_from_url(url: str) -> str:
    match = PERSON_ID_PATTERN.search(str(url or ""))
    return match.group(1) if match else ""


def find_headshot_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Every dict in this payload that carries a real player_headshot_url
    -- the main board's plays, the legacy ticket's legs (folded into the
    board client-side), and PARLAY_POLICY_V2's leg dicts."""
    found: list[dict[str, Any]] = []

    for play in payload.get("plays") or []:
        if isinstance(play, dict) and play.get("player_headshot_url"):
            found.append(play)

    ticket = (payload.get("daily_parlay") or {}).get("selected_ticket")
    if isinstance(ticket, dict):
        for leg in ticket.get("legs") or []:
            if isinstance(leg, dict) and leg.get("player_headshot_url"):
                found.append(leg)

    parlays = payload.get("parlays")
    if isinstance(parlays, dict):
        for parent_key in ("selected_parlay", "shadow_candidate"):
            parent = parlays.get(parent_key)
            if not isinstance(parent, dict):
                continue
            for leg_key in ("leg_1", "leg_2", "leg_3", "leg_4"):
                leg = parent.get(leg_key)
                if isinstance(leg, dict) and leg.get("player_headshot_url"):
                    found.append(leg)

    return found


def collect_headshot_entries(payloads: list[dict[str, Any]]) -> list[HeadshotEntry]:
    entries: list[HeadshotEntry] = []
    for payload in payloads:
        for record in find_headshot_dicts(payload):
            url = str(record.get("player_headshot_url") or "").strip()
            person_id = _person_id_from_url(url)
            if not person_id or not url:
                continue
            fallback = str(record.get("player_headshot_fallback_url") or "").strip() or None
            entries.append(HeadshotEntry(id=person_id, url=url, fallback_url=fallback))
    return entries


def rewrite_headshot_pointers(payload: dict[str, Any], *, manifest_path: Path) -> int:
    """Points every real headshot dict's player_headshot_url at the local
    cached copy when one exists, moving the original real remote URL to
    player_headshot_fallback_url. Returns how many records were
    rewritten."""
    rewritten = 0
    for record in find_headshot_dicts(payload):
        url = str(record.get("player_headshot_url") or "").strip()
        person_id = _person_id_from_url(url)
        if not person_id:
            continue
        filename = cached_relative_path(person_id, manifest_path=manifest_path)
        if not filename:
            continue
        record["player_headshot_fallback_url"] = url
        record["player_headshot_url"] = f"{LOCAL_URL_PREFIX}/{filename}"
        rewritten += 1
    return rewritten


def sync_and_rewrite(
    daily_predictions_paths: list[Path],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    force_refresh: bool = False,
    fetch_fn=None,
) -> dict[str, Any]:
    payloads = []
    for path in daily_predictions_paths:
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            payloads.append({})

    entries = collect_headshot_entries(payloads)
    sync_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "manifest_path": manifest_path,
        "force_refresh": force_refresh,
    }
    if fetch_fn is not None:
        sync_kwargs["fetch_fn"] = fetch_fn
    sync_summary = sync_headshot_cache(entries, **sync_kwargs)

    rewritten_counts: dict[str, int] = {}
    for path, payload in zip(daily_predictions_paths, payloads):
        if not path.exists():
            continue
        rewritten = rewrite_headshot_pointers(payload, manifest_path=manifest_path)
        rewritten_counts[str(path)] = rewritten
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return {
        "unique_players_seen": len({e.id for e in entries}),
        "sync": sync_summary,
        "rewritten": rewritten_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-predictions-path", type=Path, default=None, action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download every real player's image even if already cached (the weekly refresh sweep).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = sync_and_rewrite(
        args.daily_predictions_path,
        cache_dir=args.cache_dir,
        manifest_path=args.manifest_path,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
