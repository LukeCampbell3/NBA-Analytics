#!/usr/bin/env python3
"""Real, deduplicated local headshot cache for golf's real bettable players.

Mirrors sports/mlb/scripts/update_mlb_player_headshot_cache.py's design
(see its module docstring for the full rationale). Golf's real headshot
source is ESPN (a.espncdn.com/i/headshots/golf/players/full/{player_id}.png,
where player_id is ESPN's own real athlete id) -- already threaded through
to PgaCandidate.player_headshot_url for every real, priced candidate (the
board's actual "bettable player" universe; top_10 is a pure model ranking
with no market price and is intentionally not cached here).
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

DEFAULT_CACHE_DIR = REPO_ROOT / "sports" / "golf" / "web" / "data" / "headshots"
DEFAULT_MANIFEST_PATH = DEFAULT_CACHE_DIR / "manifest.json"
LOCAL_URL_PREFIX = "data/headshots"

PLAYER_ID_PATTERN = re.compile(r"/headshots/golf/players/full/(\d+)\.png")


def _player_id_from_url(url: str) -> str:
    match = PLAYER_ID_PATTERN.search(str(url or ""))
    return match.group(1) if match else ""


def find_headshot_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (payload.get("candidates") or []) if isinstance(c, dict) and c.get("player_headshot_url")]


def collect_headshot_entries(payloads: list[dict[str, Any]]) -> list[HeadshotEntry]:
    entries: list[HeadshotEntry] = []
    for payload in payloads:
        for record in find_headshot_dicts(payload):
            url = str(record.get("player_headshot_url") or "").strip()
            player_id = _player_id_from_url(url)
            if not player_id or not url:
                continue
            entries.append(HeadshotEntry(id=player_id, url=url))
    return entries


def rewrite_headshot_pointers(payload: dict[str, Any], *, manifest_path: Path) -> int:
    rewritten = 0
    for record in find_headshot_dicts(payload):
        url = str(record.get("player_headshot_url") or "").strip()
        player_id = _player_id_from_url(url)
        if not player_id:
            continue
        filename = cached_relative_path(player_id, manifest_path=manifest_path)
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
    payloads = [json.loads(p.read_text(encoding="utf-8")) if p.exists() else {} for p in daily_predictions_paths]
    entries = collect_headshot_entries(payloads)
    sync_kwargs: dict[str, Any] = {"cache_dir": cache_dir, "manifest_path": manifest_path, "force_refresh": force_refresh}
    if fetch_fn is not None:
        sync_kwargs["fetch_fn"] = fetch_fn
    sync_summary = sync_headshot_cache(entries, **sync_kwargs)

    rewritten_counts: dict[str, int] = {}
    for path, payload in zip(daily_predictions_paths, payloads):
        if not path.exists():
            continue
        rewritten_counts[str(path)] = rewrite_headshot_pointers(payload, manifest_path=manifest_path)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return {"unique_players_seen": len({e.id for e in entries}), "sync": sync_summary, "rewritten": rewritten_counts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-predictions-path", type=Path, default=None, action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = sync_and_rewrite(
        args.daily_predictions_path, cache_dir=args.cache_dir, manifest_path=args.manifest_path,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
