#!/usr/bin/env python3
"""Shared, sport-agnostic real-image cache for player headshots.

Every sport that has a real headshot source (MLB's Cloudinary-backed
img.mlbstatic.com, NBA's cdn.nba.com, golf's a.espncdn.com) currently
hot-links that remote CDN directly, from every place a player's picture is
shown -- the main board, every parlay leg, every product. A given player
who appears on both the main board and a parlay leg is fetched twice (or
more) by every visitor's browser, and the site has zero control if a CDN
goes down, rate-limits, or changes its URL scheme.

This module stores exactly one real image file per real player id, keyed
by the id already embedded in that sport's own real headshot URL (never a
guessed or synthetic id). A per-sport "collect today's real bettable
player ids" script (see sports/{sport}/scripts/update_*_headshot_cache.py)
supplies (id, source_url) pairs; this module downloads only the ones not
already on disk (or all of them, if force_refresh=True, for the weekly
refresh sweep), and maintains a manifest recording what is cached and
where each image really came from -- never a fabricated entry.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; NBA-Analytics/1.0; +read-only-headshot-cache)"
DEFAULT_TIMEOUT_SECONDS = 20.0

_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class HeadshotEntry:
    """One real player id -> real source image URL, as supplied by a
    per-sport collector. `fallback_url` is optional -- used only if the
    primary URL fails to fetch."""

    id: str
    url: str
    fallback_url: Optional[str] = None


def _safe_filename_stem(entry_id: str) -> str:
    collapsed = _SAFE_ID_PATTERN.sub("_", str(entry_id).strip())
    collapsed = collapsed.replace("..", "_").strip("._") or "unknown"
    return collapsed


def _extension_for(content_type: str, url: str) -> str:
    normalized = str(content_type or "").split(";", 1)[0].strip().lower()
    if normalized in _CONTENT_TYPE_EXTENSIONS:
        return _CONTENT_TYPE_EXTENSIONS[normalized]
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else ".jpg"


def default_fetch(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> tuple[bytes, str]:
    """Real HTTP GET -- returns (body_bytes, content_type). Raises on
    any non-2xx response or network failure; callers decide how to
    handle a failed entry (never silently fabricate image bytes)."""
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "image/*"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
    return body, content_type


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sync_headshot_cache(
    entries: list[HeadshotEntry],
    *,
    cache_dir: Path,
    manifest_path: Path,
    fetch_fn: Callable[[str], tuple[bytes, str]] = default_fetch,
    force_refresh: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
    rate_limit_seconds: float = 0.0,
) -> dict[str, Any]:
    """Downloads exactly one real image per real entry id not already
    cached (or every entry, if force_refresh=True). Never fabricates an
    image for an id whose real fetch failed -- that id is simply left
    absent from the cache/manifest, and callers fall back to the real
    remote URL rather than showing a broken or fake photo.

    Returns {"already_cached": n, "downloaded": n, "failed": [ids],
    "cache_dir": str, "manifest_path": str}.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)

    already_cached = 0
    downloaded = 0
    failed: list[str] = []
    seen_ids: set[str] = set()

    for entry in entries:
        entry_id = str(entry.id).strip()
        if not entry_id or entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)

        stem = _safe_filename_stem(entry_id)
        existing = manifest.get(entry_id)
        existing_path = cache_dir / existing["filename"] if existing else None
        if not force_refresh and existing_path is not None and existing_path.exists():
            already_cached += 1
            continue

        for candidate_url in (entry.url, entry.fallback_url):
            if not candidate_url:
                continue
            try:
                body, content_type = fetch_fn(candidate_url)
                if not body:
                    continue
                extension = _extension_for(content_type, candidate_url)
                filename = f"{stem}{extension}"
                (cache_dir / filename).write_bytes(body)
                manifest[entry_id] = {
                    "filename": filename,
                    "source_url": candidate_url,
                    "content_type": content_type,
                    "fetched_at_utc": _utc_now_iso(),
                }
                downloaded += 1
                break
            except (HTTPError, URLError, TimeoutError, OSError):
                continue
        else:
            failed.append(entry_id)

        if rate_limit_seconds > 0:
            sleep_fn(rate_limit_seconds)

    _write_manifest(manifest_path, manifest)
    return {
        "already_cached": already_cached,
        "downloaded": downloaded,
        "failed": failed,
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
    }


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def cached_relative_path(entry_id: str, *, manifest_path: Path) -> Optional[str]:
    """The manifest-recorded filename for a real cached id, or None if
    that id was never successfully cached. Callers join this onto
    whatever URL prefix their own page serves data/ from."""
    manifest = _load_manifest(manifest_path)
    record = manifest.get(str(entry_id).strip())
    return record["filename"] if record else None
