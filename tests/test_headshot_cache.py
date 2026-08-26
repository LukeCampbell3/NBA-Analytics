from __future__ import annotations

from pathlib import Path

import pytest

from sports.shared.headshots.cache import (
    HeadshotEntry,
    cached_relative_path,
    sync_headshot_cache,
)


def _fake_fetch(body_by_url: dict[str, bytes], content_type: str = "image/jpeg"):
    def fetch(url: str):
        if url not in body_by_url:
            raise OSError(f"no fixture body for {url}")
        return body_by_url[url], content_type

    return fetch


def test_sync_downloads_each_new_entry_exactly_once(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    entries = [
        HeadshotEntry(id="624413", url="https://example.com/624413.jpg"),
        HeadshotEntry(id="691718", url="https://example.com/691718.jpg"),
    ]
    fetch = _fake_fetch({e.url: b"real-bytes" for e in entries})

    summary = sync_headshot_cache(entries, cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=fetch)

    assert summary["downloaded"] == 2
    assert summary["already_cached"] == 0
    assert (cache_dir / "624413.jpg").read_bytes() == b"real-bytes"
    assert cached_relative_path("624413", manifest_path=manifest_path) == "624413.jpg"


def test_sync_skips_already_cached_entries_without_refetching(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    entry = HeadshotEntry(id="624413", url="https://example.com/624413.jpg")
    calls: list[str] = []

    def counting_fetch(url: str):
        calls.append(url)
        return b"real-bytes", "image/jpeg"

    sync_headshot_cache([entry], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=counting_fetch)
    summary = sync_headshot_cache([entry], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=counting_fetch)

    assert calls == ["https://example.com/624413.jpg"]  # fetched once, not twice
    assert summary["already_cached"] == 1
    assert summary["downloaded"] == 0


def test_sync_force_refresh_refetches_every_entry(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    entry = HeadshotEntry(id="624413", url="https://example.com/624413.jpg")
    calls: list[str] = []

    def counting_fetch(url: str):
        calls.append(url)
        return b"real-bytes", "image/jpeg"

    sync_headshot_cache([entry], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=counting_fetch)
    sync_headshot_cache([entry], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=counting_fetch, force_refresh=True)

    assert calls == ["https://example.com/624413.jpg", "https://example.com/624413.jpg"]


def test_sync_deduplicates_the_same_id_appearing_more_than_once(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    entries = [
        HeadshotEntry(id="624413", url="https://example.com/624413.jpg"),
        HeadshotEntry(id="624413", url="https://example.com/624413.jpg"),  # same player, appears twice
    ]
    calls: list[str] = []

    def counting_fetch(url: str):
        calls.append(url)
        return b"real-bytes", "image/jpeg"

    summary = sync_headshot_cache(entries, cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=counting_fetch)

    assert len(calls) == 1  # only ONE real fetch for the duplicated id -- exactly one copy stored
    assert summary["downloaded"] == 1


def test_sync_falls_back_to_secondary_url_when_primary_fetch_fails(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    entry = HeadshotEntry(id="624413", url="https://bad.example.com/x.jpg", fallback_url="https://good.example.com/x.jpg")

    def fetch(url: str):
        if url == entry.fallback_url:
            return b"real-bytes", "image/jpeg"
        raise OSError("primary failed")

    summary = sync_headshot_cache([entry], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=fetch)

    assert summary["downloaded"] == 1
    assert summary["failed"] == []


def test_sync_never_fabricates_an_image_for_a_fully_failed_entry(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    entry = HeadshotEntry(id="624413", url="https://bad.example.com/x.jpg")

    def failing_fetch(url: str):
        raise OSError("network down")

    summary = sync_headshot_cache([entry], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=failing_fetch)

    assert summary["failed"] == ["624413"]
    assert summary["downloaded"] == 0
    assert not list(cache_dir.glob("*"))
    assert cached_relative_path("624413", manifest_path=manifest_path) is None


def test_sync_picks_extension_from_content_type_not_url(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    entry = HeadshotEntry(id="9478", url="https://example.com/headshot/current")  # no extension in URL

    def fetch(url: str):
        return b"real-bytes", "image/png"

    sync_headshot_cache([entry], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=fetch)

    assert (cache_dir / "9478.png").exists()


def test_sync_sanitizes_ids_that_are_unsafe_as_filenames(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    entry = HeadshotEntry(id="../../etc/passwd", url="https://example.com/x.jpg")

    def fetch(url: str):
        return b"real-bytes", "image/jpeg"

    sync_headshot_cache([entry], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=fetch)

    written = list(cache_dir.glob("*"))
    assert len(written) == 1
    assert written[0].parent == cache_dir  # never escaped cache_dir
    assert ".." not in written[0].name
