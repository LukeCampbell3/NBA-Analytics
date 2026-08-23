"""Raw-data caching with a source manifest (section 34: "Do not
repeatedly download the same season data" / "Save source, endpoint,
retrieved_at, season, parameters, hash with cached datasets").

Every page this pipeline fetches from the network is written once to
``data/raw/<source>/`` and never re-fetched on a later run -- the
pipeline is reproducible offline from the cache. Each cached file has a
sibling ``<key>.manifest.json`` recording exactly where it came from and
when, so any real number downstream can be traced back to a specific,
re-fetchable page.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional


def _paths(root: Path, cache_key: str) -> tuple[Path, Path]:
    safe_key = cache_key.replace("/", "_")[:150]
    return root / f"{safe_key}.html", root / f"{safe_key}.manifest.json"


def read_cached_text(root: Path, cache_key: str) -> Optional[str]:
    body_path, _ = _paths(root, cache_key)
    if not body_path.is_file():
        return None
    return body_path.read_text(encoding="utf-8", errors="replace")


def write_cached_text(root: Path, cache_key: str, body: str, *, source: str, url: str, retrieved_at: str, **extra_params) -> None:
    root.mkdir(parents=True, exist_ok=True)
    body_path, manifest_path = _paths(root, cache_key)
    body_path.write_text(body, encoding="utf-8")
    manifest = {
        "source": source,
        "url": url,
        "retrieved_at": retrieved_at,
        "cache_key": cache_key,
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "content_length": len(body),
        **extra_params,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def list_cached_manifests(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    manifests = []
    for path in sorted(root.glob("*.manifest.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    return manifests
