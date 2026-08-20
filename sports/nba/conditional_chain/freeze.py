from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .protocol import ALLOCATION_PATH_PROTOCOL, FROZEN_SELECTOR_PROTOCOL


DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-learn")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_freeze_manifest(package_dir: Path | None = None) -> dict[str, Any]:
    package_dir = package_dir or Path(__file__).resolve().parent
    source_files = sorted(
        path for path in package_dir.glob("*.py") if path.name != "__pycache__"
    )
    dependencies = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    protocol_payload = {
        "selector": FROZEN_SELECTOR_PROTOCOL.as_dict(),
        "allocation_path": ALLOCATION_PATH_PROTOCOL.as_dict(),
    }
    protocol_sha = hashlib.sha256(_canonical_bytes(protocol_payload)).hexdigest()
    source_hashes: dict[str, str] = {}
    bundle = hashlib.sha256()
    bundle.update(_canonical_bytes(protocol_payload))
    bundle.update(_canonical_bytes(dependencies))
    for path in source_files:
        content = path.read_bytes()
        source_hashes[path.name] = hashlib.sha256(content).hexdigest()
        bundle.update(path.name.encode("utf-8"))
        bundle.update(content)
    return {
        "freeze_version": "NBA_CONDITIONAL_CHAIN_FREEZE_V1",
        "selector_version": FROZEN_SELECTOR_PROTOCOL.version,
        "representation_version": ALLOCATION_PATH_PROTOCOL.version,
        "protocol_sha256": protocol_sha,
        "executable_bundle_sha256": bundle.hexdigest(),
        "dependencies": dependencies,
        "source_sha256": source_hashes,
    }
