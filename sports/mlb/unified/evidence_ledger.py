from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"corrupted evidence ledger line {number}") from error
    return rows


def _atomic_write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for item in rows:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_generation(path: Path, record: dict[str, Any]) -> bool:
    generation_id = str(record.get("generation_id") or "")
    if not generation_id:
        raise ValueError("generation_id is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_ledger(path)
    prior = next((item for item in existing if item.get("generation_id") == generation_id and int(item.get("revision") or 1) == 1), None)
    if prior is not None:
        if prior != record:
            raise ValueError("generation_id collision with different evidence")
        return False
    prior_times = [str(item.get("generated_at_utc") or "") for item in existing]
    if prior_times and str(record.get("generated_at_utc") or "") < max(prior_times):
        raise ValueError("older generation cannot overwrite newer evidence")
    _atomic_write_rows(path, existing + [record])
    return True


def append_revision(path: Path, record: dict[str, Any]) -> bool:
    """Append a hash-linked settlement/correction; never mutate generation evidence."""
    generation_id = str(record.get("generation_id") or "")
    if not generation_id:
        raise ValueError("generation_id is required")
    existing = read_ledger(path)
    generations = [item for item in existing if item.get("generation_id") == generation_id]
    if not generations:
        raise ValueError("revision references unknown generation_id")
    expected = max(int(item.get("revision") or 1) for item in generations) + 1
    revision = int(record.get("revision") or 0)
    if revision != expected:
        raise ValueError(f"revision must be {expected}")
    if not record.get("supersedes_revision") or int(record["supersedes_revision"]) != expected - 1:
        raise ValueError("revision must identify the superseded revision")
    prior = next((item for item in generations if int(item.get("revision") or 1) == revision), None)
    if prior is not None:
        if prior != record:
            raise ValueError("revision collision with different evidence")
        return False
    _atomic_write_rows(path, existing + [record])
    return True
