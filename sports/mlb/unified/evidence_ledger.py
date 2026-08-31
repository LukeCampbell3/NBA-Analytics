from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def append_generation(path: Path, record: dict[str, Any]) -> bool:
    generation_id = str(record.get("generation_id") or "")
    if not generation_id:
        raise ValueError("generation_id is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                existing.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"corrupted evidence ledger line {number}") from error
    prior = next((item for item in existing if item.get("generation_id") == generation_id), None)
    if prior is not None:
        if prior != record:
            raise ValueError("generation_id collision with different evidence")
        return False
    prior_times = [str(item.get("generated_at_utc") or "") for item in existing]
    if prior_times and str(record.get("generated_at_utc") or "") < max(prior_times):
        raise ValueError("older generation cannot overwrite newer evidence")
    rows = existing + [record]
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
    return True
