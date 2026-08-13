#!/usr/bin/env python3
"""Minimal repo-root environment loader for legacy MLB shadow scripts."""
from __future__ import annotations

import os
from pathlib import Path


def load_repo_env(start_path: Path | None = None) -> Path | None:
    """Load missing values from the nearest `.env` without exposing secrets."""
    current = start_path or Path(__file__).resolve().parent
    current = current if current.is_dir() else current.parent
    for _ in range(20):
        env_path = current / ".env"
        if env_path.is_file():
            _parse_env_file(env_path)
            return env_path
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _parse_env_file(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for source_line in content.splitlines():
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
