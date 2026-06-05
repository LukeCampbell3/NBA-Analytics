#!/usr/bin/env python3
"""
MLB Provider Credentials Loader

Loads API keys from repo-root .env file without requiring python-dotenv.

Behavior:
  1. Walk upward from start_path until .env is found
  2. Parse .env manually (KEY=value, comments, blank lines, quoted values)
  3. Load SPORTSGAMEODDS_API_KEY from .env if not already in os.environ
  4. Prefer real environment variable over .env
  5. Never print the key or any prefix/suffix

Usage:
    from provider_credentials import get_sportsgameodds_api_key
    creds = get_sportsgameodds_api_key()
    # creds["api_key"] — the key (internal use only, never print)
    # creds["credentials_present"] — bool
    # creds["key_length"] — int
    # creds["key_source"] — "environment" | ".env" | "missing"
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_repo_env(start_path: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from start_path to find and parse .env file.

    Sets missing keys into os.environ. Does not overwrite existing values.
    Returns the .env path if found, else None.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path if start_path.is_dir() else start_path.parent

    # Walk upward to find .env
    for _ in range(20):  # Safety limit
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
    """Parse a .env file and set missing keys in os.environ.

    Supports:
      - KEY=value
      - KEY="value" (strips quotes)
      - KEY='value' (strips quotes)
      - # comments
      - blank lines
      - export KEY=value (strips export prefix)

    Does NOT overwrite existing os.environ values.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    for line in content.splitlines():
        line = line.strip()

        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue

        # Strip optional 'export ' prefix
        if line.startswith("export "):
            line = line[7:].strip()

        # Split on first '='
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # Strip surrounding quotes
        if len(value) >= 2:
            if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
                value = value[1:-1]

        # Only set if not already in environment
        if key not in os.environ:
            os.environ[key] = value


def get_sportsgameodds_api_key() -> Dict[str, Any]:
    """Get SportsGameOdds API key with source tracking.

    Returns dict with:
      api_key: str or None (for internal use only — never print)
      credentials_present: bool
      key_length: int
      key_source: "environment" | ".env" | "missing"
    """
    # Check if already in environment before loading .env
    pre_existing = os.environ.get("SPORTSGAMEODDS_API_KEY")

    # Load .env (won't overwrite existing)
    env_path = load_repo_env()

    key = os.environ.get("SPORTSGAMEODDS_API_KEY")

    if not key:
        return {
            "api_key": None,
            "credentials_present": False,
            "key_length": 0,
            "key_source": "missing",
        }

    # Determine source
    if pre_existing:
        source = "environment"
    elif env_path is not None:
        source = ".env"
    else:
        source = "environment"

    return {
        "api_key": key,
        "credentials_present": True,
        "key_length": len(key),
        "key_source": source,
    }
