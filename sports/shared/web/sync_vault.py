#!/usr/bin/env python3
"""Copy canonical Card Vault assets into hub and sport web directories for local dev."""

from __future__ import annotations

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VAULT_SOURCE = SCRIPT_DIR / "vault"
SPORTS_ROOT = SCRIPT_DIR.parents[1]

TARGETS = [
    SPORTS_ROOT / "site" / "web" / "vault",
    SPORTS_ROOT / "nba" / "web" / "vault",
    SPORTS_ROOT / "mlb" / "web" / "vault",
    SPORTS_ROOT / "nfl" / "web" / "vault",
    SPORTS_ROOT / "f1" / "web" / "vault",
    SPORTS_ROOT / "golf" / "web" / "vault",
]


def main() -> int:
    if not VAULT_SOURCE.is_dir():
        print(f"Missing vault source: {VAULT_SOURCE}")
        return 1

    for target in TARGETS:
        parent = target.parent
        if not parent.is_dir():
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(VAULT_SOURCE, target)
        print(f"[vault] {VAULT_SOURCE} -> {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
