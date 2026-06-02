#!/usr/bin/env python3
"""Apply SQL migrations to DATABASE_URL."""
from __future__ import annotations

import argparse

from sports.nba.backend.db.connection import get_database_url, run_migrations


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NBA Analytics DB migrations")
    parser.add_argument("--database-url", default=None, help="Override DATABASE_URL")
    args = parser.parse_args()
    url = args.database_url or get_database_url()
    run_migrations(url)
    print(f"[ok] migrations applied to {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
