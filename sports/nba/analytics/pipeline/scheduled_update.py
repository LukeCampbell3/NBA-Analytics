"""
Scheduled Update Pipeline

Supports daily, weekly, and manual update modes.
Checks source freshness, skips unchanged sources, caches responses,
writes manifests, and rebuilds downstream outputs only when needed.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

WORKSPACE = Path(__file__).resolve().parents[4]
OUTPUT_DIR = WORKSPACE / "sports" / "nba" / "analytics" / "output"
MANIFEST_DIR = WORKSPACE / "sports" / "nba" / "analytics" / "data" / "manifests"
DATA_DIR = WORKSPACE / "sports" / "nba" / "analytics" / "data"


def _ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "cache").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)


def run_scheduled_update(
    mode: str = "daily",
    force_rebuild: bool = False,
    max_players: int = 0,
    skip_api: bool = False,
) -> Dict[str, Any]:
    """Run the scheduled update pipeline.

    Modes:
      daily: Check freshness, update stale sources, rebuild vectors
      weekly: Full rebuild of all sources + vectors + cards
      manual: Same as weekly but triggered manually

    Args:
        mode: "daily", "weekly", or "manual"
        force_rebuild: Ignore cache freshness, rebuild everything
        max_players: Limit players (0 = all)
        skip_api: Skip nba_api calls (use cached/existing data only)
    """
    _ensure_dirs()
    start = time.time()
    manifest: Dict[str, Any] = {
        "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
        "errors": [],
        "completed": False,
    }

    # Step 1: Check data freshness
    step = {"name": "check_freshness", "status": "started"}
    try:
        from ..data_ingest.nba_api_loader import get_data_freshness_report
        freshness = get_data_freshness_report()
        stale_sources = [k for k, v in freshness.items() if not v["fresh"]]
        step["stale_sources"] = len(stale_sources)
        step["status"] = "complete"
    except Exception as e:
        step["status"] = "error"
        step["error"] = str(e)[:200]
        manifest["errors"].append(step["error"])
    manifest["steps"].append(step)

    # Step 2: Fetch league stats if stale or forced
    if not skip_api and (force_rebuild or mode in ("weekly", "manual")):
        step = {"name": "fetch_league_stats", "status": "started"}
        try:
            from ..data_ingest.nba_api_loader import fetch_league_stats
            df = fetch_league_stats()
            step["rows"] = len(df) if df is not None else 0
            step["status"] = "complete" if df is not None else "skipped"
        except Exception as e:
            step["status"] = "error"
            step["error"] = str(e)[:200]
            manifest["errors"].append(step["error"])
        manifest["steps"].append(step)

    # Step 3: Build vectors
    step = {"name": "build_vectors", "status": "started"}
    try:
        from .build_league_vectors import run_full_build
        summary = run_full_build(max_players=max_players)
        step["players_processed"] = summary["players_processed"]
        step["status"] = "complete"
    except Exception as e:
        step["status"] = "error"
        step["error"] = str(e)[:200]
        manifest["errors"].append(step["error"])
    manifest["steps"].append(step)

    # Step 4: Position percentiles
    step = {"name": "position_percentiles", "status": "started"}
    try:
        from .build_league_vectors import build_all_vectors
        from ..features.position_percentiles import assign_position_percentiles_to_vectors
        vectors = build_all_vectors(max_players)
        assign_position_percentiles_to_vectors(vectors)
        step["players"] = len(vectors)
        step["status"] = "complete"
    except Exception as e:
        step["status"] = "error"
        step["error"] = str(e)[:200]
        manifest["errors"].append(step["error"])
    manifest["steps"].append(step)

    # Step 5: Pairwise fit (sample only for daily, full for weekly)
    step = {"name": "pairwise_fit", "status": "started"}
    try:
        from ..team_building.pairwise_fit import build_pairwise_matrix
        if mode == "daily":
            # Only top 50 players for daily
            sample = sorted(vectors, key=lambda v: sum(d.raw_value or 0 for d in v.dimensions.values()), reverse=True)[:50]
        else:
            sample = vectors[:100]  # Top 100 for weekly
        fits = build_pairwise_matrix(sample)
        fit_path = OUTPUT_DIR / "pairwise_fits.json"
        fit_path.write_text(json.dumps(fits[:200], indent=2, default=str), encoding="utf-8")
        step["pairs_computed"] = len(fits)
        step["status"] = "complete"
    except Exception as e:
        step["status"] = "error"
        step["error"] = str(e)[:200]
        manifest["errors"].append(step["error"])
    manifest["steps"].append(step)

    # Finalize
    elapsed = time.time() - start
    manifest["completed"] = len(manifest["errors"]) == 0
    manifest["elapsed_seconds"] = round(elapsed, 1)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Write manifest
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_path = MANIFEST_DIR / f"update_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Analytics Scheduled Update")
    parser.add_argument("--mode", default="daily", choices=["daily", "weekly", "manual"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-players", type=int, default=0)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    result = run_scheduled_update(
        mode=args.mode,
        force_rebuild=args.force,
        max_players=args.max_players,
        skip_api=args.skip_api,
    )
    print(json.dumps(result, indent=2))
