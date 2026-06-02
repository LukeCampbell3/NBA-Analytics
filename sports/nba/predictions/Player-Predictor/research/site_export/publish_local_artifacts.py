#!/usr/bin/env python3
"""Publish local site export artifacts to static data dir and record artifact_runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from sports.nba.backend.db.connection import get_database_url, run_migrations
from sports.nba.backend.db.repository import insert_artifact_run

REQUIRED_FILES = [
    "safe_state_latest.json",
    "safe_state_latest.csv",
    "safe_state_cards.json",
    "site_manifest.json",
]
OPTIONAL_FILES = [
    "player_simulation_cards.json",
    "player_simulation_summary.csv",
    "simulation_credibility_gate.json",
    "site_production_status.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(source_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required export files: {', '.join(missing)}")

    latest = read_json(source_dir / "safe_state_latest.json")
    manifest = read_json(source_dir / "site_manifest.json")
    cards = read_json(source_dir / "safe_state_cards.json")
    sim_path = source_dir / "player_simulation_cards.json"
    simulations = read_json(sim_path) if sim_path.exists() else []

    if latest.get("shadow_only") is not True:
        raise ValueError("safe_state_latest.json must set shadow_only=true")
    if latest.get("production_behavior_changed") is True:
        raise ValueError("production_behavior_changed must remain false for shadow exports")
    if latest.get("staking_enabled") is True or latest.get("auto_bet_enabled") is True:
        raise ValueError("staking/autobet must remain disabled")

    card_list = cards if isinstance(cards, list) else latest.get("cards", [])
    sim_list = simulations if isinstance(simulations, list) else []
    return {
        "run_id": str(latest.get("run_id") or manifest.get("run_id") or date.today().isoformat()),
        "run_date": str(latest.get("run_date") or manifest.get("run_date") or date.today().isoformat()),
        "card_count": len(card_list) if isinstance(card_list, list) else 0,
        "simulation_count": len(sim_list),
    }


def publish(source_dir: Path, target_dir: Path, database_url: str | None) -> dict[str, Any]:
    if database_url:
        run_migrations(database_url)
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    meta = validate_artifacts(source_dir)
    copied: list[str] = []

    for name in REQUIRED_FILES + OPTIONAL_FILES:
        src = source_dir / name
        if not src.exists():
            continue
        dst = target_dir / name
        shutil.copy2(src, dst)
        copied.append(name)

        if database_url:
            insert_artifact_run(
                {
                    "run_id": meta["run_id"],
                    "run_date": meta["run_date"],
                    "sport": "nba",
                    "artifact_type": name,
                    "artifact_path": str(dst),
                    "artifact_hash": sha256_file(dst),
                    "card_count": meta["card_count"],
                    "simulation_count": meta["simulation_count"],
                    "shadow_only": True,
                    "promotion_ready": False,
                    "production_behavior_changed": False,
                }
            )

    latest_pointer = {
        "run_id": meta["run_id"],
        "run_date": meta["run_date"],
        "published_at": date.today().isoformat(),
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "files": copied,
        "shadow_only": True,
        "promotion_ready": False,
        "production_behavior_changed": False,
    }
    (target_dir / "site_publish_manifest.json").write_text(
        json.dumps(latest_pointer, indent=2),
        encoding="utf-8",
    )
    return latest_pointer


def main() -> int:
    repo_root = Path(__file__).resolve().parents[6]
    parser = argparse.ArgumentParser(description="Publish local NBA site export artifacts")
    parser.add_argument(
        "--source-dir",
        default=str(repo_root / "sports" / "nba" / "validation" / "production_shadow" / "site_exports"),
    )
    parser.add_argument(
        "--target-dir",
        default=str(repo_root / "sports" / "nba" / "web" / "data"),
    )
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    database_url = args.database_url or get_database_url()
    run_migrations(database_url)
    summary = publish(Path(args.source_dir), Path(args.target_dir), database_url)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
