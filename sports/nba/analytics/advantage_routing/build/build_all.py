"""Builds every seed player's advantage-routing artifact plus the
players.json index the frontend's player selector reads.

    python -m sports.nba.analytics.advantage_routing.build.build_all --season 2025-26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_player import OUTPUT_ROOT, _slugify, write_player_artifact

SEED_PLAYERS = [
    "Derik Queen",
    "Collin Murray-Boyles",
    "Donovan Clingan",
    "Yves Missi",
    "Jamal Murray",  # representative drive-passer, section 46
]


def build_all(season: str = "2025-26", *, players: list[str] | None = None, output_root: Path = OUTPUT_ROOT) -> Path:
    players = players if players is not None else SEED_PLAYERS
    index_entries = []
    for player_name in players:
        try:
            out_path = write_player_artifact(player_name, season, output_root=output_root)
            print(f"  wrote {out_path}")
            index_entries.append({"name": player_name, "slug": _slugify(player_name), "season": season})
        except Exception as exc:  # noqa: BLE001 -- one player's failure must not silently corrupt the rest
            print(f"  FAILED {player_name}: {exc}")

    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "players.json"
    index_path.write_text(json.dumps({"season": season, "players": index_entries}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {index_path}")
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--players", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    build_all(args.season, players=args.players, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
