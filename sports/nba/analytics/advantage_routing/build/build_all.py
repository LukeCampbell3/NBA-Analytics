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
    # Original seed population (spec section 46).
    "Derik Queen",
    "Collin Murray-Boyles",
    "Donovan Clingan",
    "Yves Missi",
    "Jamal Murray",  # representative drive-passer, section 46

    # Low-usage/rotation bigs with latent processing interest -- the
    # spec's stated primary interest area (section 46: "with particular
    # interest in low-usage bigs with latent processing ability").
    "Walker Kessler",
    "Isaiah Hartenstein",
    "Jarrett Allen",
    "Ivica Zubac",
    "Nic Claxton",
    "Zach Edey",
    "Jaxson Hayes",
    "Mark Williams",
    "Naz Reid",

    # Established elite-passing bigs -- a validation/contrast set: if the
    # gravity/recipient-network model is honest, these should register
    # clearly differently from the low-usage group above, not just
    # confirm whatever the model already assumed.
    "Nikola Jokić",
    "Domantas Sabonis",
    "Alperen Sengun",

    # Primary drive-passers, broadening beyond the single original
    # representative (Jamal Murray) to exercise the drive-pass model
    # across a wider range of usage/role.
    "Luka Dončić",
    "LeBron James",
    "Trae Young",
    "Ja Morant",

    # Versatile modern hybrid forward/bigs, to test generalization
    # outside both the "low-usage big" and "primary ball-handler" poles.
    "Victor Wembanyama",
    "Draymond Green",
]


def build_all(season: str = "2025-26", *, players: list[str] | None = None, output_root: Path = OUTPUT_ROOT) -> Path:
    """Builds the given players (SEED_PLAYERS by default) and writes
    players.json. The index is MERGED with whatever already exists on
    disk at output_root/players.json, keyed by slug -- passing a subset
    via `players` (e.g. only newly added names) never drops previously
    built players from the index, so incremental population growth
    never requires re-fetching everyone from scratch."""
    players = players if players is not None else SEED_PLAYERS

    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "players.json"
    existing_entries: dict[str, dict] = {}
    if index_path.exists():
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8"))
            for entry in existing.get("players", []):
                if isinstance(entry, dict) and entry.get("slug"):
                    existing_entries[entry["slug"]] = entry
        except (json.JSONDecodeError, OSError):
            pass  # a corrupt/missing prior index just means we start fresh

    for player_name in players:
        try:
            out_path = write_player_artifact(player_name, season, output_root=output_root)
            print(f"  wrote {out_path}")
            slug = _slugify(player_name)
            existing_entries[slug] = {"name": player_name, "slug": slug, "season": season}
        except Exception as exc:  # noqa: BLE001 -- one player's failure must not silently corrupt the rest
            print(f"  FAILED {player_name}: {exc}")

    index_entries = sorted(existing_entries.values(), key=lambda e: e["name"])
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
