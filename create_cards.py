from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils import safe_float, safe_int, sanitize_filename


def _card_from_row(row: pd.Series) -> dict:
    name = str(row.get("player_name", "Unknown Player")).strip() or "Unknown Player"
    points = safe_float(row.get("points_per_game"))
    assists = safe_float(row.get("assists_per_game"))
    rebounds = safe_float(row.get("rebounds_per_game"))
    steals = safe_float(row.get("steals_per_game"))
    blocks = safe_float(row.get("blocks_per_game"))
    turnovers = safe_float(row.get("turnovers_per_game"))
    plus_minus = safe_float(row.get("plus_minus"))
    return {
        "player": {
            "name": name,
            "team": str(row.get("team", "")),
            "position": str(row.get("position", "")),
            "season": safe_int(row.get("season")),
        },
        "identity": {
            "id": sanitize_filename(f"{name}_{row.get('season', '')}"),
            "age": safe_float(row.get("age")),
            "games_played": safe_int(row.get("games_played")),
        },
        "offense": {
            "points_per_game": points,
            "assists_per_game": assists,
            "usage_rate": safe_float(row.get("usage_rate")),
            "field_goal_attempts_per_game": safe_float(row.get("field_goal_attempts_per_game")),
            "three_point_attempts_per_game": safe_float(row.get("three_point_attempts_per_game")),
            "turnovers_per_game": turnovers,
        },
        "defense": {
            "rebounds_per_game": rebounds,
            "steals_per_game": steals,
            "blocks_per_game": blocks,
        },
        "impact": {
            "minutes_per_game": safe_float(row.get("minutes_per_game")),
            "plus_minus": plus_minus,
            "box_score_signal": points + 0.7 * rebounds + 0.7 * assists + steals + blocks - turnovers,
        },
        "metadata": {"source": "create_cards", "schema_version": "simplified_v1"},
    }


def generate_cards(input_file: str | Path, output_dir: str | Path) -> int:
    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(input_path)
    count = 0
    summary = []
    for _, row in rows.iterrows():
        card = _card_from_row(row)
        filename = sanitize_filename(card["player"]["name"]) or f"player_{count + 1}"
        path = out_dir / f"{filename}.json"
        path.write_text(json.dumps(card, indent=2), encoding="utf-8")
        summary.append({"player": card["player"]["name"], "path": str(path)})
        count += 1
    (out_dir / "cards_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simplified NBA player cards.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(generate_cards(args.input, args.output))


if __name__ == "__main__":
    main()
