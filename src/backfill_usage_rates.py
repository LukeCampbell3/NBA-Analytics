"""
backfill_usage_rates.py - Derive and backfill per-player usage rates.

Updates usage fields in existing `*_final.json` player cards using box-score
stats from `data/raw/practical_player_card_data.csv`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def normalize_player_id(value: object) -> str:
    """Normalize ids like 201939.0 and '201939' to the same string key."""
    if value is None:
        return ""
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def usage_band_from_rate(usage_rate: float) -> str:
    """Map usage rate to existing band labels used by final cards."""
    if usage_rate >= 0.28:
        return "high"
    if usage_rate >= 0.22:
        return "medium"
    return "low"


def compute_usage_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Compute USG% from available per-game stats.

    Formula:
      USG = ((FGA + 0.44*FTA + TOV) * (Team MIN / 5)) / (MIN * (Team FGA + 0.44*Team FTA + Team TOV))
    """
    required = ["PLAYER_ID", "TEAM_ABBREVIATION", "MIN", "FGA", "FTA", "TOV"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    work = df.copy()
    work["PLAYER_ID_NORM"] = work["PLAYER_ID"].apply(normalize_player_id)
    work["TEAM_ABBREVIATION"] = work["TEAM_ABBREVIATION"].astype(str).str.upper().str.strip()

    team_totals = (
        work.groupby("TEAM_ABBREVIATION", as_index=False)[["MIN", "FGA", "FTA", "TOV"]]
        .sum()
        .rename(
            columns={
                "MIN": "TEAM_MIN",
                "FGA": "TEAM_FGA",
                "FTA": "TEAM_FTA",
                "TOV": "TEAM_TOV",
            }
        )
    )
    work = work.merge(team_totals, on="TEAM_ABBREVIATION", how="left")

    player_possessions = work["FGA"] + 0.44 * work["FTA"] + work["TOV"]
    team_possessions = work["TEAM_FGA"] + 0.44 * work["TEAM_FTA"] + work["TEAM_TOV"]
    denominator = work["MIN"] * team_possessions
    numerator = player_possessions * (work["TEAM_MIN"] / 5.0)

    usage = pd.Series(0.0, index=work.index, dtype="float64")
    valid = (denominator > 0) & (work["MIN"] > 0)
    usage.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    usage = usage.clip(lower=0.0, upper=1.0)
    work["USAGE_RATE_DERIVED"] = usage

    lookup: Dict[Tuple[str, str], float] = {}
    for _, row in work.iterrows():
        key = (row["PLAYER_ID_NORM"], row["TEAM_ABBREVIATION"])
        lookup[key] = float(row["USAGE_RATE_DERIVED"])
    return lookup


def get_card_usage_key(card: dict) -> Tuple[str, str]:
    player = card.get("player", {})
    player_id = normalize_player_id(player.get("id", ""))
    team = str(player.get("team", "")).upper().strip()
    return player_id, team


def backfill_usage(raw_csv: Path, cards_dir: Path) -> Tuple[int, int]:
    df = pd.read_csv(raw_csv)
    usage_lookup = compute_usage_lookup(df)

    updated = 0
    skipped = 0

    for card_path in sorted(cards_dir.glob("*_final.json")):
        with open(card_path, "r", encoding="utf-8") as f:
            card = json.load(f)

        key = get_card_usage_key(card)
        usage_rate: Optional[float] = usage_lookup.get(key)
        if usage_rate is None:
            skipped += 1
            continue

        performance = card.setdefault("performance", {})
        advanced = performance.setdefault("advanced", {})
        advanced["usage_rate"] = round(usage_rate, 3)

        identity = card.setdefault("identity", {})
        identity["usage_band"] = usage_band_from_rate(usage_rate)

        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=True)

        updated += 1

    return updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill usage rates in final player cards")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/practical_player_card_data.csv"),
        help="Raw player stats CSV path",
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=Path("data/processed/player_cards"),
        help="Directory containing *_final.json player card files",
    )
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"Error: raw CSV not found: {args.raw}")
        return 1
    if not args.cards.exists():
        print(f"Error: cards directory not found: {args.cards}")
        return 1

    updated, skipped = backfill_usage(args.raw, args.cards)
    print(f"Updated usage for {updated} cards")
    print(f"Skipped {skipped} cards (no raw-data match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
