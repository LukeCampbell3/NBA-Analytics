#!/usr/bin/env python3
"""
Validate published NBA and MLB final pools against locally stored market data.

This script answers a practical question:
"Could a person actually place these published bets from the market data we have?"

It reports two levels of confidence:

1. line_placeable:
   A matching market line exists and at least one book contributed to it.
2. side_price_confirmed:
   We also found a concrete price for the chosen side in either the aggregated
   snapshot or the long-form market feed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NBA_PAYLOAD = REPO_ROOT / "sports" / "nba" / "web" / "data" / "daily_predictions.json"
MLB_PAYLOAD = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"
NBA_LONG_DEFAULT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor" / "data copy" / "raw" / "market_odds" / "nba" / "latest_player_props_long.csv"
MLB_LONG_DEFAULT = REPO_ROOT / "sports" / "mlb" / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io" / "latest_player_props_long.csv"
DEFAULT_REPORT = REPO_ROOT / "sports" / "validation" / "placeable_final_pools_report.json"

NBA_TARGET_TO_MARKET_KEY = {
    "PTS": "player_points",
    "TRB": "player_rebounds",
    "AST": "player_assists",
}
MLB_TARGET_TO_MARKET_KEY = {
    "H": "batter_hits",
    "TB": "batter_total_bases",
    "R": "batter_runs_scored",
    "HR": "batter_home_runs",
    "RBI": "batter_rbis",
    "K": "pitcher_strikeouts",
    "ER": "pitcher_earned_runs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NBA and MLB final pools for practical bet placeability.")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT, help="Destination JSON report path.")
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(out):
        return None
    return out


def _normalize_player_key(value: str) -> str:
    return str(value or "").strip().replace(" ", "_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def audit_nba() -> dict[str, Any]:
    payload = _load_json(NBA_PAYLOAD)
    market_snapshot_path = Path(str(payload.get("market_snapshot") or ""))
    wide = _read_table(market_snapshot_path) if market_snapshot_path.exists() else pd.DataFrame()
    long = _read_table(NBA_LONG_DEFAULT) if NBA_LONG_DEFAULT.exists() else pd.DataFrame()

    results: list[dict[str, Any]] = []
    for play in payload.get("plays", []):
        player_key = _normalize_player_key(play.get("player", ""))
        target = str(play.get("target", "")).strip().upper()
        direction = str(play.get("direction", "")).strip().upper()
        market_date = str(play.get("market_date", ""))[:10]
        market_line = _safe_float(play.get("market_line"))

        wide_part = wide.loc[
            (wide.get("Player", pd.Series(dtype="object")).astype(str) == player_key)
            & (wide.get("Market_Date", pd.Series(dtype="object")).astype(str).str[:10] == market_date)
        ].copy()
        market_col = f"Market_{target}"
        books_col = f"Market_{target}_books"
        side_col = f"Market_{target}_{'over' if direction == 'OVER' else 'under'}_price"

        line_match = False
        books = None
        side_price_wide = None
        if not wide_part.empty and market_col in wide_part.columns:
            line_values = pd.to_numeric(wide_part[market_col], errors="coerce").dropna()
            if market_line is not None:
                line_match = bool((line_values.round(6) == round(float(market_line), 6)).any())
        if not wide_part.empty and books_col in wide_part.columns:
            book_values = pd.to_numeric(wide_part[books_col], errors="coerce").dropna()
            if not book_values.empty:
                books = float(book_values.iloc[0])
        if not wide_part.empty and side_col in wide_part.columns:
            price_values = pd.to_numeric(wide_part[side_col], errors="coerce").dropna()
            if not price_values.empty:
                side_price_wide = float(price_values.iloc[0])

        long_price = None
        if not long.empty and market_line is not None:
            market_key = NBA_TARGET_TO_MARKET_KEY.get(target)
            if market_key:
                long_part = long.loc[
                    (long.get("player_name_norm", pd.Series(dtype="object")).astype(str) == player_key)
                    & (long.get("market_key", pd.Series(dtype="object")).astype(str) == market_key)
                    & (pd.to_numeric(long.get("line", pd.Series(dtype="float")), errors="coerce").round(6) == round(float(market_line), 6))
                ].copy()
                long_side_col = "over_price" if direction == "OVER" else "under_price"
                long_prices = pd.to_numeric(long_part.get(long_side_col, pd.Series(dtype="float")), errors="coerce").dropna()
                if not long_prices.empty:
                    long_price = float(long_prices.iloc[0])

        side_price_confirmed = side_price_wide is not None or long_price is not None
        results.append(
            {
                "player": play.get("player_display_name") or play.get("player"),
                "target": target,
                "direction": direction,
                "market_date": market_date,
                "market_line": market_line,
                "line_placeable": bool(line_match and (books or 0) > 0),
                "side_price_confirmed": bool(side_price_confirmed),
                "books": books,
                "wide_side_price": side_price_wide,
                "long_side_price": long_price,
            }
        )

    return {
        "sport": "nba",
        "payload_path": str(NBA_PAYLOAD),
        "wide_snapshot_path": str(market_snapshot_path),
        "long_snapshot_path": str(NBA_LONG_DEFAULT),
        "plays": results,
        "summary": {
            "play_count": len(results),
            "line_placeable_count": sum(bool(row["line_placeable"]) for row in results),
            "side_price_confirmed_count": sum(bool(row["side_price_confirmed"]) for row in results),
        },
    }


def audit_mlb() -> dict[str, Any]:
    payload = _load_json(MLB_PAYLOAD)
    run_date = str(payload.get("run_date", ""))[:10]
    run_stamp = run_date.replace("-", "")
    selected_csv = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs" / run_stamp / f"daily_prediction_pool_{run_stamp}_high_precision_predictions.csv"
    selected = pd.read_csv(selected_csv) if selected_csv.exists() else pd.DataFrame()
    long = _read_table(MLB_LONG_DEFAULT) if MLB_LONG_DEFAULT.exists() else pd.DataFrame()

    results: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        player_display = str(row.get("Player", "")).strip()
        player_key = _normalize_player_key(player_display)
        target = str(row.get("Target", "")).strip().upper()
        direction = str(row.get("Direction", "")).strip().upper()
        market_line = _safe_float(row.get("Market_Line"))
        books = _safe_float(row.get("Market_Books"))
        market_source = str(row.get("Market_Source", "")).strip().lower()
        csv_side_col = "Market_Over_Price" if direction == "OVER" else "Market_Under_Price"
        csv_side_price = _safe_float(row.get(csv_side_col))

        long_price = None
        long_line_match = False
        if not long.empty and market_line is not None:
            market_key = MLB_TARGET_TO_MARKET_KEY.get(target)
            if market_key:
                long_part = long.loc[
                    (long.get("player_name_norm", pd.Series(dtype="object")).astype(str) == player_key)
                    & (long.get("market_key", pd.Series(dtype="object")).astype(str) == market_key)
                    & (long.get("event_date_et", pd.Series(dtype="object")).astype(str).str[:10] == run_date)
                    & (pd.to_numeric(long.get("line", pd.Series(dtype="float")), errors="coerce").round(6) == round(float(market_line), 6))
                ].copy()
                long_line_match = not long_part.empty
                long_side_col = "over_price" if direction == "OVER" else "under_price"
                long_prices = pd.to_numeric(long_part.get(long_side_col, pd.Series(dtype="float")), errors="coerce").dropna()
                if not long_prices.empty:
                    long_price = float(long_prices.iloc[0])

        results.append(
            {
                "player": player_display,
                "target": target,
                "direction": direction,
                "market_date": run_date,
                "market_line": market_line,
                "market_source": market_source,
                "line_placeable": bool(market_source == "real" and (books or 0) > 0),
                "side_price_confirmed": bool(csv_side_price is not None or long_price is not None),
                "books": books,
                "csv_side_price": csv_side_price,
                "long_line_match": bool(long_line_match),
                "long_side_price": long_price,
            }
        )

    return {
        "sport": "mlb",
        "payload_path": str(MLB_PAYLOAD),
        "selected_csv_path": str(selected_csv),
        "long_snapshot_path": str(MLB_LONG_DEFAULT),
        "plays": results,
        "summary": {
            "play_count": len(results),
            "line_placeable_count": sum(bool(row["line_placeable"]) for row in results),
            "side_price_confirmed_count": sum(bool(row["side_price_confirmed"]) for row in results),
        },
    }


def main() -> None:
    args = parse_args()
    report = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "nba": audit_nba(),
        "mlb": audit_mlb(),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("PLACEABLE FINAL POOLS REPORT")
    print(f"NBA line-placeable: {report['nba']['summary']['line_placeable_count']} / {report['nba']['summary']['play_count']}")
    print(f"NBA side-price confirmed: {report['nba']['summary']['side_price_confirmed_count']} / {report['nba']['summary']['play_count']}")
    print(f"MLB line-placeable: {report['mlb']['summary']['line_placeable_count']} / {report['mlb']['summary']['play_count']}")
    print(f"MLB side-price confirmed: {report['mlb']['summary']['side_price_confirmed_count']} / {report['mlb']['summary']['play_count']}")
    print(f"Report JSON: {args.report_json}")


if __name__ == "__main__":
    main()
