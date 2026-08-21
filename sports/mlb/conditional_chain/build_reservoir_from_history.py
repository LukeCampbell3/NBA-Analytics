from __future__ import annotations

"""Build a path-conditioned reservoir CSV from published MLB board history.

Reads ``sports/mlb/web/data/history/*.json`` (the site's own published daily
board, which already carries the pool's day-of prior probability per play)
and settles each play against the same processed game-log data
``sports/mlb/scripts/generate_daily_prediction_pool.py`` reads from
(``Player-Predictor/Data-Proc-MLB/<Player>/<season>_processed_processed.csv``),
using the same OVER/UNDER-vs-line rule as
``sports/mlb/predictions/scripts/settle_mlb_production_shadow.py``:
OVER hits if actual > line, UNDER hits if actual < line, push at 0.5.

A row is written only when both a day-of prior and a real settled actual
value are available. Rows this script cannot ground in real data are
skipped and counted in the summary, never fabricated.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORY_DIR = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "history"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"

# Target -> actual-value column in the per-player processed game log. These
# are the raw realized-stat columns (not the Market_* columns), verified
# against sports/mlb/scripts/generate_daily_prediction_pool.py's own reader.
TARGET_ACTUAL_COLUMN = {
    "H": "H",
    "TB": "TB",
    "R": "R",
    "HR": "HR",
    "RBI": "RBI",
    "K": "K",
    "ER": "ER",
}


def _iter_history_files(history_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in history_dir.glob("*.json")
        if path.stem != "index"
    )


def _processed_path(processed_dir: Path, player: str, season: int) -> Path:
    return processed_dir / player.replace(" ", "_") / f"{season}_processed_processed.csv"


def _settle(direction: str, line: float, actual: float) -> float | None:
    direction = direction.upper()
    if direction not in {"OVER", "UNDER"}:
        return None
    if actual == line:
        return 0.5
    if direction == "OVER":
        return 1.0 if actual > line else 0.0
    return 1.0 if actual < line else 0.0


def build_reservoir(
    history_dir: Path,
    processed_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = {
        "missing_required_fields": 0,
        "unresolvable_target": 0,
        "no_processed_file": 0,
        "no_matching_game_row": 0,
        "not_yet_settled": 0,
    }
    processed_cache: dict[Path, pd.DataFrame] = {}

    for path in _iter_history_files(history_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for play in payload.get("plays", []):
            player = play.get("player")
            target = play.get("target")
            direction = play.get("direction")
            market_line = play.get("market_line")
            prior = play.get("estimated_hit_probability")
            robust = play.get("historical_bucket_win_rate")
            event_date = play.get("official_game_date") or play.get("market_date")
            game_id = play.get("game_id")
            if not all([player, target, direction, event_date]) or market_line is None or prior is None:
                skipped["missing_required_fields"] += 1
                continue
            actual_column = TARGET_ACTUAL_COLUMN.get(str(target))
            if actual_column is None:
                skipped["unresolvable_target"] += 1
                continue

            game_date = pd.Timestamp(str(event_date)).normalize()
            season = int(game_date.year)
            processed_path = _processed_path(processed_dir, str(player), season)
            if processed_path not in processed_cache:
                if processed_path.exists():
                    try:
                        processed_cache[processed_path] = pd.read_csv(processed_path, low_memory=False)
                    except (OSError, pd.errors.ParserError):
                        processed_cache[processed_path] = None
                else:
                    processed_cache[processed_path] = None
            processed = processed_cache[processed_path]
            if processed is None or actual_column not in processed.columns:
                skipped["no_processed_file"] += 1
                continue

            processed_dates = pd.to_datetime(processed["Date"], errors="coerce").dt.normalize()
            match = processed.loc[processed_dates.eq(game_date)]
            if match.empty:
                skipped["no_matching_game_row"] += 1
                continue
            actual_value = pd.to_numeric(match.iloc[0].get(actual_column), errors="coerce")
            if pd.isna(actual_value):
                skipped["not_yet_settled"] += 1
                continue

            leg_result = _settle(str(direction), float(market_line), float(actual_value))
            if leg_result is None:
                skipped["missing_required_fields"] += 1
                continue

            rows.append(
                {
                    "event_date": game_date,
                    "event_id": str(game_id) if game_id else "",
                    "player": str(player),
                    "market": str(target),
                    "side": str(direction).upper(),
                    "line": float(market_line),
                    "robust_score": float(robust) if robust is not None else float(prior),
                    "survival_probability": float(prior),
                    "leg_result": float(leg_result),
                }
            )

    # Always emit the reservoir's full schema, even with zero rows, so
    # downstream consumers (merge_candidates_with_paths et al.) see a
    # correctly-columned empty frame instead of failing on a schema check.
    reservoir_columns = [
        "event_date",
        "event_id",
        "player",
        "market",
        "side",
        "line",
        "robust_score",
        "survival_probability",
        "leg_result",
    ]
    reservoir = pd.DataFrame(rows, columns=reservoir_columns)
    summary = {
        "rows_written": int(len(reservoir)),
        "skipped": skipped,
        "history_files_scanned": len(_iter_history_files(history_dir)),
    }
    return reservoir, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a settled MLB reservoir CSV from published board history."
    )
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary-json", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    reservoir, summary = build_reservoir(args.history_dir, args.processed_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.to_csv(args.out_csv, index=False)
    if args.out_summary_json is not None:
        args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
