"""Build a real, settled `actuals` file for update_safe_state_shadow_settlement.py.

For every real candidate in a safe-state run's `candidate_pool.csv`, this:

1. Reads the real settled box-score stat (PTS/TRB/AST) for that player and
   date directly from the real per-player game log
   (Player-Predictor/Data-Proc/<player>/2026_processed_processed.csv --
   real ESPN/NBA box scores, not a projection).
2. Independently confirms the candidate's own recorded `market_line`
   against real historical betting-line data
   (data copy/raw/market_odds/nba/history_player_props_long.csv -- real,
   per-sportsbook rows, e.g. DraftKings and FanDuel separately) for the
   same player/date/market. A candidate's line is only `line_confirmed`
   when every real book row for that player/date/market agrees with the
   candidate's own recorded line; any real disagreement is flagged in
   `line_discrepancy`, never silently ignored.
3. Determines the real result (win/loss/push) from the real actual stat
   against the candidate's own recorded line and direction -- the line
   the pick was actually made against -- while carrying the line
   confirmation as an explicit, auditable column rather than silently
   trusting it.

This never fabricates an actual stat or a historical line: a candidate
whose player/date isn't found in Data-Proc, or whose market isn't found
in the historical props archive, is left unresolved/unconfirmed rather
than guessed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import build_candidate_id  # noqa: E402

DEFAULT_DATA_PROC_ROOT = PLAYER_PREDICTOR_ROOT / "Data-Proc"
DEFAULT_HISTORICAL_PROPS_CSV = (
    PLAYER_PREDICTOR_ROOT / "data copy" / "raw" / "market_odds" / "nba" / "history_player_props_long.csv"
)

TARGET_TO_MARKET_KEY = {"PTS": "player_points", "TRB": "player_rebounds", "AST": "player_assists"}


def _load_actual_stat(player: str, market_date: str, target: str, *, data_proc_root: Path) -> float | None:
    path = data_proc_root / player / "2026_processed_processed.csv"
    if not path.exists() or target not in ("PTS", "TRB", "AST"):
        return None
    try:
        frame = pd.read_csv(path, usecols=["Date", target])
    except (FileNotFoundError, ValueError):
        return None
    row = frame.loc[frame["Date"] == market_date, target]
    if row.empty or pd.isna(row.iloc[0]):
        return None
    return float(row.iloc[0])


_EMPTY_HISTORICAL_LINES_COLUMNS = ["player_name_norm", "event_date_et", "market_key", "bookmaker_key", "line"]


def _load_historical_lines(historical_props_csv: Path) -> pd.DataFrame:
    if not historical_props_csv.exists():
        return pd.DataFrame(columns=_EMPTY_HISTORICAL_LINES_COLUMNS)
    try:
        return pd.read_csv(historical_props_csv, usecols=_EMPTY_HISTORICAL_LINES_COLUMNS)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_EMPTY_HISTORICAL_LINES_COLUMNS)


def _confirm_line(
    historical_lines: pd.DataFrame, *, player: str, market_date: str, target: str, candidate_line: float
) -> dict[str, Any]:
    market_key = TARGET_TO_MARKET_KEY.get(target)
    if market_key is None:
        return {"historical_books_found": 0, "historical_consensus_line": None, "line_confirmed": False, "line_discrepancy": None}

    rows = historical_lines[
        (historical_lines["player_name_norm"] == player)
        & (historical_lines["event_date_et"] == market_date)
        & (historical_lines["market_key"] == market_key)
    ]
    lines = pd.to_numeric(rows["line"], errors="coerce").dropna()
    if lines.empty:
        return {"historical_books_found": 0, "historical_consensus_line": None, "line_confirmed": False, "line_discrepancy": None}

    distinct_lines = sorted(lines.unique())
    consensus_line = float(lines.median())
    discrepancy = None if candidate_line is None else round(consensus_line - float(candidate_line), 4)
    confirmed = bool(len(distinct_lines) == 1 and candidate_line is not None and abs(consensus_line - float(candidate_line)) < 1e-9)
    return {
        "historical_books_found": int(rows["bookmaker_key"].nunique()),
        "historical_consensus_line": consensus_line,
        "historical_distinct_lines": distinct_lines,
        "line_confirmed": confirmed,
        "line_discrepancy": discrepancy,
    }


def _grade(actual_stat: float | None, line: float, direction: str) -> str | None:
    if actual_stat is None:
        return None
    if actual_stat == line:
        return "push"
    over_wins = actual_stat > line
    direction = str(direction).strip().upper()
    if direction == "OVER":
        return "win" if over_wins else "loss"
    if direction == "UNDER":
        return "loss" if over_wins else "win"
    return None


def build_real_actuals(
    candidate_pool: pd.DataFrame,
    *,
    data_proc_root: Path = DEFAULT_DATA_PROC_ROOT,
    historical_props_csv: Path = DEFAULT_HISTORICAL_PROPS_CSV,
) -> pd.DataFrame:
    if candidate_pool.empty:
        return pd.DataFrame()

    historical_lines = _load_historical_lines(historical_props_csv)
    working = candidate_pool.copy()
    if "candidate_id" not in working.columns or working["candidate_id"].isna().any():
        working["candidate_id"] = build_candidate_id(working)

    rows: list[dict[str, Any]] = []
    for _, candidate in working.iterrows():
        player = str(candidate["player"])
        market_date = str(candidate["market_date"])
        target = str(candidate["target"]).upper()
        direction = str(candidate["direction"])
        candidate_line = pd.to_numeric(pd.Series([candidate.get("market_line", candidate.get("line"))]), errors="coerce").iloc[0]
        candidate_line = None if pd.isna(candidate_line) else float(candidate_line)

        actual_stat = _load_actual_stat(player, market_date, target, data_proc_root=data_proc_root)
        confirmation = _confirm_line(historical_lines, player=player, market_date=market_date, target=target, candidate_line=candidate_line)
        result = _grade(actual_stat, candidate_line, direction) if candidate_line is not None else None

        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "player": player,
                "market_date": market_date,
                "target": target,
                "direction": direction,
                "line": candidate_line,
                "actual_stat": actual_stat,
                "actual_result": result,
                "result": result,
                "settlement_status": "RESOLVED" if result is not None else "PENDING_NO_ACTUAL_STAT",
                **confirmation,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Safe-state run directory containing candidate_pool.csv")
    parser.add_argument("--data-proc-root", type=Path, default=DEFAULT_DATA_PROC_ROOT)
    parser.add_argument("--historical-props-csv", type=Path, default=DEFAULT_HISTORICAL_PROPS_CSV)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_pool = pd.read_csv(args.run_dir / "candidate_pool.csv")
    actuals = build_real_actuals(candidate_pool, data_proc_root=args.data_proc_root, historical_props_csv=args.historical_props_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    actuals.to_csv(args.output_csv, index=False)

    resolved = int((actuals["settlement_status"] == "RESOLVED").sum()) if not actuals.empty else 0
    confirmed = int(actuals["line_confirmed"].sum()) if not actuals.empty else 0
    discrepancies = actuals.loc[actuals["line_discrepancy"].notna() & (actuals["line_discrepancy"].abs() > 1e-9)] if not actuals.empty else pd.DataFrame()
    print(f"candidates: {len(actuals)}  resolved: {resolved}  line_confirmed: {confirmed}  line_discrepancies: {len(discrepancies)}")
    if not discrepancies.empty:
        print(discrepancies[["candidate_id", "line", "historical_consensus_line", "line_discrepancy"]].to_string(index=False))
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
