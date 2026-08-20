from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .frozen_selector import select_frozen_board
from .protocol import FROZEN_SELECTOR_PROTOCOL, FrozenSelectorProtocol


VALIDATION_POOL_COLUMNS = {
    "market_date",
    "player",
    "target",
    "direction",
    "market_line",
    "actual",
    "estimated_win_rate",
}


def adapt_validation_pool_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(VALIDATION_POOL_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"validation pool ledger is missing required columns: {missing}")
    adapted = pd.DataFrame(index=frame.index)
    adapted["event_date"] = pd.to_datetime(frame["market_date"], errors="raise").dt.normalize()
    adapted["player"] = frame["player"].astype(str)
    adapted["market"] = frame["target"].astype(str).str.lower().map(
        {"pts": "player_points", "trb": "player_rebounds", "ast": "player_assists"}
    )
    adapted["side"] = frame["direction"].astype(str).str.upper()
    adapted["line"] = pd.to_numeric(frame["market_line"], errors="coerce")
    adapted["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    selected_probability = pd.to_numeric(frame["estimated_win_rate"], errors="coerce")
    adapted["p_over"] = np.where(
        adapted["side"].eq("OVER"), selected_probability, 1.0 - selected_probability
    )
    adapted["event_id"] = (
        frame["game_key"].astype(str)
        if "game_key" in frame
        else adapted["event_date"].astype(str)
    )
    for column in (
        "event_start_time_utc",
        "snapshot_time_utc",
        "book",
        "decimal_odds",
        "source",
        "raw_source_hash",
        "parser_version",
    ):
        if column in frame:
            adapted[column] = frame[column]
    return adapted


def _leg_result(actual: float, line: float, side: str) -> float:
    if np.isclose(actual, line):
        return 0.5
    if side == "OVER":
        return float(actual > line)
    return float(actual < line)


def load_data_proc_history(data_proc_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(data_proc_dir.glob("*/*processed*.csv")):
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in {"Date", "Player", "PTS", "TRB", "AST"},
            low_memory=False,
        )
        required = {"Date", "Player", "PTS", "TRB", "AST"}
        if not required.issubset(frame.columns):
            continue
        long = frame.melt(
            id_vars=["Date", "Player"],
            value_vars=["PTS", "TRB", "AST"],
            var_name="market",
            value_name="actual",
        )
        long = long.rename(columns={"Date": "event_date", "Player": "player"})
        long["market"] = long["market"].str.lower().map(
            {"pts": "player_points", "trb": "player_rebounds", "ast": "player_assists"}
        )
        rows.append(long)
    if not rows:
        raise ValueError(f"no processed player history found under {data_proc_dir}")
    history = pd.concat(rows, ignore_index=True)
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce").dt.normalize()
    history["actual"] = pd.to_numeric(history["actual"], errors="coerce")
    return history.dropna(subset=["event_date", "player", "market", "actual"]).drop_duplicates(
        ["event_date", "player", "market"], keep="last"
    )


def replay_frozen_selector(
    validation_pool: pd.DataFrame,
    *,
    historical_actuals: pd.DataFrame | None = None,
    protocol: FrozenSelectorProtocol = FROZEN_SELECTOR_PROTOCOL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay one frozen decision per slate on a completely observed candidate pool."""

    adapted = adapt_validation_pool_ledger(validation_pool)
    adapted = adapted.dropna(subset=["market", "line", "actual", "p_over"])
    if historical_actuals is None:
        history = (
            adapted[["event_date", "player", "market", "actual"]]
            .drop_duplicates(["event_date", "player", "market"], keep="last")
            .copy()
        )
        history_source = "candidate_pool_only"
    else:
        required_history = {"event_date", "player", "market", "actual"}
        missing_history = sorted(required_history - set(historical_actuals.columns))
        if missing_history:
            raise ValueError(f"historical actuals are missing required columns: {missing_history}")
        history = historical_actuals[list(required_history)].copy()
        history_source = "external_date_safe_player_game_history"
    decision_rows: list[pd.DataFrame] = []
    slate_rows: list[dict[str, Any]] = []
    for event_date in sorted(adapted["event_date"].unique()):
        slate = adapted.loc[adapted["event_date"] == event_date].copy()
        selection = select_frozen_board(slate, history, protocol=protocol)
        board = selection.control_parlay.copy()
        slate_result = {
            "event_date": pd.Timestamp(event_date),
            "candidate_rows": int(len(slate)),
            "eligible_players": int(selection.reservoir["player"].nunique()),
            "published": bool(selection.published),
            "status": selection.status,
            "legs": int(len(board)),
            "parlay_hit": False,
        }
        if selection.published:
            board = board.reset_index(drop=True)
            board["rank"] = np.arange(1, len(board) + 1)
            board["leg_result"] = [
                _leg_result(float(row["actual"]), float(row["line"]), str(row["side"]))
                for _, row in board.iterrows()
            ]
            board["decision_id"] = pd.Timestamp(event_date).strftime("%Y-%m-%d")
            board["published"] = True
            board["parlay_hit"] = bool(board["leg_result"].eq(1.0).all())
            slate_result["parlay_hit"] = bool(board["parlay_hit"].iloc[0])
            decision_rows.append(board)
        slate_rows.append(slate_result)

    decisions = pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame()
    slates = pd.DataFrame(slate_rows)
    published_slates = slates.loc[slates["published"]]
    resolved_legs = decisions.loc[decisions["leg_result"].isin([0.0, 1.0])] if not decisions.empty else decisions
    core = (
        decisions.loc[decisions["rank"].isin([1, 4])]
        if "rank" in decisions
        else decisions.copy()
    )
    core_decisions = (
        core.groupby("decision_id")["leg_result"].agg(
            legs="size", parlay_hit=lambda values: bool(values.eq(1.0).all())
        )
        if len(core)
        else pd.DataFrame(columns=["legs", "parlay_hit"])
    )
    core_decisions = core_decisions.loc[core_decisions["legs"].eq(2)]
    line_fraction = np.mod(adapted["line"].to_numpy(dtype=float), 1.0)
    conventional_line = np.isclose(line_fraction, 0.0) | np.isclose(line_fraction, 0.5)
    report = {
        "selector_version": protocol.version,
        "candidate_universe_evidence": "FULL_CANDIDATE_UNIVERSE_FOR_COMMITTED_WINDOW",
        "market_evidence": "SYNTHETIC_THRESHOLD_HISTORY",
        "production_authorizable": False,
        "production_blockers": [
            "NO_BOOK_QUOTE_PROVENANCE",
            "NO_OBSERVED_PRICE_PROVENANCE",
            "NO_QUOTE_TIMESTAMP_OR_RAW_SOURCE_HASH",
        ],
        "line_audit": {
            "conventional_integer_or_half_lines": int(conventional_line.sum()),
            "conventional_integer_or_half_fraction": float(conventional_line.mean()),
            "verified_book_quotes": 0,
        },
        "warning": (
            "The committed window contains model-generated thresholds, not verified executable "
            "sportsbook quotes. Its outcomes are research diagnostics only."
        ),
        "history_source": history_source,
        "history_rows": int(len(history)),
        "candidate_rows": int(len(adapted)),
        "eligible_slates": int(len(slates)),
        "published_slates": int(len(published_slates)),
        "slate_coverage": float(len(published_slates) / len(slates)) if len(slates) else 0.0,
        "resolved_legs": int(len(resolved_legs)),
        "individual_leg_wr": (
            float(resolved_legs["leg_result"].mean()) if len(resolved_legs) else None
        ),
        "four_leg_parlay_wins": int(published_slates["parlay_hit"].sum()),
        "four_leg_parlay_wr": (
            float(published_slates["parlay_hit"].mean()) if len(published_slates) else None
        ),
        "research_core_ranks_1_4": {
            "decisions": int(len(core_decisions)),
            "parlay_wins": int(core_decisions["parlay_hit"].sum()),
            "parlay_wr": (
                float(core_decisions["parlay_hit"].mean())
                if len(core_decisions)
                else None
            ),
            "status": "SHADOW_ONLY_DIFFERENT_PREDICTOR_VERSION",
        },
        "publication_floor": protocol.publication_floor,
        "slates": slates.to_dict(orient="records"),
    }
    return decisions, report
