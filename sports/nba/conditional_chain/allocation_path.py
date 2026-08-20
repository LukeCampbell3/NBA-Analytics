from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd

from .protocol import ALLOCATION_PATH_PROTOCOL, AllocationPathProtocol


REQUIRED_QUOTE_COLUMNS = {
    "event_id",
    "event_start_time_utc",
    "snapshot_time_utc",
    "player",
    "team",
    "market",
    "line",
    "book",
}
REQUIRED_OUTCOME_COLUMNS = {"event_id", "team", "player", "actual"}


class PathQualityStatus(str, Enum):
    VALID = "VALID"
    MARKET_UNAVAILABLE = "MARKET_UNAVAILABLE"
    EVENT_TIME_AMBIGUOUS = "EVENT_TIME_AMBIGUOUS"
    TEAM_IDENTITY_MISSING = "TEAM_IDENTITY_MISSING"
    INVALID_LINE = "INVALID_LINE"
    INSUFFICIENT_ENGINES = "INSUFFICIENT_ENGINES"
    MISSING_CHECKPOINT = "MISSING_CHECKPOINT"
    INSUFFICIENT_STABLE_PLAYERS = "INSUFFICIENT_STABLE_PLAYERS"
    INSUFFICIENT_STABLE_COVERAGE = "INSUFFICIENT_STABLE_COVERAGE"
    DUPLICATE_OUTCOME = "DUPLICATE_OUTCOME"
    MISSING_ACTUAL = "MISSING_ACTUAL"
    INVALID_ACTUAL_TOTAL = "INVALID_ACTUAL_TOTAL"


@dataclass(frozen=True)
class PathBuildResult:
    player_features: pd.DataFrame
    event_features: pd.DataFrame
    quality_ledger: pd.DataFrame


@dataclass(frozen=True)
class SettlementResult:
    settled_player_features: pd.DataFrame
    quality_ledger: pd.DataFrame


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _entropy(values: np.ndarray) -> float:
    positive = values[values > 0]
    return float(-(positive * np.log(positive)).sum())


def _direction_reversals(values: np.ndarray) -> int:
    signs = np.sign(np.diff(values))
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _quality_row(
    *,
    event_id: str,
    team: str,
    status: PathQualityStatus,
    reason: str,
    protocol: AllocationPathProtocol,
    stable_players: int = 0,
    union_players: int = 0,
    stable_coverage: float | None = None,
) -> dict[str, object]:
    return {
        "unit_id": f"{event_id}::{team or 'UNKNOWN'}::{protocol.market}",
        "event_id": event_id,
        "team": team,
        "market": protocol.market,
        "representation_version": protocol.version,
        "status": status.value,
        "reason": reason,
        "stable_players": int(stable_players),
        "union_players": int(union_players),
        "stable_coverage": stable_coverage,
    }


def _checkpoint_consensus(
    team_quotes: pd.DataFrame,
    checkpoint_time: pd.Timestamp,
    protocol: AllocationPathProtocol,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    eligible = team_quotes.loc[team_quotes["snapshot_time_utc"] <= checkpoint_time].copy()
    eligible["quote_age_minutes"] = (
        checkpoint_time - eligible["snapshot_time_utc"]
    ).dt.total_seconds() / 60.0
    eligible = eligible.loc[
        eligible["quote_age_minutes"].between(0.0, protocol.max_checkpoint_age_minutes)
    ]
    if eligible.empty:
        return {}, {}, {}

    latest = (
        eligible.sort_values("snapshot_time_utc")
        .groupby(["player", "engine"], as_index=False, sort=False)
        .tail(1)
    )
    engine_counts = latest.groupby("player")["engine"].nunique()
    valid_players = engine_counts.loc[
        engine_counts >= protocol.minimum_independent_engines
    ].index
    latest = latest.loc[latest["player"].isin(valid_players)]
    if latest.empty:
        return {}, {}, engine_counts.astype(int).to_dict()

    lines = latest.groupby("player")["line"].median().astype(float).to_dict()
    ages = latest.groupby("player")["quote_age_minutes"].max().astype(float).to_dict()
    return lines, ages, engine_counts.astype(int).to_dict()


def build_allocation_paths(
    quotes: pd.DataFrame,
    *,
    engine_map: Mapping[str, str] | None = None,
    protocol: AllocationPathProtocol = ALLOCATION_PATH_PROTOCOL,
) -> PathBuildResult:
    """Build stable-subset allocation paths with deterministic pre-outcome exclusions."""

    missing = sorted(REQUIRED_QUOTE_COLUMNS - set(quotes.columns))
    if missing:
        raise ValueError(f"quotes are missing required columns: {missing}")

    frame = quotes.copy()
    frame["event_id"] = frame["event_id"].astype(str)
    frame["player"] = frame["player"].astype(str)
    frame["team"] = frame["team"].astype("string")
    frame["market"] = frame["market"].astype(str).str.lower()
    frame["book"] = frame["book"].astype(str).str.lower()
    frame["event_start_time_utc"] = pd.to_datetime(
        frame["event_start_time_utc"], utc=True, errors="coerce"
    )
    frame["snapshot_time_utc"] = pd.to_datetime(
        frame["snapshot_time_utc"], utc=True, errors="coerce"
    )
    frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    if "engine" not in frame:
        frame["engine"] = frame["book"]
    else:
        frame["engine"] = frame["engine"].fillna(frame["book"]).astype(str).str.lower()
    if engine_map:
        normalized_map = {str(key).lower(): str(value).lower() for key, value in engine_map.items()}
        frame["engine"] = frame["book"].map(normalized_map).fillna(frame["engine"])

    event_rows: list[dict[str, object]] = []
    player_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    for event_id, all_event_quotes in frame.groupby("event_id", sort=True):
        event_quotes = all_event_quotes.loc[
            all_event_quotes["market"].eq(protocol.market)
        ].copy()
        if event_quotes.empty:
            quality_rows.append(
                _quality_row(
                    event_id=event_id,
                    team="",
                    status=PathQualityStatus.MARKET_UNAVAILABLE,
                    reason=f"no {protocol.market} quotes",
                    protocol=protocol,
                )
            )
            continue

        start_times = event_quotes["event_start_time_utc"].dropna().unique()
        if len(start_times) != 1:
            quality_rows.append(
                _quality_row(
                    event_id=event_id,
                    team="",
                    status=PathQualityStatus.EVENT_TIME_AMBIGUOUS,
                    reason=f"expected one event start, observed {len(start_times)}",
                    protocol=protocol,
                )
            )
            continue
        event_start = pd.Timestamp(start_times[0])

        missing_team = event_quotes["team"].isna() | event_quotes["team"].str.strip().eq("")
        if bool(missing_team.any()):
            quality_rows.append(
                _quality_row(
                    event_id=event_id,
                    team="",
                    status=PathQualityStatus.TEAM_IDENTITY_MISSING,
                    reason="pregame quote rows must carry team identity; outcomes cannot repair it",
                    protocol=protocol,
                )
            )
            continue
        invalid_line = (
            event_quotes["line"].isna()
            | ~np.isfinite(event_quotes["line"])
            | event_quotes["line"].le(0.0)
        )
        if bool(invalid_line.any()):
            quality_rows.append(
                _quality_row(
                    event_id=event_id,
                    team="",
                    status=PathQualityStatus.INVALID_LINE,
                    reason="all player-point lines must be finite and positive",
                    protocol=protocol,
                )
            )
            continue

        for team, team_quotes in event_quotes.groupby("team", sort=True):
            checkpoint_lines: dict[int, dict[str, float]] = {}
            checkpoint_ages: dict[int, dict[str, float]] = {}
            checkpoint_engines: dict[int, dict[str, int]] = {}
            for offset in protocol.checkpoints_minutes:
                checkpoint = event_start + pd.Timedelta(minutes=offset)
                lines, ages, engines = _checkpoint_consensus(team_quotes, checkpoint, protocol)
                checkpoint_lines[offset] = lines
                checkpoint_ages[offset] = ages
                checkpoint_engines[offset] = engines

            player_sets = [set(checkpoint_lines[offset]) for offset in protocol.checkpoints_minutes]
            union_players = set().union(*player_sets)
            stable_players = set.intersection(*player_sets) if player_sets else set()
            coverage = len(stable_players) / len(union_players) if union_players else 0.0
            unit_kwargs = {
                "event_id": event_id,
                "team": str(team),
                "protocol": protocol,
                "stable_players": len(stable_players),
                "union_players": len(union_players),
                "stable_coverage": float(coverage),
            }

            if not union_players:
                any_engines = max(
                    (max(values.values(), default=0) for values in checkpoint_engines.values()),
                    default=0,
                )
                status = (
                    PathQualityStatus.INSUFFICIENT_ENGINES
                    if any_engines < protocol.minimum_independent_engines
                    else PathQualityStatus.MISSING_CHECKPOINT
                )
                quality_rows.append(
                    _quality_row(
                        **unit_kwargs,
                        status=status,
                        reason="no player has a fresh two-engine checkpoint quote",
                    )
                )
                continue
            if any(not players for players in player_sets):
                quality_rows.append(
                    _quality_row(
                        **unit_kwargs,
                        status=PathQualityStatus.MISSING_CHECKPOINT,
                        reason="at least one fixed checkpoint has no valid players",
                    )
                )
                continue
            if len(stable_players) < protocol.minimum_stable_players:
                quality_rows.append(
                    _quality_row(
                        **unit_kwargs,
                        status=PathQualityStatus.INSUFFICIENT_STABLE_PLAYERS,
                        reason=(
                            f"requires {protocol.minimum_stable_players} stable players at all checkpoints"
                        ),
                    )
                )
                continue
            if coverage < protocol.minimum_stable_coverage:
                quality_rows.append(
                    _quality_row(
                        **unit_kwargs,
                        status=PathQualityStatus.INSUFFICIENT_STABLE_COVERAGE,
                        reason=f"stable intersection/union coverage {coverage:.3f} is below floor",
                    )
                )
                continue

            ordered_players = sorted(stable_players)
            line_matrix = np.asarray(
                [
                    [checkpoint_lines[offset][player] for player in ordered_players]
                    for offset in protocol.checkpoints_minutes
                ],
                dtype=float,
            )
            team_totals = line_matrix.sum(axis=1)
            if bool(np.any(team_totals <= 0.0)):
                quality_rows.append(
                    _quality_row(
                        **unit_kwargs,
                        status=PathQualityStatus.INVALID_LINE,
                        reason="stable-subset team line total must be positive",
                    )
                )
                continue
            shares = line_matrix / team_totals[:, None]
            hhi = np.square(shares).sum(axis=1)
            entropy = np.asarray([_entropy(state) for state in shares], dtype=float)
            step_l1 = np.abs(np.diff(shares, axis=0)).sum(axis=1)
            displacement = float(np.abs(shares[-1] - shares[0]).sum())
            path_length = float(step_l1.sum())
            team_efficiency = displacement / path_length if path_length > 0 else 0.0
            unit_id = f"{event_id}::{team}::{protocol.market}"
            max_quote_age = max(
                checkpoint_ages[offset][player]
                for offset in protocol.checkpoints_minutes
                for player in ordered_players
            )
            min_engine_count = min(
                checkpoint_engines[offset][player]
                for offset in protocol.checkpoints_minutes
                for player in ordered_players
            )

            event_row = {
                "unit_id": unit_id,
                "event_id": event_id,
                "event_date": event_start.normalize().tz_localize(None),
                "event_start_time_utc": event_start,
                "team": str(team),
                "market": protocol.market,
                "representation_version": protocol.version,
                "stable_players": len(ordered_players),
                "union_players": len(union_players),
                "stable_coverage": float(coverage),
                "open_team_total": float(team_totals[0]),
                "close_team_total": float(team_totals[-1]),
                "open_hhi": float(hhi[0]),
                "close_hhi": float(hhi[-1]),
                "delta_hhi": float(hhi[-1] - hhi[0]),
                "open_entropy": float(entropy[0]),
                "close_entropy": float(entropy[-1]),
                "delta_entropy": float(entropy[-1] - entropy[0]),
                "allocation_displacement_l1": displacement,
                "allocation_path_length_l1": path_length,
                "allocation_path_efficiency": float(team_efficiency),
                "max_quote_age_minutes": float(max_quote_age),
                "minimum_engine_count": int(min_engine_count),
            }
            event_rows.append(event_row)

            for player_index, player in enumerate(ordered_players):
                player_shares = shares[:, player_index]
                player_tv = float(np.abs(np.diff(player_shares)).sum())
                player_displacement = float(abs(player_shares[-1] - player_shares[0]))
                player_efficiency = player_displacement / player_tv if player_tv > 0 else 0.0
                row = dict(event_row)
                row.update(
                    {
                        "player": player,
                        "open_line": float(line_matrix[0, player_index]),
                        "close_line": float(line_matrix[-1, player_index]),
                        "open_share": float(player_shares[0]),
                        "close_share": float(player_shares[-1]),
                        "delta_share": float(player_shares[-1] - player_shares[0]),
                        "player_total_variation": player_tv,
                        "player_path_efficiency": float(player_efficiency),
                        "direction_reversals": _direction_reversals(player_shares),
                    }
                )
                for offset_index, offset in enumerate(protocol.checkpoints_minutes):
                    row[f"share_m{abs(offset)}"] = float(player_shares[offset_index])
                    row[f"line_m{abs(offset)}"] = float(line_matrix[offset_index, player_index])
                player_rows.append(row)

            quality_rows.append(
                _quality_row(
                    **unit_kwargs,
                    status=PathQualityStatus.VALID,
                    reason="all frozen V1.1 representation gates passed",
                )
            )

    return PathBuildResult(
        player_features=pd.DataFrame(player_rows),
        event_features=pd.DataFrame(event_rows),
        quality_ledger=pd.DataFrame(quality_rows),
    )


def attach_realized_allocations(
    player_features: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> SettlementResult:
    """Attach outcomes without changing the pre-outcome stable player universe."""

    missing = sorted(REQUIRED_OUTCOME_COLUMNS - set(outcomes.columns))
    if missing:
        raise ValueError(f"outcomes are missing required columns: {missing}")
    if player_features.empty:
        return SettlementResult(
            settled_player_features=player_features.copy(),
            quality_ledger=_empty_frame(["unit_id", "event_id", "team", "status", "reason"]),
        )

    actuals = outcomes.copy()
    actuals["event_id"] = actuals["event_id"].astype(str)
    actuals["team"] = actuals["team"].astype(str)
    actuals["player"] = actuals["player"].astype(str)
    actuals["actual"] = pd.to_numeric(actuals["actual"], errors="coerce")

    settled_rows: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []
    keys = ["event_id", "team", "player"]
    duplicate_keys = actuals.duplicated(keys, keep=False)

    for unit_id, unit in player_features.groupby("unit_id", sort=True):
        event_id = str(unit["event_id"].iloc[0])
        team = str(unit["team"].iloc[0])
        relevant_duplicates = actuals.loc[
            duplicate_keys
            & actuals["event_id"].eq(event_id)
            & actuals["team"].eq(team)
            & actuals["player"].isin(unit["player"])
        ]
        if not relevant_duplicates.empty:
            quality_rows.append(
                {
                    "unit_id": unit_id,
                    "event_id": event_id,
                    "team": team,
                    "status": PathQualityStatus.DUPLICATE_OUTCOME.value,
                    "reason": "outcome keys are not unique",
                }
            )
            continue

        merged = unit.merge(actuals[keys + ["actual"]], on=keys, how="left", validate="one_to_one")
        if bool(merged["actual"].isna().any()):
            missing_players = sorted(merged.loc[merged["actual"].isna(), "player"].tolist())
            quality_rows.append(
                {
                    "unit_id": unit_id,
                    "event_id": event_id,
                    "team": team,
                    "status": PathQualityStatus.MISSING_ACTUAL.value,
                    "reason": f"entire unit rejected; missing actuals for {missing_players}",
                }
            )
            continue
        actual_total = float(merged["actual"].sum())
        if actual_total <= 0.0:
            quality_rows.append(
                {
                    "unit_id": unit_id,
                    "event_id": event_id,
                    "team": team,
                    "status": PathQualityStatus.INVALID_ACTUAL_TOTAL.value,
                    "reason": "stable-subset actual total must be positive",
                }
            )
            continue
        merged["realized_share"] = merged["actual"] / actual_total
        settled_rows.append(merged)
        quality_rows.append(
            {
                "unit_id": unit_id,
                "event_id": event_id,
                "team": team,
                "status": PathQualityStatus.VALID.value,
                "reason": "all stable players settled",
            }
        )

    settled = pd.concat(settled_rows, ignore_index=True) if settled_rows else player_features.iloc[0:0].copy()
    if "realized_share" not in settled:
        settled["realized_share"] = pd.Series(dtype=float)
    return SettlementResult(
        settled_player_features=settled,
        quality_ledger=pd.DataFrame(quality_rows),
    )
