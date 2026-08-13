"""Team-constrained opportunity and lineup scenario model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
BUDGET_STATS = (
    "attempts",
    "targets",
    "carries",
    "passing_tds",
    "receiving_tds",
    "rushing_tds",
)


@dataclass(frozen=True)
class LineupConfig:
    recent_games: int = 17
    budget_seasons: int = 2
    league_shrinkage: float = 0.15
    maximum_multiplier: float = 1.75
    minimum_multiplier: float = 0.20


DEPTH_USAGE = {
    "QB": {1: 0.90, 2: 0.075, 3: 0.02, 4: 0.005},
    "RB": {1: 0.92, 2: 0.72, 3: 0.38, 4: 0.20, 5: 0.10},
    "WR": {1: 0.95, 2: 0.91, 3: 0.85, 4: 0.42, 5: 0.24, 6: 0.14},
    "TE": {1: 0.92, 2: 0.52, 3: 0.25, 4: 0.12},
}

POSITION_PRIORS = {
    "QB": {"attempts": 18.0, "targets": 0.0, "carries": 2.2, "passing_tds": 0.9, "receiving_tds": 0.0, "rushing_tds": 0.12},
    "RB": {"attempts": 0.0, "targets": 1.4, "carries": 4.5, "passing_tds": 0.0, "receiving_tds": 0.08, "rushing_tds": 0.18},
    "WR": {"attempts": 0.0, "targets": 2.4, "carries": 0.12, "passing_tds": 0.0, "receiving_tds": 0.16, "rushing_tds": 0.01},
    "TE": {"attempts": 0.0, "targets": 1.8, "carries": 0.02, "passing_tds": 0.0, "receiving_tds": 0.11, "rushing_tds": 0.0},
}


def merge_roster_with_depth_chart(
    roster: pd.DataFrame,
    depth_chart: pd.DataFrame | None,
    *,
    season: int,
) -> pd.DataFrame:
    """Make the latest depth chart authoritative and retain unranked camp players."""

    frame = roster.rename(
        columns={"gsis_id": "player_id", "full_name": "player_display_name", "team": "recent_team"}
    ).copy()
    if "season" in frame:
        frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(season)]
    if "status" in frame:
        frame = frame.loc[~frame["status"].astype(str).isin({"CUT", "RET"})]
    frame = frame.loc[frame["position"].isin(SKILL_POSITIONS)].drop_duplicates("player_id", keep="last")
    frame["depth_rank"] = np.nan
    frame["depth_as_of_utc"] = None
    if depth_chart is None or depth_chart.empty:
        return frame.reset_index(drop=True)

    depth = depth_chart.copy()
    if "dt" in depth.columns:
        depth = depth.loc[depth["dt"].astype(str).eq(str(depth["dt"].max()))]
    depth = depth.rename(
        columns={
            "gsis_id": "player_id",
            "player_name": "depth_player_name",
            "team": "depth_team",
            "pos_abb": "depth_position",
            "pos_rank": "depth_rank_value",
            "dt": "depth_as_of_value",
        }
    )
    depth = depth.loc[depth["depth_position"].isin(SKILL_POSITIONS)].dropna(subset=["player_id"])
    depth = depth.sort_values("depth_rank_value").drop_duplicates("player_id", keep="first")
    depth_fields = [
        "player_id",
        "depth_player_name",
        "depth_team",
        "depth_position",
        "depth_rank_value",
        "depth_as_of_value",
    ]
    merged = frame.merge(depth[depth_fields], on="player_id", how="outer")
    merged["player_display_name"] = merged["depth_player_name"].fillna(merged["player_display_name"])
    merged["recent_team"] = merged["depth_team"].fillna(merged["recent_team"])
    merged["position"] = merged["depth_position"].fillna(merged["position"])
    merged["depth_rank"] = pd.to_numeric(merged["depth_rank_value"], errors="coerce")
    merged["depth_as_of_utc"] = merged["depth_as_of_value"]
    merged["status"] = merged.get("status", pd.Series(index=merged.index, dtype=object)).fillna("ACT")
    merged["years_exp"] = pd.to_numeric(
        merged.get("years_exp", pd.Series(index=merged.index, dtype=float)), errors="coerce"
    ).fillna(0)
    history_available = merged.get(
        "history_available", pd.Series(index=merged.index, dtype=object)
    )
    merged["history_available"] = history_available.where(
        history_available.notna(), False
    ).astype(bool)
    return merged.loc[
        merged["position"].isin(SKILL_POSITIONS)
        & merged["player_id"].notna()
        & merged["recent_team"].notna()
    ].drop(columns=[column for column in depth_fields[1:] if column in merged]).reset_index(drop=True)


def _weighted_rate(logs: pd.DataFrame, column: str, prior: float) -> float:
    if logs.empty or column not in logs:
        return prior
    values = pd.to_numeric(logs[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ages = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / 6.0)
    observed = float(np.average(values, weights=weights))
    credibility = len(values) / (len(values) + 3.0)
    return credibility * observed + (1.0 - credibility) * prior


def _health_probability(logs: pd.DataFrame, years_exp: float) -> float:
    if logs.empty:
        return 0.62 if years_exp <= 1 else 0.74
    season_games = logs.groupby("season")["week"].nunique().sort_index().tail(3)
    recency = np.array([0.15, 0.30, 0.55], dtype=float)[-len(season_games) :]
    recency /= recency.sum()
    observed = float(np.average(np.clip(season_games.to_numpy(dtype=float) / 17.0, 0, 1), weights=recency))
    probability = (observed * len(season_games) + 0.84 * 2.0) / (len(season_games) + 2.0)
    if years_exp > 8:
        probability -= min(0.10, 0.015 * (years_exp - 8))
    return float(np.clip(probability, 0.45, 0.96))


def _depth_usage(position: str, rank: float | None) -> float:
    if rank is None or pd.isna(rank):
        return 0.04
    return DEPTH_USAGE[position].get(int(rank), 0.05 if int(rank) <= 8 else 0.025)


def _team_budgets(history: pd.DataFrame, config: LineupConfig) -> dict[str, dict[str, float]]:
    last_season = int(history["season"].max())
    recent = history.loc[history["season"].ge(last_season - config.budget_seasons + 1)].copy()
    for column in BUDGET_STATS:
        if column not in recent:
            recent[column] = 0.0
    weekly = recent.groupby(["season", "week", "recent_team"], as_index=False)[list(BUDGET_STATS)].sum()
    league = weekly[list(BUDGET_STATS)].mean()
    budgets: dict[str, dict[str, float]] = {}
    for team, rows in weekly.groupby("recent_team"):
        seasons = rows.groupby("season")[list(BUDGET_STATS)].mean().sort_index()
        weights = np.array([0.35, 0.65], dtype=float)[-len(seasons) :]
        weights /= weights.sum()
        values = np.average(seasons.to_numpy(dtype=float), axis=0, weights=weights)
        shrunk = (1.0 - config.league_shrinkage) * values + config.league_shrinkage * league.to_numpy(dtype=float)
        budgets[str(team)] = {column: float(shrunk[index]) for index, column in enumerate(BUDGET_STATS)}
    return budgets


def build_lineup_contexts(
    history: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    config: LineupConfig | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Allocate finite team opportunities and return player scenario contexts."""

    cfg = config or LineupConfig()
    budgets = _team_budgets(history, cfg)
    league_budget = {
        column: float(np.mean([values[column] for values in budgets.values()]))
        for column in BUDGET_STATS
    }
    contexts: dict[str, dict[str, Any]] = {}
    team_audit: list[dict[str, Any]] = []
    latest_history = history.sort_values(["season", "week"]).drop_duplicates(
        "player_id", keep="last"
    ).copy()
    latest_history.index = latest_history["player_id"].astype(str)
    for team, team_roster in roster.groupby("recent_team", sort=True):
        team = str(team)
        budget = budgets.get(team, league_budget)
        rows: list[dict[str, Any]] = []
        for player in team_roster.itertuples(index=False):
            player_id = str(player.player_id)
            position = str(player.position)
            logs = history.loc[history["player_id"].astype(str).eq(player_id)].tail(cfg.recent_games)
            years_exp = float(getattr(player, "years_exp", 0) or 0)
            rank_value = getattr(player, "depth_rank", np.nan)
            usage = _depth_usage(position, rank_value)
            health = _health_probability(logs, years_exp)
            latest_team = str(latest_history.loc[player_id]["recent_team"]) if player_id in latest_history.index else None
            changed_team = bool(latest_team and latest_team != team)
            active_probability = usage * health
            rates = {
                column: _weighted_rate(logs, column, POSITION_PRIORS[position][column])
                for column in BUDGET_STATS
            }
            rows.append(
                {
                    "player_id": player_id,
                    "player": str(player.player_display_name),
                    "position": position,
                    "depth_rank": None if pd.isna(rank_value) else int(rank_value),
                    "depth_usage": usage,
                    "health_probability": health,
                    "active_probability": active_probability,
                    "changed_team": changed_team,
                    "rates": rates,
                }
            )

        qbs = [row for row in rows if row["position"] == "QB"]
        qb_total = sum(row["active_probability"] for row in qbs) or 1.0
        for row in qbs:
            row["active_probability"] /= qb_total

        def allocate(column: str, eligible_positions: set[str]) -> None:
            eligible = [row for row in rows if row["position"] in eligible_positions]
            weights = [row["active_probability"] * max(row["rates"][column], 0.01) for row in eligible]
            total_weight = sum(weights) or 1.0
            for row, weight in zip(eligible, weights):
                expected = budget[column] * weight / total_weight
                conditional = expected / max(row["active_probability"], 0.01)
                raw = max(row["rates"][column], 0.05)
                row.setdefault("allocated", {})[column] = expected
                row.setdefault("conditional", {})[column] = conditional
                row.setdefault("multipliers", {})[column] = float(
                    np.clip(conditional / raw, cfg.minimum_multiplier, cfg.maximum_multiplier)
                )

        allocate("attempts", {"QB"})
        allocate("targets", {"RB", "WR", "TE"})
        allocate("carries", {"QB", "RB", "WR", "TE"})
        allocate("passing_tds", {"QB"})
        allocate("receiving_tds", {"RB", "WR", "TE"})
        allocate("rushing_tds", {"QB", "RB", "WR", "TE"})

        for row in rows:
            uncertainty = 0.10
            if row["changed_team"]:
                uncertainty += 0.12
            if row["depth_rank"] is None or row["depth_rank"] > 2:
                uncertainty += 0.12
            if row["active_probability"] < 0.65:
                uncertainty += 0.10
            contexts[row["player_id"]] = {
                "team": team,
                "position": row["position"],
                "depth_rank": row["depth_rank"],
                "active_probability": round(float(row["active_probability"]), 5),
                "health_probability": round(float(row["health_probability"]), 5),
                "changed_team": row["changed_team"],
                "role_uncertainty": round(min(0.42, uncertainty), 4),
                "expected_games": round(17.0 * float(row["active_probability"]), 2),
                "conditional_opportunities": {
                    key: round(float(value), 3) for key, value in row.get("conditional", {}).items()
                },
                "multipliers": row.get("multipliers", {}),
            }
        audit = {
            "team": team,
            "budgets_per_game": {key: round(float(value), 3) for key, value in budget.items()},
            "allocated_per_game": {
                column: round(sum(row.get("allocated", {}).get(column, 0.0) for row in rows), 3)
                for column in BUDGET_STATS
            },
            "qb_scenarios": [
                {
                    "player": row["player"],
                    "depth_rank": row["depth_rank"],
                    "start_probability": round(float(row["active_probability"]), 4),
                }
                for row in sorted(qbs, key=lambda value: value["active_probability"], reverse=True)
            ],
            "target_tree": [
                {
                    "player": row["player"],
                    "position": row["position"],
                    "depth_rank": row["depth_rank"],
                    "expected_targets_per_team_game": round(float(row.get("allocated", {}).get("targets", 0.0)), 3),
                    "conditional_targets": round(float(row.get("conditional", {}).get("targets", 0.0)), 3),
                    "active_probability": round(float(row["active_probability"]), 4),
                }
                for row in sorted(rows, key=lambda value: value.get("allocated", {}).get("targets", 0.0), reverse=True)
                if row["position"] in {"RB", "WR", "TE"}
            ][:12],
        }
        team_audit.append(audit)
    return contexts, team_audit


def apply_lineup_context(
    simulated: np.ndarray,
    *,
    stat_names: tuple[str, ...],
    context: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply role budgets and availability scenarios to simulated stat lines."""

    output = simulated.copy()
    index = {name: idx for idx, name in enumerate(stat_names)}
    multipliers = context.get("multipliers", {})
    mappings = {
        "attempts": ("passing_yards", "interceptions", "passing_2pt_conversions"),
        "passing_tds": ("passing_tds",),
        "targets": ("receptions", "receiving_yards", "receiving_2pt_conversions"),
        "receiving_tds": ("receiving_tds",),
        "carries": ("rushing_yards", "rushing_2pt_conversions"),
        "rushing_tds": ("rushing_tds",),
    }
    role_sigma = float(context.get("role_uncertainty", 0.10))
    role_noise = rng.lognormal(
        mean=-0.5 * role_sigma**2,
        sigma=role_sigma,
        size=(output.shape[0], 1),
    )
    for opportunity, stats in mappings.items():
        multiplier = float(multipliers.get(opportunity, 1.0))
        for stat in stats:
            if stat in index:
                output[..., index[stat]] *= multiplier * role_noise
    active_probability = float(context.get("active_probability", 1.0))
    active = rng.binomial(1, active_probability, size=output.shape[:2]).astype(float)
    output *= active[..., None]
    return np.maximum(output, 0.0)
