"""Monte Carlo PPR projections and position-aware NFL draft rankings.

The model intentionally stays compact: recent player game logs establish each
player's role and correlated stat-line distribution, position priors stabilize
small samples, and last-season opponent results add a capped matchup adjustment.
Entire game vectors are resampled together so receptions, yards, and touchdowns
do not become independent synthetic events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .accuracy import train_accuracy_model, upcoming_accuracy_forecasts
from .lineup import apply_lineup_context, build_lineup_contexts, merge_roster_with_depth_chart


POSITIONS = ("QB", "RB", "WR", "TE")
DRAFT_DEPTH_CAPS = {"QB": 2, "RB": 5, "WR": 6, "TE": 4}
MODEL_STATS = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "special_teams_tds",
)
DISPLAY_STATS = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)


@dataclass(frozen=True)
class ScoringSettings:
    """Default full-PPR scoring."""

    passing_yards: float = 0.04
    passing_tds: float = 4.0
    interceptions: float = -2.0
    rushing_yards: float = 0.1
    rushing_tds: float = 6.0
    receptions: float = 1.0
    receiving_yards: float = 0.1
    receiving_tds: float = 6.0
    fumbles_lost: float = -2.0
    two_point_conversions: float = 2.0
    special_teams_tds: float = 6.0


@dataclass(frozen=True)
class FantasyConfig:
    season: int = 2026
    simulations: int = 2_000
    recent_games: int = 17
    half_life_games: float = 6.0
    prior_games: float = 4.0
    random_seed: int = 20260813
    published_players: int = 200
    # Twelve-team, 1QB, 2RB, 2WR, 1TE and one FLEX replacement levels.
    replacement_ranks: tuple[tuple[str, int], ...] = (
        ("QB", 13),
        ("RB", 31),
        ("WR", 37),
        ("TE", 13),
    )
    # Keep the published draft pool usable; NFL rosters contain far more
    # reserve quarterbacks and tight ends than a 12-team league would draft.
    draft_pool_caps: tuple[tuple[str, int], ...] = (
        ("QB", 24),
        ("RB", 70),
        ("WR", 80),
        ("TE", 30),
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def fantasy_points(
    frame: pd.DataFrame,
    scoring: ScoringSettings | None = None,
) -> pd.Series:
    """Translate a frame of player stats into full-PPR fantasy points."""

    rules = scoring or ScoringSettings()
    fumbles = _numeric(frame, "rushing_fumbles_lost") + _numeric(
        frame, "receiving_fumbles_lost"
    )
    conversions = sum(
        (_numeric(frame, column) for column in (
            "passing_2pt_conversions",
            "rushing_2pt_conversions",
            "receiving_2pt_conversions",
        )),
        start=pd.Series(0.0, index=frame.index, dtype=float),
    )
    return (
        _numeric(frame, "passing_yards") * rules.passing_yards
        + _numeric(frame, "passing_tds") * rules.passing_tds
        + _numeric(frame, "interceptions") * rules.interceptions
        + _numeric(frame, "rushing_yards") * rules.rushing_yards
        + _numeric(frame, "rushing_tds") * rules.rushing_tds
        + _numeric(frame, "receptions") * rules.receptions
        + _numeric(frame, "receiving_yards") * rules.receiving_yards
        + _numeric(frame, "receiving_tds") * rules.receiving_tds
        + fumbles * rules.fumbles_lost
        + conversions * rules.two_point_conversions
        + _numeric(frame, "special_teams_tds") * rules.special_teams_tds
    )


def _clean_history(history: pd.DataFrame, scoring: ScoringSettings) -> pd.DataFrame:
    frame = history.copy()
    if "season_type" in frame.columns:
        frame = frame.loc[frame["season_type"].eq("REG")].copy()
    if "player_display_name" not in frame.columns and "player_name" in frame.columns:
        frame["player_display_name"] = frame["player_name"]
    required = {"player_id", "position", "season", "week", "opponent_team"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fantasy history is missing required columns: {', '.join(missing)}")
    frame = frame.loc[frame["position"].isin(POSITIONS)].copy()
    for column in MODEL_STATS:
        frame[column] = _numeric(frame, column)
    frame["fantasy_points_ppr_model"] = fantasy_points(frame, scoring)
    return frame.sort_values(["season", "week", "player_id"]).reset_index(drop=True)


def _recency_weights(rows: int, half_life: float) -> np.ndarray:
    age = np.arange(rows - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / max(half_life, 0.1))
    return weights / weights.sum()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(values, axis=0, weights=weights)


def _position_priors(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    recent = frame.loc[frame["season"].ge(int(frame["season"].max()) - 1)].copy()
    priors: dict[str, np.ndarray] = {}
    for position in POSITIONS:
        group = recent.loc[recent["position"].eq(position)]
        if group.empty:
            priors[position] = np.zeros(len(MODEL_STATS), dtype=float)
            continue
        player_means = group.groupby("player_id")[list(MODEL_STATS)].mean()
        # The 60th percentile of all rostered players is a conservative proxy
        # for an unknown player's usable role without manufacturing starter status.
        priors[position] = player_means.quantile(0.60).to_numpy(dtype=float)
    return priors


def _opponent_factors(frame: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    last_season = int(frame["season"].max())
    recent = frame.loc[frame["season"].eq(last_season)].copy()
    factors: dict[tuple[str, str, str], float] = {}
    for position in POSITIONS:
        group = recent.loc[recent["position"].eq(position)]
        if group.empty:
            continue
        for stat in DISPLAY_STATS:
            allowed = group.groupby("opponent_team")[stat].mean()
            league = float(group[stat].mean())
            if league <= 0:
                continue
            # Shrink and cap noisy one-year defense effects.
            shrunk = 1.0 + 0.45 * (allowed / league - 1.0)
            for opponent, value in shrunk.clip(0.82, 1.18).items():
                factors[(position, str(opponent), stat)] = float(value)
    return factors


def _normalize_roster(
    roster: pd.DataFrame,
    season: int,
    depth_chart: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if depth_chart is not None:
        frame = merge_roster_with_depth_chart(roster, depth_chart, season=season)
        required = {"player_id", "player_display_name", "recent_team", "position"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Fantasy roster is missing required columns: {', '.join(missing)}")
        return frame.dropna(subset=list(required)).reset_index(drop=True)
    frame = roster.rename(
        columns={
            "gsis_id": "player_id",
            "full_name": "player_display_name",
            "team": "recent_team",
        }
    ).copy()
    required = {"player_id", "player_display_name", "recent_team", "position"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fantasy roster is missing required columns: {', '.join(missing)}")
    if "season" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(season)]
    if "status" in frame.columns:
        frame = frame.loc[~frame["status"].astype(str).isin({"CUT", "RET"})]
    return (
        frame.loc[frame["position"].isin(POSITIONS)]
        .dropna(subset=list(required))
        .drop_duplicates("player_id", keep="last")
        .reset_index(drop=True)
    )


def _player_schedule(schedule: pd.DataFrame, team: str, season: int) -> list[dict[str, Any]]:
    frame = schedule.copy()
    if "game_type" in frame.columns:
        frame = frame.loc[frame["game_type"].eq("REG")]
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(season)]
    games = frame.loc[frame["home_team"].eq(team) | frame["away_team"].eq(team)].copy()
    output: list[dict[str, Any]] = []
    for row in games.sort_values("week").itertuples(index=False):
        opponent = row.away_team if row.home_team == team else row.home_team
        output.append({"week": int(row.week), "opponent": str(opponent)})
    return output


def _stat_points_array(stats: np.ndarray, scoring: ScoringSettings) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(MODEL_STATS)}
    return (
        stats[..., index["passing_yards"]] * scoring.passing_yards
        + stats[..., index["passing_tds"]] * scoring.passing_tds
        + stats[..., index["interceptions"]] * scoring.interceptions
        + stats[..., index["rushing_yards"]] * scoring.rushing_yards
        + stats[..., index["rushing_tds"]] * scoring.rushing_tds
        + stats[..., index["receptions"]] * scoring.receptions
        + stats[..., index["receiving_yards"]] * scoring.receiving_yards
        + stats[..., index["receiving_tds"]] * scoring.receiving_tds
        + (
            stats[..., index["rushing_fumbles_lost"]]
            + stats[..., index["receiving_fumbles_lost"]]
        )
        * scoring.fumbles_lost
        + (
            stats[..., index["passing_2pt_conversions"]]
            + stats[..., index["rushing_2pt_conversions"]]
            + stats[..., index["receiving_2pt_conversions"]]
        )
        * scoring.two_point_conversions
        + stats[..., index["special_teams_tds"]] * scoring.special_teams_tds
    )


def _round_map(values: np.ndarray, divisor: float = 1.0) -> dict[str, float]:
    return {
        stat: round(float(values[index]) / divisor, 2)
        for index, stat in enumerate(MODEL_STATS)
        if stat in DISPLAY_STATS
    }


def _distribution_curve(values: np.ndarray, points: int = 33) -> list[dict[str, float]]:
    """Compress simulated season outcomes into a smooth, frontend-ready KDE."""

    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return []
    q01, q99 = np.quantile(sample, [0.01, 0.99])
    mean = float(sample.mean())
    median = float(np.median(sample))
    low = float(min(q01, mean, median))
    high = float(max(q99, mean, median))
    if high - low < 1.0:
        low -= 0.5
        high += 0.5
    spread = float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0
    iqr = float(np.subtract(*np.quantile(sample, [0.75, 0.25])))
    robust_spread = min(spread, iqr / 1.34) if spread > 0 and iqr > 0 else max(spread, iqr / 1.34)
    bandwidth = max(0.9 * robust_spread * sample.size ** (-0.2), (high - low) / 80.0, 0.5)
    grid = np.linspace(low, high, points)
    z = (grid[:, None] - sample[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))
    peak = float(density.max())
    normalized = density / peak if peak > 0 else np.zeros_like(density)
    return [
        {"value": round(float(value), 2), "density": round(float(weight), 4)}
        for value, weight in zip(grid, normalized)
    ]


def _simulate_player(
    row: Any,
    history: pd.DataFrame,
    games: list[dict[str, Any]],
    priors: dict[str, np.ndarray],
    matchup_factors: dict[tuple[str, str, str], float],
    config: FantasyConfig,
    scoring: ScoringSettings,
    rng: np.random.Generator,
    accuracy_forecasts: dict[tuple[str, int, str], dict[str, float]] | None = None,
    lineup_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not games:
        return None
    player_history = history.loc[history["player_id"].astype(str).eq(str(row.player_id))].tail(
        config.recent_games
    )
    position = str(row.position)
    prior = priors[position]
    if player_history.empty:
        values = np.atleast_2d(prior)
        weights = np.ones(1)
        player_mean = prior.copy()
        confidence = "prior only"
    else:
        values = player_history[list(MODEL_STATS)].to_numpy(dtype=float)
        weights = _recency_weights(len(values), config.half_life_games)
        player_mean = _weighted_mean(values, weights)
        confidence = "established" if len(values) >= 10 else "developing"
    credibility = len(player_history) / (len(player_history) + config.prior_games)
    baseline = credibility * player_mean + (1.0 - credibility) * prior
    centered = values - _weighted_mean(values, weights)
    samples = rng.choice(len(values), size=(config.simulations, len(games)), p=weights)
    residuals = centered[samples]
    role_sigma = 0.08 if confidence == "established" else (0.13 if confidence == "developing" else 0.22)
    season_role = rng.lognormal(-0.5 * role_sigma**2, role_sigma, size=(config.simulations, 1, 1))
    simulated = baseline.reshape(1, 1, -1) * season_role + residuals
    for game_index, game in enumerate(games):
        for stat_index, stat in enumerate(MODEL_STATS):
            factor = matchup_factors.get((position, game["opponent"], stat), 1.0)
            simulated[:, game_index, stat_index] *= factor
    simulated = np.maximum(simulated, 0.0)
    forecast_widths: list[float] = []
    if accuracy_forecasts:
        for game_index, game in enumerate(games):
            forecast = accuracy_forecasts.get(
                (str(row.player_id), int(game["week"]), str(game["opponent"]))
            )
            if not forecast:
                continue
            raw_points = _stat_points_array(simulated[:, game_index, :], scoring)
            raw_mean = float(raw_points.mean())
            if raw_mean > 0.1:
                # Preserve the sampled stat-line shape while centering each game
                # on the regularized model's forward prediction.
                adjustment = np.clip(float(forecast["mean"]) / raw_mean, 0.50, 1.50)
                simulated[:, game_index, :] *= adjustment
            forecast_widths.append(float(forecast["interval_half_width"]))
    if lineup_context:
        simulated = apply_lineup_context(
            simulated,
            stat_names=MODEL_STATS,
            context=lineup_context,
            rng=rng,
        )
    points = _stat_points_array(simulated, scoring)
    season_points = points.sum(axis=1)
    season_stats = simulated.sum(axis=1)
    mean_totals = season_stats.mean(axis=0)
    p10, p50, p90 = np.quantile(season_points, [0.10, 0.50, 0.90])
    mean_points = float(season_points.mean())
    expected_games = (
        float(lineup_context.get("expected_games", len(games)))
        if lineup_context
        else float(len(games))
    )
    ppg = mean_points / max(expected_games, 1.0)
    volatility = float(points.std() / max(points.mean(), 0.1))
    interval_width = float(np.mean(forecast_widths)) if forecast_widths else None
    if interval_width is not None:
        confidence = "high" if interval_width <= 6.0 else "medium" if interval_width <= 9.0 else "low"
    if lineup_context:
        role_uncertainty = float(lineup_context.get("role_uncertainty", 0.10))
        active_probability = float(lineup_context.get("active_probability", 1.0))
        if lineup_context.get("changed_team") or role_uncertainty >= 0.30 or active_probability < 0.65:
            confidence = "low"
        elif role_uncertainty >= 0.22 or active_probability < 0.85:
            confidence = "medium" if confidence == "high" else confidence
    return {
        "player_id": str(row.player_id),
        "player": str(row.player_display_name),
        "team": str(row.recent_team),
        "position": position,
        "games": round(expected_games, 2),
        "schedule_games": len(games),
        "projection_confidence": confidence,
        "history_games": int(len(player_history)),
        "fantasy_points": {
            "per_game": round(ppg, 2),
            "season_mean": round(mean_points, 1),
            "season_p10": round(float(p10), 1),
            "season_median": round(float(p50), 1),
            "season_p90": round(float(p90), 1),
            "distribution": _distribution_curve(season_points),
        },
        "projected_stats": {
            "per_game": _round_map(mean_totals, max(expected_games, 1.0)),
            "season_total": _round_map(mean_totals),
        },
        "weekly_volatility": round(volatility, 3),
        "weekly_interval_half_width": round(interval_width, 2) if interval_width is not None else None,
        "lineup": {
            key: value
            for key, value in (lineup_context or {}).items()
            if key != "multipliers"
        },
    }


def _replacement_values(players: list[dict[str, Any]], config: FantasyConfig) -> dict[str, float]:
    levels: dict[str, float] = {}
    for position, rank in dict(config.replacement_ranks).items():
        values = sorted(
            (item["fantasy_points"]["per_game"] for item in players if item["position"] == position),
            reverse=True,
        )
        levels[position] = float(values[min(rank - 1, len(values) - 1)]) if values else 0.0
    return levels


def _tier(rank: int) -> int:
    if rank <= 12:
        return 1
    if rank <= 30:
        return 2
    if rank <= 60:
        return 3
    if rank <= 100:
        return 4
    if rank <= 150:
        return 5
    return 6


def _assessment(item: dict[str, Any]) -> str:
    pos_rank = item["position_rank"]
    position = item["position"]
    label = (
        f"Elite {position}1"
        if pos_rank <= 5
        else f"Starting {position}{1 if pos_rank <= 12 else 2}"
        if pos_rank <= 24
        else f"Depth {position}"
    )
    volatility = item["weekly_volatility"]
    shape = "high weekly floor" if volatility < 0.65 else "balanced range" if volatility < 0.95 else "boom/bust range"
    confidence = item["projection_confidence"]
    return f"{label} · {shape} · {confidence} projection"


def build_draft_rankings(
    history: pd.DataFrame,
    roster: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    config: FantasyConfig | None = None,
    scoring: ScoringSettings | None = None,
    accuracy_bundle: dict[str, Any] | None = None,
    depth_chart: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate the upcoming season and return a frontend-ready ranking payload."""

    cfg = config or FantasyConfig()
    rules = scoring or ScoringSettings()
    clean = _clean_history(history, rules)
    current = _normalize_roster(roster, cfg.season, depth_chart)
    lineup_contexts: dict[str, dict[str, Any]] = {}
    team_opportunity_audit: list[dict[str, Any]] = []
    if depth_chart is not None:
        lineup_contexts, team_opportunity_audit = build_lineup_contexts(clean, current)
    priors = _position_priors(clean)
    factors = _opponent_factors(clean)
    accuracy_forecasts: dict[tuple[str, int, str], dict[str, float]] = {}
    if accuracy_bundle:
        accuracy_forecasts = upcoming_accuracy_forecasts(
            clean,
            current,
            schedule,
            accuracy_bundle,
            season=cfg.season,
        )
    rng = np.random.default_rng(cfg.random_seed)
    players: list[dict[str, Any]] = []
    schedule_cache: dict[str, list[dict[str, Any]]] = {}
    for row in current.sort_values(["position", "recent_team", "player_id"]).itertuples(index=False):
        team = str(row.recent_team)
        games = schedule_cache.setdefault(team, _player_schedule(schedule, team, cfg.season))
        projection = _simulate_player(
            row,
            clean,
            games,
            priors,
            factors,
            cfg,
            rules,
            rng,
            accuracy_forecasts,
            lineup_contexts.get(str(row.player_id)),
        )
        if projection is not None:
            players.append(projection)

    replacement = _replacement_values(players, cfg)
    for item in players:
        points = item["fantasy_points"]
        replacement_ppg = replacement[item["position"]]
        # Keep VORP signed. Flooring it at zero lets high-volume backup QBs
        # outrank genuinely draftable RB/WR depth on raw season points alone.
        value = (points["per_game"] - replacement_ppg) * item["games"]
        upside = max(0.0, points["season_p90"] - points["season_mean"])
        item["replacement_ppg"] = round(replacement_ppg, 2)
        item["value_over_replacement"] = round(value, 1)
        item["draft_score"] = round(value + 0.14 * points["season_mean"] + 0.06 * upside, 2)
    simulated_count = len(players)
    excluded_by_lineup = 0
    draft_pool: list[dict[str, Any]] = []
    for position, cap in dict(cfg.draft_pool_caps).items():
        eligible_players = []
        for item in players:
            if item["position"] != position:
                continue
            depth_rank = item.get("lineup", {}).get("depth_rank")
            if depth_chart is not None and (
                depth_rank is None or int(depth_rank) > DRAFT_DEPTH_CAPS[position]
            ):
                excluded_by_lineup += 1
                continue
            eligible_players.append(item)
        position_players = sorted(
            eligible_players,
            key=lambda item: (item["draft_score"], item["fantasy_points"]["season_mean"]),
            reverse=True,
        )
        draft_pool.extend(position_players[:cap])
    players = draft_pool
    players.sort(
        key=lambda item: (item["draft_score"], item["fantasy_points"]["season_mean"], item["player"]),
        reverse=True,
    )
    position_counts = {position: 0 for position in POSITIONS}
    for rank, item in enumerate(players, start=1):
        position_counts[item["position"]] += 1
        item["rank"] = rank
        item["position_rank"] = position_counts[item["position"]]
        item["tier"] = _tier(rank)
        item["assessment"] = _assessment(item)

    published = players[: cfg.published_players]
    audit_frame = clean[["player_id", "season", "week", *MODEL_STATS]]
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(audit_frame, index=False).values.tobytes()
    ).hexdigest()
    allocation_tolerance = 0.002
    allocation_checks = [
        abs(float(audit["budgets_per_game"][stat]) - float(audit["allocated_per_game"][stat]))
        <= allocation_tolerance
        for audit in team_opportunity_audit
        for stat in audit["budgets_per_game"]
    ]
    depth_coverage = (
        float(pd.to_numeric(current.get("depth_rank"), errors="coerce").notna().mean())
        if "depth_rank" in current and len(current)
        else 0.0
    )
    lineup_status = (
        "passed"
        if allocation_checks and all(allocation_checks)
        else "not_applied"
        if depth_chart is None
        else "failed"
    )
    lineup_validation = {
        "status": lineup_status,
        "teams": len(team_opportunity_audit),
        "depth_chart_player_coverage": round(depth_coverage, 4),
        "finite_budget_checks": len(allocation_checks),
        "finite_budget_checks_passed": int(sum(allocation_checks)),
        "allocation_tolerance_per_game": allocation_tolerance,
    }
    depth_dates = current.get("depth_as_of_utc", pd.Series(dtype=object)).dropna().astype(str)
    return {
        "schema_version": 1,
        "league": "NFL",
        "season": cfg.season,
        "format": "12-team full PPR",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": {
            "name": "weighted-game-resampling-monte-carlo",
            "simulations": cfg.simulations,
            "recent_games": cfg.recent_games,
            "half_life_games": cfg.half_life_games,
            "opponent_adjustment_range": [0.82, 1.18],
            "random_seed": cfg.random_seed,
            "accuracy_layer": "position-specific regularized CatBoost MAE" if accuracy_bundle else "recency baseline",
            "lineup_layer": "depth-chart scenarios with finite team opportunity budgets" if depth_chart is not None else "not applied",
            "depth_chart_as_of_utc": depth_dates.max() if not depth_dates.empty else None,
            "source_history_sha256": source_hash,
        },
        "scoring": asdict(rules),
        "replacement_levels": {key: round(value, 2) for key, value in replacement.items()},
        "players_simulated": simulated_count,
        "players_excluded_by_lineup": excluded_by_lineup,
        "draft_pool_players": len(players),
        "players_published": len(published),
        "rankings": published,
        "lineup_validation": lineup_validation,
        "team_opportunity_audit": team_opportunity_audit,
        "method_note": (
            "Draft score combines position-adjusted value over replacement, mean season points, "
            "and simulated upside. Current depth-chart scenarios share each team's finite target, "
            "carry, pass-attempt, and touchdown budgets; rankings are not guarantees or injury reports."
        ),
    }


def validate_projection_model(
    history: pd.DataFrame,
    *,
    holdout_season: int = 2025,
    scoring: ScoringSettings | None = None,
    minimum_prior_games: int = 4,
) -> dict[str, Any]:
    """Return seen/unseen, overfit, and calibrated-confidence diagnostics."""

    report, _ = fit_accuracy_layer(
        history,
        holdout_season=holdout_season,
        scoring=scoring,
        minimum_prior_games=minimum_prior_games,
    )
    return report


def fit_accuracy_layer(
    history: pd.DataFrame,
    *,
    holdout_season: int = 2025,
    scoring: ScoringSettings | None = None,
    minimum_prior_games: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare history, select the point model, and return report plus artifact."""

    rules = scoring or ScoringSettings()
    frame = _clean_history(history, rules)
    return train_accuracy_model(
        frame,
        holdout_season=holdout_season,
        minimum_prior_games=minimum_prior_games,
    )
