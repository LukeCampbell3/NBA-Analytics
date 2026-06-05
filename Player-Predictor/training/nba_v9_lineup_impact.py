#!/usr/bin/env python3
"""
NBA v9 Lineup Impact & Teammate Dependency Model

Research basis:
  - One player's usage rises when another player sits
  - One player's points affects another player's assists
  - One player's rebounds depend on teammate/opponent shot profile
  - Blowouts reduce ALL starter minutes (correlated unders)
  - NBA props are coupled, not independent

This module:
  1. Models how lineup changes shift individual player distributions
  2. Quantifies teammate-out usage/opportunity spikes
  3. Detects role changes from lineup context
  4. Provides lineup-adjusted stat projections
  5. Identifies "opportunity environment" for each prop
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class LineupContext:
    """Context about tonight's lineup and its impact on a player's prop."""
    player: str
    teammates_out: list[str] = field(default_factory=list)
    teammates_in: list[str] = field(default_factory=list)
    usage_boost: float = 0.0          # Expected USG% increase from lineup changes
    minutes_boost: float = 0.0        # Expected minutes increase
    opportunity_score: float = 0.5    # 0-1 scale of opportunity environment
    role_change_detected: bool = False
    confidence: float = 0.5           # How confident we are in the adjustment


@dataclass
class TeammateImpact:
    """Quantified impact of a specific teammate being out."""
    teammate: str
    target_player: str
    stat: str
    games_without: int = 0
    games_with: int = 0
    stat_mean_without: float = 0.0
    stat_mean_with: float = 0.0
    stat_delta: float = 0.0           # mean_without - mean_with
    stat_delta_pct: float = 0.0       # percentage change
    usage_delta: float = 0.0          # USG% change
    minutes_delta: float = 0.0        # MP change
    significance: float = 0.0         # Statistical significance (0-1)


@dataclass
class OpportunityEnvironment:
    """The opportunity environment for a player's prop tonight."""
    player: str
    stat: str
    base_projection: float = 0.0      # Projection without lineup context
    lineup_adjusted: float = 0.0      # Projection with lineup context
    adjustment_magnitude: float = 0.0  # How much lineup changes the projection
    pace_factor: float = 1.0          # Pace multiplier
    minutes_projection: float = 0.0   # Expected minutes
    usage_projection: float = 0.0     # Expected usage rate
    opportunity_rating: str = ""      # "elevated", "normal", "suppressed"
    key_factors: list[str] = field(default_factory=list)


class LineupImpactModel:
    """
    Models how lineup changes affect individual player stat distributions.

    Core insight: A player's prop outcome is heavily influenced by who else
    is on the court. When a high-usage teammate is out, the remaining players
    absorb those touches, shots, and opportunities.

    This is NOT just "player X averages more when player Y is out."
    It's about how the DISTRIBUTION shifts - the median, the variance,
    and the tail probabilities all change.
    """

    def __init__(self, min_games: int = 5):
        self.min_games = min_games
        self._teammate_impacts: dict[str, dict[str, list[TeammateImpact]]] = {}
        self._player_baselines: dict[str, dict] = {}
        self._lineup_history: dict[str, pd.DataFrame] = {}

    def fit(
        self,
        game_logs: pd.DataFrame,
        player_col: str = "Player",
        date_col: str = "Date",
        teammates_col: str = "Teammates",
    ) -> "LineupImpactModel":
        """
        Fit lineup impact model from historical game logs.

        Args:
            game_logs: DataFrame with player game logs including teammate info
            player_col: column identifying the player
            date_col: column identifying the game date
            teammates_col: column with list of teammates who played
        """
        stat_cols = [c for c in ["PTS", "TRB", "AST"] if c in game_logs.columns]
        if not stat_cols or player_col not in game_logs.columns:
            return self

        players = game_logs[player_col].unique()

        for player in players:
            player_games = game_logs[game_logs[player_col] == player].copy()
            if len(player_games) < self.min_games * 2:
                continue

            # Compute baseline stats
            self._player_baselines[player] = {
                stat: {
                    "mean": float(player_games[stat].mean()),
                    "std": float(player_games[stat].std()),
                    "median": float(player_games[stat].median()),
                }
                for stat in stat_cols
            }

            # If we have teammate information, compute impacts
            if teammates_col in game_logs.columns:
                self._compute_teammate_impacts(player, player_games, stat_cols, teammates_col)

        return self

    def fit_from_splits(
        self,
        player: str,
        with_teammate_stats: dict[str, pd.DataFrame],
        without_teammate_stats: dict[str, pd.DataFrame],
    ) -> None:
        """
        Fit from pre-computed with/without splits.

        Args:
            player: player name
            with_teammate_stats: {teammate: DataFrame of games WITH teammate}
            without_teammate_stats: {teammate: DataFrame of games WITHOUT teammate}
        """
        stat_cols = ["PTS", "TRB", "AST"]
        impacts = {}

        for teammate in with_teammate_stats:
            with_df = with_teammate_stats[teammate]
            without_df = without_teammate_stats.get(teammate)

            if without_df is None or len(without_df) < self.min_games:
                continue

            teammate_impacts = []
            for stat in stat_cols:
                if stat not in with_df.columns or stat not in without_df.columns:
                    continue

                mean_with = float(with_df[stat].mean())
                mean_without = float(without_df[stat].mean())
                delta = mean_without - mean_with

                # Statistical significance via t-test approximation
                n_with = len(with_df)
                n_without = len(without_df)
                std_with = float(with_df[stat].std())
                std_without = float(without_df[stat].std())

                if std_with > 0 and std_without > 0:
                    se = np.sqrt(std_with**2 / n_with + std_without**2 / n_without)
                    t_stat = abs(delta) / max(se, 0.01)
                    # Rough significance: t > 2 is significant
                    significance = min(1.0, t_stat / 3.0)
                else:
                    significance = 0.0

                # Usage delta
                usg_delta = 0.0
                if "USG%" in with_df.columns and "USG%" in without_df.columns:
                    usg_delta = float(without_df["USG%"].mean() - with_df["USG%"].mean())

                # Minutes delta
                mp_delta = 0.0
                if "MP" in with_df.columns and "MP" in without_df.columns:
                    mp_delta = float(without_df["MP"].mean() - with_df["MP"].mean())

                impact = TeammateImpact(
                    teammate=teammate,
                    target_player=player,
                    stat=stat,
                    games_without=n_without,
                    games_with=n_with,
                    stat_mean_without=mean_without,
                    stat_mean_with=mean_with,
                    stat_delta=delta,
                    stat_delta_pct=delta / max(mean_with, 0.1),
                    usage_delta=usg_delta,
                    minutes_delta=mp_delta,
                    significance=significance,
                )
                teammate_impacts.append(impact)

            if teammate_impacts:
                impacts[teammate] = teammate_impacts

        self._teammate_impacts[player] = impacts

    def _compute_teammate_impacts(
        self,
        player: str,
        player_games: pd.DataFrame,
        stat_cols: list[str],
        teammates_col: str,
    ) -> None:
        """Compute teammate impacts from game logs with teammate lists."""
        # This requires the teammates_col to contain lists of teammates
        # who played in each game. We compare games with/without each teammate.
        if teammates_col not in player_games.columns:
            return

        # Get all teammates who appeared
        all_teammates = set()
        for teammates in player_games[teammates_col].dropna():
            if isinstance(teammates, (list, set)):
                all_teammates.update(teammates)
            elif isinstance(teammates, str):
                all_teammates.update(teammates.split(","))

        impacts = {}
        for teammate in all_teammates:
            if teammate == player:
                continue

            # Games with and without this teammate
            with_mask = player_games[teammates_col].apply(
                lambda x: teammate in x if isinstance(x, (list, set, str)) else False
            )
            without_mask = ~with_mask

            with_games = player_games[with_mask]
            without_games = player_games[without_mask]

            if len(without_games) < self.min_games or len(with_games) < self.min_games:
                continue

            teammate_impacts = []
            for stat in stat_cols:
                mean_with = float(with_games[stat].mean())
                mean_without = float(without_games[stat].mean())
                delta = mean_without - mean_with

                impact = TeammateImpact(
                    teammate=teammate,
                    target_player=player,
                    stat=stat,
                    games_without=len(without_games),
                    games_with=len(with_games),
                    stat_mean_without=mean_without,
                    stat_mean_with=mean_with,
                    stat_delta=delta,
                    stat_delta_pct=delta / max(mean_with, 0.1),
                    significance=0.5,  # Simplified
                )
                teammate_impacts.append(impact)

            if teammate_impacts:
                impacts[teammate] = teammate_impacts

        self._teammate_impacts[player] = impacts

    def get_lineup_context(
        self,
        player: str,
        teammates_out: list[str] = None,
        teammates_in: list[str] = None,
    ) -> LineupContext:
        """
        Get lineup context for a player given tonight's lineup changes.

        Args:
            player: target player
            teammates_out: list of teammates who are OUT tonight
            teammates_in: list of teammates who are IN (returned from injury, etc.)

        Returns:
            LineupContext with usage/minutes boosts and opportunity score
        """
        teammates_out = teammates_out or []
        teammates_in = teammates_in or []

        usage_boost = 0.0
        minutes_boost = 0.0
        role_change = False
        confidence_factors = []

        player_impacts = self._teammate_impacts.get(player, {})

        for teammate in teammates_out:
            if teammate in player_impacts:
                impacts = player_impacts[teammate]
                for impact in impacts:
                    if impact.stat == "PTS":  # Use PTS as primary signal
                        usage_boost += impact.usage_delta
                        minutes_boost += impact.minutes_delta
                        confidence_factors.append(impact.significance)
                        if abs(impact.stat_delta_pct) > 0.15:
                            role_change = True

        # Teammates returning might suppress opportunity
        for teammate in teammates_in:
            if teammate in player_impacts:
                impacts = player_impacts[teammate]
                for impact in impacts:
                    if impact.stat == "PTS":
                        usage_boost -= impact.usage_delta * 0.7  # Partial reversal
                        minutes_boost -= impact.minutes_delta * 0.7

        # Opportunity score: 0.5 = neutral, >0.5 = elevated, <0.5 = suppressed
        opportunity_score = 0.5 + np.clip(usage_boost * 5, -0.4, 0.4)

        confidence = float(np.mean(confidence_factors)) if confidence_factors else 0.3

        return LineupContext(
            player=player,
            teammates_out=teammates_out,
            teammates_in=teammates_in,
            usage_boost=usage_boost,
            minutes_boost=minutes_boost,
            opportunity_score=opportunity_score,
            role_change_detected=role_change,
            confidence=confidence,
        )

    def compute_opportunity_environment(
        self,
        player: str,
        stat: str,
        base_projection: float,
        lineup_context: LineupContext,
        pace_factor: float = 1.0,
    ) -> OpportunityEnvironment:
        """
        Compute the full opportunity environment for a prop.

        This combines lineup context with pace and role information
        to produce an adjusted projection and opportunity rating.
        """
        # Apply lineup adjustment
        player_impacts = self._teammate_impacts.get(player, {})
        stat_adjustment = 0.0

        for teammate in lineup_context.teammates_out:
            if teammate in player_impacts:
                for impact in player_impacts[teammate]:
                    if impact.stat == stat:
                        # Weight by significance
                        stat_adjustment += impact.stat_delta * impact.significance

        # Apply pace factor
        pace_adjusted = base_projection * pace_factor

        # Combined adjustment
        lineup_adjusted = pace_adjusted + stat_adjustment

        # Determine opportunity rating
        adjustment_pct = (lineup_adjusted - base_projection) / max(base_projection, 0.1)
        if adjustment_pct > 0.10:
            rating = "elevated"
        elif adjustment_pct < -0.10:
            rating = "suppressed"
        else:
            rating = "normal"

        # Key factors
        factors = []
        if lineup_context.teammates_out:
            factors.append(f"Teammates out: {', '.join(lineup_context.teammates_out[:3])}")
        if abs(lineup_context.usage_boost) > 0.02:
            factors.append(f"USG% boost: {lineup_context.usage_boost:+.1%}")
        if abs(lineup_context.minutes_boost) > 2:
            factors.append(f"Minutes boost: {lineup_context.minutes_boost:+.1f}")
        if pace_factor != 1.0:
            factors.append(f"Pace factor: {pace_factor:.2f}")

        baseline = self._player_baselines.get(player, {})
        minutes_proj = baseline.get("MP", {}).get("mean", 32) + lineup_context.minutes_boost
        usage_proj = baseline.get("USG%", {}).get("mean", 0.25) + lineup_context.usage_boost

        return OpportunityEnvironment(
            player=player,
            stat=stat,
            base_projection=base_projection,
            lineup_adjusted=lineup_adjusted,
            adjustment_magnitude=abs(lineup_adjusted - base_projection),
            pace_factor=pace_factor,
            minutes_projection=minutes_proj,
            usage_projection=usage_proj,
            opportunity_rating=rating,
            key_factors=factors,
        )

    def get_significant_impacts(
        self,
        player: str,
        stat: str = None,
        min_significance: float = 0.4,
    ) -> list[TeammateImpact]:
        """Get all significant teammate impacts for a player."""
        impacts = self._teammate_impacts.get(player, {})
        results = []

        for teammate, impact_list in impacts.items():
            for impact in impact_list:
                if stat and impact.stat != stat:
                    continue
                if impact.significance >= min_significance:
                    results.append(impact)

        # Sort by significance * magnitude
        results.sort(key=lambda x: x.significance * abs(x.stat_delta), reverse=True)
        return results

    def save(self, path: str | Path) -> None:
        """Save lineup impact model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        impacts_serializable = {}
        for player, teammates in self._teammate_impacts.items():
            impacts_serializable[player] = {}
            for teammate, impact_list in teammates.items():
                impacts_serializable[player][teammate] = [
                    {
                        "teammate": i.teammate,
                        "target_player": i.target_player,
                        "stat": i.stat,
                        "games_without": i.games_without,
                        "games_with": i.games_with,
                        "stat_mean_without": i.stat_mean_without,
                        "stat_mean_with": i.stat_mean_with,
                        "stat_delta": i.stat_delta,
                        "stat_delta_pct": i.stat_delta_pct,
                        "usage_delta": i.usage_delta,
                        "minutes_delta": i.minutes_delta,
                        "significance": i.significance,
                    }
                    for i in impact_list
                ]

        data = {
            "teammate_impacts": impacts_serializable,
            "player_baselines": self._player_baselines,
            "min_games": self.min_games,
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LineupImpactModel":
        """Load lineup impact model."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        model = cls(min_games=data.get("min_games", 5))
        model._player_baselines = data.get("player_baselines", {})

        # Reconstruct TeammateImpact objects
        for player, teammates in data.get("teammate_impacts", {}).items():
            model._teammate_impacts[player] = {}
            for teammate, impact_dicts in teammates.items():
                model._teammate_impacts[player][teammate] = [
                    TeammateImpact(**d) for d in impact_dicts
                ]

        return model


if __name__ == "__main__":
    np.random.seed(42)
    print("Testing Lineup Impact Model...")

    # Simulate game logs with teammate effects
    n_games = 80
    # Jalen Brunson's stats when Randle plays vs doesn't
    games_with_randle = pd.DataFrame({
        "PTS": np.random.normal(26, 5, 50),
        "TRB": np.random.normal(3.5, 1.5, 50),
        "AST": np.random.normal(7, 2, 50),
        "MP": np.random.normal(35, 3, 50),
        "USG%": np.random.normal(0.30, 0.03, 50),
    })
    games_without_randle = pd.DataFrame({
        "PTS": np.random.normal(30, 6, 30),  # More points without Randle
        "TRB": np.random.normal(4, 1.5, 30),
        "AST": np.random.normal(8.5, 2.5, 30),  # More assists too
        "MP": np.random.normal(37, 2, 30),  # More minutes
        "USG%": np.random.normal(0.34, 0.03, 30),  # Higher usage
    })

    model = LineupImpactModel(min_games=5)
    model.fit_from_splits(
        player="Jalen Brunson",
        with_teammate_stats={"Julius Randle": games_with_randle},
        without_teammate_stats={"Julius Randle": games_without_randle},
    )

    # Get lineup context when Randle is out
    context = model.get_lineup_context(
        player="Jalen Brunson",
        teammates_out=["Julius Randle"],
    )
    print(f"  Lineup Context (Randle OUT):")
    print(f"    Usage boost: {context.usage_boost:+.3f}")
    print(f"    Minutes boost: {context.minutes_boost:+.1f}")
    print(f"    Opportunity score: {context.opportunity_score:.2f}")
    print(f"    Role change: {context.role_change_detected}")

    # Compute opportunity environment
    env = model.compute_opportunity_environment(
        player="Jalen Brunson",
        stat="PTS",
        base_projection=26.0,
        lineup_context=context,
        pace_factor=1.05,
    )
    print(f"\n  Opportunity Environment:")
    print(f"    Base projection: {env.base_projection:.1f}")
    print(f"    Lineup adjusted: {env.lineup_adjusted:.1f}")
    print(f"    Rating: {env.opportunity_rating}")
    print(f"    Key factors: {env.key_factors}")

    # Get significant impacts
    impacts = model.get_significant_impacts("Jalen Brunson", stat="PTS")
    for imp in impacts:
        print(f"\n  Impact: {imp.teammate} OUT")
        print(f"    PTS delta: {imp.stat_delta:+.1f} ({imp.stat_delta_pct:+.1%})")
        print(f"    Significance: {imp.significance:.2f}")

    print("\nLineup Impact Model smoke test PASSED")
