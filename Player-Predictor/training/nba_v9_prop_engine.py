#!/usr/bin/env python3
"""
NBA v9 Hierarchical Stochastic Player-Prop Engine

This is the master integration module that implements the full system
described in the research:

  1. Player role model - estimates minutes, usage, shot diet, assist role
  2. Hidden-state model - detects regimes via HMM
  3. Stat generator - produces full distributions for PTS, REB, AST
  4. Dependency model - captures correlation between stats and teammates
  5. Market layer - converts sportsbook line/odds into no-vig implied probability
  6. Calibration layer - corrects raw model probabilities (adaptive)
  7. Uncertainty layer - measures whether the model actually knows enough
  8. Selection layer - bets only when calibrated edge clears threshold

The model output is NOT:
  "Player over = yes"

It IS:
  Line: 22.5 points
  Median: 21.4
  Mean: 22.8
  P(over): 47.6%
  P(under): 52.4%
  Uncertainty: high
  Market no-vig over: 51.2%
  Regime: normal
  Lineup impact: neutral
  Decision: no bet

The big conceptual shift:
  You are not trying to predict "will he go over."
  You are trying to estimate:
    "Is the sportsbook line wrong relative to the player's
     full conditional distribution tonight?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


def american_to_prob(odds: float) -> float:
    """Convert American odds to raw implied probability."""
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def no_vig_probs(over_odds: float, under_odds: float) -> tuple[float, float]:
    """Return two-way no-vig probabilities for over and under."""
    p_over = american_to_prob(over_odds)
    p_under = american_to_prob(under_odds)
    total = p_over + p_under
    if total <= 0:
        return 0.5, 0.5
    return p_over / total, p_under / total


def prob_to_american(probability: float) -> int:
    """Convert a probability in (0, 1) to American odds."""
    p = float(np.clip(probability, 0.001, 0.999))
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def expected_value(p_win: float, odds: float) -> float:
    """Expected value per unit stake."""
    decimal = american_to_decimal(odds)
    return float(p_win * (decimal - 1.0) - (1.0 - p_win))


@dataclass
class PropAssessment:
    """
    Complete assessment of a single player prop bet.

    This is the final output of the v9 engine - a full probabilistic
    assessment that supports the betting decision.
    """
    # Identity
    player: str
    stat: str
    line: float
    direction: str = ""  # "over" or "under" or "" (undecided)

    # Distribution
    predicted_mean: float = 0.0
    predicted_median: float = 0.0
    predicted_std: float = 0.0
    quantiles: dict[str, float] = field(default_factory=dict)  # Q10/Q25/Q50/Q75/Q90

    # Probabilities
    p_over_raw: float = 0.5
    p_over_calibrated: float = 0.5
    p_over_regime_mixture: float = 0.5
    p_over_lineup_adjusted: float = 0.5
    p_over_final: float = 0.5  # The number we bet on

    # Market comparison
    market_implied_over: float = 0.5
    market_implied_under: float = 0.5
    market_implied_no_vig: float = 0.5
    market_no_vig_over: float = 0.5
    market_no_vig_under: float = 0.5
    p_under_calibrated: float = 0.5
    edge: float = 0.0
    edge_over: float = 0.0
    edge_under: float = 0.0
    ev: float = 0.0
    ev_over: float = 0.0
    ev_under: float = 0.0

    # Regime context
    regime: str = "normal"
    regime_confidence: float = 0.0
    regime_uncertainty: float = 0.0  # Entropy of regime distribution

    # Lineup context
    lineup_impact: str = "neutral"  # "elevated", "neutral", "suppressed"
    lineup_adjustment: float = 0.0
    teammates_out: list[str] = field(default_factory=list)

    # Uncertainty
    model_uncertainty: float = 0.0    # From sigma head
    calibration_uncertainty: float = 0.0  # From calibration CI width
    total_uncertainty: float = 0.0    # Combined
    confidence_level: str = ""        # "high", "moderate", "low", "very_low"

    # Correlation context
    same_game_correlations: dict[str, float] = field(default_factory=dict)
    diversification_score: float = 1.0

    # Decision
    decision: str = ""  # "bet_over", "bet_under", "no_bet", "monitor"
    decision_reasons: list[str] = field(default_factory=list)
    edge_category: str = ""  # "strong", "moderate", "marginal", "none"

    # Chaos diagnostics
    residual_entropy: float = 0.0
    is_bettable: bool = False


@dataclass
class SlateAssessment:
    """Assessment of an entire slate of props for one night."""
    date: str
    n_props_evaluated: int = 0
    n_bettable: int = 0
    n_strong_edge: int = 0
    n_moderate_edge: int = 0
    n_no_bet: int = 0
    props: list[PropAssessment] = field(default_factory=list)
    portfolio_correlation: float = 0.0
    expected_hit_rate: float = 0.0
    chaos_level: str = ""


class PropEngine:
    """
    The v9 Hierarchical Stochastic Player-Prop Engine.

    This orchestrates all v9 components into a single prediction pipeline:
      HMM regime detection → distribution modeling → copula dependencies →
      lineup adjustment → calibration → market comparison → decision

    Usage:
        engine = PropEngine()
        engine.load_models(model_dir)
        assessment = engine.assess_prop(player, stat, line, market_odds, context)
    """

    # Edge thresholds
    STRONG_EDGE = 0.06
    MODERATE_EDGE = 0.04
    MARGINAL_EDGE = 0.02

    # Uncertainty thresholds
    HIGH_UNCERTAINTY = 0.70
    MODERATE_UNCERTAINTY = 0.50

    # Minimum confidence to bet
    MIN_BET_CONFIDENCE = 0.55

    def __init__(
        self,
        targets: list[str] = None,
        edge_threshold: float = 0.04,
        uncertainty_cap: float = 0.75,
    ):
        self.targets = targets or ["PTS", "TRB", "AST"]
        self.edge_threshold = edge_threshold
        self.uncertainty_cap = uncertainty_cap

        # Component models (loaded separately)
        self._regime_model = None
        self._copula_model = None
        self._calibrator = None
        self._lineup_model = None
        self._chaos_analyzer = None

    def assess_prop(
        self,
        player: str,
        stat: str,
        line: float,
        market_over_odds: float = -110,
        market_under_odds: float = -110,
        recent_games: pd.DataFrame = None,
        teammates_out: list[str] = None,
        pace_factor: float = 1.0,
        quantiles: np.ndarray = None,
        sigma: float = 5.0,
    ) -> PropAssessment:
        """
        Produce a complete probabilistic assessment of a player prop.

        This is the main entry point. It runs the full pipeline:
          1. Infer regime from recent games
          2. Compute distribution-based P(over)
          3. Apply regime mixture
          4. Apply lineup adjustment
          5. Calibrate
          6. Compare to market
          7. Assess uncertainty
          8. Make decision

        Args:
            player: player name
            stat: stat type (PTS, TRB, AST)
            line: sportsbook prop line
            market_over_odds: American odds for the over
            market_under_odds: American odds for the under
            recent_games: last N games DataFrame
            teammates_out: list of teammates who are OUT
            pace_factor: pace multiplier for tonight's game
            quantiles: (5,) predicted quantiles [Q10, Q25, Q50, Q75, Q90]
            sigma: predicted uncertainty from model

        Returns:
            PropAssessment with full analysis
        """
        assessment = PropAssessment(player=player, stat=stat, line=line)
        teammates_out = teammates_out or []
        assessment.teammates_out = teammates_out

        # --- Step 1: Distribution ---
        if quantiles is not None:
            assessment.quantiles = {
                "Q10": float(quantiles[0]),
                "Q25": float(quantiles[1]),
                "Q50": float(quantiles[2]),
                "Q75": float(quantiles[3]),
                "Q90": float(quantiles[4]),
            }
            assessment.predicted_median = float(quantiles[2])
            assessment.predicted_mean = float(np.mean(quantiles))
            assessment.predicted_std = sigma
        elif recent_games is not None and stat in recent_games.columns:
            vals = recent_games[stat].dropna().values
            if len(vals) > 3:
                assessment.predicted_mean = float(np.mean(vals))
                assessment.predicted_median = float(np.median(vals))
                assessment.predicted_std = float(np.std(vals))
                assessment.quantiles = {
                    "Q10": float(np.percentile(vals, 10)),
                    "Q25": float(np.percentile(vals, 25)),
                    "Q50": float(np.percentile(vals, 50)),
                    "Q75": float(np.percentile(vals, 75)),
                    "Q90": float(np.percentile(vals, 90)),
                }

        # --- Step 2: Raw P(over) from distribution ---
        assessment.p_over_raw = self._compute_p_over(
            line, assessment.predicted_median, assessment.predicted_std, quantiles
        )

        # --- Step 3: Regime detection ---
        if self._regime_model is not None and recent_games is not None:
            regime_state = self._regime_model.infer_regime(recent_games)
            assessment.regime = regime_state.regime_name
            assessment.regime_confidence = regime_state.confidence
            # Compute regime mixture P(over)
            regime_dist = self._regime_model.compute_regime_conditional_p_over(
                regime_state, stat, line
            )
            assessment.p_over_regime_mixture = regime_dist.mixture_p_over
            assessment.regime_uncertainty = regime_dist.uncertainty
        else:
            assessment.p_over_regime_mixture = assessment.p_over_raw
            assessment.regime_uncertainty = 0.5

        # --- Step 4: Lineup adjustment ---
        if self._lineup_model is not None and teammates_out:
            lineup_ctx = self._lineup_model.get_lineup_context(
                player, teammates_out=teammates_out
            )
            env = self._lineup_model.compute_opportunity_environment(
                player, stat, assessment.predicted_median, lineup_ctx, pace_factor
            )
            assessment.lineup_impact = env.opportunity_rating
            assessment.lineup_adjustment = env.lineup_adjusted - env.base_projection

            # Adjust P(over) based on lineup shift
            if assessment.predicted_std > 0:
                shift_z = assessment.lineup_adjustment / assessment.predicted_std
                # Shift the distribution mean, recompute P(over)
                adjusted_median = assessment.predicted_median + assessment.lineup_adjustment
                assessment.p_over_lineup_adjusted = self._compute_p_over(
                    line, adjusted_median, assessment.predicted_std, None
                )
            else:
                assessment.p_over_lineup_adjusted = assessment.p_over_regime_mixture
        else:
            assessment.p_over_lineup_adjusted = assessment.p_over_regime_mixture
            assessment.lineup_impact = "neutral"

        # --- Step 5: Blend probabilities ---
        # Weighted blend of raw, regime, and lineup-adjusted
        blend_weights = self._get_blend_weights(assessment)
        assessment.p_over_final = (
            blend_weights["raw"] * assessment.p_over_raw +
            blend_weights["regime"] * assessment.p_over_regime_mixture +
            blend_weights["lineup"] * assessment.p_over_lineup_adjusted
        )
        assessment.p_over_final = float(np.clip(assessment.p_over_final, 0.01, 0.99))

        # --- Step 6: Calibration ---
        if self._calibrator is not None:
            calibrated = self._calibrator.predict(np.array([assessment.p_over_final]))
            assessment.p_over_calibrated = float(calibrated[0])
            # Get calibration CI
            ci = self._calibrator.confidence_interval(assessment.p_over_final)
            assessment.calibration_uncertainty = (ci[1] - ci[0]) / 2
        else:
            assessment.p_over_calibrated = assessment.p_over_final
            assessment.calibration_uncertainty = 0.10

        # --- Step 7: Market comparison ---
        market_info = self._compute_market_edge(
            assessment.p_over_calibrated, market_over_odds, market_under_odds
        )
        assessment.market_implied_over = market_info["implied_raw"]
        assessment.market_implied_under = market_info["implied_under_raw"]
        assessment.market_implied_no_vig = market_info["implied_no_vig"]
        assessment.market_no_vig_over = market_info["market_no_vig_over"]
        assessment.market_no_vig_under = market_info["market_no_vig_under"]
        assessment.p_under_calibrated = market_info["p_under_model"]
        assessment.edge = market_info["edge"]
        assessment.edge_over = market_info["edge_over"]
        assessment.edge_under = market_info["edge_under"]
        assessment.ev = market_info["ev"]
        assessment.ev_over = market_info["ev_over"]
        assessment.ev_under = market_info["ev_under"]

        # --- Step 8: Uncertainty assessment ---
        assessment.model_uncertainty = sigma / max(assessment.predicted_mean, 1.0)
        assessment.total_uncertainty = np.sqrt(
            assessment.model_uncertainty ** 2 +
            assessment.calibration_uncertainty ** 2 +
            assessment.regime_uncertainty ** 2
        )
        assessment.confidence_level = self._classify_confidence(assessment.total_uncertainty)

        # --- Step 9: Decision ---
        assessment = self._make_decision(assessment)

        return assessment

    def assess_slate(
        self,
        props: list[dict],
        date: str = "",
    ) -> SlateAssessment:
        """
        Assess an entire slate of props and compute portfolio-level metrics.

        Args:
            props: list of dicts with keys matching assess_prop args
            date: date string for the slate
        """
        assessments = []
        for prop in props:
            assessment = self.assess_prop(**prop)
            assessments.append(assessment)

        bettable = [a for a in assessments if a.is_bettable]
        strong = [a for a in bettable if a.edge_category == "strong"]
        moderate = [a for a in bettable if a.edge_category == "moderate"]

        # Portfolio correlation (simplified)
        if len(bettable) > 1:
            # Check how many are from the same game
            # (simplified - in production, use copula model)
            portfolio_corr = 0.0
        else:
            portfolio_corr = 0.0

        expected_hit = float(np.mean([a.p_over_calibrated for a in bettable])) if bettable else 0.0

        return SlateAssessment(
            date=date,
            n_props_evaluated=len(assessments),
            n_bettable=len(bettable),
            n_strong_edge=len(strong),
            n_moderate_edge=len(moderate),
            n_no_bet=len(assessments) - len(bettable),
            props=assessments,
            portfolio_correlation=portfolio_corr,
            expected_hit_rate=expected_hit,
        )

    def _compute_p_over(
        self,
        line: float,
        median: float,
        std: float,
        quantiles: np.ndarray = None,
    ) -> float:
        """Compute P(stat > line) from distribution parameters."""
        from scipy.stats import norm as scipy_norm

        if std <= 0:
            return 0.5

        # Use Gaussian approximation centered on median
        # (median is better than mean for prop betting due to skewness)
        z = (line - median) / std
        p_over = float(1.0 - scipy_norm.cdf(z))
        return float(np.clip(p_over, 0.01, 0.99))

    def _get_blend_weights(self, assessment: PropAssessment) -> dict[str, float]:
        """
        Determine blending weights for different P(over) estimates.

        When regime confidence is high, weight regime mixture more.
        When lineup impact is significant, weight lineup adjustment more.
        """
        # Base weights
        w_raw = 0.30
        w_regime = 0.40
        w_lineup = 0.30

        # Adjust based on regime confidence
        if assessment.regime_confidence > 0.7:
            w_regime += 0.10
            w_raw -= 0.10
        elif assessment.regime_confidence < 0.3:
            w_regime -= 0.15
            w_raw += 0.15

        # Adjust based on lineup impact
        if assessment.lineup_impact == "elevated" or assessment.lineup_impact == "suppressed":
            w_lineup += 0.10
            w_raw -= 0.10
        elif not assessment.teammates_out:
            w_lineup -= 0.15
            w_regime += 0.15

        # Normalize
        total = w_raw + w_regime + w_lineup
        return {
            "raw": w_raw / total,
            "regime": w_regime / total,
            "lineup": w_lineup / total,
        }

    def _compute_market_edge(
        self,
        p_over_model: float,
        market_over_odds: float,
        market_under_odds: float = -110,
    ) -> dict:
        """Compute side-aware edge and EV against a two-way market."""
        implied_raw = american_to_prob(market_over_odds)
        implied_under_raw = american_to_prob(market_under_odds)
        market_no_vig_over, market_no_vig_under = no_vig_probs(
            market_over_odds, market_under_odds
        )

        p_over_model = float(np.clip(p_over_model, 0.01, 0.99))
        p_under_model = 1.0 - p_over_model

        edge_over = p_over_model - market_no_vig_over
        edge_under = p_under_model - market_no_vig_under
        ev_over = expected_value(p_over_model, market_over_odds)
        ev_under = expected_value(p_under_model, market_under_odds)

        if edge_over >= edge_under:
            edge = edge_over
            ev = ev_over
            implied_no_vig = market_no_vig_over
        else:
            edge = edge_under
            ev = ev_under
            implied_no_vig = market_no_vig_under

        return {
            "implied_raw": float(implied_raw),
            "implied_under_raw": float(implied_under_raw),
            "implied_no_vig": float(implied_no_vig),
            "market_no_vig_over": float(market_no_vig_over),
            "market_no_vig_under": float(market_no_vig_under),
            "p_under_model": float(p_under_model),
            "edge": float(edge),
            "edge_over": float(edge_over),
            "edge_under": float(edge_under),
            "ev": float(ev),
            "ev_over": float(ev_over),
            "ev_under": float(ev_under),
        }

    def _classify_confidence(self, uncertainty: float) -> str:
        """Classify confidence level from uncertainty."""
        if uncertainty < 0.25:
            return "high"
        elif uncertainty < 0.45:
            return "moderate"
        elif uncertainty < 0.65:
            return "low"
        else:
            return "very_low"

    def _make_decision(self, assessment: PropAssessment) -> PropAssessment:
        """Make the final bet/no-bet decision."""
        reasons = []

        # Classify edge
        abs_edge = abs(assessment.edge)
        if abs_edge >= self.STRONG_EDGE:
            assessment.edge_category = "strong"
        elif abs_edge >= self.MODERATE_EDGE:
            assessment.edge_category = "moderate"
        elif abs_edge >= self.MARGINAL_EDGE:
            assessment.edge_category = "marginal"
        else:
            assessment.edge_category = "none"

        # Direction
        if assessment.edge_over >= assessment.edge_under:
            assessment.direction = "over"
        else:
            assessment.direction = "under"

        # Decision logic
        # 1. Must have edge
        if abs_edge < self.edge_threshold:
            assessment.decision = "no_bet"
            reasons.append(f"Edge {abs_edge:.3f} below threshold {self.edge_threshold}")
            assessment.is_bettable = False
        # 2. Must not be too uncertain
        elif assessment.total_uncertainty > self.uncertainty_cap:
            assessment.decision = "no_bet"
            reasons.append(f"Uncertainty {assessment.total_uncertainty:.2f} exceeds cap")
            assessment.is_bettable = False
        # 3. Must have minimum confidence
        elif assessment.confidence_level == "very_low":
            assessment.decision = "no_bet"
            reasons.append("Confidence too low - model doesn't know enough")
            assessment.is_bettable = False
        # 4. Regime uncertainty check
        elif assessment.regime_uncertainty > 0.85:
            assessment.decision = "monitor"
            reasons.append("Regime highly uncertain - unclear game state")
            assessment.is_bettable = False
        else:
            # Bet!
            assessment.decision = f"bet_{assessment.direction}"
            assessment.is_bettable = True
            reasons.append(f"Edge: {assessment.edge:+.3f}")
            reasons.append(f"EV: {assessment.ev:+.3f}")
            reasons.append(f"Confidence: {assessment.confidence_level}")
            if assessment.lineup_impact != "neutral":
                reasons.append(f"Lineup: {assessment.lineup_impact}")
            if assessment.regime != "normal":
                reasons.append(f"Regime: {assessment.regime}")

        assessment.decision_reasons = reasons
        return assessment

    def load_models(self, model_dir: str | Path) -> None:
        """Load all component models from a directory."""
        model_dir = Path(model_dir)

        # Try to load each component
        try:
            from nba_v9_regime_hmm import PlayerRegimeHMM
            hmm_path = model_dir / "regime_hmm.pkl"
            nested_hmm_path = model_dir / "regime" / "regime_hmm.pkl"
            if not hmm_path.exists() and nested_hmm_path.exists():
                hmm_path = nested_hmm_path
            if hmm_path.exists():
                self._regime_model = PlayerRegimeHMM.load(hmm_path)
        except (ImportError, Exception):
            pass

        try:
            from nba_v9_adaptive_calibration import AdaptiveCalibrator
            cal_path = model_dir / "adaptive_calibrator.pkl"
            nested_cal_path = model_dir / "calibration" / "global_adaptive_calibrator.pkl"
            if not cal_path.exists() and nested_cal_path.exists():
                cal_path = nested_cal_path
            if cal_path.exists():
                self._calibrator = AdaptiveCalibrator.load(cal_path)
        except (ImportError, Exception):
            pass

        try:
            from nba_v9_lineup_impact import LineupImpactModel
            lineup_path = model_dir / "lineup_impact.json"
            nested_lineup_path = model_dir / "lineup" / "lineup_impact.json"
            if not lineup_path.exists() and nested_lineup_path.exists():
                lineup_path = nested_lineup_path
            if lineup_path.exists():
                self._lineup_model = LineupImpactModel.load(lineup_path)
        except (ImportError, Exception):
            pass

    def format_assessment(self, assessment: PropAssessment) -> str:
        """Format a prop assessment for display."""
        lines = [
            f"{'=' * 50}",
            f"  {assessment.player} - {assessment.stat} {'Over' if assessment.direction == 'over' else 'Under'} {assessment.line}",
            f"{'=' * 50}",
            f"",
            f"  Distribution:",
            f"    Mean: {assessment.predicted_mean:.1f}  |  Median: {assessment.predicted_median:.1f}  |  Std: {assessment.predicted_std:.1f}",
        ]

        if assessment.quantiles:
            q_str = "  |  ".join([f"{k}: {v:.1f}" for k, v in assessment.quantiles.items()])
            lines.append(f"    {q_str}")

        lines.extend([
            f"",
            f"  Probabilities:",
            f"    P(over) raw:        {assessment.p_over_raw:.3f}",
            f"    P(over) regime mix: {assessment.p_over_regime_mixture:.3f}",
            f"    P(over) lineup adj: {assessment.p_over_lineup_adjusted:.3f}",
            f"    P(over) calibrated: {assessment.p_over_calibrated:.3f}",
            f"    P(over) FINAL:      {assessment.p_over_final:.3f}",
            f"",
            f"  Market:",
            f"    Implied (raw):      {assessment.market_implied_over:.3f}",
            f"    Implied (no-vig):   {assessment.market_implied_no_vig:.3f}",
            f"    Edge:               {assessment.edge:+.3f}",
            f"    EV:                 {assessment.ev:+.3f}",
            f"",
            f"  Context:",
            f"    Regime:             {assessment.regime} (conf: {assessment.regime_confidence:.2f})",
            f"    Lineup impact:      {assessment.lineup_impact}",
            f"    Confidence:         {assessment.confidence_level}",
            f"    Total uncertainty:  {assessment.total_uncertainty:.3f}",
            f"",
            f"  DECISION: {assessment.decision.upper()}",
        ])

        for reason in assessment.decision_reasons:
            lines.append(f"    - {reason}")

        lines.append("")
        return "\n".join(lines)

    def assessment_to_dict(self, assessment: PropAssessment) -> dict:
        """Serialize an assessment using the production board schema."""
        side = assessment.direction.upper() if assessment.direction else "NONE"
        side_prob = (
            assessment.p_over_calibrated
            if assessment.direction == "over"
            else assessment.p_under_calibrated
        )
        side_market = (
            assessment.market_no_vig_over
            if assessment.direction == "over"
            else assessment.market_no_vig_under
        )
        return {
            "player": assessment.player,
            "market": assessment.stat,
            "line": assessment.line,
            "side": side,
            "mean_projection": assessment.predicted_mean,
            "median_projection": assessment.predicted_median,
            "p_over_raw": assessment.p_over_raw,
            "p_over_calibrated": assessment.p_over_calibrated,
            "p_under_calibrated": assessment.p_under_calibrated,
            "market_no_vig_over": assessment.market_no_vig_over,
            "market_no_vig_under": assessment.market_no_vig_under,
            "p_calibrated": side_prob,
            "market_no_vig": side_market,
            "edge": assessment.edge,
            "edge_over": assessment.edge_over,
            "edge_under": assessment.edge_under,
            "ev": assessment.ev,
            "uncertainty": assessment.total_uncertainty,
            "regime": assessment.regime,
            "decision": assessment.decision,
            "reason_codes": assessment.decision_reasons,
            "quantiles": assessment.quantiles,
        }

    def price_market(
        self,
        assessment: PropAssessment,
        base_hold: float = 0.045,
        uncertainty_hold_weight: float = 0.02,
        public_bias: float = 0.0,
        liability: float = 0.0,
    ) -> dict:
        """Create a sportsbook-style price recommendation from a distribution."""
        effective_hold = float(
            np.clip(
                base_hold + assessment.total_uncertainty * uncertainty_hold_weight,
                base_hold,
                0.075,
            )
        )
        fair_over = float(np.clip(assessment.p_over_calibrated, 0.01, 0.99))
        fair_under = 1.0 - fair_over

        over_posted_prob = np.clip(
            fair_over + effective_hold / 2.0 + public_bias + liability,
            0.01,
            0.99,
        )
        under_posted_prob = np.clip(
            fair_under + effective_hold / 2.0 - public_bias - liability,
            0.01,
            0.99,
        )

        if assessment.total_uncertainty > self.uncertainty_cap:
            risk_action = "keep_market_open_low_limit"
        elif assessment.regime_uncertainty > 0.85:
            risk_action = "manual_review"
        else:
            risk_action = "keep_market_open"

        return {
            "fair_line": assessment.predicted_median,
            "recommended_line": round(assessment.predicted_median * 2.0) / 2.0,
            "recommended_over_price": prob_to_american(over_posted_prob),
            "recommended_under_price": prob_to_american(under_posted_prob),
            "effective_hold": effective_hold,
            "risk_action": risk_action,
        }


if __name__ == "__main__":
    np.random.seed(42)
    print("Testing v9 Prop Engine...")

    # Create engine
    engine = PropEngine(edge_threshold=0.03)

    # Simulate recent games for a player
    n_games = 15
    recent_games = pd.DataFrame({
        "PTS": np.random.normal(26, 5, n_games),
        "TRB": np.random.normal(4, 1.5, n_games),
        "AST": np.random.normal(7, 2, n_games),
        "MP": np.random.normal(35, 3, n_games),
        "USG%": np.random.normal(0.30, 0.03, n_games),
        "TS%": np.random.normal(0.58, 0.05, n_games),
        "FGA": np.random.normal(19, 3, n_games),
        "PLUS_MINUS": np.random.normal(3, 8, n_games),
    })

    # Assess a prop
    assessment = engine.assess_prop(
        player="Jalen Brunson",
        stat="PTS",
        line=27.5,
        market_over_odds=-110,
        recent_games=recent_games,
        teammates_out=[],
        pace_factor=1.0,
        quantiles=np.array([18.0, 22.0, 26.0, 30.0, 35.0]),
        sigma=5.5,
    )

    print(engine.format_assessment(assessment))

    # Test with lineup impact
    assessment2 = engine.assess_prop(
        player="Jalen Brunson",
        stat="PTS",
        line=27.5,
        market_over_odds=+105,  # Better odds
        recent_games=recent_games,
        teammates_out=["Julius Randle"],
        pace_factor=1.08,
        quantiles=np.array([20.0, 24.0, 28.0, 32.0, 37.0]),
        sigma=6.0,
    )

    print(engine.format_assessment(assessment2))

    # Test slate assessment
    slate = engine.assess_slate(
        props=[
            {
                "player": "Jalen Brunson",
                "stat": "PTS",
                "line": 27.5,
                "market_over_odds": -110,
                "recent_games": recent_games,
                "quantiles": np.array([18.0, 22.0, 26.0, 30.0, 35.0]),
                "sigma": 5.5,
            },
            {
                "player": "Jalen Brunson",
                "stat": "AST",
                "line": 6.5,
                "market_over_odds": -115,
                "recent_games": recent_games,
                "quantiles": np.array([3.0, 5.0, 7.0, 9.0, 11.0]),
                "sigma": 2.0,
            },
        ],
        date="2026-05-08",
    )

    print(f"\nSlate Summary:")
    print(f"  Props evaluated: {slate.n_props_evaluated}")
    print(f"  Bettable: {slate.n_bettable}")
    print(f"  Strong edge: {slate.n_strong_edge}")
    print(f"  No bet: {slate.n_no_bet}")

    print("\nv9 Prop Engine smoke test PASSED")
