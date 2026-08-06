"""Leakage-safe batter/pitcher profile network for MLB prop projections."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Mapping

import pandas as pd


NETWORK_VERSION = "batter_pitcher_profile_network_v1"
HITTER_TARGETS = ("H", "TB", "R", "HR", "RBI")
TARGET_ADJUSTMENT_CAPS: Mapping[str, float] = {
    "H": 0.10,
    "TB": 0.16,
    "R": 0.05,
    "HR": 0.025,
    "RBI": 0.06,
}
TARGET_BASELINES: Mapping[str, float] = {
    "H": 1.0,
    "TB": 1.5,
    "R": 0.5,
    "HR": 0.15,
    "RBI": 0.5,
}


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def normalize_player_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char)).strip().lower()
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        text = text.replace(old, new)
    return text


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)


def aligned_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def recent_mean(frame: pd.DataFrame, column: str, window: int, default: float) -> float:
    values = numeric_series(frame, column).tail(int(window))
    return float(values.mean()) if not values.empty else float(default)


def safe_rate(numerator: pd.Series, denominator: pd.Series, default: float) -> float:
    numerator_sum = float(numerator.sum()) if not numerator.empty else 0.0
    denominator_sum = float(denominator.sum()) if not denominator.empty else 0.0
    return numerator_sum / denominator_sum if denominator_sum > 0.0 else float(default)


def starter_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "Was_Starter" in frame.columns:
        flag = pd.to_numeric(frame["Was_Starter"], errors="coerce").fillna(0.0)
        return frame.loc[flag.gt(0.0)].copy()
    innings = aligned_numeric_series(frame, "IP")
    pitches = aligned_numeric_series(frame, "Pitches")
    return frame.loc[innings.ge(3.0) | pitches.ge(45.0)].copy()


def _batter_profile(history: pd.DataFrame) -> tuple[dict[str, float], float]:
    recent = history.tail(30)
    pa = numeric_series(recent, "PA")
    strikeouts = numeric_series(recent, "SO")
    contact_rate = 1.0 - safe_rate(strikeouts, pa, 0.225)
    woba = recent_mean(recent, "wOBA", 30, 0.315)
    iso = recent_mean(recent, "ISO", 30, 0.150)
    hard_hit = recent_mean(recent, "HardHit%", 30, 38.0)
    barrel = recent_mean(recent, "Barrel%", 30, 7.0)
    order = recent_mean(recent, "Batting_Order", 15, 6.0)

    contact = clamp((contact_rate - 0.775) / 0.12)
    on_base = clamp((woba - 0.315) / 0.08)
    power = clamp((iso - 0.150) / 0.12)
    impact = clamp((hard_hit - 38.0) / 18.0)
    barrel_skill = clamp((barrel - 7.0) / 8.0)
    lineup = clamp((5.0 - order) / 4.0)

    strengths = {
        "H": clamp((0.45 * contact) + (0.35 * on_base) + (0.20 * impact)),
        "TB": clamp((0.30 * contact) + (0.30 * on_base) + (0.25 * power) + (0.15 * impact)),
        "R": clamp((0.45 * on_base) + (0.30 * contact) + (0.25 * lineup)),
        "HR": clamp((0.45 * power) + (0.35 * barrel_skill) + (0.20 * impact)),
        "RBI": clamp((0.35 * on_base) + (0.30 * power) + (0.20 * impact) + (0.15 * lineup)),
    }
    support = clamp(len(history) / 30.0, 0.0, 1.0)
    return strengths, support


def _pitcher_profile(history: pd.DataFrame) -> tuple[dict[str, float], float, float]:
    starts = starter_history(history).tail(20)
    if starts.empty:
        return {target: 0.0 for target in HITTER_TARGETS}, 1.0, 0.0

    innings = numeric_series(starts, "IP")
    batters_faced = numeric_series(starts, "BF")
    strikeouts = numeric_series(starts, "K")
    hits = numeric_series(starts, "H_allowed")
    home_runs = numeric_series(starts, "HR_allowed")
    walks = numeric_series(starts, "BB_allowed")
    era = recent_mean(starts, "ERA", 15, 4.1)
    fip = recent_mean(starts, "FIP", 15, era)
    k9 = safe_rate(strikeouts * 9.0, innings, 8.2)
    hit_rate = safe_rate(hits, batters_faced, 0.225)
    home_run_rate = safe_rate(home_runs, batters_faced, 0.030)
    walk_rate = safe_rate(walks, batters_faced, 0.080)

    run_vulnerability = clamp((era - 4.1) / 2.5)
    fielding_vulnerability = clamp((fip - 4.1) / 2.5)
    contact_vulnerability = clamp((hit_rate - 0.225) / 0.08)
    power_vulnerability = clamp((home_run_rate - 0.030) / 0.025)
    control_vulnerability = clamp((walk_rate - 0.080) / 0.06)
    low_strikeout_vulnerability = clamp((8.2 - k9) / 4.0)

    vulnerabilities = {
        "H": clamp(
            (0.40 * contact_vulnerability)
            + (0.25 * low_strikeout_vulnerability)
            + (0.20 * run_vulnerability)
            + (0.15 * fielding_vulnerability)
        ),
        "TB": clamp(
            (0.25 * contact_vulnerability)
            + (0.25 * power_vulnerability)
            + (0.20 * low_strikeout_vulnerability)
            + (0.15 * run_vulnerability)
            + (0.15 * fielding_vulnerability)
        ),
        "R": clamp(
            (0.30 * run_vulnerability)
            + (0.25 * fielding_vulnerability)
            + (0.20 * control_vulnerability)
            + (0.15 * contact_vulnerability)
            + (0.10 * power_vulnerability)
        ),
        "HR": clamp(
            (0.50 * power_vulnerability)
            + (0.20 * low_strikeout_vulnerability)
            + (0.15 * run_vulnerability)
            + (0.15 * fielding_vulnerability)
        ),
        "RBI": clamp(
            (0.30 * run_vulnerability)
            + (0.25 * fielding_vulnerability)
            + (0.20 * contact_vulnerability)
            + (0.15 * power_vulnerability)
            + (0.10 * control_vulnerability)
        ),
    }

    support = clamp(len(starts) / 20.0, 0.0, 1.0)
    era_std = finite_float(numeric_series(starts, "ERA").tail(10).std(), 2.5)
    k9_by_start = (
        aligned_numeric_series(starts, "K")
        * 9.0
        / aligned_numeric_series(starts, "IP").clip(lower=1.0)
    ).dropna()
    k9_std = finite_float(k9_by_start.tail(10).std(), 4.0)
    ip_std = finite_float(numeric_series(starts, "IP").tail(10).std(), 2.0)
    variability = clamp(
        (0.40 * clamp(era_std / 3.0, 0.0, 1.0))
        + (0.35 * clamp(k9_std / 4.0, 0.0, 1.0))
        + (0.25 * clamp(ip_std / 2.0, 0.0, 1.0)),
        0.0,
        1.0,
    )
    uncertainty = clamp((0.55 * (1.0 - support)) + (0.45 * variability), 0.0, 1.0)
    return vulnerabilities, uncertainty, support


def _direct_matchup_rows(
    batter_history: pd.DataFrame,
    *,
    opposing_pitcher_id: object = None,
    opposing_pitcher_name: object = None,
) -> pd.DataFrame:
    if batter_history.empty:
        return batter_history.copy()
    pitcher_id = int(finite_float(opposing_pitcher_id, 0.0))
    if pitcher_id > 0 and "Opp_Starter_ID" in batter_history.columns:
        ids = pd.to_numeric(batter_history["Opp_Starter_ID"], errors="coerce").fillna(0).astype(int)
        matched = batter_history.loc[ids.eq(pitcher_id)].copy()
        if not matched.empty:
            return matched
    pitcher_key = normalize_player_key(opposing_pitcher_name)
    if pitcher_key and "Opp_Starter_Player" in batter_history.columns:
        names = batter_history["Opp_Starter_Player"].map(normalize_player_key)
        return batter_history.loc[names.eq(pitcher_key)].copy()
    return batter_history.iloc[0:0].copy()


@dataclass(frozen=True)
class MatchupNetworkSignal:
    version: str
    batter_support: float
    pitcher_support: float
    pitcher_uncertainty: float
    direct_matchup_games: int
    confidence: float
    batter_strength: dict[str, float]
    pitcher_vulnerability: dict[str, float]
    direct_matchup_lift: dict[str, float]
    network_score: dict[str, float]
    adjustment: dict[str, float]

    @classmethod
    def neutral(cls) -> "MatchupNetworkSignal":
        zeros = {target: 0.0 for target in HITTER_TARGETS}
        return cls(
            version=NETWORK_VERSION,
            batter_support=0.0,
            pitcher_support=0.0,
            pitcher_uncertainty=1.0,
            direct_matchup_games=0,
            confidence=0.0,
            batter_strength=dict(zeros),
            pitcher_vulnerability=dict(zeros),
            direct_matchup_lift=dict(zeros),
            network_score=dict(zeros),
            adjustment=dict(zeros),
        )


def build_matchup_network_signal(
    batter_history: pd.DataFrame,
    pitcher_history: pd.DataFrame,
    *,
    opposing_pitcher_id: object = None,
    opposing_pitcher_name: object = None,
) -> MatchupNetworkSignal:
    """Build a target-specific pregame batter/pitcher network signal.

    Both history frames must contain only rows strictly before the predicted game.
    Prior batter-versus-starter games are treated as noisy game-level evidence and
    are heavily shrunk because later plate appearances may have faced relievers.
    """

    if batter_history.empty:
        return MatchupNetworkSignal.neutral()

    batter_strength, batter_support = _batter_profile(batter_history)
    pitcher_vulnerability, pitcher_uncertainty, pitcher_support = _pitcher_profile(pitcher_history)
    direct_rows = _direct_matchup_rows(
        batter_history,
        opposing_pitcher_id=opposing_pitcher_id,
        opposing_pitcher_name=opposing_pitcher_name,
    )
    direct_games = int(len(direct_rows))
    direct_support = clamp(direct_games / 8.0, 0.0, 1.0)

    direct_lift: dict[str, float] = {}
    scores: dict[str, float] = {}
    adjustments: dict[str, float] = {}
    confidence = clamp(
        (0.50 * batter_support)
        + (0.35 * pitcher_support * (1.0 - (0.35 * pitcher_uncertainty)))
        + (0.15 * direct_support),
        0.0,
        1.0,
    )

    for target in HITTER_TARGETS:
        baseline = recent_mean(batter_history, target, 30, TARGET_BASELINES[target])
        direct_mean = recent_mean(direct_rows, target, 8, baseline)
        lift_scale = max(TARGET_BASELINES[target], 0.15)
        raw_direct_lift = clamp((direct_mean - baseline) / lift_scale)
        direct_lift[target] = raw_direct_lift * (direct_games / (direct_games + 8.0))

        batter_value = batter_strength[target]
        pitcher_value = pitcher_vulnerability[target]
        if batter_value >= 0.0:
            profile_synergy = batter_value * pitcher_value
        else:
            profile_synergy = -abs(batter_value) * max(-pitcher_value, 0.0)
        score = (
            (batter_support * (1.0 - pitcher_uncertainty) * profile_synergy)
            + ((0.50 + (0.50 * pitcher_uncertainty)) * direct_lift[target])
        )
        score = clamp(score)
        scores[target] = score
        adjustments[target] = TARGET_ADJUSTMENT_CAPS[target] * score

    return MatchupNetworkSignal(
        version=NETWORK_VERSION,
        batter_support=batter_support,
        pitcher_support=pitcher_support,
        pitcher_uncertainty=pitcher_uncertainty,
        direct_matchup_games=direct_games,
        confidence=confidence,
        batter_strength=batter_strength,
        pitcher_vulnerability=pitcher_vulnerability,
        direct_matchup_lift=direct_lift,
        network_score=scores,
        adjustment=adjustments,
    )
