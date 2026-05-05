"""MLB pitcher-matchup adjustment.

Adjusts hit probability for hitter props based on the opposing pitcher's
recent performance.  A hitter facing a top K-rate pitcher should have their
H/TB UNDER probability boosted; facing a weak pitcher boosts OVER.

Uses Opp_Pitcher_ERA_3 and Opp_Pitcher_K9_3 from the processed player data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PitcherMatchupConfig:
    enabled: bool = True
    # ERA thresholds (lower ERA = better pitcher = harder for hitters)
    elite_pitcher_era: float = 2.5
    weak_pitcher_era: float = 4.5
    # K/9 thresholds (higher K/9 = more strikeouts = harder for hitters)
    elite_pitcher_k9: float = 10.0
    weak_pitcher_k9: float = 6.5
    # Adjustments
    elite_pitcher_under_boost: float = 0.025
    weak_pitcher_over_boost: float = 0.020
    # Only apply to hitter targets
    applicable_targets: tuple[str, ...] = ("H", "TB", "R", "HR", "RBI")


def _sf(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def compute_pitcher_adjustment(
    *,
    target: str,
    direction: str,
    player_type: str,
    opp_pitcher_era: float,
    opp_pitcher_k9: float,
    config: PitcherMatchupConfig | None = None,
) -> dict[str, Any]:
    """Compute pitcher-matchup adjustment for a single candidate."""
    cfg = config or PitcherMatchupConfig()

    if not cfg.enabled:
        return {"adjustment": 0.0, "pitcher_quality": "unknown", "source": "disabled"}

    target_upper = str(target).upper().strip()
    direction_upper = str(direction).upper().strip()
    ptype = str(player_type).lower().strip()

    # Only apply to hitter targets
    if ptype == "pitcher" or target_upper not in cfg.applicable_targets:
        return {"adjustment": 0.0, "pitcher_quality": "n/a", "source": "not_applicable"}

    era = _sf(opp_pitcher_era, default=float("nan"))
    k9 = _sf(opp_pitcher_k9, default=float("nan"))

    if not np.isfinite(era) and not np.isfinite(k9):
        return {"adjustment": 0.0, "pitcher_quality": "unknown", "source": "no_data"}

    # Score the opposing pitcher (higher = tougher for hitters)
    pitcher_score = 0.0
    signals = 0

    if np.isfinite(era):
        if era <= cfg.elite_pitcher_era:
            pitcher_score += 1.0
        elif era >= cfg.weak_pitcher_era:
            pitcher_score -= 1.0
        else:
            # Linear interpolation
            pitcher_score += (cfg.weak_pitcher_era - era) / (cfg.weak_pitcher_era - cfg.elite_pitcher_era) * 2.0 - 1.0
        signals += 1

    if np.isfinite(k9):
        if k9 >= cfg.elite_pitcher_k9:
            pitcher_score += 1.0
        elif k9 <= cfg.weak_pitcher_k9:
            pitcher_score -= 1.0
        else:
            pitcher_score += (k9 - cfg.weak_pitcher_k9) / (cfg.elite_pitcher_k9 - cfg.weak_pitcher_k9) * 2.0 - 1.0
        signals += 1

    if signals > 0:
        pitcher_score /= signals

    # Determine quality label
    if pitcher_score >= 0.5:
        pitcher_quality = "elite"
    elif pitcher_score <= -0.5:
        pitcher_quality = "weak"
    else:
        pitcher_quality = "average"

    # Apply adjustment based on direction
    adjustment = 0.0
    source = "none"

    if pitcher_quality == "elite" and direction_upper == "UNDER":
        # Tough pitcher → UNDER is more likely to hit
        adjustment = cfg.elite_pitcher_under_boost * min(pitcher_score, 1.0)
        source = "elite_pitcher_under_boost"
    elif pitcher_quality == "elite" and direction_upper == "OVER":
        # Tough pitcher → OVER is less likely
        adjustment = -cfg.elite_pitcher_under_boost * 0.5 * min(pitcher_score, 1.0)
        source = "elite_pitcher_over_penalty"
    elif pitcher_quality == "weak" and direction_upper == "OVER":
        # Weak pitcher → OVER is more likely
        adjustment = cfg.weak_pitcher_over_boost * min(abs(pitcher_score), 1.0)
        source = "weak_pitcher_over_boost"
    elif pitcher_quality == "weak" and direction_upper == "UNDER":
        # Weak pitcher → UNDER is less likely
        adjustment = -cfg.weak_pitcher_over_boost * 0.5 * min(abs(pitcher_score), 1.0)
        source = "weak_pitcher_under_penalty"

    return {
        "adjustment": float(np.clip(adjustment, -0.03, 0.03)),
        "pitcher_quality": pitcher_quality,
        "pitcher_score": pitcher_score,
        "source": source,
    }


def annotate_pitcher_matchup(
    candidates: pd.DataFrame,
    *,
    data_proc_root=None,
    config: PitcherMatchupConfig | None = None,
) -> pd.DataFrame:
    """Annotate MLB candidates with pitcher-matchup adjustments.

    If data_proc_root is provided, looks up Opp_Pitcher_ERA_3 and
    Opp_Pitcher_K9_3 from the player's processed CSV.  Otherwise uses
    columns already present in the DataFrame.
    """
    cfg = config or PitcherMatchupConfig()
    out = candidates.copy()

    if out.empty or not cfg.enabled:
        out["pitcher_matchup_adj"] = 0.0
        out["pitcher_quality"] = "disabled"
        return out

    adjustments = []
    qualities = []

    for _, row in out.iterrows():
        # Try to get pitcher stats from the row or from data_proc
        era = _sf(row.get("Opp_Pitcher_ERA_3", row.get("opp_pitcher_era_3")), default=float("nan"))
        k9 = _sf(row.get("Opp_Pitcher_K9_3", row.get("opp_pitcher_k9_3")), default=float("nan"))

        # If not in the row, try to look up from data_proc
        if (not np.isfinite(era) or not np.isfinite(k9)) and data_proc_root is not None:
            player_id = str(row.get("Player_ID", row.get("player_id", ""))).strip()
            if player_id:
                from pathlib import Path
                player_dir = Path(data_proc_root) / player_id.replace(" ", "_")
                csv_path = player_dir / "2026_processed_processed.csv"
                if csv_path.exists():
                    try:
                        pdf = pd.read_csv(csv_path)
                        if not pdf.empty:
                            last_row = pdf.iloc[-1]
                            if not np.isfinite(era) and "Opp_Pitcher_ERA_3" in pdf.columns:
                                era = _sf(last_row.get("Opp_Pitcher_ERA_3"), float("nan"))
                            if not np.isfinite(k9) and "Opp_Pitcher_K9_3" in pdf.columns:
                                k9 = _sf(last_row.get("Opp_Pitcher_K9_3"), float("nan"))
                    except Exception:
                        pass

        result = compute_pitcher_adjustment(
            target=str(row.get("Target", row.get("target", ""))),
            direction=str(row.get("Direction", row.get("direction", ""))),
            player_type=str(row.get("Player_Type", row.get("player_type", "hitter"))),
            opp_pitcher_era=era,
            opp_pitcher_k9=k9,
            config=cfg,
        )
        adjustments.append(result["adjustment"])
        qualities.append(result["pitcher_quality"])

    out["pitcher_matchup_adj"] = adjustments
    out["pitcher_quality"] = qualities
    return out
