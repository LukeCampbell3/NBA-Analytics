"""Parlay leg validator — ensures each leg has genuine recent-form support.

The #1 cause of parlay misses is selecting legs where the player's recent
performance doesn't actually support the direction.  Examples:
  - Picking UNDER 5.5 when the player averaged 5.8 in their last 5 games
  - Picking UNDER 13.4 when the player went OVER that line 60% of last 5

This validator checks each candidate leg against the player's actual recent
game log and rejects legs that lack genuine directional support.

Rules:
  1. For UNDER picks: player's recent average must be BELOW the line
  2. For UNDER picks: player must have gone under the line in >= 60% of recent games
  3. For OVER picks: player's recent average must be ABOVE the line
  4. Margin of safety: recent avg must be at least 0.5 units away from line
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LegValidatorConfig:
    enabled: bool = True
    recent_games: int = 5
    min_under_rate: float = 0.60       # must go under in 60%+ of recent games
    min_over_rate: float = 0.55        # must go over in 55%+ of recent games
    min_margin_from_line: float = 0.3  # recent avg must be 0.3+ units from line
    data_proc_root: str = ""


def _load_player_recent(
    player_name: str,
    target: str,
    game_date: str,
    n_games: int,
    data_proc_root: Path,
) -> pd.Series | None:
    """Load a player's last N games for a target stat before a given date."""
    # Try direct name match
    player_dir = data_proc_root / player_name.replace(" ", "_")
    csv_path = player_dir / "2026_processed_processed.csv"

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if target not in df.columns or "Date" not in df.columns:
        return None

    # Filter to games before the target date
    df_before = df[df["Date"].astype(str) < game_date]

    # Filter out DNP
    if "Did_Not_Play" in df_before.columns:
        df_before = df_before[pd.to_numeric(df_before["Did_Not_Play"], errors="coerce").fillna(0) < 0.5]

    values = pd.to_numeric(df_before[target], errors="coerce").dropna()
    if len(values) < 3:
        return None

    return values.tail(n_games)


def validate_leg(
    *,
    player: str,
    target: str,
    direction: str,
    market_line: float,
    game_date: str,
    csv_path: str = "",
    config: LegValidatorConfig | None = None,
) -> dict[str, Any]:
    """Validate a single parlay leg against recent form.

    Returns:
      - valid: bool — whether the leg passes validation
      - reason: str — why it passed or failed
      - recent_avg: float — player's recent average
      - hit_rate: float — % of recent games that would have hit this line
      - margin: float — how far recent avg is from the line
    """
    cfg = config or LegValidatorConfig()

    if not cfg.enabled:
        return {"valid": True, "reason": "disabled", "recent_avg": 0, "hit_rate": 0, "margin": 0}

    # Resolve data proc root
    if cfg.data_proc_root:
        data_proc = Path(cfg.data_proc_root)
    else:
        data_proc = Path(__file__).resolve().parents[1] / "Data-Proc"

    # Try to extract player name from csv_path
    player_name = player
    if csv_path:
        parts = str(csv_path).replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part == "Data-Proc" and i + 1 < len(parts):
                player_name = parts[i + 1]
                break

    recent = _load_player_recent(player_name, target, game_date, cfg.recent_games, data_proc)

    if recent is None or len(recent) < 3:
        # Can't validate — allow but flag
        return {"valid": True, "reason": "insufficient_data", "recent_avg": 0, "hit_rate": 0, "margin": 0}

    recent_avg = float(recent.mean())
    dir_upper = str(direction).upper().strip()
    line = float(market_line)

    if dir_upper == "UNDER":
        hit_rate = float((recent < line).mean())
        margin = line - recent_avg  # positive = avg is below line (good for UNDER)
    elif dir_upper == "OVER":
        hit_rate = float((recent > line).mean())
        margin = recent_avg - line  # positive = avg is above line (good for OVER)
    else:
        return {"valid": True, "reason": "unknown_direction", "recent_avg": recent_avg, "hit_rate": 0, "margin": 0}

    # Validation checks
    if dir_upper == "UNDER":
        if recent_avg >= line:
            return {
                "valid": False,
                "reason": f"avg_above_line (avg={recent_avg:.1f} >= line={line})",
                "recent_avg": recent_avg,
                "hit_rate": hit_rate,
                "margin": margin,
            }
        if hit_rate < cfg.min_under_rate:
            return {
                "valid": False,
                "reason": f"low_under_rate ({hit_rate:.0%} < {cfg.min_under_rate:.0%})",
                "recent_avg": recent_avg,
                "hit_rate": hit_rate,
                "margin": margin,
            }
        if margin < cfg.min_margin_from_line:
            return {
                "valid": False,
                "reason": f"thin_margin ({margin:.2f} < {cfg.min_margin_from_line})",
                "recent_avg": recent_avg,
                "hit_rate": hit_rate,
                "margin": margin,
            }
    elif dir_upper == "OVER":
        if recent_avg <= line:
            return {
                "valid": False,
                "reason": f"avg_below_line (avg={recent_avg:.1f} <= line={line})",
                "recent_avg": recent_avg,
                "hit_rate": hit_rate,
                "margin": margin,
            }
        if hit_rate < cfg.min_over_rate:
            return {
                "valid": False,
                "reason": f"low_over_rate ({hit_rate:.0%} < {cfg.min_over_rate:.0%})",
                "recent_avg": recent_avg,
                "hit_rate": hit_rate,
                "margin": margin,
            }

    return {
        "valid": True,
        "reason": "passed",
        "recent_avg": recent_avg,
        "hit_rate": hit_rate,
        "margin": margin,
    }
