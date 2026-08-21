"""MLB path-conditioned evidence research package (shadow-only).

This package is a scoped port of ``sports/nba/conditional_chain``'s newest
path-evidence layer (``path_world_evidence`` / ``path_conditioned_backtest`` /
``path_conditioned_cli``) plus the sport-agnostic joint-outcome-world
machinery it depends on (``outcome_worlds`` / ``proof_trajectory``).

It intentionally does NOT port the NBA package's frozen selector, chain
resolver, authorization, freeze/confirmation, or survival-builder modules.
Instead it consumes an already-scored MLB candidate reservoir (player,
market, side, a stable historical ``robust_score``, and a day-of
``survival_probability``) produced by the existing MLB prediction-pool
pipeline (see ``sports/mlb/scripts/generate_daily_prediction_pool.py`` and
``sports/mlb/scripts/pick_survival_model.py``) rather than re-deriving that
score itself. See ``README.md`` in this directory for the full scope note.
"""

from .protocol import (
    ALLOCATION_PATH_PROTOCOL,
    BINARY_OUTCOME_SET_PROTOCOL,
)

__all__ = [
    "ALLOCATION_PATH_PROTOCOL",
    "BINARY_OUTCOME_SET_PROTOCOL",
]
