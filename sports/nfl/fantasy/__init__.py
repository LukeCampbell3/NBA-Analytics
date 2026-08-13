"""NFL fantasy-football projection and draft-ranking tools."""

from .model import (
    FantasyConfig,
    ScoringSettings,
    build_draft_rankings,
    fantasy_points,
    fit_accuracy_layer,
    validate_projection_model,
)
from .accuracy import train_accuracy_model

__all__ = [
    "FantasyConfig",
    "ScoringSettings",
    "build_draft_rankings",
    "fantasy_points",
    "fit_accuracy_layer",
    "validate_projection_model",
    "train_accuracy_model",
]
