"""NFL prediction and evaluation pipeline."""

from .pipeline import TARGET_SPECS, build_features, train_and_backtest

__all__ = ["TARGET_SPECS", "build_features", "train_and_backtest"]
