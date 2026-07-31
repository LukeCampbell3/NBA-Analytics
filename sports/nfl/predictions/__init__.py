"""NFL prediction and evaluation pipeline."""

from .pipeline import TARGET_SPECS, build_features, train_and_backtest
from .latent_pipeline import predict_week_latent, train_and_backtest_latent

__all__ = [
    "TARGET_SPECS",
    "build_features",
    "predict_week_latent",
    "train_and_backtest",
    "train_and_backtest_latent",
]
