"""Fold-local predictive latent states for NFL player histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .pipeline import HISTORY_COLUMNS


LATENT_KEY_COLUMNS = ["player_id", "season", "week"]


def build_sequence_table(
    stats: pd.DataFrame,
    *,
    sequence_length: int = 8,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Flatten the preceding games while retaining explicit history masks."""

    frame = stats.sort_values(["player_id", "season", "week"]).reset_index(drop=True).copy()
    values = [column for column in HISTORY_COLUMNS if column in frame.columns]
    grouped = frame.groupby("player_id", sort=False)
    feature_columns: list[str] = []
    generated: dict[str, pd.Series] = {}
    for lag in range(1, sequence_length + 1):
        lagged = grouped[values].shift(lag)
        for column in values:
            feature = f"seq_lag{lag}_{column}"
            generated[feature] = pd.to_numeric(lagged[column], errors="coerce")
            feature_columns.append(feature)
        mask_feature = f"seq_lag{lag}_observed"
        generated[mask_feature] = grouped["season"].shift(lag).notna().astype(float)
        feature_columns.append(mask_feature)
    generated["sequence_games_prior"] = grouped.cumcount()
    feature_columns.append("sequence_games_prior")
    for column in values:
        generated[f"next_{column}"] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    target_columns = [f"next_{column}" for column in values]
    generated_frame = pd.DataFrame(generated, index=frame.index)
    output = pd.concat([frame[LATENT_KEY_COLUMNS], generated_frame], axis=1)
    return output[LATENT_KEY_COLUMNS + feature_columns + target_columns], feature_columns, target_columns


@dataclass
class PredictiveLatentEncoder:
    sequence_length: int = 8
    latent_dimensions: int = 16
    random_state: int = 42
    minimum_prior_games: int = 3

    def fit(self, sequence_table: pd.DataFrame, feature_columns: list[str], target_columns: list[str]) -> "PredictiveLatentEncoder":
        eligible = sequence_table["sequence_games_prior"].ge(self.minimum_prior_games)
        train = sequence_table.loc[eligible]
        if len(train) < 500:
            raise ValueError("At least 500 historical sequences are required for latent pretraining.")
        self.feature_columns_ = list(feature_columns)
        self.target_columns_ = list(target_columns)
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=False)
        self.input_scaler_ = StandardScaler()
        self.target_scaler_ = StandardScaler()
        x_imputed = self.imputer_.fit_transform(train[self.feature_columns_])
        x_scaled = self.input_scaler_.fit_transform(x_imputed)
        y_scaled = self.target_scaler_.fit_transform(train[self.target_columns_].astype(float))
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(64, self.latent_dimensions),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=160,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=self.random_state,
        )
        self.model_.fit(x_scaled, y_scaled)
        self.training_rows_ = int(len(train))
        self.iterations_ = int(self.model_.n_iter_)
        self.pretraining_validation_score_ = float(self.model_.best_validation_score_)
        return self

    def transform(self, sequence_table: pd.DataFrame) -> np.ndarray:
        x_imputed = self.imputer_.transform(sequence_table[self.feature_columns_])
        activation = self.input_scaler_.transform(x_imputed)
        # The final hidden layer is the predictive player-state bottleneck.
        for weights, bias in zip(self.model_.coefs_[:-1], self.model_.intercepts_[:-1]):
            activation = np.maximum(0.0, activation @ weights + bias)
        return np.asarray(activation, dtype=float)

    def transform_frame(self, sequence_table: pd.DataFrame) -> pd.DataFrame:
        latent = self.transform(sequence_table)
        output = sequence_table[LATENT_KEY_COLUMNS].copy()
        for index in range(latent.shape[1]):
            output[f"latent_{index:02d}"] = latent[:, index]
        return output
