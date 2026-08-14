#!/usr/bin/env python3
"""Leakage-safe NumPy inference for the GPU-trained MLB latent parlay model."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_VERSION = "mlb_latent_parlay_set_attention_v3"
EVIDENCE_LABEL = "SYNTHETIC_H05_OUTCOME_SHADOW_NO_EXECUTABLE_ROI_CLAIM"
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "predictions"
    / "calibration"
    / "latent_parlay_model_2026.json"
)

NUMERIC_FEATURES = (
    "baseline",
    "last_hits",
    "batting_order",
    "log_history_rows",
    "is_home",
    "batter_strength",
    "pitcher_vulnerability",
    "pitcher_uncertainty",
    "batter_vs_starter_games",
    "batter_vs_starter_lift",
    "archetype_neighbor_games",
    "archetype_neighbor_support",
    "archetype_neighbor_lift",
    "matchup_network_score",
    "matchup_network_confidence",
    "matchup_network_adjustment",
)
CATEGORICAL_FEATURES = ("player", "pitcher", "team", "opponent")
SUPPORT_FEATURES = (
    "baseline",
    "batting_order",
    "is_home",
    "batter_strength",
    "pitcher_vulnerability",
    "pitcher_uncertainty",
    "batter_vs_starter_lift",
    "archetype_neighbor_lift",
    "matchup_network_score",
    "matchup_network_confidence",
    "matchup_network_adjustment",
)
MARKET_RESIDUAL_LATENT_WEIGHT = 0.20
MARKET_RESIDUAL_MARKET_WEIGHT = 0.80
MARKET_RESIDUAL_UNCERTAINTY_PENALTY = 0.25


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if math.isfinite(output) else float(default)


def market_residual_probability(
    latent_probability: Any,
    market_probability: Any,
    ensemble_std: Any,
) -> float:
    probability = (
        MARKET_RESIDUAL_LATENT_WEIGHT * _finite(latent_probability, 0.5)
        + MARKET_RESIDUAL_MARKET_WEIGHT * _finite(market_probability, 0.5)
        - MARKET_RESIDUAL_UNCERTAINTY_PENALTY * max(0.0, _finite(ensemble_std, 0.0))
    )
    return max(0.01, min(0.99, probability))


def _hash_bucket(value: Any, buckets: int) -> int:
    text = unicodedata.normalize("NFKD", str(value or "unknown"))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    normalized = re.sub(r"[^a-z0-9]", "", text) or "unknown"
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(buckets)


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (
        1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * np.power(values, 3)))
    )


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    positive = array >= 0.0
    output = np.empty_like(array)
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    negative_exp = np.exp(array[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def _layer_norm(values: np.ndarray, weight: np.ndarray, bias: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.square(values - mean).mean(axis=-1, keepdims=True)
    return ((values - mean) / np.sqrt(variance + epsilon)) * weight + bias


def candidate_features(
    candidate: Any,
    *,
    last_hits: float,
    batting_order: float,
) -> tuple[dict[str, float], dict[str, str]]:
    raw = getattr(candidate, "raw", {}) or {}
    numeric = {
        "baseline": _finite(raw.get("Baseline"), _finite(getattr(candidate, "prediction", 0.0))),
        "last_hits": _finite(last_hits),
        "batting_order": max(1.0, min(9.0, _finite(batting_order, 6.0))),
        "log_history_rows": math.log1p(max(0.0, _finite(getattr(candidate, "history_rows", 0.0)))),
        "is_home": _finite(raw.get("Is_Home")),
        "batter_strength": _finite(raw.get("Batter_Profile_Strength")),
        "pitcher_vulnerability": _finite(raw.get("Pitcher_Profile_Vulnerability")),
        "pitcher_uncertainty": _finite(raw.get("Pitcher_Profile_Uncertainty"), 1.0),
        "batter_vs_starter_games": _finite(raw.get("Batter_Vs_Starter_Games")),
        "batter_vs_starter_lift": _finite(raw.get("Batter_Vs_Starter_Lift")),
        "archetype_neighbor_games": _finite(raw.get("Archetype_Neighbor_Games")),
        "archetype_neighbor_support": _finite(raw.get("Archetype_Neighbor_Effective_Support")),
        "archetype_neighbor_lift": _finite(raw.get("Archetype_Neighbor_Lift")),
        "matchup_network_score": _finite(raw.get("Matchup_Network_Score")),
        "matchup_network_confidence": _finite(raw.get("Matchup_Network_Confidence")),
        "matchup_network_adjustment": _finite(raw.get("Matchup_Network_Adjustment")),
    }
    categorical = {
        "player": str(getattr(candidate, "player_id", "") or getattr(candidate, "player", "")),
        "pitcher": str(raw.get("Opposing_Pitcher_ID") or raw.get("Opposing_Pitcher") or "unknown"),
        "team": str(getattr(candidate, "team", "") or "unknown"),
        "opponent": str(raw.get("Opponent_ID") or raw.get("Opponent") or "unknown"),
    }
    return numeric, categorical


@dataclass(frozen=True)
class LatentPrediction:
    probability: float
    raw_probability: float
    ensemble_std: float
    support_fraction: float
    in_support: bool


class LatentParlayBundle:
    """Portable inference bundle exported by the CUDA trainer."""

    def __init__(self, artifact: Mapping[str, Any]):
        self.artifact = dict(artifact)
        self.model_version = str(artifact.get("model_version", ""))
        self.evidence_label = str(artifact.get("evidence_label", EVIDENCE_LABEL))
        self.numeric_features = tuple(artifact["schema"]["numeric_features"])
        if self.numeric_features != NUMERIC_FEATURES:
            raise ValueError("Latent artifact numeric feature schema does not match runtime")
        self.mean = np.asarray(artifact["scaler"]["mean"], dtype=np.float64)
        self.scale = np.asarray(artifact["scaler"]["scale"], dtype=np.float64)
        self.support_low = np.asarray(artifact["support"]["standardized_low"], dtype=np.float64)
        self.support_high = np.asarray(artifact["support"]["standardized_high"], dtype=np.float64)
        self.clip_low = np.asarray(artifact["support"]["clip_low"], dtype=np.float64)
        self.clip_high = np.asarray(artifact["support"]["clip_high"], dtype=np.float64)
        support_names = tuple(artifact["support"]["features"])
        self.support_indices = np.asarray([self.numeric_features.index(name) for name in support_names], dtype=int)
        self.minimum_support_fraction = float(artifact["support"].get("minimum_fraction", 0.80))
        self.category_buckets = {key: int(value) for key, value in artifact["schema"]["category_buckets"].items()}
        self.models = [self._arrays(model) for model in artifact["ensemble"]]
        self.leg_calibration = artifact["calibration"]["leg"]
        self.ticket_calibration = artifact["calibration"]["ticket"]

    @staticmethod
    def _arrays(model: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return {key: np.asarray(value, dtype=np.float64) for key, value in model.items() if key != "seed"}

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT_PATH) -> "LatentParlayBundle | None":
        if not path.exists():
            return None
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("model_version") != MODEL_VERSION or artifact.get("status") not in {"shadow", "active"}:
            return None
        return cls(artifact)

    def _input(self, numeric: Mapping[str, float], categorical: Mapping[str, str]) -> tuple[np.ndarray, dict[str, int], float]:
        values = np.asarray([_finite(numeric.get(name)) for name in self.numeric_features], dtype=np.float64)
        raw_standardized = (values - self.mean) / self.scale
        supported = (
            (raw_standardized[self.support_indices] >= self.support_low[self.support_indices])
            & (raw_standardized[self.support_indices] <= self.support_high[self.support_indices])
        )
        fraction = float(supported.mean())
        standardized = np.clip(raw_standardized, self.clip_low, self.clip_high)
        categories = {
            name: _hash_bucket(categorical.get(name, "unknown"), self.category_buckets[name])
            for name in CATEGORICAL_FEATURES
        }
        return standardized, categories, fraction

    @staticmethod
    def _encode(model: Mapping[str, np.ndarray], numeric: np.ndarray, categories: Mapping[str, int]) -> tuple[np.ndarray, float]:
        embeddings = [model[f"embedding_{name}"][categories[name]] for name in CATEGORICAL_FEATURES]
        inputs = np.concatenate((numeric, *embeddings), axis=-1)
        hidden = _gelu(inputs @ model["encoder_0_weight"].T + model["encoder_0_bias"])
        latent = _gelu(hidden @ model["encoder_2_weight"].T + model["encoder_2_bias"])
        logit = float(latent @ model["leg_head_weight"].reshape(-1) + model["leg_head_bias"].reshape(-1)[0])
        return latent, logit

    @staticmethod
    def _attention(model: Mapping[str, np.ndarray], latents: np.ndarray) -> np.ndarray:
        dimensions = latents.shape[-1]
        heads = int(model["attention_heads"].reshape(-1)[0])
        head_dimensions = dimensions // heads
        query = latents @ model["query_weight"].T
        key = latents @ model["key_weight"].T
        value = latents @ model["value_weight"].T
        query = query.reshape(len(latents), heads, head_dimensions).transpose(1, 0, 2)
        key = key.reshape(len(latents), heads, head_dimensions).transpose(1, 0, 2)
        value = value.reshape(len(latents), heads, head_dimensions).transpose(1, 0, 2)
        weights = _softmax(query @ key.transpose(0, 2, 1) / math.sqrt(head_dimensions), axis=-1)
        attended = (weights @ value).transpose(1, 0, 2).reshape(len(latents), dimensions)
        attended = attended @ model["attention_out_weight"].T
        normalized = _layer_norm(
            latents + attended,
            model["attention_norm_weight"],
            model["attention_norm_bias"],
        )
        feed_forward = _gelu(normalized @ model["ff_0_weight"].T + model["ff_0_bias"])
        feed_forward = feed_forward @ model["ff_2_weight"].T + model["ff_2_bias"]
        return _layer_norm(
            normalized + feed_forward,
            model["ff_norm_weight"],
            model["ff_norm_bias"],
        )

    @staticmethod
    def _calibrate(logit: float, calibration: Mapping[str, Any]) -> float:
        return float(_sigmoid(float(calibration["slope"]) * logit + float(calibration["intercept"])))

    def predict_leg(self, numeric: Mapping[str, float], categorical: Mapping[str, str]) -> LatentPrediction:
        standardized, categories, support_fraction = self._input(numeric, categorical)
        logits = np.asarray([self._encode(model, standardized, categories)[1] for model in self.models])
        raw_probabilities = _sigmoid(logits)
        mean_logit = float(logits.mean())
        return LatentPrediction(
            probability=self._calibrate(mean_logit, self.leg_calibration),
            raw_probability=float(raw_probabilities.mean()),
            ensemble_std=float(raw_probabilities.std()),
            support_fraction=support_fraction,
            in_support=support_fraction >= self.minimum_support_fraction,
        )

    def predict_ticket(
        self,
        legs: Sequence[tuple[Mapping[str, float], Mapping[str, str]]],
    ) -> LatentPrediction:
        if not 2 <= len(legs) <= 4:
            raise ValueError("Latent parlay model supports two through four legs")
        prepared = [self._input(numeric, categorical) for numeric, categorical in legs]
        support_fraction = min(item[2] for item in prepared)
        logits: list[float] = []
        raw_probabilities: list[float] = []
        for model in self.models:
            encoded = [self._encode(model, numeric, categories) for numeric, categories, _ in prepared]
            latents = np.asarray([item[0] for item in encoded])
            leg_probabilities = _sigmoid(np.asarray([item[1] for item in encoded]))
            contextual = self._attention(model, latents)
            pooled = contextual.mean(axis=0)
            statistics = np.asarray(
                [leg_probabilities.min(), leg_probabilities.mean(), np.prod(leg_probabilities), len(legs) / 4.0]
            )
            ticket_input = np.concatenate((pooled, statistics))
            hidden = _gelu(ticket_input @ model["ticket_head_0_weight"].T + model["ticket_head_0_bias"])
            residual = 0.35 * math.tanh(
                float(hidden @ model["ticket_head_2_weight"].reshape(-1) + model["ticket_head_2_bias"].reshape(-1)[0])
            )
            independent_probability = float(np.clip(np.prod(leg_probabilities), 1e-5, 1.0 - 1e-5))
            logit = math.log(independent_probability) - math.log1p(-independent_probability) + residual
            logits.append(logit)
            raw_probabilities.append(float(_sigmoid(logit)))
        mean_logit = float(np.mean(logits))
        return LatentPrediction(
            probability=self._calibrate(mean_logit, self.ticket_calibration),
            raw_probability=float(np.mean(raw_probabilities)),
            ensemble_std=float(np.std(raw_probabilities)),
            support_fraction=support_fraction,
            in_support=support_fraction >= self.minimum_support_fraction,
        )
