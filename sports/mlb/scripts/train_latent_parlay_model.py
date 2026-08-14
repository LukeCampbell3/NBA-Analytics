#!/usr/bin/env python3
"""Train the leakage-safe MLB latent parlay ensemble on a CUDA device."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.optimize import minimize
from torch import nn

try:
    from .latent_parlay_model import (
        CATEGORICAL_FEATURES,
        EVIDENCE_LABEL,
        LatentParlayBundle,
        MODEL_VERSION,
        NUMERIC_FEATURES,
        _hash_bucket,
        calendar_features,
    )
except ImportError:
    from latent_parlay_model import (
        CATEGORICAL_FEATURES,
        EVIDENCE_LABEL,
        LatentParlayBundle,
        MODEL_VERSION,
        NUMERIC_FEATURES,
        _hash_bucket,
        calendar_features,
    )


SEEDS = (20260814, 20260831, 20260917)
CATEGORY_BUCKETS = {"player": 512, "pitcher": 512, "team": 64, "opponent": 64}
CATEGORY_DIMENSIONS = {"player": 8, "pitcher": 8, "team": 4, "opponent": 4}
LATENT_DIMENSIONS = 32
ATTENTION_HEADS = 4
MAX_LEGS = 4
MIN_HISTORY_ROWS = 35

SOURCE_COLUMNS = {
    "Date",
    "Player",
    "Player_MLBAM_ID",
    "Player_Type",
    "Team",
    "Opponent",
    "Game_ID",
    "Is_Home",
    "Opp_Starter_ID",
    "Opp_Starter_Player",
    "H",
    "Batting_Order",
    "Did_Not_Play",
    "H_rolling_avg",
    "H_lag1",
    "Matchup_Network_Batter_Support",
    "Matchup_Network_Pitcher_Support",
    "Pitcher_Profile_Uncertainty",
    "Batter_Vs_Starter_Games",
    "Matchup_Network_Confidence",
    "Batter_Profile_H_Strength",
    "Pitcher_Profile_H_Vulnerability",
    "Batter_Vs_Starter_H_Lift",
    "Archetype_Neighbor_H_Games",
    "Archetype_Neighbor_H_Effective_Support",
    "Archetype_Neighbor_H_Lift",
    "Matchup_Network_H_Score",
    "Matchup_Network_H_Adjustment",
}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if math.isfinite(output) else float(default)


def build_rows(processed_root: Path, *, before_date: date) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(processed_root.glob("*/20*_processed_processed.csv")):
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in SOURCE_COLUMNS, low_memory=False)
        except Exception:
            continue
        if frame.empty or "Player_Type" not in frame or not frame["Player_Type"].eq("hitter").any():
            continue
        frame = frame.loc[frame["Player_Type"].eq("hitter")].copy()
        frame["_date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.loc[frame["_date"].dt.date < before_date].sort_values(["_date", "Game_ID"], kind="stable")
        frame["_history_rows"] = np.arange(len(frame), dtype=int)
        for _, row in frame.iterrows():
            batting_order = finite(row.get("Batting_Order"), 0.0)
            if (
                int(row["_history_rows"]) < MIN_HISTORY_ROWS
                or finite(row.get("Did_Not_Play"), 0.0) != 0.0
                or not 1.0 <= batting_order <= 9.0
                or not math.isfinite(finite(row.get("H"), float("nan")))
            ):
                continue
            row_date = row["_date"].date()
            numeric = {
                "baseline": finite(row.get("H_rolling_avg")),
                "last_hits": finite(row.get("H_lag1")),
                "batting_order": batting_order,
                "log_history_rows": math.log1p(int(row["_history_rows"])),
                "is_home": finite(row.get("Is_Home")),
                **calendar_features(row_date),
                "batter_strength": finite(row.get("Batter_Profile_H_Strength")),
                "pitcher_vulnerability": finite(row.get("Pitcher_Profile_H_Vulnerability")),
                "pitcher_uncertainty": finite(row.get("Pitcher_Profile_Uncertainty"), 1.0),
                "batter_vs_starter_games": finite(row.get("Batter_Vs_Starter_Games")),
                "batter_vs_starter_lift": finite(row.get("Batter_Vs_Starter_H_Lift")),
                "archetype_neighbor_games": finite(row.get("Archetype_Neighbor_H_Games")),
                "archetype_neighbor_support": finite(row.get("Archetype_Neighbor_H_Effective_Support")),
                "archetype_neighbor_lift": finite(row.get("Archetype_Neighbor_H_Lift")),
                "matchup_network_score": finite(row.get("Matchup_Network_H_Score")),
                "matchup_network_confidence": finite(row.get("Matchup_Network_Confidence")),
                "matchup_network_adjustment": finite(row.get("Matchup_Network_H_Adjustment")),
            }
            records.append(
                {
                    "date": row_date.isoformat(),
                    "game_id": str(row.get("Game_ID", "")),
                    "player": str(row.get("Player_MLBAM_ID") or row.get("Player") or path.parent.name),
                    "pitcher": str(row.get("Opp_Starter_ID") or row.get("Opp_Starter_Player") or "unknown"),
                    "team": str(row.get("Team") or "unknown"),
                    "opponent": str(row.get("Opponent") or "unknown"),
                    "win": int(finite(row.get("H")) > 0.5),
                    **numeric,
                }
            )
    output = pd.DataFrame.from_records(records)
    if output.empty:
        return output
    return output.drop_duplicates(["date", "game_id", "player"], keep="last").sort_values(
        ["date", "game_id", "player"], kind="stable"
    ).reset_index(drop=True)


def split_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    dates = sorted(rows["date"].unique())
    if len(dates) < 60:
        raise ValueError(f"At least 60 dates are required; found {len(dates)}")
    development_end = max(1, int(len(dates) * 0.68))
    calibration_end = max(development_end + 1, int(len(dates) * 0.84))
    development_dates = dates[:development_end]
    calibration_dates = dates[development_end:calibration_end]
    holdout_dates = dates[calibration_end:]
    partitions = {
        "development": {"start": development_dates[0], "end": development_dates[-1]},
        "calibration": {"start": calibration_dates[0], "end": calibration_dates[-1]},
        "locked_holdout": {"start": holdout_dates[0], "end": holdout_dates[-1]},
    }
    return (
        rows.loc[rows["date"].isin(development_dates)].copy(),
        rows.loc[rows["date"].isin(calibration_dates)].copy(),
        rows.loc[rows["date"].isin(holdout_dates)].copy(),
        partitions,
    )


@dataclass(frozen=True)
class PreparedRows:
    numeric: np.ndarray
    categories: np.ndarray
    labels: np.ndarray
    frame: pd.DataFrame


def prepare_rows(frame: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> PreparedRows:
    numeric = frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    numeric = np.clip((numeric - mean) / scale, -8.0, 8.0).astype(np.float32)
    categories = np.column_stack(
        [
            frame[name].map(lambda value, key=name: _hash_bucket(value, CATEGORY_BUCKETS[key])).to_numpy(dtype=np.int64)
            for name in CATEGORICAL_FEATURES
        ]
    )
    return PreparedRows(
        numeric=numeric,
        categories=categories,
        labels=frame["win"].to_numpy(dtype=np.float32),
        frame=frame.reset_index(drop=True),
    )


def heuristic_score(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["baseline"]
        - (0.025 * frame["batting_order"])
        + (0.08 * frame["batter_strength"])
        + (0.08 * frame["pitcher_vulnerability"])
        - (0.04 * frame["pitcher_uncertainty"])
        + (0.05 * frame["matchup_network_score"])
    )


def build_ticket_indices(frame: pd.DataFrame, *, seed: int, samples_per_date: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tickets: list[list[int]] = []
    masks: list[list[float]] = []
    labels: list[float] = []
    for _, slate in frame.groupby("date", sort=True):
        slate = slate.copy()
        slate["_score"] = heuristic_score(slate)
        slate = slate.sort_values("_score", ascending=False, kind="stable").head(36)
        games = [group.index.to_numpy(dtype=int) for _, group in slate.groupby("game_id", sort=False)]
        if len(games) < 2:
            continue
        for leg_count in range(2, MAX_LEGS + 1):
            if len(games) < leg_count:
                continue
            count = max(16, samples_per_date // 3)
            for _ in range(count):
                selected_games = rng.choice(len(games), size=leg_count, replace=False)
                selected = [int(rng.choice(games[index])) for index in selected_games]
                padded = selected + [selected[-1]] * (MAX_LEGS - leg_count)
                mask = [1.0] * leg_count + [0.0] * (MAX_LEGS - leg_count)
                tickets.append(padded)
                masks.append(mask)
                labels.append(float(frame.loc[selected, "win"].min()))
    return (
        np.asarray(tickets, dtype=np.int64),
        np.asarray(masks, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


class LatentSetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(CATEGORY_BUCKETS[name], CATEGORY_DIMENSIONS[name])
                for name in CATEGORICAL_FEATURES
            }
        )
        input_dimensions = len(NUMERIC_FEATURES) + sum(CATEGORY_DIMENSIONS.values())
        self.encoder = nn.Sequential(
            nn.Linear(input_dimensions, 64),
            nn.GELU(approximate="tanh"),
            nn.Linear(64, LATENT_DIMENSIONS),
            nn.GELU(approximate="tanh"),
        )
        self.leg_head = nn.Linear(LATENT_DIMENSIONS, 1)
        self.query = nn.Linear(LATENT_DIMENSIONS, LATENT_DIMENSIONS, bias=False)
        self.key = nn.Linear(LATENT_DIMENSIONS, LATENT_DIMENSIONS, bias=False)
        self.value = nn.Linear(LATENT_DIMENSIONS, LATENT_DIMENSIONS, bias=False)
        self.attention_out = nn.Linear(LATENT_DIMENSIONS, LATENT_DIMENSIONS, bias=False)
        self.attention_norm = nn.LayerNorm(LATENT_DIMENSIONS)
        self.ff = nn.Sequential(
            nn.Linear(LATENT_DIMENSIONS, 64),
            nn.GELU(approximate="tanh"),
            nn.Linear(64, LATENT_DIMENSIONS),
        )
        self.ff_norm = nn.LayerNorm(LATENT_DIMENSIONS)
        self.ticket_head = nn.Sequential(
            nn.Linear(LATENT_DIMENSIONS + 4, 32),
            nn.GELU(approximate="tanh"),
            nn.Linear(32, 1),
        )
        self.decoder = nn.Linear(LATENT_DIMENSIONS, len(NUMERIC_FEATURES))

    def encode(self, numeric: torch.Tensor, categories: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = [self.embeddings[name](categories[..., index]) for index, name in enumerate(CATEGORICAL_FEATURES)]
        latent = self.encoder(torch.cat((numeric, *embedded), dim=-1))
        return latent, self.leg_head(latent).squeeze(-1)

    def ticket(self, numeric: torch.Tensor, categories: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, leg_logits = self.encode(numeric, categories)
        batch, legs, dimensions = latent.shape
        head_dimensions = dimensions // ATTENTION_HEADS
        query = self.query(latent).view(batch, legs, ATTENTION_HEADS, head_dimensions).transpose(1, 2)
        key = self.key(latent).view(batch, legs, ATTENTION_HEADS, head_dimensions).transpose(1, 2)
        value = self.value(latent).view(batch, legs, ATTENTION_HEADS, head_dimensions).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_dimensions)
        scores = scores.masked_fill(mask[:, None, None, :].eq(0.0), -1e9)
        attended = (torch.softmax(scores, dim=-1) @ value).transpose(1, 2).reshape(batch, legs, dimensions)
        contextual = self.attention_norm(latent + self.attention_out(attended))
        contextual = self.ff_norm(contextual + self.ff(contextual))
        pooled = (contextual * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        probabilities = torch.sigmoid(leg_logits)
        minimum = probabilities.masked_fill(mask.eq(0.0), 1.0).min(dim=1).values
        mean = (probabilities * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        product = torch.where(mask.gt(0.0), probabilities, torch.ones_like(probabilities)).prod(dim=1)
        statistics = torch.stack((minimum, mean, product, mask.sum(dim=1) / MAX_LEGS), dim=1)
        residual = 0.35 * torch.tanh(self.ticket_head(torch.cat((pooled, statistics), dim=1)).squeeze(-1))
        independent_probability = product.clamp(1e-5, 1.0 - 1e-5)
        independent_logit = torch.log(independent_probability) - torch.log1p(-independent_probability)
        return independent_logit + residual, leg_logits


def batches(length: int, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = rng.permutation(length)
    return [indices[start : start + size] for start in range(0, length, size)]


def tensor(values: np.ndarray, device: torch.device, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


@torch.no_grad()
def predict_leg_logits(model: LatentSetModel, rows: PreparedRows, device: torch.device) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(rows.labels), 2048):
        _, logits = model.encode(
            tensor(rows.numeric[start : start + 2048], device),
            tensor(rows.categories[start : start + 2048], device),
        )
        output.append(logits.cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def predict_ticket_logits(
    model: LatentSetModel,
    rows: PreparedRows,
    ticket_indices: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(ticket_indices), 1024):
        indices = ticket_indices[start : start + 1024]
        logits, _ = model.ticket(
            tensor(rows.numeric[indices], device),
            tensor(rows.categories[indices], device),
            tensor(masks[start : start + 1024], device),
        )
        output.append(logits.cpu().numpy())
    return np.concatenate(output)


def validation_score(
    model: LatentSetModel,
    rows: PreparedRows,
    ticket_indices: np.ndarray,
    masks: np.ndarray,
    ticket_labels: np.ndarray,
    device: torch.device,
) -> float:
    leg_probability = 1.0 / (1.0 + np.exp(-predict_leg_logits(model, rows, device)))
    ticket_probability = 1.0 / (1.0 + np.exp(-predict_ticket_logits(model, rows, ticket_indices, masks, device)))
    return float(0.4 * brier_score_loss(rows.labels, leg_probability) + 0.6 * brier_score_loss(ticket_labels, ticket_probability))


def train_model(
    development: PreparedRows,
    calibration: PreparedRows,
    development_tickets: tuple[np.ndarray, np.ndarray, np.ndarray],
    calibration_tickets: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    seed: int,
    device: torch.device,
) -> LatentSetModel:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = LatentSetModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    leg_loss = nn.BCEWithLogitsLoss()
    ticket_loss = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    stale_epochs = 0
    rng = np.random.default_rng(seed)
    dev_ticket_indices, dev_masks, dev_ticket_labels = development_tickets
    cal_ticket_indices, cal_masks, cal_ticket_labels = calibration_tickets
    for _epoch in range(60):
        model.train()
        leg_batches = batches(len(development.labels), 512, rng)
        ticket_batches = batches(len(dev_ticket_labels), 256, rng)
        steps = max(len(leg_batches), len(ticket_batches))
        for step in range(steps):
            leg_index = leg_batches[step % len(leg_batches)]
            ticket_index = ticket_batches[step % len(ticket_batches)]
            numeric = tensor(development.numeric[leg_index], device)
            categories = tensor(development.categories[leg_index], device)
            labels = tensor(development.labels[leg_index], device)
            noise_mask = torch.rand_like(numeric).lt(0.15)
            latent, logits = model.encode(numeric.masked_fill(noise_mask, 0.0), categories)
            reconstruction = model.decoder(latent)
            reconstruction_loss = torch.square(reconstruction - numeric).masked_select(noise_mask).mean()
            selected_tickets = dev_ticket_indices[ticket_index]
            set_logits, set_leg_logits = model.ticket(
                tensor(development.numeric[selected_tickets], device),
                tensor(development.categories[selected_tickets], device),
                tensor(dev_masks[ticket_index], device),
            )
            set_labels = tensor(dev_ticket_labels[ticket_index], device)
            member_labels = tensor(development.labels[selected_tickets], device)
            member_mask = tensor(dev_masks[ticket_index], device)
            member_loss = nn.functional.binary_cross_entropy_with_logits(
                set_leg_logits,
                member_labels,
                reduction="none",
            )
            member_loss = (member_loss * member_mask).sum() / member_mask.sum().clamp_min(1.0)
            loss = (
                0.35 * leg_loss(logits, labels)
                + 0.20 * member_loss
                + 0.45 * ticket_loss(set_logits, set_labels)
                + 0.04 * reconstruction_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        score = validation_score(model, calibration, cal_ticket_indices, cal_masks, cal_ticket_labels, device)
        if score < best_score - 1e-5:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 10:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model


def fit_calibrator(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = labels.astype(float)

    def objective(parameters: np.ndarray) -> float:
        slope = math.exp(float(parameters[0]))
        values = np.clip(slope * logits + float(parameters[1]), -30.0, 30.0)
        return float(np.mean(np.logaddexp(0.0, values) - labels * values))

    prior = float(np.clip(labels.mean(), 1e-5, 1.0 - 1e-5))
    result = minimize(
        objective,
        np.asarray([0.0, math.log(prior / (1.0 - prior))]),
        method="L-BFGS-B",
        bounds=[(math.log(0.05), math.log(5.0)), (-10.0, 10.0)],
    )
    if not result.success:
        raise RuntimeError(f"Calibration failed: {result.message}")
    return {"slope": math.exp(float(result.x[0])), "intercept": float(result.x[1])}


def calibrated_probability(logits: np.ndarray, calibrator: dict[str, float]) -> np.ndarray:
    values = calibrator["slope"] * logits + calibrator["intercept"]
    return 1.0 / (1.0 + np.exp(-values))


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(int)
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(labels)),
        "wins": int(labels.sum()),
        "hit_rate": float(labels.mean()),
        "mean_probability": float(probabilities.mean()),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else None,
    }


def wilson(wins: int, rows: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if rows <= 0:
        return None, None
    probability = wins / rows
    denominator = 1.0 + z * z / rows
    center = (probability + z * z / (2.0 * rows)) / denominator
    margin = z * math.sqrt(probability * (1.0 - probability) / rows + z * z / (4.0 * rows * rows)) / denominator
    return center - margin, center + margin


def ranking_metrics(
    rows: PreparedRows,
    models: list[LatentSetModel],
    leg_calibrator: dict[str, float],
    ticket_calibrator: dict[str, float],
    device: torch.device,
) -> dict[str, Any]:
    leg_logits = np.mean([predict_leg_logits(model, rows, device) for model in models], axis=0)
    leg_probabilities = calibrated_probability(leg_logits, leg_calibrator)
    results: dict[str, Any] = {}
    for leg_count in range(2, MAX_LEGS + 1):
        latent_hits: list[int] = []
        independent_hits: list[int] = []
        for _, slate in rows.frame.groupby("date", sort=True):
            slate_indices = slate.index.to_numpy(dtype=int)
            ranking = np.argsort(leg_probabilities[slate_indices])[::-1]
            candidate_indices = slate_indices[ranking[:12]]
            combinations = [
                combination
                for combination in itertools.combinations(candidate_indices, leg_count)
                if len({str(rows.frame.loc[index, "game_id"]) for index in combination}) == leg_count
            ]
            if not combinations:
                continue
            ticket_indices = np.asarray(
                [list(combo) + [combo[-1]] * (MAX_LEGS - leg_count) for combo in combinations], dtype=np.int64
            )
            masks = np.asarray(
                [[1.0] * leg_count + [0.0] * (MAX_LEGS - leg_count) for _ in combinations], dtype=np.float32
            )
            ticket_logits = np.mean(
                [predict_ticket_logits(model, rows, ticket_indices, masks, device) for model in models], axis=0
            )
            ticket_probabilities = calibrated_probability(ticket_logits, ticket_calibrator)
            latent_choice = combinations[int(np.argmax(ticket_probabilities))]
            independent_choice = combinations[
                int(np.argmax([np.prod(leg_probabilities[list(combo)]) for combo in combinations]))
            ]
            latent_hits.append(int(rows.labels[list(latent_choice)].min()))
            independent_hits.append(int(rows.labels[list(independent_choice)].min()))
        latent_wins = int(sum(latent_hits))
        independent_wins = int(sum(independent_hits))
        latent_low, latent_high = wilson(latent_wins, len(latent_hits))
        independent_low, independent_high = wilson(independent_wins, len(independent_hits))
        results[str(leg_count)] = {
            "slates": len(latent_hits),
            "latent_ticket_wins": latent_wins,
            "latent_ticket_hit_rate": latent_wins / len(latent_hits) if latent_hits else None,
            "latent_wilson_95": [latent_low, latent_high],
            "independent_product_wins": independent_wins,
            "independent_product_hit_rate": independent_wins / len(independent_hits) if independent_hits else None,
            "independent_wilson_95": [independent_low, independent_high],
        }
    return results


def daily_top_two_leg_metrics(rows: PreparedRows, probabilities: np.ndarray) -> dict[str, Any]:
    selected: list[int] = []
    for _, slate in rows.frame.groupby("date", sort=True):
        slate_indices = slate.index.to_numpy(dtype=int)
        games: set[str] = set()
        for index in slate_indices[np.argsort(probabilities[slate_indices])[::-1]]:
            game_id = str(rows.frame.loc[index, "game_id"])
            if not game_id or game_id in games:
                continue
            selected.append(int(index))
            games.add(game_id)
            if len(games) == 2:
                break
    labels = rows.labels[selected].astype(int)
    wins = int(labels.sum())
    low, high = wilson(wins, len(labels))
    return {
        "legs": len(labels),
        "wins": wins,
        "hit_rate": float(labels.mean()) if len(labels) else None,
        "wilson_95": [low, high],
    }


def array(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().numpy().astype(np.float32).tolist()


def export_model(model: LatentSetModel, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        **{f"embedding_{name}": array(model.embeddings[name].weight) for name in CATEGORICAL_FEATURES},
        "encoder_0_weight": array(model.encoder[0].weight),
        "encoder_0_bias": array(model.encoder[0].bias),
        "encoder_2_weight": array(model.encoder[2].weight),
        "encoder_2_bias": array(model.encoder[2].bias),
        "leg_head_weight": array(model.leg_head.weight),
        "leg_head_bias": array(model.leg_head.bias),
        "attention_heads": [ATTENTION_HEADS],
        "query_weight": array(model.query.weight),
        "key_weight": array(model.key.weight),
        "value_weight": array(model.value.weight),
        "attention_out_weight": array(model.attention_out.weight),
        "attention_norm_weight": array(model.attention_norm.weight),
        "attention_norm_bias": array(model.attention_norm.bias),
        "ff_0_weight": array(model.ff[0].weight),
        "ff_0_bias": array(model.ff[0].bias),
        "ff_2_weight": array(model.ff[2].weight),
        "ff_2_bias": array(model.ff[2].bias),
        "ff_norm_weight": array(model.ff_norm.weight),
        "ff_norm_bias": array(model.ff_norm.bias),
        "ticket_head_0_weight": array(model.ticket_head[0].weight),
        "ticket_head_0_bias": array(model.ticket_head[0].bias),
        "ticket_head_2_weight": array(model.ticket_head[2].weight),
        "ticket_head_2_bias": array(model.ticket_head[2].bias),
    }


def verify_export_parity(
    artifact: dict[str, Any],
    rows: PreparedRows,
    tickets: tuple[np.ndarray, np.ndarray, np.ndarray],
    models: list[LatentSetModel],
    leg_calibrator: dict[str, float],
    ticket_calibrator: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    bundle = LatentParlayBundle(artifact)
    sample_index = 0
    source = rows.frame.loc[sample_index]
    numeric = {name: float(source[name]) for name in NUMERIC_FEATURES}
    categories = {name: str(source[name]) for name in CATEGORICAL_FEATURES}
    runtime_leg = bundle.predict_leg(numeric, categories)
    leg_logits = np.asarray([predict_leg_logits(model, rows, device)[sample_index] for model in models])
    expected_leg = float(calibrated_probability(np.asarray([leg_logits.mean()]), leg_calibrator)[0])
    expected_leg_raw = float((1.0 / (1.0 + np.exp(-leg_logits))).mean())

    ticket_indices = tickets[0][0:1]
    ticket_masks = tickets[1][0:1]
    leg_count = int(ticket_masks[0].sum())
    source_indices = ticket_indices[0, :leg_count]
    runtime_ticket = bundle.predict_ticket(
        [
            (
                {name: float(rows.frame.loc[index, name]) for name in NUMERIC_FEATURES},
                {name: str(rows.frame.loc[index, name]) for name in CATEGORICAL_FEATURES},
            )
            for index in source_indices
        ]
    )
    ticket_logits = np.asarray(
        [predict_ticket_logits(model, rows, ticket_indices, ticket_masks, device)[0] for model in models]
    )
    expected_ticket = float(calibrated_probability(np.asarray([ticket_logits.mean()]), ticket_calibrator)[0])
    expected_ticket_raw = float((1.0 / (1.0 + np.exp(-ticket_logits))).mean())
    differences = {
        "leg_probability_max_abs": abs(runtime_leg.probability - expected_leg),
        "leg_raw_probability_max_abs": abs(runtime_leg.raw_probability - expected_leg_raw),
        "ticket_probability_max_abs": abs(runtime_ticket.probability - expected_ticket),
        "ticket_raw_probability_max_abs": abs(runtime_ticket.raw_probability - expected_ticket_raw),
    }
    if max(differences.values()) > 1e-5:
        raise RuntimeError(f"NumPy export parity failed: {differences}")
    return differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--before-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for latent parlay training")
    device = torch.device("cuda")
    rows = build_rows(args.processed_root.resolve(), before_date=args.before_date)
    development_frame, calibration_frame, holdout_frame, partitions = split_rows(rows)
    mean = development_frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float64).mean(axis=0)
    scale = development_frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float64).std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    development = prepare_rows(development_frame, mean, scale)
    calibration = prepare_rows(calibration_frame, mean, scale)
    holdout = prepare_rows(holdout_frame, mean, scale)
    development_tickets = build_ticket_indices(development.frame, seed=7001, samples_per_date=240)
    calibration_tickets = build_ticket_indices(calibration.frame, seed=7002, samples_per_date=360)
    holdout_tickets = build_ticket_indices(holdout.frame, seed=7003, samples_per_date=360)
    models = [
        train_model(
            development,
            calibration,
            development_tickets,
            calibration_tickets,
            seed=seed,
            device=device,
        )
        for seed in SEEDS
    ]
    calibration_leg_logits = np.mean([predict_leg_logits(model, calibration, device) for model in models], axis=0)
    calibration_ticket_logits = np.mean(
        [
            predict_ticket_logits(model, calibration, calibration_tickets[0], calibration_tickets[1], device)
            for model in models
        ],
        axis=0,
    )
    leg_calibrator = fit_calibrator(calibration_leg_logits, calibration.labels)
    ticket_calibrator = fit_calibrator(calibration_ticket_logits, calibration_tickets[2])
    safe_baseline_calibrator = fit_calibrator(
        calibration.frame["baseline"].to_numpy(dtype=float),
        calibration.labels,
    )
    holdout_leg_logits = np.mean([predict_leg_logits(model, holdout, device) for model in models], axis=0)
    holdout_ticket_logits = np.mean(
        [predict_ticket_logits(model, holdout, holdout_tickets[0], holdout_tickets[1], device) for model in models],
        axis=0,
    )
    calibration_leg_probability = calibrated_probability(calibration_leg_logits, leg_calibrator)
    calibration_baseline_probability = calibrated_probability(
        calibration.frame["baseline"].to_numpy(dtype=float), safe_baseline_calibrator
    )
    holdout_leg_probability = calibrated_probability(holdout_leg_logits, leg_calibrator)
    holdout_baseline_probability = calibrated_probability(
        holdout.frame["baseline"].to_numpy(dtype=float), safe_baseline_calibrator
    )
    standardized_development = development.numeric.astype(np.float64)
    support_low = np.quantile(standardized_development, 0.005, axis=0)
    support_high = np.quantile(standardized_development, 0.995, axis=0)
    report = {
        "model_version": MODEL_VERSION,
        "status": "development_shadow",
        "evidence_label": EVIDENCE_LABEL,
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "claim_scope": "synthetic H over 0.5 hit-chain ranking only; no executable price or ROI claim",
        "leakage_exclusions": [
            "same-game H, PA, AB, Team_PA_share, wOBA, xwOBA, ISO, Barrel%, HardHit%",
            "historical H_market_gap and projection derived from same-game fields",
        ],
        "rows": int(len(rows)),
        "dates": int(rows["date"].nunique()),
        "partitions": partitions,
        "partition_rows": {
            "development": len(development.labels),
            "calibration": len(calibration.labels),
            "locked_holdout": len(holdout.labels),
        },
        "sampled_tickets": {
            "development": len(development_tickets[2]),
            "calibration": len(calibration_tickets[2]),
            "locked_holdout": len(holdout_tickets[2]),
        },
        "calibration": {
            "legs": metrics(calibration.labels, calibration_leg_probability),
            "safe_rolling_baseline": metrics(calibration.labels, calibration_baseline_probability),
            "sampled_tickets": metrics(
                calibration_tickets[2], calibrated_probability(calibration_ticket_logits, ticket_calibrator)
            ),
        },
        "locked_holdout": {
            "legs": metrics(holdout.labels, holdout_leg_probability),
            "safe_rolling_baseline": metrics(holdout.labels, holdout_baseline_probability),
            "daily_top_two_legs": {
                "latent": daily_top_two_leg_metrics(holdout, holdout_leg_probability),
                "safe_rolling_baseline": daily_top_two_leg_metrics(holdout, holdout_baseline_probability),
            },
            "sampled_tickets": metrics(
                holdout_tickets[2], calibrated_probability(holdout_ticket_logits, ticket_calibrator)
            ),
            "daily_ranking": ranking_metrics(holdout, models, leg_calibrator, ticket_calibrator, device),
        },
    }
    artifact = {
        "model_version": MODEL_VERSION,
        "status": "shadow",
        "evidence_label": EVIDENCE_LABEL,
        "trained_before_date": args.before_date.isoformat(),
        "schema": {
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "category_buckets": CATEGORY_BUCKETS,
            "maximum_legs": MAX_LEGS,
        },
        "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
        "support": {
            "standardized_low": support_low.tolist(),
            "standardized_high": support_high.tolist(),
            "minimum_fraction": 0.80,
        },
        "calibration": {"leg": leg_calibrator, "ticket": ticket_calibrator},
        "ensemble": [export_model(model, seed) for model, seed in zip(models, SEEDS)],
        "validation": report,
    }
    report["export_parity"] = verify_export_parity(
        artifact,
        holdout,
        holdout_tickets,
        models,
        leg_calibrator,
        ticket_calibrator,
        device,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["locked_holdout"], indent=2))


if __name__ == "__main__":
    main()
