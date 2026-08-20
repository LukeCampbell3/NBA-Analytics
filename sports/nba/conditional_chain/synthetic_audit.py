from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .confirmation import chronological_confirmation
from .protocol import ALLOCATION_PATH_PROTOCOL, AllocationPathProtocol


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-5, None)
    return values / values.sum()


def generate_synthetic_settled_paths(
    *,
    events: int,
    players: int = 6,
    path_effect: float = 0.0,
    seed: int,
) -> pd.DataFrame:
    """Generate event-clustered simplex data for null/power software audits."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    base_date = pd.Timestamp("2024-01-01")
    for event_index in range(events):
        open_share = rng.dirichlet(np.full(players, 4.0))
        movement = rng.normal(0.0, 0.055, size=players)
        movement -= movement.mean()
        close_share = _normalize(open_share + movement)
        delta = close_share - open_share
        realized = _normalize(
            close_share + path_effect * delta + rng.normal(0.0, 0.006, size=players)
        )
        middle = np.vstack(
            [
                open_share,
                _normalize(open_share + 0.30 * delta + rng.normal(0.0, 0.004, players)),
                _normalize(open_share + 0.60 * delta + rng.normal(0.0, 0.004, players)),
                _normalize(open_share + 0.82 * delta + rng.normal(0.0, 0.003, players)),
                close_share,
            ]
        )
        team_hhi = np.square(middle).sum(axis=1)
        team_entropy = -(middle * np.log(middle)).sum(axis=1)
        step_l1 = np.abs(np.diff(middle, axis=0)).sum(axis=1)
        displacement = float(np.abs(middle[-1] - middle[0]).sum())
        path_length = float(step_l1.sum())
        team_efficiency = displacement / path_length if path_length > 0 else 0.0
        event_date = base_date + pd.Timedelta(days=event_index // 4)
        event_id = f"synthetic_{event_index:04d}"
        for player_index in range(players):
            trajectory = middle[:, player_index]
            variation = float(np.abs(np.diff(trajectory)).sum())
            player_displacement = float(abs(trajectory[-1] - trajectory[0]))
            signs = np.sign(np.diff(trajectory))
            signs = signs[signs != 0]
            reversals = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
            rows.append(
                {
                    "unit_id": f"{event_id}::SYN::player_points",
                    "event_id": event_id,
                    "event_date": event_date,
                    "team": "SYN",
                    "player": f"player_{player_index}",
                    "realized_share": float(realized[player_index]),
                    "open_share": float(open_share[player_index]),
                    "close_share": float(close_share[player_index]),
                    "close_team_total": float(rng.normal(112.0, 3.0)),
                    "close_hhi": float(team_hhi[-1]),
                    "close_entropy": float(team_entropy[-1]),
                    "delta_share": float(delta[player_index]),
                    "player_total_variation": variation,
                    "player_path_efficiency": (
                        player_displacement / variation if variation > 0 else 0.0
                    ),
                    "direction_reversals": reversals,
                    "allocation_displacement_l1": displacement,
                    "allocation_path_length_l1": path_length,
                    "allocation_path_efficiency": team_efficiency,
                    "delta_hhi": float(team_hhi[-1] - team_hhi[0]),
                    "delta_entropy": float(team_entropy[-1] - team_entropy[0]),
                }
            )
    return pd.DataFrame(rows)


def run_null_power_audit(
    *,
    simulations: int = 20,
    events: int = 80,
    injected_path_effect: float = 0.40,
    protocol: AllocationPathProtocol = ALLOCATION_PATH_PROTOCOL,
) -> dict[str, Any]:
    audit_protocol = replace(
        protocol,
        bootstrap_samples=min(protocol.bootstrap_samples, 3_000),
        sign_flip_samples=min(protocol.sign_flip_samples, 10_000),
    )
    null_passes = 0
    effect_passes = 0
    null_improvements: list[float] = []
    effect_improvements: list[float] = []
    for simulation in range(simulations):
        seed = protocol.random_seed + simulation * 13
        null_result = chronological_confirmation(
            generate_synthetic_settled_paths(
                events=events, path_effect=0.0, seed=seed
            ),
            protocol=audit_protocol,
        )
        effect_result = chronological_confirmation(
            generate_synthetic_settled_paths(
                events=events, path_effect=injected_path_effect, seed=seed
            ),
            protocol=audit_protocol,
        )
        null_passes += int(bool(null_result.report["path_authorized"]))
        effect_passes += int(bool(effect_result.report["path_authorized"]))
        if not null_result.event_evaluations.empty:
            null_improvements.append(float(null_result.event_evaluations["mae_improvement"].mean()))
        if not effect_result.event_evaluations.empty:
            effect_improvements.append(float(effect_result.event_evaluations["mae_improvement"].mean()))
    return {
        "audit": "conditional_null_and_injected_path_effect",
        "simulations": simulations,
        "events_per_simulation": events,
        "practical_mae_improvement": protocol.practical_mae_improvement,
        "injected_path_effect": injected_path_effect,
        "null_passes": null_passes,
        "null_false_pass_rate": null_passes / simulations,
        "effect_passes": effect_passes,
        "effect_detection_rate": effect_passes / simulations,
        "mean_null_mae_improvement": float(np.mean(null_improvements)),
        "mean_effect_mae_improvement": float(np.mean(effect_improvements)),
        "note": "Monte Carlo sample counts are reduced for runtime; the frozen real-data gate uses full counts.",
    }
