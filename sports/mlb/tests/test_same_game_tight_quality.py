from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "parlay_v2"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

from same_game_quality_selector import quality_safe_candidates  # noqa: E402


def _combo(*, joint: float, edge: float, ev: float):
    return SimpleNamespace(
        real_joint_model_probability=joint,
        probability_edge=edge,
        expected_value_per_unit=ev,
    )


def test_same_game_requires_material_edge() -> None:
    thin_edge = _combo(joint=0.28, edge=0.01, ev=0.20)
    assert quality_safe_candidates([thin_edge]) == []


def test_same_game_requires_material_synthetic_ev() -> None:
    thin_ev = _combo(joint=0.28, edge=0.06, ev=0.03)
    assert quality_safe_candidates([thin_ev]) == []


def test_same_game_accepts_a_low_joint_candidate_once_edge_and_ev_clear() -> None:
    """Real evidence removed the joint-probability gate 2026-08-29 (see
    same_game_quality_selector.py's module docstring): a real same-game
    combo has never once reached even 36% joint probability, so a low
    joint probability must no longer block headline eligibility -- edge
    and EV are the real bar."""
    low_joint_but_safe = _combo(joint=0.28, edge=0.05, ev=0.12)
    assert quality_safe_candidates([low_joint_but_safe]) == [low_joint_but_safe]


def test_same_game_accepts_only_candidates_that_clear_both_gates() -> None:
    safe = _combo(joint=0.28, edge=0.05, ev=0.12)
    low_edge = _combo(joint=0.30, edge=0.02, ev=0.30)
    low_ev = _combo(joint=0.30, edge=0.08, ev=0.04)
    assert quality_safe_candidates([low_edge, low_ev, safe]) == [safe]
