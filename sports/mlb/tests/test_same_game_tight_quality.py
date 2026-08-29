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


def test_same_game_requires_material_edge_after_joint_gate() -> None:
    thin_edge = _combo(joint=0.58, edge=0.01, ev=0.20)
    assert quality_safe_candidates([thin_edge]) == []


def test_same_game_requires_material_synthetic_ev_after_joint_gate() -> None:
    thin_ev = _combo(joint=0.58, edge=0.06, ev=0.03)
    assert quality_safe_candidates([thin_ev]) == []


def test_same_game_accepts_only_candidate_that_clears_all_three_gates() -> None:
    safe = _combo(joint=0.56, edge=0.05, ev=0.12)
    low_joint = _combo(joint=0.49, edge=0.20, ev=0.50)
    low_edge = _combo(joint=0.60, edge=0.02, ev=0.30)
    low_ev = _combo(joint=0.60, edge=0.08, ev=0.04)
    assert quality_safe_candidates([low_joint, low_edge, low_ev, safe]) == [safe]
