from __future__ import annotations

from sports.nba.analytics.par.config import MODEL_CONFIG
from sports.nba.analytics.par.engine import (
    aggregate_player_components,
    build_forecasts,
    infer_role,
    validate_atom_rules,
    validate_overlap,
)
from sports.nba.analytics.par.models import PlayerMeta, ReplacementBaseline, ValueAtom


def atom(**overrides) -> ValueAtom:
    values = {
        "atom_id": "a1",
        "possession_id": "p1",
        "game_id": "g1",
        "event_time": "2025-10-01",
        "season": "2025-26",
        "player_id": "p",
        "team_id": "t",
        "opponent_id": "o",
        "primary_value_label": "scoring_volume_above_replacement",
        "category": "SCORING",
        "overlap_group_id": "og1",
        "context_labels": ["box_visible"],
        "source_event_ids": ["e1"],
        "source_type": "box_score_direct_bootstrap",
        "source_tier": "TIER_A_DIRECT",
        "raw_value": 12.0,
        "replacement_baseline": 4.0,
        "value_above_replacement": 8.0,
        "reliability_weight": 1.0,
        "shrinkage_factor": 1.0,
        "overlap_adjustment": 0.0,
        "par_value": 8.0,
        "label_entropy": 0.0,
        "confidence_tier": "high",
        "player_credit_json": {"player_id": "p", "credit": 1.0},
        "category_rollup_json": {"category": "SCORING"},
        "residual_value": 0.0,
        "par_model_version": MODEL_CONFIG.par_model_version,
    }
    values.update(overrides)
    return ValueAtom(**values)


def baseline() -> ReplacementBaseline:
    return ReplacementBaseline(
        season="2025-26",
        role="primary_creator",
        atom_type="scoring_volume_above_replacement",
        sample_size=100,
        replacement_value=4.0,
        uncertainty=0.5,
        baseline_version="test",
    )


def test_player_par_accounting_identity_and_presentation_formulas() -> None:
    meta = PlayerMeta("p", "Player", "t", "TST", "2025-26", "primary_creator", 1000.0, 10)
    atoms = [
        atom(atom_id="score", par_value=80.0, value_above_replacement=80.0),
        atom(
            atom_id="create",
            primary_value_label="passing_creation",
            category="CREATION",
            overlap_group_id="og2",
            source_event_ids=["e2"],
            par_value=20.0,
            value_above_replacement=20.0,
        ),
    ]
    rows, validation = aggregate_player_components([meta], atoms)
    row = rows[0]
    assert validation["status"] == "pass"
    assert row["total_par"] == row["scoring_par"] + row["creation_par"]
    assert row["par_1000"] == 100.0
    assert row["war_equivalent"] == round(100.0 / 30.4, 9)
    assert row["model_version"] == "par_pvg_v0_5"


def test_overlap_validator_blocks_duplicate_source_event_credit() -> None:
    duplicate = atom(atom_id="a2")
    report = validate_overlap([atom(), duplicate])
    assert report["status"] == "fail"
    assert report["overlap_groups_failed"] > 0


def test_atom_rule_validator_blocks_unsupported_value_and_unshrunk_proxy() -> None:
    unsupported = atom(source_tier="TIER_E_UNSUPPORTED", par_value=1.0)
    proxy = atom(atom_id="a2", source_tier="TIER_D_SHRUNK_PROXY", shrinkage_factor=1.0)
    cv = atom(atom_id="a3", source_type="cv_tracking", context_labels=[], par_value=1.0)
    report = validate_atom_rules([unsupported, proxy, cv], [baseline()])
    rules = {failure["rule"] for failure in report["failures"]}
    assert report["status"] == "fail"
    assert "unsupported_atoms_contribute_zero" in rules
    assert "proxy_atoms_require_shrinkage" in rules
    assert "cv_source_readiness" in rules


def test_parf_forecast_uses_atom_specific_persistence_and_bridge_reconciles() -> None:
    meta = PlayerMeta("p", "Player", "t", "TST", "2025-26", "primary_creator", 2000.0, 60)
    atoms = [
        atom(atom_id="score", par_value=100.0, value_above_replacement=100.0),
        atom(
            atom_id="steal",
            primary_value_label="steals",
            category="PERIMETER_DISRUPTION",
            overlap_group_id="og2",
            source_event_ids=["e2"],
            par_value=50.0,
            value_above_replacement=50.0,
        ),
    ]
    components, _ = aggregate_player_components([meta], atoms)
    forecasts, ledger, validation = build_forecasts(components, atoms, "2026-27")
    assert validation["status"] == "pass"
    assert {row["persistence"] for row in ledger} == {0.70, 0.60}
    forecast = forecasts[0]
    assert forecast["forecast_bridge"]["projected_par_f"] == forecast["projected_par"]
    assert forecast["parf_model_version"] == "parf_v0_6"


def test_role_inference_classifies_high_rebounding_scorer_as_big() -> None:
    role = infer_role(
        {
            "minutes": 2400.0,
            "games": 70.0,
            "pts": 1700.0,
            "ast": 180.0,
            "trb": 780.0,
            "stl": 80.0,
        }
    )
    assert role == "roll_big"
