"""Test roster scoring and cap legality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.schema import PlayerCapabilityVector, ObservationStatus
from analytics.team_building.roster_score import score_roster, RosterScoreResult
from analytics.cap.cap_rules import CapConstraints, PlayerSalary, validate_roster_legality


def _make_vector(name: str, **dims) -> PlayerCapabilityVector:
    v = PlayerCapabilityVector(player_id=name, player_name=name)
    for dim_name, val in dims.items():
        v.set_dimension(dim_name, raw_value=val, observation_status=ObservationStatus.OBSERVED, confidence=0.8)
    return v


def test_fatal_holes_penalized():
    """Roster with fatal holes (no creator, no rim protector) is penalized."""
    # All non-shooters with no creation or rim protection
    vectors = [_make_vector(f"P{i}", spacing_gravity=20, rim_protection=5, on_ball_creation=10) for i in range(5)]
    result = score_roster(vectors)
    assert result.fatal_hole_penalty > 0
    assert len(result.fatal_holes) > 0


def test_balanced_roster_scores_higher():
    """A balanced roster scores higher than a one-dimensional one."""
    # Balanced: creator + spacers + rim protector
    balanced = [
        _make_vector("Creator", on_ball_creation=70, passing_creation=60, spacing_gravity=40, decision_quality=60),
        _make_vector("Spacer1", spacing_gravity=75, shooting_gravity=70, off_ball_scalability=65),
        _make_vector("Spacer2", spacing_gravity=70, catch_and_shoot_gravity=65, off_ball_scalability=60),
        _make_vector("Rim", rim_protection=70, rebounding_value=65, physical_translation=60),
        _make_vector("Wing", defensive_disruption=55, spacing_gravity=50, transition_value=50),
    ]
    # One-dimensional: all shooters, no defense/creation
    one_dim = [_make_vector(f"Shooter{i}", spacing_gravity=80, shooting_gravity=75) for i in range(5)]

    balanced_score = score_roster(balanced)
    one_dim_score = score_roster(one_dim)

    assert balanced_score.roster_score > one_dim_score.roster_score


def test_non_shooter_conflict_penalty():
    """Too many non-shooters triggers conflict penalty."""
    vectors = [_make_vector(f"Big{i}", rim_pressure=60, spacing_gravity=15, rebounding_value=50) for i in range(5)]
    result = score_roster(vectors)
    assert result.non_shooter_conflict_penalty > 0
    assert any("non_shooter" in c for c in result.conflicts)


def test_cap_legality_rejects_over_cap():
    """Illegal rosters (over cap) are rejected."""
    salaries = [PlayerSalary(player_id=f"P{i}", salary=30_000_000) for i in range(5)]
    # 5 * 30M = 150M > 141M cap
    constraints = CapConstraints(cap_limit=141_000_000, allow_luxury_tax=False)
    result = validate_roster_legality(salaries, constraints)
    assert result["legal"] is False or result["over_cap"] > 0


def test_cap_legality_accepts_under_cap():
    """Legal roster under cap is accepted."""
    salaries = [PlayerSalary(player_id=f"P{i}", salary=8_000_000) for i in range(15)]
    # 15 * 8M = 120M < 141M
    result = validate_roster_legality(salaries)
    assert result["legal"] is True


def test_roster_size_constraints():
    """Roster size constraints are enforced."""
    too_small = [PlayerSalary(player_id=f"P{i}", salary=5_000_000) for i in range(10)]
    result = validate_roster_legality(too_small)
    assert not result["legal"]
    assert any("too_small" in v for v in result["violations"])


def test_no_duplicate_players():
    """Duplicate players are detected."""
    salaries = [PlayerSalary(player_id="same", salary=10_000_000) for _ in range(15)]
    result = validate_roster_legality(salaries)
    assert not result["legal"]
    assert any("duplicate" in v for v in result["violations"])
