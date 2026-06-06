"""Test vector schema integrity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.schema import (
    CAPABILITY_DIMENSIONS,
    ObservationStatus,
    PlayerCapabilityVector,
    VectorDimension,
)


def test_all_dimensions_present():
    """Capability vector has all 22 required dimensions."""
    v = PlayerCapabilityVector(player_name="Test Player")
    assert len(v.dimensions) == 22
    for dim in CAPABILITY_DIMENSIONS:
        assert dim in v.dimensions


def test_missing_dimensions_are_unavailable():
    """Missing dimensions are marked unavailable, not silently zeroed."""
    v = PlayerCapabilityVector(player_name="Test Player")
    for dim in v.dimensions.values():
        assert dim.observation_status == ObservationStatus.UNAVAILABLE
        assert dim.raw_value is None


def test_set_dimension():
    """Setting a dimension updates correctly."""
    v = PlayerCapabilityVector(player_name="Test")
    v.set_dimension("shooting_gravity", raw_value=85.0, confidence=0.9,
                    observation_status=ObservationStatus.OBSERVED, sample_size=60)
    d = v.get("shooting_gravity")
    assert d.raw_value == 85.0
    assert d.confidence == 0.9
    assert d.observation_status == ObservationStatus.OBSERVED
    assert d.sample_size == 60


def test_confidence_summary():
    """Confidence summary reports observed/inferred/unavailable correctly."""
    v = PlayerCapabilityVector(player_name="Test")
    v.set_dimension("shooting_gravity", observation_status=ObservationStatus.OBSERVED, confidence=0.9)
    v.set_dimension("rim_pressure", observation_status=ObservationStatus.INFERRED, confidence=0.5)
    summary = v.confidence_summary()
    assert summary["observed_dimensions"] == 1
    assert summary["inferred_dimensions"] == 1
    assert summary["unavailable_dimensions"] == 20


def test_to_dict_serializable():
    """Vector serializes to dict without errors."""
    v = PlayerCapabilityVector(player_name="Test", team="LAL", position="F")
    v.set_dimension("on_ball_creation", raw_value=42.0, confidence=0.8)
    d = v.to_dict()
    assert d["player_name"] == "Test"
    assert "on_ball_creation" in d["dimensions"]
    assert d["dimensions"]["on_ball_creation"]["raw_value"] == 42.0


def test_no_percentiles_outside_0_100():
    """No percentile value can exceed 0-100 range."""
    from analytics.features.percentiles import compute_percentile
    import numpy as np
    pop = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for val in [-100, 0, 5, 10, 1000]:
        pct = compute_percentile(val, pop)
        assert 0 <= pct <= 100
