"""Test spacing ecology system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.features.spacing_features import PlayerSpacingProfile, compute_player_spacing
from analytics.team_building.spacing_ecology import evaluate_spacing_ecology


def _make_spacer(gravity: float = 75, rim_freq: float = 0.2, name: str = "Spacer") -> PlayerSpacingProfile:
    return compute_player_spacing(
        three_pct=0.38, three_pa_rate=0.4, three_pa_per_game=6.0,
        ft_pct=0.85, assisted_rate=0.6, rim_frequency=rim_freq,
        usage_rate=0.18, games_played=60, player_name=name,
    )

def _make_non_shooter(rim_freq: float = 0.5, name: str = "Big") -> PlayerSpacingProfile:
    return compute_player_spacing(
        three_pct=0.25, three_pa_rate=0.1, three_pa_per_game=1.0,
        ft_pct=0.60, assisted_rate=0.8, rim_frequency=rim_freq,
        usage_rate=0.15, games_played=60, player_name=name,
    )


def test_one_shooter_four_non_shooters_low_spacing():
    """1 high-gravity shooter + 4 non-shooters must NOT receive high spacing score."""
    lineup = [_make_spacer()] + [_make_non_shooter(name=f"Big{i}") for i in range(4)]
    eco = evaluate_spacing_ecology(lineup)
    assert eco.spacing_density < 50, f"Expected low density, got {eco.spacing_density}"


def test_three_spacers_increase_density():
    """3 reliable spacers increase spacing_density."""
    lineup = [_make_spacer(name=f"S{i}") for i in range(3)] + [_make_non_shooter(name=f"B{i}") for i in range(2)]
    eco = evaluate_spacing_ecology(lineup)
    assert eco.spacing_density > 50
    assert eco.count_above_60 >= 3


def test_two_non_shooting_bigs_trigger_warning():
    """2 non-shooting bigs trigger paint_congestion_warning."""
    lineup = [_make_spacer(name=f"S{i}") for i in range(3)] + [_make_non_shooter(rim_freq=0.5, name=f"B{i}") for i in range(2)]
    eco = evaluate_spacing_ecology(lineup)
    assert eco.paint_congestion_warning is True


def test_creator_with_spacers_high_lane_context():
    """Creator + spacers increases lane_creation_context."""
    creator = compute_player_spacing(
        three_pct=0.35, three_pa_rate=0.3, three_pa_per_game=5.0,
        ft_pct=0.82, assisted_rate=0.3, rim_frequency=0.3,
        usage_rate=0.30, games_played=60, player_name="Creator",
    )
    spacers = [_make_spacer(name=f"S{i}") for i in range(4)]
    lineup = [creator] + spacers
    eco = evaluate_spacing_ecology(lineup, creator_indices=[0])
    assert eco.lane_creation_context > 50


def test_spacing_fragility_single_shooter():
    """Spacing fragility increases when spacing depends on one player."""
    top_spacer = compute_player_spacing(
        three_pct=0.42, three_pa_rate=0.5, three_pa_per_game=9.0,
        ft_pct=0.90, assisted_rate=0.5, rim_frequency=0.15,
        usage_rate=0.22, games_played=70, player_name="Star Shooter",
    )
    non_shooters = [_make_non_shooter(name=f"NS{i}") for i in range(4)]
    lineup = [top_spacer] + non_shooters
    eco = evaluate_spacing_ecology(lineup)
    assert eco.spacing_fragility > 20
