from dataclasses import replace

from sports.mlb.advanced.game_conditioned_moe import build_expert_state
from sports.mlb.advanced.schema import (
    AdvancedCandidateContext,
    BatterProcessProfile,
    DirectMatchupProcess,
    PitcherProcessProfile,
)
from sports.mlb.advanced.sequential_pa_model import simulate_hitter_market


def _batter(**overrides):
    base = BatterProcessProfile(
        player_id=101,
        player_name="Test Batter",
        as_of_date="2026-09-05",
        sample_pa=420,
        sample_bbe=275,
        handedness="L",
        k_rate=0.20,
        bb_rate=0.09,
        hr_rate=0.04,
        contact_rate=0.80,
        whiff_rate=0.20,
        chase_rate=0.28,
        xwoba=0.350,
        xba=0.275,
        xslg=0.490,
        avg_ev=89.5,
        ev90=105.0,
        hard_hit_rate=0.43,
        barrel_rate=0.10,
        sweet_spot_rate=0.34,
        support=0.95,
    )
    return replace(base, **overrides)


def _pitcher(**overrides):
    base = PitcherProcessProfile(
        player_id=202,
        player_name="Test Pitcher",
        as_of_date="2026-09-05",
        sample_pa=520,
        sample_bbe=330,
        handedness="R",
        k_rate=0.25,
        bb_rate=0.08,
        hr_rate=0.03,
        k_minus_bb_rate=0.17,
        whiff_rate=0.25,
        xwoba_allowed=0.320,
        xba_allowed=0.250,
        xslg_allowed=0.420,
        avg_ev_allowed=88.5,
        hard_hit_rate_allowed=0.38,
        barrel_rate_allowed=0.075,
        sweet_spot_rate_allowed=0.33,
        gb_rate=0.43,
        xfip=3.80,
        siera=3.75,
        projected_ip=5.7,
        support=0.95,
    )
    return replace(base, **overrides)


def _context(*, batter=None, pitcher=None, direct=None):
    return AdvancedCandidateContext(
        game_id="test-game",
        run_date="2026-09-05",
        batter=batter or _batter(),
        pitcher=pitcher or _pitcher(),
        direct_matchup=direct,
        batting_order=2,
        is_home=False,
        team_expected_runs=4.8,
        park_factor=1.0,
        defense_residual=0.0,
        defense_status="AVERAGE_CONTEXT_RESIDUAL_ONLY",
        data_freshness_status="FRESH",
        missing_components=(),
        temperature_f=75.0,
    )


def _sequential(context):
    return simulate_hitter_market(context, target="H", market_line=0.5, trials=3000)


def _direct(*, strong: bool):
    if strong:
        return DirectMatchupProcess(
            batter_id=101,
            pitcher_id=202,
            pa=24,
            strikeouts=2,
            walks=3,
            hbp=0,
            home_runs=3,
            non_hr_contacts=16,
            hard_contacts=10,
            weak_contacts=2,
            xwoba_contact=0.430,
            xba_contact=0.335,
            xslg_contact=0.650,
            avg_ev=93.5,
            barrel_rate=0.16,
            whiff_rate=0.12,
            shrinkage_weight=0.50,
        )
    return DirectMatchupProcess(
        batter_id=101,
        pitcher_id=202,
        pa=24,
        strikeouts=9,
        walks=1,
        hbp=0,
        home_runs=0,
        non_hr_contacts=10,
        hard_contacts=2,
        weak_contacts=7,
        xwoba_contact=0.230,
        xba_contact=0.185,
        xslg_contact=0.285,
        avg_ev=84.0,
        barrel_rate=0.02,
        whiff_rate=0.36,
        shrinkage_weight=0.50,
    )


def test_lower_chase_rate_strengthens_contact_survival_state():
    base_context = _context()
    sequential = _sequential(base_context)
    disciplined = build_expert_state(
        _context(batter=_batter(chase_rate=0.18)),
        sequential,
        target="H",
        pitch_compatibility_score=0.0,
    )
    chase_heavy = build_expert_state(
        _context(batter=_batter(chase_rate=0.40)),
        sequential,
        target="H",
        pitch_compatibility_score=0.0,
    )

    assert disciplined.signals["strikeout_contact"] > chase_heavy.signals["strikeout_contact"]
    assert disciplined.diagnostics["batter_chase_rate"] == 0.18


def test_ev_shape_changes_contact_quality_and_power_state():
    base_context = _context()
    sequential = _sequential(base_context)
    loud = build_expert_state(
        _context(batter=_batter(avg_ev=94.0, ev90=112.0, sweet_spot_rate=0.45)),
        sequential,
        target="TB",
        pitch_compatibility_score=0.0,
    )
    weak = build_expert_state(
        _context(batter=_batter(avg_ev=84.0, ev90=96.0, sweet_spot_rate=0.22)),
        sequential,
        target="TB",
        pitch_compatibility_score=0.0,
    )

    assert loud.signals["contact_quality"] > weak.signals["contact_quality"]
    assert loud.signals["power_tb"] > weak.signals["power_tb"]
    assert loud.diagnostics["batter_ev90"] == 112.0


def test_direct_matchup_is_directional_but_heavily_shrunk():
    neutral_context = _context()
    sequential = _sequential(neutral_context)
    strong = build_expert_state(
        _context(direct=_direct(strong=True)),
        sequential,
        target="TB",
        pitch_compatibility_score=0.0,
    )
    weak = build_expert_state(
        _context(direct=_direct(strong=False)),
        sequential,
        target="TB",
        pitch_compatibility_score=0.0,
    )

    assert 0.0 < strong.diagnostics["direct_matchup_weight"] <= 0.30
    assert strong.diagnostics["direct_matchup_weight"] == weak.diagnostics["direct_matchup_weight"]
    assert strong.signals["strikeout_contact"] > weak.signals["strikeout_contact"]
    assert strong.signals["contact_quality"] > weak.signals["contact_quality"]
    assert strong.signals["power_tb"] > weak.signals["power_tb"]


def test_handedness_is_preserved_as_state_without_fabricated_split_direction():
    base_context = _context()
    sequential = _sequential(base_context)
    left_vs_right = build_expert_state(
        _context(batter=_batter(handedness="L"), pitcher=_pitcher(handedness="R")),
        sequential,
        target="H",
        pitch_compatibility_score=0.0,
    )
    right_vs_right = build_expert_state(
        _context(batter=_batter(handedness="R"), pitcher=_pitcher(handedness="R")),
        sequential,
        target="H",
        pitch_compatibility_score=0.0,
    )

    assert left_vs_right.diagnostics["handedness_matchup"] == "L_VS_R"
    assert right_vs_right.diagnostics["handedness_matchup"] == "R_VS_R"
    assert left_vs_right.diagnostics["handedness_context_available"] is True
    # Until handedness-specific split evidence is preserved, identity is state
    # metadata only; no arbitrary fixed platoon coefficient is fabricated.
    assert left_vs_right.signals == right_vs_right.signals
