from __future__ import annotations

import pandas as pd

from sports.mlb.advanced.data_layer import build_batter_profile, build_direct_matchup, build_pitcher_profile


def statcast_fixture() -> pd.DataFrame:
    rows = []
    events = [
        ("single", 101.0, 12.0, 0.82, 0.74, 1.25, 6, "FF", "hit_into_play", 2),
        ("field_out", 98.0, 18.0, 0.64, 0.58, 0.91, 5, "FF", "hit_into_play", 2),
        ("strikeout", None, None, None, None, None, None, "SL", "swinging_strike", 2),
        ("walk", None, None, None, None, None, None, "CH", "ball", 2),
        ("home_run", 105.0, 27.0, 0.96, 0.91, 3.45, 6, "FF", "hit_into_play", 2),
        ("double", 99.0, 16.0, 0.76, 0.69, 1.72, 5, "SL", "hit_into_play", 3),
        ("field_out", 73.0, -9.0, 0.14, 0.11, 0.14, 1, "CH", "hit_into_play", 3),
        ("strikeout", None, None, None, None, None, None, "SL", "called_strike", 3),
    ]
    for idx, (event, ev, la, xba, xwoba, xslg, lsa, pitch, desc, pitcher) in enumerate(events, start=1):
        rows.append({
            "game_date": "2026-08-01",
            "at_bat_number": idx,
            "pitch_number": 1,
            "events": event,
            "launch_speed": ev,
            "launch_angle": la,
            "estimated_ba_using_speedangle": xba,
            "estimated_woba_using_speedangle": xwoba,
            "estimated_slg_using_speedangle": xslg,
            "launch_speed_angle": lsa,
            "bb_type": "line_drive" if la is not None and 10 <= la <= 25 else "ground_ball",
            "pitch_type": pitch,
            "release_speed": 95.0 if pitch == "FF" else 86.0,
            "pfx_x": 0.4,
            "pfx_z": 1.1,
            "description": desc,
            "zone": 11 if idx % 3 == 0 else 5,
            "woba_value": 0.9 if event == "home_run" else 0.6 if event in {"single", "double"} else 0.0,
            "woba_denom": 1,
            "stand": "R",
            "p_throws": "R",
            "pitcher": pitcher,
        })
    return pd.DataFrame(rows)


def test_batter_profile_contains_expected_contact_and_process_metrics():
    profile = build_batter_profile(statcast_fixture(), player_id=1, player_name="Batter", as_of_date="2026-09-05")
    assert profile.sample_pa == 8
    assert profile.sample_bbe == 5
    assert profile.xba is not None
    assert profile.xslg is not None
    assert profile.xwoba is not None
    assert profile.hard_hit_rate is not None
    assert profile.barrel_rate is not None
    assert profile.k_rate == 0.25
    assert profile.bb_rate == 0.125
    assert profile.hr_rate == 0.125
    assert profile.pitch_type_xwoba


def test_pitcher_profile_contains_arsenal_and_expected_contact_allowed():
    profile = build_pitcher_profile(statcast_fixture(), player_id=2, player_name="Pitcher", as_of_date="2026-09-05")
    assert profile.sample_pa == 8
    assert profile.xba_allowed is not None
    assert profile.xslg_allowed is not None
    assert profile.xwoba_allowed is not None
    assert "FF" in profile.arsenal
    assert profile.k_minus_bb_rate == profile.k_rate - profile.bb_rate


def test_direct_bvp_uses_process_events_and_shrinkage():
    direct = build_direct_matchup(statcast_fixture(), batter_id=1, pitcher_id=2)
    assert direct is not None
    assert direct.pa == 5
    assert direct.strikeouts == 1
    assert direct.walks == 1
    assert direct.home_runs == 1
    assert 0.0 < direct.shrinkage_weight < 0.5
    assert direct.avg_ev is not None


def test_direct_bvp_missing_pitcher_returns_none():
    assert build_direct_matchup(statcast_fixture(), batter_id=1, pitcher_id=999) is None
