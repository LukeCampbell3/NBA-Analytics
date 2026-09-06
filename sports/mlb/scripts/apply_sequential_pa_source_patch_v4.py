#!/usr/bin/env python3
"""Upgrade sequential PA v1 production wiring with arsenal compatibility,
team opportunity context, per-side game means, and same-day advanced-profile
reuse.  All patches are idempotent and assertion-backed.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def replace_once(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched {path.relative_to(REPO)}: {label}")
        return False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor for {label}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path.relative_to(REPO)}: {label}")
    return True


def patch_sequential_model() -> None:
    path = REPO / "sports/mlb/advanced/sequential_pa_model.py"
    replace_once(
        path,
        '''from .schema import (
    AdvancedCandidateContext,
''',
        '''from .compatibility import build_pitch_compatibility_signal
from .schema import (
    AdvancedCandidateContext,
''',
        "compatibility import",
    )
    replace_once(
        path,
        '''    k = matchup_rate(batter.k_rate, pitcher.k_rate, LEAGUE_K_RATE, shrink=0.45 + 0.55 * support)
    bb = matchup_rate(batter.bb_rate, pitcher.bb_rate, LEAGUE_BB_RATE, shrink=0.45 + 0.55 * support)
    hbp = matchup_rate(batter.hbp_rate, pitcher.hbp_rate, LEAGUE_HBP_RATE, shrink=0.35 + 0.45 * support)
    hr = matchup_rate(batter.hr_rate, pitcher.hr_rate, LEAGUE_HR_RATE, shrink=0.40 + 0.55 * support)

    # Later trips through the order generally shift a small amount of mass away
''',
        '''    k = matchup_rate(batter.k_rate, pitcher.k_rate, LEAGUE_K_RATE, shrink=0.45 + 0.55 * support)
    bb = matchup_rate(batter.bb_rate, pitcher.bb_rate, LEAGUE_BB_RATE, shrink=0.45 + 0.55 * support)
    hbp = matchup_rate(batter.hbp_rate, pitcher.hbp_rate, LEAGUE_HBP_RATE, shrink=0.35 + 0.45 * support)
    hr = matchup_rate(batter.hr_rate, pitcher.hr_rate, LEAGUE_HR_RATE, shrink=0.40 + 0.55 * support)

    compatibility = build_pitch_compatibility_signal(batter, pitcher)
    k += compatibility.k_probability_delta

    # Later trips through the order generally shift a small amount of mass away
''',
        "arsenal K compatibility",
    )
    replace_once(
        path,
        '''    p_hit = _weighted_optional(bxba, pxba, 0.58, 0.42, LEAGUE_CONTACT_XBA)

    if direct_matchup is not None and direct_matchup.xba_contact is not None:
''',
        '''    p_hit = _weighted_optional(bxba, pxba, 0.58, 0.42, LEAGUE_CONTACT_XBA)
    compatibility = build_pitch_compatibility_signal(batter, pitcher)
    p_hit += compatibility.contact_hit_probability_delta

    if direct_matchup is not None and direct_matchup.xba_contact is not None:
''',
        "arsenal contact compatibility",
    )
    replace_once(
        path,
        '''    xslg = _weighted_optional(bxslg, pxslg, 0.58, 0.42, LEAGUE_CONTACT_XSLG)
    power_tilt = _clip((xslg - LEAGUE_CONTACT_XSLG) * 0.35, -0.08, 0.12)
''',
        '''    xslg = _weighted_optional(bxslg, pxslg, 0.58, 0.42, LEAGUE_CONTACT_XSLG)
    xslg += compatibility.xslg_delta
    power_tilt = _clip((xslg - LEAGUE_CONTACT_XSLG) * 0.35, -0.08, 0.12)
''',
        "arsenal power compatibility",
    )
    replace_once(
        path,
        '''    pa_uncertainty = 0.15 if context.batting_order is not None else 0.55
    mc = _clip(mc_standard_error / 0.01, 0.0, 1.0)
    return {
''',
        '''    pa_uncertainty = 0.15 if context.batting_order is not None else 0.55
    pitch_compatibility = build_pitch_compatibility_signal(context.batter, context.pitcher)
    pitch_mix = 1.0 - _clip(pitch_compatibility.support, 0.0, 1.0)
    mc = _clip(mc_standard_error / 0.01, 0.0, 1.0)
    return {
''',
        "pitch mix uncertainty",
    )
    replace_once(
        path,
        '''        "expected_pa": pa_uncertainty,
        "monte_carlo": mc,
    }
''',
        '''        "expected_pa": pa_uncertainty,
        "pitch_mix_support": pitch_mix,
        "monte_carlo": mc,
    }
''',
        "pitch mix uncertainty output",
    )
    replace_once(
        path,
        '''        "expected_pa": 0.10,
        "monte_carlo": 0.06,
    }
''',
        '''        "expected_pa": 0.09,
        "pitch_mix_support": 0.06,
        "monte_carlo": 0.05,
    }
''',
        "pitch mix uncertainty weight",
    )
    replace_once(
        path,
        '''    contact_probs = contact_outcome_probabilities(
        context.batter,
        context.pitcher,
''',
        '''    compatibility = build_pitch_compatibility_signal(context.batter, context.pitcher)
    contact_probs = contact_outcome_probabilities(
        context.batter,
        context.pitcher,
''',
        "simulation compatibility diagnostics setup",
    )
    replace_once(
        path,
        '''            "contact_probabilities": contact_probs,
            "walks_per_game": float(walks.mean()),
''',
        '''            "contact_probabilities": contact_probs,
            "pitch_compatibility": {
                "support": compatibility.support,
                "matched_usage": compatibility.matched_usage,
                "k_probability_delta": compatibility.k_probability_delta,
                "contact_hit_probability_delta": compatibility.contact_hit_probability_delta,
                "xslg_delta": compatibility.xslg_delta,
                "expected_xwoba_contact": compatibility.expected_xwoba_contact,
            },
            "walks_per_game": float(walks.mean()),
''',
        "simulation compatibility diagnostics",
    )


def patch_same_game_expected_runs() -> None:
    path = REPO / "sports/mlb/scripts/run_mlb_same_game_daily.py"
    replace_once(
        path,
        '''        home_expected, away_expected = sides

        result = sim.simulate_game_outcomes(
''',
        '''        home_expected, away_expected = sides
        # Shared team run means are also consumed by the hitter PA-opportunity
        # model. Persisting them here keeps player and game simulations on the
        # same pitching-enriched state instead of independently re-deriving it.
        entry["home_expected_runs"] = float(home_expected)
        entry["away_expected_runs"] = float(away_expected)

        result = sim.simulate_game_outcomes(
''',
        "persist per-side expected runs",
    )


def patch_integration_team_context() -> None:
    path = REPO / "sports/mlb/advanced/integration.py"
    replace_once(
        path,
        '''from .sequential_pa_model import MODEL_VERSION, simulate_hitter_market

MLB_LIVE_FEED_ROOT = "https://statsapi.mlb.com/api/v1.1/game"
''',
        '''from .sequential_pa_model import MODEL_VERSION, simulate_hitter_market
from .team_context import build_team_run_environment

MLB_LIVE_FEED_ROOT = "https://statsapi.mlb.com/api/v1.1/game"
''',
        "team context import",
    )
    replace_once(
        path,
        '''    output = hydrated.copy()
    columns = {
''',
        '''    output = hydrated.copy()
    team_run_context = build_team_run_environment(
        hydrated,
        same_game_json=Path(__file__).resolve().parents[1] / "web" / "data" / "same_game_predictions.json",
    )
    columns = {
''',
        "team context build",
    )
    replace_once(
        path,
        '''        context = AdvancedCandidateContext(
            game_id=str(row.get("Game_ID") or ""), run_date=run_date, batter=batter, pitcher=pitcher,
            direct_matchup=direct, batting_order=batting_order, is_home=is_home,
            team_expected_runs=_team_expected_runs(row), park_factor=float(park_factor),
''',
        '''        team_key = (str(row.get("Game_ID") or ""), str(row.get("Team") or "").upper())
        shared_team_context = team_run_context.get(team_key) or {}
        expected_team_runs = _float(shared_team_context.get("expected_runs"), _team_expected_runs(row))
        context = AdvancedCandidateContext(
            game_id=str(row.get("Game_ID") or ""), run_date=run_date, batter=batter, pitcher=pitcher,
            direct_matchup=direct, batting_order=batting_order, is_home=is_home,
            team_expected_runs=expected_team_runs, park_factor=float(park_factor),
''',
        "team context consume",
    )
    replace_once(
        path,
        '''            "expected_h": result.expected_hits, "expected_tb": result.expected_tb, "support": result.support,
            "uncertainty": result.uncertainty,
        })
''',
        '''            "expected_h": result.expected_hits, "expected_tb": result.expected_tb, "support": result.support,
            "uncertainty": result.uncertainty,
            "team_expected_runs": expected_team_runs,
            "team_run_context_source": shared_team_context.get("source", "NEUTRAL_FALLBACK"),
        })
''',
        "team context report",
    )


def patch_data_cache_reuse() -> None:
    path = REPO / "sports/mlb/advanced/data_layer.py"
    replace_once(
        path,
        '''    identities = read_pool_candidate_identities(pool_csv)[: max(1, int(max_candidates))]
    tools = _load_pybaseball()
    fg_pitchers = _fangraphs_pitching_map(tools["pitching_stats"], run_day.year)

    batters: dict[str, dict[str, Any]] = {}
    pitchers: dict[str, dict[str, Any]] = {}
    matchups: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    batter_frames: dict[int, pd.DataFrame] = {}
''',
        '''    identities = read_pool_candidate_identities(pool_csv)[: max(1, int(max_candidates))]
    existing_batter_payload, existing_pitcher_payload, existing_matchup_payload, existing_manifest = load_profile_partition(
        advanced_root, run_date
    )
    batters: dict[str, dict[str, Any]] = dict(existing_batter_payload.get("profiles") or {})
    pitchers: dict[str, dict[str, Any]] = dict(existing_pitcher_payload.get("profiles") or {})
    matchups: dict[str, dict[str, Any]] = dict(existing_matchup_payload.get("matchups") or {})
    reused_batters = len(batters)
    reused_pitchers = len(pitchers)
    reused_matchups = len(matchups)

    needs_remote = any(
        (int(identity.get("batter_id") or 0) > 0 and str(int(identity.get("batter_id") or 0)) not in batters)
        or (int(identity.get("pitcher_id") or 0) > 0 and str(int(identity.get("pitcher_id") or 0)) not in pitchers)
        or (
            int(identity.get("batter_id") or 0) > 0
            and int(identity.get("pitcher_id") or 0) > 0
            and f"{int(identity.get('batter_id') or 0)}:{int(identity.get('pitcher_id') or 0)}" not in matchups
        )
        for identity in identities
    )
    tools = _load_pybaseball() if needs_remote else {}
    missing_pitcher_profile = any(
        int(identity.get("pitcher_id") or 0) > 0 and str(int(identity.get("pitcher_id") or 0)) not in pitchers
        for identity in identities
    )
    fg_pitchers = _fangraphs_pitching_map(tools["pitching_stats"], run_day.year) if missing_pitcher_profile else {}

    failures: list[dict[str, Any]] = []
    batter_frames: dict[int, pd.DataFrame] = {}
''',
        "reuse same-day profile partitions",
    )
    replace_once(
        path,
        '''        if batter_id > 0 and batter_id not in batter_frames:
            try:
                batter_frames[batter_id] = _safe_statcast_fetch(
                    tools["statcast_batter"], start_day.isoformat(), (run_day - timedelta(days=1)).isoformat(), batter_id
                )
                profile = build_batter_profile(
                    batter_frames[batter_id], player_id=batter_id, player_name=identity["batter_name"], as_of_date=run_date
                )
                batters[str(batter_id)] = profile.to_dict()
            except Exception as exc:
                failures.append({"entity": "batter", "player_id": batter_id, "error": str(exc)})
''',
        '''        matchup_key = f"{batter_id}:{pitcher_id}" if batter_id > 0 and pitcher_id > 0 else ""
        needs_batter_frame = str(batter_id) not in batters or (matchup_key and matchup_key not in matchups)
        if batter_id > 0 and needs_batter_frame and batter_id not in batter_frames:
            try:
                batter_frames[batter_id] = _safe_statcast_fetch(
                    tools["statcast_batter"], start_day.isoformat(), (run_day - timedelta(days=1)).isoformat(), batter_id
                )
                if str(batter_id) not in batters:
                    profile = build_batter_profile(
                        batter_frames[batter_id], player_id=batter_id, player_name=identity["batter_name"], as_of_date=run_date
                    )
                    batters[str(batter_id)] = profile.to_dict()
            except Exception as exc:
                failures.append({"entity": "batter", "player_id": batter_id, "error": str(exc)})
''',
        "reuse batter profiles unless new BvP needed",
    )
    replace_once(
        path,
        '''        if batter_id > 0 and pitcher_id > 0 and batter_id in batter_frames:
            direct = build_direct_matchup(batter_frames[batter_id], batter_id=batter_id, pitcher_id=pitcher_id)
            if direct is not None:
                matchups[f"{batter_id}:{pitcher_id}"] = direct.to_dict()
''',
        '''        if batter_id > 0 and pitcher_id > 0 and matchup_key not in matchups and batter_id in batter_frames:
            direct = build_direct_matchup(batter_frames[batter_id], batter_id=batter_id, pitcher_id=pitcher_id)
            if direct is not None:
                matchups[matchup_key] = direct.to_dict()
''',
        "reuse direct matchups",
    )
    replace_once(
        path,
        '''        "direct_matchups": len(matchups),
        "failures": failures,
''',
        '''        "direct_matchups": len(matchups),
        "reused_batter_profiles": reused_batters,
        "reused_pitcher_profiles": reused_pitchers,
        "reused_direct_matchups": reused_matchups,
        "remote_refresh_required": needs_remote,
        "failures": failures,
''',
        "cache reuse manifest",
    )


def main() -> int:
    patch_sequential_model()
    patch_same_game_expected_runs()
    patch_integration_team_context()
    patch_data_cache_reuse()
    print("sequential PA v4 source upgrades complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
