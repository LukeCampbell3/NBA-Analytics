from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "sports" / "site" / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from validate_daily_publication import (
    MLB_MATCHUP_NETWORK_VERSION,
    MLB_POLICY_PROFILE,
    MLB_REQUIRED_TARGETS,
    as_float,
    validate_nba_payload,
    validate_mlb_payload,
    validate_publication,
)


def test_as_float_rejects_nonfinite_values() -> None:
    assert as_float(float("nan")) is None
    assert as_float(float("inf")) is None


def write_payload(path: Path, *, run_date: str, status: str = "ready", sport: str = "nba") -> None:
    payload = {
        "run_date": run_date,
        "publication_status": status,
        "plays": [],
    }
    if sport == "nba":
        payload["confidence_calibration"] = {
            "status": "passed",
            "method": "segment_monotonic_safety",
            "evidence_scope": "FULL_CANDIDATE_POOL_REPLAY",
            "locked_metrics": {"rows": 1067},
            "historical_support": {
                key: [0.49, 0.93]
                for key in (
                    "GLOBAL",
                    "PTS_OVER",
                    "PTS_UNDER",
                    "TRB_OVER",
                    "TRB_UNDER",
                    "AST_OVER",
                    "AST_UNDER",
                )
            },
        }
    if sport == "mlb":
        payload.update(
            {
                "policy_profile": MLB_POLICY_PROFILE,
                "publication_state": "published_current_pool",
                "daily_parlay": {
                    "status": "withheld",
                    "available": False,
                    "selected_ticket": None,
                    "reason": "no ticket cleared",
                },
                "policy_governance": {
                    "candidate_authorization_enabled": False,
                    "staking_enabled": False,
                    "publication_mode": "SHADOW_RESEARCH_ONLY",
                    "certificate_status": "NO_ACTIVE_PROSPECTIVE_CERTIFICATE"
                },
                "selection": {
                    "matchup_network_enabled": True,
                    "matchup_network_version": MLB_MATCHUP_NETWORK_VERSION,
                    "top_n": 3,
                    "targets": sorted(MLB_REQUIRED_TARGETS),
                    "max_per_market_bucket": 2,
                    "optimized_over_max_per_market_bucket": None,
                    "min_expected_value": 0.0,
                    "min_market_books": 5,
                    "min_common_market_books": 2,
                    "require_real_market_source": True,
                    "allow_unpriced_side": False,
                    "optimized_over_profile": "r_tb_over_moderate_edge_v1",
                    "optimized_over_profile_status": "probation",
                    "optimized_over_targets": [],
                    "over_min_abs_edge": None,
                    "over_max_abs_edge": None,
                    "over_min_model_hit_probability": None,
                    "over_max_model_hit_probability": None,
                    "over_min_expected_value": None,
                    "over_min_history_rows": None,
                    "over_max_american_price": None,
                    "pitcher_k_over_profile_enabled": False,
                    "pitcher_k_over_profile": "pitcher_k_over_workload_v1",
                    "pitcher_k_over_profile_status": "probation",
                    "pitcher_k_min_starter_history": 15,
                    "pitcher_k_min_projected_ip": 5.25,
                    "pitcher_k_min_projected_pitches": 75.0,
                    "pitcher_k_max_days_since_history": 14,
                    "pitcher_k_min_abs_edge": 0.15,
                    "pitcher_k_max_abs_edge": 1.0,
                    "pitcher_k_min_model_hit_probability": 0.50,
                    "pitcher_k_max_model_hit_probability": 0.65,
                    "pitcher_k_min_expected_value": 0.0,
                    "pitcher_k_min_american_price": -130.0,
                    "pitcher_k_max_american_price": 130.0,
                    "max_pitcher_k_picks": 1,
                    "core_min_american_price": -180.0,
                    "core_max_american_price": 125.0,
                    "min_over_picks": 0,
                    "max_over_picks": 3,
                    "max_under_picks": 1,
                    "daily_pick_soft_cap": 3,
                    "post_cap_min_selection_score": 0.80,
                    "min_hit_probability": 0.825,
                    "min_graded_hit_rate": 0.825,
                    "historical_calibration_evidence_scope": "real_price_confirmed_markets_only_v1",
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_static_shell(root: Path) -> None:
    for relative_path in ("dist/index.html", "dist/app.js", "dist/styles.css"):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")


def test_validate_publication_accepts_current_payloads(tmp_path: Path) -> None:
    build_static_shell(tmp_path)
    for sport in ("nba", "mlb"):
        write_payload(
            tmp_path / f"sports/{sport}/web/data/daily_predictions.json",
            run_date="2026-04-28",
            sport=sport,
        )
        write_payload(
            tmp_path / f"dist/{sport}/data/daily_predictions.json",
            run_date="2026-04-28",
            sport=sport,
        )
        route = tmp_path / f"dist/{sport}/predictions/index.html"
        route.parent.mkdir(parents=True, exist_ok=True)
        route.write_text("ok", encoding="utf-8")

    summaries = validate_publication(
        repo_root=tmp_path,
        output_dir=Path("dist"),
        run_date="2026-04-28",
        sports=["nba", "mlb"],
    )

    assert summaries == [
        "NBA: 2026-04-28, status=ready, plays=0",
        "MLB: 2026-04-28, status=ready, plays=0, mode=SHADOW_RESEARCH_ONLY",
    ]


def test_validate_publication_rejects_stale_payload(tmp_path: Path) -> None:
    build_static_shell(tmp_path)
    write_payload(
        tmp_path / "sports/mlb/web/data/daily_predictions.json",
        run_date="2026-04-27",
        sport="mlb",
    )
    write_payload(
        tmp_path / "dist/mlb/data/daily_predictions.json",
        run_date="2026-04-27",
        sport="mlb",
    )
    route = tmp_path / "dist/mlb/predictions/index.html"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text("ok", encoding="utf-8")

    with pytest.raises(ValueError, match="MLB source payload is stale"):
        validate_publication(
            repo_root=tmp_path,
            output_dir=Path("dist"),
            run_date="2026-04-28",
            sports=["mlb"],
        )


def test_validate_nba_payload_rejects_uncertified_confidence() -> None:
    payload = {
        "confidence_calibration": {
            "status": "failed",
            "method": "segment_monotonic_safety",
            "evidence_scope": "FULL_CANDIDATE_POOL_REPLAY",
        },
        "plays": [],
    }

    with pytest.raises(ValueError, match="locked selected-board calibration policy"):
        validate_nba_payload(payload, label="test")


def test_validate_publication_allows_stale_payloads_when_requested(tmp_path: Path) -> None:
    build_static_shell(tmp_path)
    write_payload(
        tmp_path / "sports/mlb/web/data/daily_predictions.json",
        run_date="2026-04-27",
        sport="mlb",
    )
    write_payload(
        tmp_path / "dist/mlb/data/daily_predictions.json",
        run_date="2026-04-27",
        sport="mlb",
    )
    route = tmp_path / "dist/mlb/predictions/index.html"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text("ok", encoding="utf-8")

    summaries = validate_publication(
        repo_root=tmp_path,
        output_dir=Path("dist"),
        run_date="2026-04-28",
        sports=["mlb"],
        allow_stale_payloads=True,
    )

    assert summaries == ["MLB: 2026-04-28, status=ready, plays=0, mode=SHADOW_RESEARCH_ONLY"]


def test_validate_publication_allows_empty_payloads_when_requested(tmp_path: Path) -> None:
    build_static_shell(tmp_path)
    for relative_path in (
        "sports/mlb/web/data/daily_predictions.json",
        "dist/mlb/data/daily_predictions.json",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    route = tmp_path / "dist/mlb/predictions/index.html"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text("ok", encoding="utf-8")

    summaries = validate_publication(
        repo_root=tmp_path,
        output_dir=Path("dist"),
        run_date="2026-04-28",
        sports=["mlb"],
        allow_stale_payloads=True,
    )

    assert summaries == ["MLB: 2026-04-28, status=unavailable, plays=0"]


def test_validate_publication_rejects_legacy_mlb_pool_policy(tmp_path: Path) -> None:
    build_static_shell(tmp_path)
    for relative_path in (
        "sports/mlb/web/data/daily_predictions.json",
        "dist/mlb/data/daily_predictions.json",
    ):
        write_payload(tmp_path / relative_path, run_date="2026-04-28", sport="mlb")
    source_path = tmp_path / "sports/mlb/web/data/daily_predictions.json"
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload["policy_profile"] = "walk_forward_balanced_v1"
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")
    route = tmp_path / "dist/mlb/predictions/index.html"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text("ok", encoding="utf-8")

    with pytest.raises(ValueError, match="expected premium_evidence_gated_v7"):
        validate_publication(
            repo_root=tmp_path,
            output_dir=Path("dist"),
            run_date="2026-04-28",
            sports=["mlb"],
        )


def test_validate_mlb_payload_rejects_probationary_over_profile_enablement(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    write_payload(payload_path, run_date="2026-04-28", sport="mlb")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["selection"]["optimized_over_targets"] = ["R", "TB"]

    with pytest.raises(ValueError, match="enabled the probationary R/TB OVER target set"):
        validate_mlb_payload(payload, label="test")


def test_validate_mlb_payload_rejects_adaptive_volume_drift(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    write_payload(payload_path, run_date="2026-04-28", sport="mlb")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["selection"]["daily_pick_soft_cap"] = 4

    with pytest.raises(ValueError, match="changed the adaptive daily pick soft cap"):
        validate_mlb_payload(payload, label="test")


def test_validate_mlb_payload_rejects_nonexecutable_parlay(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    write_payload(payload_path, run_date="2026-04-28", sport="mlb")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["parlay_pairs"] = [
        {
            "same_sportsbook_confirmed": False,
            "sportsbook_key": "",
            "expected_return_per_unit": 0.25,
        }
    ]

    with pytest.raises(ValueError, match="not executable at one confirmed sportsbook"):
        validate_mlb_payload(payload, label="test")


def test_validate_mlb_payload_accepts_adaptive_over_parlay(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    write_payload(payload_path, run_date="2026-04-28", sport="mlb")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["daily_parlay"] = {
        "status": "review",
        "available": True,
        "selected_ticket": {
            "leg_count": 2,
            "sportsbook_key": "draftkings",
            "same_sportsbook_confirmed": True,
            "projected_probability": 0.44,
            "combined_decimal_price": 2.25,
            "expected_return_per_unit": 0.01,
            "risk_flags": ["lineup_unconfirmed"],
            "legs": [
                {
                    "player": "First",
                    "player_id": "first",
                    "game_id": "g1",
                    "direction": "OVER",
                    "market_source": "real",
                    "price_confirmed": True,
                    "selected_side_price": -180,
                    "selected_sportsbook_key": "draftkings",
                    "market_books": 6,
                    "market_common_books": 3,
                    "estimated_graded_hit_rate": 0.67,
                },
                {
                    "player": "Second",
                    "player_id": "second",
                    "game_id": "g2",
                    "direction": "OVER",
                    "market_source": "real",
                    "price_confirmed": True,
                    "selected_side_price": -170,
                    "selected_sportsbook_key": "draftkings",
                    "market_books": 6,
                    "market_common_books": 3,
                    "estimated_graded_hit_rate": 0.66,
                },
            ],
        },
    }
    selected_ticket = payload["daily_parlay"]["selected_ticket"]
    balanced_ticket = json.loads(json.dumps(selected_ticket))
    balanced_ticket.update(
        {
            "leg_count": 3,
            "projected_probability": 0.28,
            "combined_decimal_price": 4.0,
            "expected_return_per_unit": 0.12,
        }
    )
    balanced_ticket["legs"].append(
        {
            "player": "Third",
            "player_id": "third",
            "game_id": "g3",
            "direction": "OVER",
            "market_source": "real",
            "price_confirmed": True,
            "selected_side_price": -160,
            "selected_sportsbook_key": "draftkings",
            "market_books": 6,
            "market_common_books": 3,
            "estimated_graded_hit_rate": 0.65,
        }
    )
    profit_ticket = json.loads(json.dumps(selected_ticket))
    profit_ticket.update(
        {
            "ticket_id": "profit_boost_2_leg",
            "ticket_tier": "profit_boost",
            "projected_probability": 0.14,
            "combined_decimal_price": 9.0,
            "expected_return_per_unit": 0.26,
            "candidate_authorized": False,
        }
    )
    for index, leg in enumerate(profit_ticket["legs"], start=1):
        leg.update(
            {
                "line_variant": "alternate",
                "base_market_line": 0.5,
                "market_line": 1.5,
                "provider_source_market_id": f"alt-{index}",
                "alternate_line_observed_at_utc": "2026-04-28T14:00:00Z",
                "alternate_line_books": 1,
                "selected_side_price": 200,
                "estimated_graded_hit_rate": 0.38,
                "expected_value_per_unit": 0.14,
            }
        )
    payload["daily_parlay"]["ticket_ladder"] = [selected_ticket, balanced_ticket, profit_ticket]

    validate_mlb_payload(payload, label="test")

    balanced_ticket["combined_decimal_price"] = 11.0
    with pytest.raises(ValueError, match="outside its declared payout scope"):
        validate_mlb_payload(payload, label="test")


def test_validate_mlb_payload_rejects_probationary_over_pick(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    write_payload(payload_path, run_date="2026-04-28", sport="mlb")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["plays"] = [
        {
            "selection_profile": "r_tb_over_moderate_edge_v1",
            "market_source": "real",
            "market_books": 5,
            "market_common_books": 2,
            "price_confirmed": True,
            "selected_side_price": 120,
            "selected_sportsbook_key": "fanduel",
            "selected_sportsbook": "FanDuel",
            "expected_value_per_unit": 0.12,
            "direction": "OVER",
            "target": "R",
            "abs_edge": 0.25,
            "model_hit_probability": 0.52,
        }
    ]

    with pytest.raises(ValueError, match="used the disabled probationary OVER profile"):
        validate_mlb_payload(payload, label="test")


def test_validate_mlb_payload_rejects_uncertified_pitcher_profile(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    write_payload(payload_path, run_date="2026-04-28", sport="mlb")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["plays"] = [
        {
            "selection_profile": "pitcher_k_over_workload_v1",
            "market_source": "real",
            "market_books": 6,
            "market_common_books": 3,
            "price_confirmed": True,
            "selected_side_price": 105,
            "selected_sportsbook_key": "fanduel",
            "selected_sportsbook": "FanDuel",
            "expected_value_per_unit": 0.08,
            "direction": "OVER",
            "target": "K",
            "starter_confirmed": True,
            "starter_history_rows": 18,
            "projected_ip": 5.6,
            "projected_pitches": 88,
            "days_since_history": 6,
            "abs_edge": 0.6,
            "model_hit_probability": 0.58,
        }
    ]

    with pytest.raises(ValueError, match="disabled probationary pitcher K profile"):
        validate_mlb_payload(payload, label="test")


def test_validate_mlb_payload_requires_networked_hitter_play(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    write_payload(payload_path, run_date="2026-04-28", sport="mlb")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["plays"] = [
        {
            "selection_profile": "core_market_v1",
            "market_source": "real",
            "market_books": 5,
            "market_common_books": 2,
            "price_confirmed": True,
            "selected_side_price": 120,
            "selected_sportsbook_key": "fanduel",
            "selected_sportsbook": "FanDuel",
            "expected_value_per_unit": 0.12,
            "direction": "OVER",
            "target": "R",
            "abs_edge": 0.25,
            "model_hit_probability": 0.52,
            "player_type": "hitter",
            "opposing_pitcher": "Test Starter",
            "matchup_network_version": MLB_MATCHUP_NETWORK_VERSION,
            "pitcher_profile_uncertainty": 0.35,
            "matchup_network_confidence": 0.70,
            "matchup_network_adjustment": 0.02,
            "archetype_neighbor_games": 18,
            "archetype_neighbor_effective_support": 7.5,
            "archetype_neighbor_lift": 0.08,
        }
    ]

    validate_mlb_payload(payload, label="test")

    payload["plays"][0]["matchup_network_version"] = ""
    with pytest.raises(ValueError, match="missing the current matchup network version"):
        validate_mlb_payload(payload, label="test")
