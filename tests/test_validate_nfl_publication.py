from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "sports" / "site" / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from validate_nfl_publication import validate_nfl_publication


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_publication(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "dist"
    daily = {
        "run_date": "2026-07-31",
        "publication_status": "research_only",
        "mode": "historical_holdout",
    }
    validation = {
        "publication_status": "research_only_source_blocked",
        "validated_targets": ["passing"],
        "gates": {"deployment": {"status": "blocked"}},
    }
    write_payload(tmp_path / "sports/nfl/web/data/daily_predictions.json", daily)
    write_payload(output / "nfl/data/daily_predictions.json", daily)
    write_payload(
        tmp_path / "sports/nfl/web/data/market_validation_summary.json", validation
    )
    write_payload(output / "nfl/data/market_validation_summary.json", validation)
    write_payload(
        tmp_path / "sports/nfl/data/evaluation/daily_policy_backtest.json",
        {"gates": {"singles": {"status": "passed"}, "parlay": {"status": "failed"}}},
    )
    write_payload(
        tmp_path / "sports/nfl/data/evaluation/pick_meta_backtest.json",
        {
            "sport": "NFL",
            "locked_recent_validation": {"status": "passed"},
            "confidence_calibration": {"status": "passed"},
            "deployment": {"status": "shadow_only"},
        },
    )
    route = output / "nfl/predictions/index.html"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text("<!doctype html>", encoding="utf-8")
    return tmp_path, output


def test_validate_nfl_publication_accepts_research_only_payload(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)

    summary = validate_nfl_publication(repo_root=repo_root, output_dir=output)

    assert summary == (
        "NFL: status=research_only, date=2026-07-31, plays=0, "
        "validated_targets=passing"
    )


def test_validate_nfl_publication_rejects_live_claim(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)
    daily_path = repo_root / "sports/nfl/web/data/daily_predictions.json"
    daily = json.loads(daily_path.read_text(encoding="utf-8"))
    daily["publication_status"] = "ready"
    write_payload(daily_path, daily)
    write_payload(output / "nfl/data/daily_predictions.json", daily)

    with pytest.raises(ValueError, match="legacy payload must remain research_only"):
        validate_nfl_publication(repo_root=repo_root, output_dir=output)


def test_validate_nfl_publication_rejects_output_drift(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)
    public_path = output / "nfl/data/market_validation_summary.json"
    public_payload = json.loads(public_path.read_text(encoding="utf-8"))
    public_payload["validated_targets"] = ["rushing"]
    write_payload(public_path, public_payload)

    with pytest.raises(ValueError, match="validation source and public payloads differ"):
        validate_nfl_publication(repo_root=repo_root, output_dir=output)


def test_validate_nfl_publication_accepts_withheld_live_shadow(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)
    payload = {
        "schema_version": 2,
        "run_date": "2026-08-11",
        "publication_status": "withheld_current_pool",
        "mode": "live_shadow",
        "policy_profile": "nfl_passing_loss_aware_meta_policy_v2",
        "selection": {
            "loss_aware_meta_policy": {
                "minimum_side_probability": 0.58,
                "minimum_no_vig_advantage": 0.1,
                "minimum_price": -130,
                "maximum_price": 130,
                "weekly_cap": 6,
            },
            "confidence_calibration": {
                "method": "identity",
                "status": "passed",
                "historical_support": [0.585605, 0.806799],
            },
        },
        "plays": [],
        "daily_parlay": {
            "status": "withheld",
            "validation_status": "failed_locked_holdout",
            "candidate_authorized": False,
        },
        "policy_governance": {
            "publication_mode": "SHADOW_RESEARCH_ONLY",
            "candidate_authorization_enabled": False,
            "staking_enabled": False,
        },
    }
    write_payload(repo_root / "sports/nfl/web/data/daily_predictions.json", payload)
    write_payload(output / "nfl/data/daily_predictions.json", payload)

    summary = validate_nfl_publication(
        repo_root=repo_root, output_dir=output, run_date="2026-08-11"
    )

    assert "status=withheld_current_pool" in summary


def test_validate_nfl_publication_rejects_authorized_live_pick(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)
    payload = {
        "schema_version": 2,
        "run_date": "2026-08-11",
        "publication_status": "shadow_current_pool",
        "mode": "live_shadow",
        "policy_profile": "nfl_passing_loss_aware_meta_policy_v2",
        "selection": {
            "loss_aware_meta_policy": {
                "minimum_side_probability": 0.58,
                "minimum_no_vig_advantage": 0.1,
                "minimum_price": -130,
                "maximum_price": 130,
                "weekly_cap": 6,
            },
            "confidence_calibration": {
                "method": "identity",
                "status": "passed",
                "historical_support": [0.585605, 0.806799],
            },
        },
        "plays": [
            {
                "target": "passing",
                "market_source": "the_odds_api_live",
                "price_confirmed": True,
                "selected_side_price": -110,
                "model_hit_probability": 0.64,
                "probability_advantage": 0.12,
                "meta_policy_score": 0.76,
                "raw_model_probability": 0.64,
                "calibrated_hit_probability": 0.64,
                "confidence_in_support": True,
                "market_books": 2,
                "market_common_books": 1,
                "candidate_authorized": True,
            }
        ],
        "daily_parlay": {
            "status": "withheld",
            "validation_status": "failed_locked_holdout",
            "candidate_authorized": False,
        },
        "policy_governance": {
            "publication_mode": "SHADOW_RESEARCH_ONLY",
            "candidate_authorization_enabled": False,
            "staking_enabled": False,
        },
    }
    write_payload(repo_root / "sports/nfl/web/data/daily_predictions.json", payload)
    write_payload(output / "nfl/data/daily_predictions.json", payload)

    with pytest.raises(ValueError, match="cannot be authorized"):
        validate_nfl_publication(repo_root=repo_root, output_dir=output)


def test_validate_nfl_publication_accepts_sportsgameodds_source(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)
    payload = {
        "schema_version": 2,
        "run_date": "2026-08-11",
        "publication_status": "shadow_current_pool",
        "mode": "live_shadow",
        "policy_profile": "nfl_passing_loss_aware_meta_policy_v2",
        "selection": {
            "loss_aware_meta_policy": {
                "minimum_side_probability": 0.58,
                "minimum_no_vig_advantage": 0.1,
                "minimum_price": -130,
                "maximum_price": 130,
                "weekly_cap": 6,
            },
            "confidence_calibration": {
                "method": "identity",
                "status": "passed",
                "historical_support": [0.585605, 0.806799],
            },
        },
        "plays": [
            {
                "target": "passing",
                "market_source": "sportsgameodds_live",
                "price_confirmed": True,
                "selected_side_price": -110,
                "model_hit_probability": 0.64,
                "probability_advantage": 0.12,
                "meta_policy_score": 0.76,
                "raw_model_probability": 0.64,
                "calibrated_hit_probability": 0.64,
                "confidence_in_support": True,
                "market_books": 2,
                "market_common_books": 2,
                "candidate_authorized": False,
            }
        ],
        "daily_parlay": {
            "status": "withheld",
            "validation_status": "failed_locked_holdout",
            "candidate_authorized": False,
        },
        "policy_governance": {
            "publication_mode": "SHADOW_RESEARCH_ONLY",
            "candidate_authorization_enabled": False,
            "staking_enabled": False,
        },
    }
    write_payload(repo_root / "sports/nfl/web/data/daily_predictions.json", payload)
    write_payload(output / "nfl/data/daily_predictions.json", payload)

    summary = validate_nfl_publication(repo_root=repo_root, output_dir=output)

    assert "plays=1" in summary
