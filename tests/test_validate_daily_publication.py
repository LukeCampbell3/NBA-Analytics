from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "sports" / "site" / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from validate_daily_publication import (
    MLB_POLICY_PROFILE,
    MLB_REQUIRED_TARGETS,
    as_float,
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
    if sport == "mlb":
        payload.update(
            {
                "policy_profile": MLB_POLICY_PROFILE,
                "publication_state": "published_current_pool",
                "selection": {
                    "targets": sorted(MLB_REQUIRED_TARGETS),
                    "max_per_market_bucket": 4,
                    "min_expected_value": 0.0,
                    "min_market_books": 5,
                    "min_common_market_books": 2,
                    "require_real_market_source": True,
                    "allow_unpriced_side": False,
                    "optimized_over_profile": "r_tb_over_moderate_edge_v1",
                    "optimized_over_profile_status": "probation",
                    "optimized_over_targets": ["R", "TB"],
                    "over_min_abs_edge": 0.15,
                    "over_max_abs_edge": 0.35,
                    "over_min_model_hit_probability": 0.45,
                    "over_max_model_hit_probability": 0.55,
                    "over_min_expected_value": 0.10,
                    "over_max_american_price": 125.0,
                    "min_over_picks": 3,
                    "max_over_picks": 3,
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
        "MLB: 2026-04-28, status=ready, plays=0",
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

    with pytest.raises(ValueError, match="expected premium_price_defended_v1"):
        validate_publication(
            repo_root=tmp_path,
            output_dir=Path("dist"),
            run_date="2026-04-28",
            sports=["mlb"],
        )


def test_validate_mlb_payload_rejects_over_profile_threshold_drift(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    write_payload(payload_path, run_date="2026-04-28", sport="mlb")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["selection"]["over_max_american_price"] = 150

    with pytest.raises(ValueError, match="changed validated OVER threshold over_max_american_price"):
        validate_mlb_payload(payload, label="test")


def test_validate_mlb_payload_checks_each_profiled_over_pick(tmp_path: Path) -> None:
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

    validate_mlb_payload(payload, label="test")

    payload["plays"][0]["model_hit_probability"] = 0.60
    with pytest.raises(ValueError, match="outside the validated OVER probability corridor"):
        validate_mlb_payload(payload, label="test")
