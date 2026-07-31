from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "sports" / "site" / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from validate_daily_publication import MLB_POLICY_PROFILE, MLB_REQUIRED_TARGETS, as_float, validate_publication


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
                    "require_real_market_source": True,
                    "allow_unpriced_side": False,
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

    with pytest.raises(ValueError, match="expected walk_forward_balanced_v2"):
        validate_publication(
            repo_root=tmp_path,
            output_dir=Path("dist"),
            run_date="2026-04-28",
            sports=["mlb"],
        )
