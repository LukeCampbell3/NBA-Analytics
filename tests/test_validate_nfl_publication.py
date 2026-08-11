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
    route = output / "nfl/predictions/index.html"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text("<!doctype html>", encoding="utf-8")
    return tmp_path, output


def test_validate_nfl_publication_accepts_research_only_payload(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)

    summary = validate_nfl_publication(repo_root=repo_root, output_dir=output)

    assert summary == "NFL: research_only, holdout=2026-07-31, validated_targets=passing"


def test_validate_nfl_publication_rejects_live_claim(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)
    daily_path = repo_root / "sports/nfl/web/data/daily_predictions.json"
    daily = json.loads(daily_path.read_text(encoding="utf-8"))
    daily["publication_status"] = "ready"
    write_payload(daily_path, daily)
    write_payload(output / "nfl/data/daily_predictions.json", daily)

    with pytest.raises(ValueError, match="must remain research_only"):
        validate_nfl_publication(repo_root=repo_root, output_dir=output)


def test_validate_nfl_publication_rejects_output_drift(tmp_path: Path) -> None:
    repo_root, output = make_publication(tmp_path)
    public_path = output / "nfl/data/market_validation_summary.json"
    public_payload = json.loads(public_path.read_text(encoding="utf-8"))
    public_payload["validated_targets"] = ["rushing"]
    write_payload(public_path, public_payload)

    with pytest.raises(ValueError, match="validation source and public payloads differ"):
        validate_nfl_publication(repo_root=repo_root, output_dir=output)
