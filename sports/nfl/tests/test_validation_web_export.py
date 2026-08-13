from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
NFL_ROOT = REPO_ROOT / "sports" / "nfl"
sys.path.insert(0, str(NFL_ROOT / "scripts"))

import export_nfl_validation_web as exporter


def test_committed_validation_payload_matches_locked_replay() -> None:
    evaluation = NFL_ROOT / "data" / "evaluation"
    selector = json.loads(
        (evaluation / "market_selector_report.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (evaluation / "production_replay_report.json").read_text(encoding="utf-8")
    )
    weekly = pd.read_csv(evaluation / "production_replay_weekly.csv")

    payload = exporter.build_payload(selector, replay, weekly)

    assert payload["publication_status"] == "research_only_source_blocked"
    assert payload["validated_targets"] == ["passing"]
    assert payload["locked_policy"]["weekly_top_n"] == 12
    assert payload["methodology"]["selected_architecture"] == (
        "regularized_logistic_raw"
    )
    assert payload["final_test"]["wins"] == 127
    assert payload["final_test"]["losses"] == 83
    assert payload["final_test"]["hit_rate"] == 0.6048
    assert payload["gates"]["deployment"]["status"] == "blocked"
    assert len(payload["weekly"]) == 18


def test_frontend_exposes_validation_sections_and_payload() -> None:
    index_html = (NFL_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    predictions_html = (NFL_ROOT / "web" / "predictions.html").read_text(
        encoding="utf-8"
    )
    predictions_js = (NFL_ROOT / "web" / "predictions.js").read_text(
        encoding="utf-8"
    )
    fantasy_html = (NFL_ROOT / "web" / "fantasy.html").read_text(encoding="utf-8")
    fantasy_js = (NFL_ROOT / "web" / "fantasy.js").read_text(encoding="utf-8")
    fantasy_payload = json.loads(
        (NFL_ROOT / "web" / "data" / "fantasy_draft_rankings.json").read_text(
            encoding="utf-8"
        )
    )
    about_html = (NFL_ROOT / "web" / "prediction-about.html").read_text(
        encoding="utf-8"
    )

    assert 'url=predictions/' in index_html
    assert 'url=/nfl/fantasy/' in predictions_html
    assert 'id="rankingTable"' in fantasy_html
    assert 'id="confidenceMetrics"' in fantasy_html
    assert "data/fantasy_draft_rankings.json" in fantasy_js
    assert fantasy_payload["validation"]["status"] == "passed"
    assert len(fantasy_payload["rankings"]) == 200
    # The legacy prop implementation remains available in source for future
    # routing, but it is no longer the primary /nfl/predictions/ experience.
    assert "data/market_validation_summary.json" in predictions_js
    assert "data/daily_predictions.json" in predictions_js
    assert 'id="marketMethodFacts"' in about_html
    assert 'id="marketLimitations"' in about_html
