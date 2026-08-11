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
    about_html = (NFL_ROOT / "web" / "prediction-about.html").read_text(
        encoding="utf-8"
    )

    assert 'url=predictions/' in index_html
    assert 'id="currentBoard"' in predictions_html
    assert 'id="dailyParlay"' in predictions_html
    assert 'id="marketReplayMetrics"' in predictions_html
    assert 'id="marketWeekly"' in predictions_html
    assert "data/market_validation_summary.json" in predictions_js
    assert "data/daily_predictions.json" in predictions_js
    assert 'id="marketMethodFacts"' in about_html
    assert 'id="marketLimitations"' in about_html
