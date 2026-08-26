from __future__ import annotations

import sys
import json
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "sports" / "site" / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

import build_static_site


def test_prune_keeps_daily_parlay_stylesheet(tmp_path: Path) -> None:
    sport_output = tmp_path / "mlb"
    sport_output.mkdir()
    (sport_output / "predictions.html").write_text("predictions", encoding="utf-8")
    (sport_output / "parlay-board.css").write_text(".parlay {}", encoding="utf-8")
    (sport_output / "unrelated.txt").write_text("remove", encoding="utf-8")

    build_static_site.prune_non_prediction_assets(sport_output)

    assert (sport_output / "predictions.html").exists()
    assert (sport_output / "parlay-board.css").exists()
    assert not (sport_output / "unrelated.txt").exists()


def test_prune_keeps_only_prediction_history_json(tmp_path: Path) -> None:
    history = tmp_path / "mlb" / "data" / "history"
    history.mkdir(parents=True)
    (history / "index.json").write_text('{"dates":[]}', encoding="utf-8")
    (history / "2026-08-09.json").write_text('{"run_date":"2026-08-09"}', encoding="utf-8")
    (history / "notes.json").write_text("{}", encoding="utf-8")
    (history / "debug.txt").write_text("debug", encoding="utf-8")

    build_static_site.prune_non_prediction_assets(tmp_path / "mlb")

    assert (history / "index.json").exists()
    assert (history / "2026-08-09.json").exists()
    assert not (history / "notes.json").exists()
    assert not (history / "debug.txt").exists()


def test_prune_keeps_real_headshot_cache_assets(tmp_path: Path) -> None:
    headshots = tmp_path / "mlb" / "data" / "headshots"
    headshots.mkdir(parents=True)
    (headshots / "624413.jpg").write_bytes(b"real-image-bytes")
    (headshots / "manifest.json").write_text('{"624413": {"filename": "624413.jpg"}}', encoding="utf-8")
    (headshots / "debug.txt").write_text("debug", encoding="utf-8")

    build_static_site.prune_non_prediction_assets(tmp_path / "mlb")

    assert (headshots / "624413.jpg").exists()
    assert (headshots / "manifest.json").exists()
    assert not (headshots / "debug.txt").exists()


def test_build_publishes_sport_routes_and_keeps_compatibility_copy(tmp_path: Path, monkeypatch) -> None:
    public_source = tmp_path / "public-source"
    private_source = tmp_path / "private-source"
    sport_source = tmp_path / "sport-source"
    public_source.mkdir()
    private_source.mkdir()
    (sport_source / "data" / "history").mkdir(parents=True)
    (public_source / "index.html").write_text("public landing", encoding="utf-8")
    (public_source / "app.js").write_text("public catalog", encoding="utf-8")
    (private_source / "index.html").write_text("member shell", encoding="utf-8")
    (sport_source / "predictions.html").write_text("<head></head>prediction board", encoding="utf-8")
    (sport_source / "prediction-about.html").write_text("<head></head>prediction method", encoding="utf-8")
    (sport_source / "fantasy.html").write_text("<head></head>fantasy draft", encoding="utf-8")
    (sport_source / "predictions.js").write_text("prediction script", encoding="utf-8")
    (sport_source / "fantasy.js").write_text("fantasy script", encoding="utf-8")
    (sport_source / "fantasy.css").write_text(".fantasy {}", encoding="utf-8")
    (sport_source / "data" / "daily_predictions.json").write_text('{"published":true}', encoding="utf-8")
    (sport_source / "data" / "fantasy_draft_rankings.json").write_text('{"rankings":[]}', encoding="utf-8")
    (sport_source / "data" / "history" / "index.json").write_text('{"dates":["2026-08-09"]}', encoding="utf-8")
    (sport_source / "data" / "history" / "2026-08-09.json").write_text('{"run_date":"2026-08-09"}', encoding="utf-8")
    monkeypatch.setattr(build_static_site, "VAULT_SOURCE_DIR", tmp_path / "missing-vault")
    monkeypatch.setattr(build_static_site, "discover_sports", lambda: [{
        "slug": "nba", "source_dir": sport_source, "title": "NBA Analytics",
        "tagline": "NBA desk", "summary": "NBA signals", "status": "active",
        "status_label": "Active", "accent": "#c02c3a", "surface": "#172131",
        "pages": [
            {"slug": "predictions", "label": "Predictions", "href": "/nba/predictions/"},
            {"slug": "fantasy", "label": "Fantasy Draft", "href": "/nba/fantasy/"},
            {"slug": "prediction-about", "label": "Method", "href": "/nba/prediction-about/"},
        ],
    }])
    public_output = tmp_path / "dist"
    private_output = tmp_path / "private-content" / "app"
    result = build_static_site.build_static_site(
        public_source, public_output, None, private_source, private_output,
    )
    assert result == 0
    assert (public_output / "index.html").read_text(encoding="utf-8") == "public landing"
    assert (public_output / "nba" / "predictions" / "index.html").exists()
    assert (public_output / "nba" / "fantasy" / "index.html").exists()
    assert (public_output / "nba" / "data" / "daily_predictions.json").exists()
    assert (public_output / "nba" / "data" / "fantasy_draft_rankings.json").exists()
    assert (public_output / "nba" / "data" / "history" / "2026-08-09.json").exists()
    assert (private_output / "nba" / "data" / "daily_predictions.json").exists()
    assert (private_output / "nba" / "data" / "history" / "index.json").exists()
    assert (private_output / "nba" / "predictions" / "index.html").exists()
    manifest = json.loads((public_output / "data" / "sports.json").read_text(encoding="utf-8"))
    assert manifest[0]["entry_href"] == "/nba/predictions/"
