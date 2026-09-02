from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nfl_projections_and_picks_are_separate_routes():
    web = ROOT / "sports/nfl/web"
    projections = (web / "projections.html").read_text(encoding="utf-8")
    picks = (web / "picks.html").read_text(encoding="utf-8")
    legacy = (web / "predictions.html").read_text(encoding="utf-8")
    script = (web / "predictions.js").read_text(encoding="utf-8")

    assert '<h1>Projections</h1>' in projections
    assert 'id="weekProjectionPool"' in projections
    assert 'id="currentBoard"' not in projections
    assert '<h1>Picks</h1>' in picks
    assert 'id="currentBoard"' in picks
    assert 'id="parlayWatchlists"' in picks
    assert 'id="weekProjectionPool"' not in picks
    assert 'url=/nfl/projections/' in legacy
    assert 'label: "Projections"' in script
    assert 'label: "Picks"' in script
    assert 'this.pageMode === "projections"' in script


def test_static_builder_publishes_new_nfl_route_assets():
    builder = (ROOT / "sports/site/pipeline/build_static_site.py").read_text(encoding="utf-8")
    assert '"projections", "picks"' in builder
    assert '"projections.html"' in builder
    assert '"picks.html"' in builder
    assert 'in {"predictions", "projections"}' in builder
