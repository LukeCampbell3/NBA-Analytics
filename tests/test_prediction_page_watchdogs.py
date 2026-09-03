from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prediction_pages_have_self_contained_startup_watchdogs():
    cases = {
        "mlb": ("predictions.html", "predictionRunMeta", "Loading board details...", "predictionCards", True),
        "nba": ("predictions.html", "openingRunFacts", "Loading opening-night pool...", "predictionCards", True),
        "nfl": ("picks.html", "runFacts", "Loading Week 1 picks...", "currentBoard", False),
    }

    for sport, (filename, status_id, loading_text, board_id, has_reload) in cases.items():
        html = (REPO_ROOT / "sports" / sport / "web" / filename).read_text(encoding="utf-8")

        watchdog_position = html.index("12000")
        first_external_script = html.index('<script src="')
        assert watchdog_position < first_external_script
        assert f'document.getElementById("{status_id}")' in html
        assert loading_text in html
        assert f'document.getElementById("{board_id}")' in html
        if has_reload:
            assert "data-prediction-reload" in html
            assert "window.location.reload()" in html
        assert "localStorage.getItem" not in html
        assert "localStorage.setItem" not in html
        assert "sessionStorage.getItem" not in html
        assert "sessionStorage.setItem" not in html


def test_mlb_watchdog_has_dependency_free_board_recovery():
    html = (REPO_ROOT / "sports" / "mlb" / "web" / "predictions.html").read_text(encoding="utf-8")

    assert "data/daily_predictions.json" in html
    assert 'cache: "no-store"' in html
    assert "Array.isArray(payload.plays)" in html
    assert 'document.createElement("article")' in html
    assert "Limited compatibility mode." in html
    assert 'vault-components.js?v=17' in html
    assert 'predictions.js?v=39' in html
