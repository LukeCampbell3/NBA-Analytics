from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prediction_pages_have_self_contained_startup_watchdogs():
    cases = {
        "mlb": ("predictionRunMeta", "Loading board details...", "predictionCards"),
        "nba": ("openingRunFacts", "Loading opening-night pool...", "predictionCards"),
        "nfl": ("runFacts", "Loading Week 1 pool...", "currentBoard"),
    }

    for sport, (status_id, loading_text, board_id) in cases.items():
        html = (REPO_ROOT / "sports" / sport / "web" / "predictions.html").read_text(encoding="utf-8")

        watchdog_position = html.index("const timeoutMs = 12000;")
        first_external_script = html.index('<script src="')
        assert watchdog_position < first_external_script
        assert f'document.getElementById("{status_id}")' in html
        assert loading_text in html
        assert f'document.getElementById("{board_id}")' in html
        assert "data-prediction-reload" in html
        assert "window.location.reload()" in html
        assert "localStorage" not in html
        assert "sessionStorage" not in html
