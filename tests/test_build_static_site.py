from __future__ import annotations

import sys
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
