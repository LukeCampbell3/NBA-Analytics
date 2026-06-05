import pandas as pd
import importlib.util
from pathlib import Path


SCRIPT = Path("Player-Predictor/scripts/market_odds_quality.py")
SPEC = importlib.util.spec_from_file_location("market_odds_quality", SCRIPT)
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)


def test_valid_american_odds_rule():
    assert QUALITY.is_valid_american_odds(-110)
    assert QUALITY.is_valid_american_odds(120)
    assert not QUALITY.is_valid_american_odds(-11)
    assert not QUALITY.is_valid_american_odds(49)
    assert not QUALITY.is_valid_american_odds(0)


def test_odds_quality_report_counts_invalid_rows():
    frame = pd.DataFrame({"over_odds": [-110, -11], "under_odds": [100, -120]})
    report = QUALITY.odds_quality_report(frame)
    assert report["rows_with_valid_american_odds"] == 1
    assert report["rows_with_invalid_american_odds"] == 1
    assert report["rows_dropped"] == 1
