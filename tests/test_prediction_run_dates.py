from __future__ import annotations

import importlib.util
from datetime import date, datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


SITE_PIPELINE = load_module(
    "site_run_daily_predictions",
    "sports/site/pipeline/run_daily_predictions.py",
)
NBA_DAILY_PIPELINE = load_module(
    "nba_run_daily_market_pipeline",
    "sports/nba/predictions/Player-Predictor/scripts/run_daily_market_pipeline.py",
)


def test_shared_pipeline_defaults_to_eastern_run_date() -> None:
    utc_now = datetime(2026, 5, 4, 0, 30, tzinfo=timezone.utc)
    resolved = SITE_PIPELINE.resolve_effective_run_date(None, now=utc_now)
    assert resolved == date(2026, 5, 3)


def test_shared_pipeline_preserves_explicit_run_date() -> None:
    utc_now = datetime(2026, 5, 4, 0, 30, tzinfo=timezone.utc)
    resolved = SITE_PIPELINE.resolve_effective_run_date("2026-05-02", now=utc_now)
    assert resolved == date(2026, 5, 2)


def test_nba_pipeline_defaults_to_eastern_run_date() -> None:
    utc_now = datetime(2026, 5, 4, 0, 30, tzinfo=timezone.utc)
    resolved = NBA_DAILY_PIPELINE.resolve_run_timestamp(None, now=utc_now)
    assert resolved.strftime("%Y-%m-%d") == "2026-05-03"


def test_nba_pipeline_preserves_explicit_run_date() -> None:
    utc_now = datetime(2026, 5, 4, 0, 30, tzinfo=timezone.utc)
    resolved = NBA_DAILY_PIPELINE.resolve_run_timestamp("2026-05-02", now=utc_now)
    assert resolved.strftime("%Y-%m-%d") == "2026-05-02"
