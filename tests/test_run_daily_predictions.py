from __future__ import annotations

import json
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_PIPELINE_ROOT = REPO_ROOT / "sports" / "site" / "pipeline"
sys.path.insert(0, str(SITE_PIPELINE_ROOT))

import run_daily_predictions as shared_daily_predictions


EASTERN = ZoneInfo("America/New_York")


class FrozenDateTime(datetime):
    current = datetime(2026, 4, 28, 15, 0, tzinfo=EASTERN)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is not None:
            return cls.current.astimezone(tz)
        return cls.current


def _default_args(**overrides) -> Namespace:
    values = {
        "python": "python",
        "run_date": None,
        "output_dir": REPO_ROOT / "dist",
        "scheduled_hour": 2,
        "scheduled_minute": 0,
        "force_run": False,
        "skip_nba": False,
        "skip_mlb": True,
        "skip_build_site": False,
        "nba_manifest": None,
        "nba_season": None,
        "nba_latest": False,
        "nba_policy_profile": "production_board_objective_b12",
        "nba_shadow_policy_profiles": None,
        "nba_allow_heuristic_fallback": False,
        "nba_skip_update_data": False,
        "nba_skip_collect_market": False,
        "nba_skip_align": False,
        "nba_skip_backtest": False,
        "nba_skip_cutoff_meta_monitor": False,
        "mlb_pool_csv": None,
        "mlb_skip_fetch_market": False,
        "mlb_skip_update_data": False,
        "mlb_skip_generate": False,
        "mlb_data_dir": REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB",
        "mlb_manifest": REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB" / "update_manifest_2026.json",
        "mlb_market_provider": "rotowire",
        "mlb_market_input_path": None,
        "mlb_fallback_policy": "exact_or_latest",
        "mlb_min_publish_plays": 4,
        "mlb_top_n": 10,
    }
    values.update(overrides)
    return Namespace(**values)


def _write_payload(path: Path, run_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_date": run_date}), encoding="utf-8")


def test_check_schedule_gate_runs_after_schedule_when_nba_payload_is_stale(tmp_path, monkeypatch) -> None:
    nba_payload = tmp_path / "nba" / "data" / "daily_predictions.json"
    _write_payload(nba_payload, "2026-04-27")

    monkeypatch.setattr(shared_daily_predictions, "datetime", FrozenDateTime)
    monkeypatch.setattr(shared_daily_predictions, "NBA_WEB_JSON", nba_payload)

    should_run, message = shared_daily_predictions.check_schedule_gate(_default_args())

    assert should_run is True
    assert "2026-04-28" in message
    assert "NBA" in message


def test_check_schedule_gate_skips_when_payloads_are_already_current(tmp_path, monkeypatch) -> None:
    nba_payload = tmp_path / "nba" / "data" / "daily_predictions.json"
    _write_payload(nba_payload, "2026-04-28")

    monkeypatch.setattr(shared_daily_predictions, "datetime", FrozenDateTime)
    monkeypatch.setattr(shared_daily_predictions, "NBA_WEB_JSON", nba_payload)

    should_run, message = shared_daily_predictions.check_schedule_gate(_default_args())

    assert should_run is False
    assert "already current" in message


def test_run_nba_exports_expected_same_day_manifest(tmp_path, monkeypatch) -> None:
    predictor_root = tmp_path / "Player-Predictor"
    manifest_path = predictor_root / "model" / "analysis" / "daily_runs" / "20260428" / "daily_market_pipeline_manifest_20260428.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    commands: list[tuple[str, list[str]]] = []

    def fake_run_step(label: str, command: list[str], cwd: Path = shared_daily_predictions.REPO_ROOT) -> None:
        commands.append((label, command))

    monkeypatch.setattr(shared_daily_predictions, "datetime", FrozenDateTime)
    monkeypatch.setattr(shared_daily_predictions, "run_step", fake_run_step)
    monkeypatch.setattr(shared_daily_predictions, "NBA_PREDICTOR_ROOT", predictor_root)
    monkeypatch.setattr(shared_daily_predictions, "NBA_RUNNER", tmp_path / "run_daily_market_pipeline.py")
    monkeypatch.setattr(shared_daily_predictions, "NBA_EXPORTER", tmp_path / "export_daily_predictions_web.py")
    monkeypatch.setattr(shared_daily_predictions, "NBA_WEB_JSON", tmp_path / "sports" / "nba" / "web" / "data" / "daily_predictions.json")
    monkeypatch.setattr(shared_daily_predictions, "NBA_CARDS_JSON", tmp_path / "sports" / "nba" / "web" / "data" / "cards.json")

    shared_daily_predictions.run_nba(_default_args(), tmp_path / "dist")

    assert len(commands) == 2
    export_command = commands[1][1]
    assert export_command[:2] == ["python", str(tmp_path / "export_daily_predictions_web.py")]
    assert "--manifest" in export_command
    assert str(manifest_path) in export_command
