from __future__ import annotations

import json
import subprocess
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


def test_mlb_primary_policy_uses_validated_portfolio_limits() -> None:
    assert shared_daily_predictions.MLB_PICK_SURVIVAL_TOP_K == 3
    assert shared_daily_predictions.MLB_PARLAY_SELECTOR.name == "select_daily_parlay.py"
    assert shared_daily_predictions.MLB_GOVERNANCE_CAPTURE.name == "capture_complete_slate.py"
    top_n_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--top-n")
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[top_n_index + 1] == "3"
    index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--max-per-market-bucket")
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[index + 1] == "2"
    over_bucket_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index(
        "--optimized-over-max-per-market-bucket"
    )
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[over_bucket_index + 1] == "3"
    ev_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--min-expected-value")
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[ev_index + 1] == "0.0"
    common_books_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--min-common-market-books")
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[common_books_index + 1] == "2"
    availability_rate_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index(
        "--min-historical-market-availability-rate"
    )
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[availability_rate_index + 1] == "0"
    optimized_over_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--optimized-over-targets")
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[optimized_over_index + 1 : optimized_over_index + 3] == [
        "R",
        "TB",
    ]
    min_over_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--min-over-picks")
    max_over_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--max-over-picks")
    max_under_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--max-under-picks")
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[min_over_index + 1] == "3"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[max_over_index + 1] == "3"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[max_under_index + 1] == "1"
    soft_cap_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--daily-pick-soft-cap")
    expansion_score_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index(
        "--post-cap-min-selection-score"
    )
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[soft_cap_index + 1] == "3"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[expansion_score_index + 1] == "0.80"
    over_history_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--over-min-history-rows")
    core_min_price_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--core-min-american-price")
    core_price_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--core-max-american-price")
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[over_history_index + 1] == "55"
    assert "--enable-pitcher-k-over-profile" in shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS
    pitcher_history_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index(
        "--pitcher-k-min-starter-history"
    )
    pitcher_ip_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index(
        "--pitcher-k-min-projected-ip"
    )
    pitcher_recency_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index(
        "--pitcher-k-max-days-since-history"
    )
    pitcher_cap_index = shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS.index("--max-pitcher-k-picks")
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[pitcher_history_index + 1] == "15"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[pitcher_ip_index + 1] == "5.25"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[pitcher_recency_index + 1] == "14"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[pitcher_cap_index + 1] == "1"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[core_min_price_index + 1] == "-250"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS[core_price_index + 1] == "-200"
    assert shared_daily_predictions.MLB_PRIMARY_POLICY_PROFILE == "premium_over_pitcher_v5"


def test_annotate_mlb_summary_keeps_policy_identity_separate_from_publication_state(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    shared_daily_predictions.annotate_mlb_summary(
        summary_path,
        publication_strategy=shared_daily_predictions.MLB_PRIMARY_POLICY_PROFILE,
        publication_state="withheld_current_pool",
        market_profile={"rows": 2},
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["publication_strategy"] == "premium_over_pitcher_v5"
    assert summary["publication_state"] == "withheld_current_pool"


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
        "nba_market_provider": "rotowire",
        "nba_market_bookmakers": "draftkings,fanduel",
        "nba_snapshot_policy": "auto",
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
        "mlb_min_publish_plays": 1,
        "mlb_min_rescue_plays": 1,
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


def test_run_nba_forwards_scraped_live_market_configuration(tmp_path, monkeypatch) -> None:
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

    args = _default_args(
        run_date="2026-04-28",
        nba_market_provider="rotowire",
        nba_market_bookmakers="draftkings,fanduel",
        nba_snapshot_policy="live_only",
    )
    shared_daily_predictions.run_nba(args, tmp_path / "dist")

    runner_command = commands[0][1]
    assert runner_command[runner_command.index("--market-provider") + 1] == "rotowire"
    assert runner_command[runner_command.index("--market-bookmakers") + 1] == "draftkings,fanduel"
    assert runner_command[runner_command.index("--snapshot-policy") + 1] == "live_only"


def test_main_continues_mlb_and_build_when_nba_fails(monkeypatch, tmp_path, capsys) -> None:
    args = _default_args(skip_mlb=False, force_run=True, output_dir=tmp_path / "dist")
    calls: list[str] = []

    def fail_nba(_args: Namespace, _output_dir: Path) -> None:
        calls.append("nba")
        raise subprocess.CalledProcessError(1, ["python", "nba_runner.py"])

    def fake_run_mlb(_args: Namespace, _output_dir: Path) -> tuple[Path, Path, Path]:
        calls.append("mlb")
        return (tmp_path / "pool.csv", tmp_path / "selected.csv", tmp_path / "summary.json")

    def fake_build_site(_args: Namespace, _output_dir: Path) -> None:
        calls.append("build")

    monkeypatch.setattr(shared_daily_predictions, "parse_args", lambda: args)
    monkeypatch.setattr(shared_daily_predictions, "check_schedule_gate", lambda _args: (True, "forced test run"))
    monkeypatch.setattr(shared_daily_predictions, "run_nba", fail_nba)
    monkeypatch.setattr(shared_daily_predictions, "run_mlb", fake_run_mlb)
    monkeypatch.setattr(shared_daily_predictions, "build_site", fake_build_site)

    shared_daily_predictions.main()

    assert calls == ["nba", "mlb", "build"]
    output = capsys.readouterr().out
    assert "NBA Prediction Refresh Failed Safely" in output
    assert "SHARED DAILY PREDICTION REFRESH COMPLETE" in output


def test_main_raises_nba_failure_for_nba_only_runs(monkeypatch, tmp_path) -> None:
    args = _default_args(skip_mlb=True, force_run=True, output_dir=tmp_path / "dist")

    def fail_nba(_args: Namespace, _output_dir: Path) -> None:
        raise subprocess.CalledProcessError(1, ["python", "nba_runner.py"])

    monkeypatch.setattr(shared_daily_predictions, "parse_args", lambda: args)
    monkeypatch.setattr(shared_daily_predictions, "check_schedule_gate", lambda _args: (True, "forced test run"))
    monkeypatch.setattr(shared_daily_predictions, "run_nba", fail_nba)

    try:
        shared_daily_predictions.main()
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("Expected NBA-only run to propagate the NBA failure")
