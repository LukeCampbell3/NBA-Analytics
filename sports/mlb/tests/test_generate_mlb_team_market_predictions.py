from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"))

import generate_mlb_team_market_predictions as team_market  # noqa: E402
from test_run_mlb_same_game_daily import (  # noqa: E402
    _fake_fanduel_provider,
    _fake_odds_provider,
    _schedule_payload,
    _write_historical_fixtures,
)
from the_odds_api_mlb_team_market_provider import TheOddsApiMlbTeamMarketProvider  # noqa: E402


def test_build_team_market_predictions_reports_no_real_games_scheduled(tmp_path) -> None:
    team_csv, pitcher_csv = _write_historical_fixtures(tmp_path)
    payload = team_market.build_team_market_predictions(
        run_date=date(2026, 6, 20), team_universe_csv=team_csv, pitcher_universe_csv=pitcher_csv,
        calibration_ledger=None, num_trials=500,
        schedule_payload={"dates": []}, fanduel_provider=_fake_fanduel_provider(), odds_api_provider=_fake_odds_provider(),
    )
    assert payload["status"] == "no_real_games_scheduled_today"
    assert payload["picks"] == []


def test_build_team_market_predictions_starts_unauthorized_with_no_calibration_evidence(tmp_path) -> None:
    """Same honest shadow-only posture as run_mlb_same_game_daily.py's own
    combo pipeline: brand-new buckets, empty calibration ledger -> zero
    real picks, never a guessed one."""
    team_csv, pitcher_csv = _write_historical_fixtures(tmp_path)
    payload = team_market.build_team_market_predictions(
        run_date=date(2026, 6, 20), team_universe_csv=team_csv, pitcher_universe_csv=pitcher_csv,
        calibration_ledger=None, num_trials=2000,
        schedule_payload=_schedule_payload(), fanduel_provider=_fake_fanduel_provider(), odds_api_provider=_fake_odds_provider(),
    )
    assert payload["status"] == "ok"
    assert payload["model"] == "mlb_team_market_joint_sim_v1"
    assert payload["picks"] == []
    assert payload["authorized_pick_count"] == 0


def test_build_team_market_predictions_reads_real_legs_when_calibration_authorizes(tmp_path, monkeypatch) -> None:
    """With calibration support forced True, real legs (moneyline/game_total/
    first_5_innings_total) built from real odds actually flow through as
    authorized picks -- proves the wiring end to end, not just the zero case."""
    team_csv, pitcher_csv = _write_historical_fixtures(tmp_path)

    import select_mlb_same_game_bets as select

    monkeypatch.setattr(select, "evaluate_support", lambda *a, **k: type("S", (), {"in_support": True, "blocking_dimensions": []})())

    payload = team_market.build_team_market_predictions(
        run_date=date(2026, 6, 20), team_universe_csv=team_csv, pitcher_universe_csv=pitcher_csv,
        calibration_ledger=tmp_path / "empty_but_present_ledger.jsonl", num_trials=2000,
        schedule_payload=_schedule_payload(), fanduel_provider=_fake_fanduel_provider(), odds_api_provider=_fake_odds_provider(),
    )
    assert payload["authorized_pick_count"] > 0
    for pick in payload["picks"]:
        assert pick["sport"] == "mlb"
        assert pick["market_type"] == "mlb_team_market_joint_sim_v1"
        assert pick["leg_authorized"] is True
        assert pick["market"] in {"moneyline", "game_total", "first_5_innings_total"}


def test_build_team_market_predictions_reports_no_real_odds_priced_yet(tmp_path) -> None:
    team_csv, pitcher_csv = _write_historical_fixtures(tmp_path)
    empty_odds_provider = TheOddsApiMlbTeamMarketProvider(api_key="fixture", fixture_payloads=[])
    payload = team_market.build_team_market_predictions(
        run_date=date(2026, 6, 20), team_universe_csv=team_csv, pitcher_universe_csv=pitcher_csv,
        calibration_ledger=None, num_trials=500,
        schedule_payload=_schedule_payload(), fanduel_provider=_fake_fanduel_provider(empty=True), odds_api_provider=empty_odds_provider,
    )
    assert payload["status"] == "ok"
    assert payload["picks"] == []  # no real odds -> no real legs to build


def test_merge_into_daily_predictions_is_additive_only(tmp_path) -> None:
    daily_predictions_path = tmp_path / "daily_predictions.json"
    daily_predictions_path.write_text(json.dumps({"plays": [{"player": "Real Player"}], "summary": {"x": 1}}), encoding="utf-8")

    team_market_payload = {
        "status": "ok", "generated_at_utc": "2026-06-20T00:00:00+00:00",
        "picks": [{"market": "moneyline", "leg_authorized": True}],
    }
    merged = team_market.merge_into_daily_predictions(team_market_payload, daily_predictions_path=daily_predictions_path)

    # existing keys untouched
    assert merged["plays"] == [{"player": "Real Player"}]
    assert merged["summary"] == {"x": 1}
    # new keys added
    assert merged["mlb_team_market_plays"] == [{"market": "moneyline", "leg_authorized": True}]
    assert merged["mlb_team_market_status"] == "ok"

    on_disk = json.loads(daily_predictions_path.read_text(encoding="utf-8"))
    assert on_disk == merged


def test_merge_into_daily_predictions_creates_file_if_missing(tmp_path) -> None:
    daily_predictions_path = tmp_path / "nested" / "daily_predictions.json"
    merged = team_market.merge_into_daily_predictions(
        {"status": "ok", "generated_at_utc": "x", "picks": []}, daily_predictions_path=daily_predictions_path
    )
    assert daily_predictions_path.exists()
    assert merged["mlb_team_market_plays"] == []
