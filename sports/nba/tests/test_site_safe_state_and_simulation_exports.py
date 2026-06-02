from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.player_simulation.simulate_next_season_player_states import (
    load_player_logs,
    simulate_next_season_player_states,
)
from research.player_simulation.audit_frozen_sample_leakage import audit_frozen_sample_leakage
from research.player_simulation.build_frozen_preseason_backtest_sample import build_frozen_preseason_backtest_sample
from research.player_simulation.backfill_pre_cutoff_player_state_logs import backfill_pre_cutoff_player_state_logs
from research.player_simulation.discover_pre_cutoff_player_logs import discover_pre_cutoff_player_logs
from research.player_simulation.simulation_credibility_gate import evaluate_simulation_credibility
from research.run_site_production_exports import run_site_production_exports
from research.site_export.export_safe_state_site_cards import export_safe_state_site_cards


def _safe_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate::safe",
        "game_id": "game_1",
        "market_date": "2026-05-27",
        "game_date": "2026-05-27",
        "player": "Safe Player",
        "player_name": "Safe Player",
        "target": "PTS",
        "direction": "OVER",
        "side": "OVER",
        "market_type": "PTS_OVER",
        "line": 12.5,
        "market_line": 12.5,
        "team": "AAA",
        "opponent": "BBB",
        "market_side_price": -110,
        "market_side_break_even": 0.5238,
        "stress_probability": 0.60,
        "lcb_probability": 0.56,
        "stress_edge": 0.076,
        "lcb_edge": 0.036,
        "edge_defendability_tier": "EDGE_DEFENDABLE",
        "forecastability_tier": "HIGH_FORECASTABILITY",
        "structural_mispricing_tier": "STRUCTURAL_MISPRICE_ACCEPTABLE",
        "similar_state_reliability_tier": "TIGHT",
        "safe_state_tier": "SAFE_STATE_CORE",
        "timestamp_safety_basis": "EVENT_START_VERIFIED",
        "safe_state_explanation": "Accepted in shadow only because price, forecastability, and structural evidence pass.",
    }
    row.update(overrides)
    return row


def _write_safe_state_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "safe_state_run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run_site_test",
        "run_date": "2026-05-27",
        "provider": "sportsgameodds",
        "production_behavior_changed": False,
        "promotion_ready": False,
    }
    (run_dir / "safe_state_production_shadow_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        _safe_row(),
        _safe_row(
            candidate_id="candidate::dependent",
            player="Dependent Player",
            player_name="Dependent Player",
            edge_defendability_tier="EDGE_PRICE_DEPENDENT",
            safe_state_tier="SAFE_STATE_REJECT",
            forecastability_tier="LOW_FORECASTABILITY",
        ),
        _safe_row(
            candidate_id="candidate::unstable",
            player="Volatile Player",
            player_name="Volatile Player",
            safe_state_tier="SAFE_STATE_UNSTABLE",
            forecastability_gap_primary="FORECASTABILITY_GAP_MINUTES_STATE",
        ),
        _safe_row(
            candidate_id="candidate::sample",
            player="Sample Player",
            player_name="Sample Player",
            safe_state_tier="SAFE_STATE_INSUFFICIENT_EVIDENCE",
        ),
    ]
    pd.DataFrame(rows).to_csv(run_dir / "safe_state_annotated_candidates.csv", index=False)
    pd.DataFrame([rows[0]]).to_csv(run_dir / "production_board_as_is.csv", index=False)
    pd.DataFrame([{"candidate_id": "candidate::unstable", "root_cause_primary": "REAL_MINUTES_VOLATILITY", "recommended_repair": "KEEP_UNSAFE_TRUE_VOLATILITY"}]).to_csv(
        run_dir / "forecastability_root_cause_rows.csv", index=False
    )
    pd.DataFrame([{"candidate_id": "candidate::sample", "root_cause_primary": "INSUFFICIENT_COMPARABLE_STATES", "recommended_repair": "NEEDS_MORE_SAMPLE"}]).to_csv(
        run_dir / "needs_more_sample_queue.csv", index=False
    )
    pd.DataFrame([{"candidate_id": "candidate::unstable", "root_cause_primary": "REAL_MINUTES_VOLATILITY", "recommended_action": "KEEP_UNSAFE_TRUE_VOLATILITY"}]).to_csv(
        run_dir / "true_unstable_shadow_rejections.csv", index=False
    )
    pd.DataFrame(
        [
            {"candidate_id": "candidate::safe", "primary_blocker": ""},
            {"candidate_id": "candidate::dependent", "primary_blocker": "PRICE_GAP"},
            {"candidate_id": "candidate::unstable", "primary_blocker": "FORECASTABILITY_GAP_MINUTES_STATE"},
            {"candidate_id": "candidate::sample", "primary_blocker": "SIMILAR_STATE_GAP"},
        ]
    ).to_csv(run_dir / "safe_state_candidate_blockers.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_id": "candidate::safe", "settlement_status": "PENDING"},
            {"candidate_id": "candidate::dependent", "settlement_status": "PENDING"},
            {"candidate_id": "candidate::unstable", "settlement_status": "PENDING"},
            {"candidate_id": "candidate::sample", "settlement_status": "PENDING"},
        ]
    ).to_csv(run_dir / "safe_state_settlement_status_audit.csv", index=False)
    return run_dir


def _write_logs(tmp_path: Path, *, sparse: bool = False) -> Path:
    data_proc = tmp_path / "Data-Proc"
    player_dir = data_proc / "Safe_Player"
    player_dir.mkdir(parents=True)
    rows = []
    count = 4 if sparse else 20
    for idx in range(count):
        rows.append(
            {
                "Date": f"2026-04-{idx + 1:02d}",
                "Player": "Safe Player",
                "Team": "AAA",
                "MP": 28 + (idx % 3),
                "PTS": 15 + (idx % 5),
                "TRB": 5 + (idx % 3),
                "AST": 4 + (idx % 2),
                "STL": 1,
                "BLK": 0,
                "FG3M": 2,
            }
        )
    rows.append({"Date": "2026-06-01", "Player": "Safe Player", "Team": "AAA", "MP": 40, "PTS": 80, "TRB": 30, "AST": 30, "FG3M": 10})
    pd.DataFrame(rows).to_csv(player_dir / "2026_processed_processed.csv", index=False)
    return data_proc


def _write_historical_logs(tmp_path: Path) -> Path:
    data_proc = tmp_path / "Historical-Data-Proc"
    player_dir = data_proc / "Sample_Player"
    player_dir.mkdir(parents=True)
    rows = []
    for idx in range(14):
        rows.append(
            {
                "Date": f"2025-03-{idx + 1:02d}",
                "Player": "Sample Player",
                "Player_ID": "p1",
                "Team": "AAA",
                "MP": 24 + (idx % 4),
                "PTS": 10 + (idx % 6),
                "TRB": 4 + (idx % 3),
                "AST": 3 + (idx % 2),
                "STL": 1,
                "BLK": 0,
                "FG3M": 1,
                "FGA": 8 + (idx % 4),
            }
        )
    for idx in range(12):
        rows.append(
            {
                "Date": f"2025-11-{idx + 1:02d}",
                "Player": "Sample Player",
                "Player_ID": "p1",
                "Team": "AAA",
                "MP": 26 + (idx % 4),
                "PTS": 12 + (idx % 5),
                "TRB": 5 + (idx % 2),
                "AST": 4 + (idx % 2),
                "STL": 1,
                "BLK": 0,
                "FG3M": 2,
                "FGA": 9 + (idx % 3),
            }
        )
    pd.DataFrame(rows).to_csv(player_dir / "historical_processed.csv", index=False)
    return data_proc


def test_safe_state_site_cards_include_shadow_labels_and_badges(tmp_path: Path) -> None:
    run_dir = _write_safe_state_run(tmp_path)
    report = export_safe_state_site_cards(safe_state_run_dir=run_dir, output_dir=tmp_path / "site", run_date="2026-05-27")
    payload = json.loads((tmp_path / "site" / "safe_state_latest.json").read_text(encoding="utf-8"))
    cards = {card["candidate_id"]: card for card in payload["cards"]}

    assert report["validation"]["safe_state_cards_include_shadow_status"] is True
    assert "SAFE_STATE_CORE_SHADOW" in cards["candidate::safe"]["warning_badges"]
    assert "SAFE_STATE_CORE_SHADOW" not in cards["candidate::dependent"]["warning_badges"]
    assert "TRUE_UNSTABLE_REJECTED" in cards["candidate::unstable"]["warning_badges"]
    assert "NEEDS_MORE_SAMPLE" in cards["candidate::sample"]["warning_badges"]
    assert all(card["shadow_only"] is True for card in payload["cards"])


def test_simulation_uses_only_data_before_cutoff_and_outputs_ranges(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path)
    logs, _manifest = load_player_logs(data_proc, cutoff_date="2026-05-01")
    assert logs["Date"].max().strftime("%Y-%m-%d") <= "2026-05-01"

    report = simulate_next_season_player_states(
        data_proc_dir=data_proc,
        output_dir=tmp_path / "sim",
        cutoff_date="2026-05-01",
        simulation_count=300,
        seed=11,
    )
    cards = json.loads((tmp_path / "sim" / "player_simulation_cards.json").read_text(encoding="utf-8"))

    assert report["production_behavior_changed"] is False
    assert cards
    for stat in ["pts", "reb", "ast", "pra"]:
        assert cards[0][stat]["p10"] is not None
        assert cards[0][stat]["p50"] is not None
        assert cards[0][stat]["p90"] is not None


def test_missing_or_sparse_player_data_lowers_confidence(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, sparse=True)
    simulate_next_season_player_states(
        data_proc_dir=data_proc,
        output_dir=tmp_path / "sim",
        cutoff_date="2026-05-01",
        simulation_count=100,
        seed=12,
    )
    cards = json.loads((tmp_path / "sim" / "player_simulation_cards.json").read_text(encoding="utf-8"))

    assert cards[0]["confidence_tier"] == "INSUFFICIENT_DATA"


def test_site_runner_writes_required_files_without_promotion_or_staking(tmp_path: Path) -> None:
    run_dir = _write_safe_state_run(tmp_path)
    report = run_site_production_exports(
        season=2026,
        run_date="2026-05-27",
        safe_state_run_dir=run_dir,
        site_output_dir=tmp_path / "site",
        simulate_next_season=False,
        shadow_only=True,
        skip_safe_state_run=True,
        copy_to_web_data=None,
    )
    manifest = json.loads((tmp_path / "site" / "site_manifest.json").read_text(encoding="utf-8"))

    assert (tmp_path / "site" / "safe_state_latest.json").exists()
    assert (tmp_path / "site" / "safe_state_cards.json").exists()
    assert manifest["production_behavior_changed"] is False
    assert manifest["promotion_ready"] is False
    assert manifest["staking_enabled"] is False
    assert manifest["auto_bet_enabled"] is False
    assert report["shadow_only"] is True


def test_frontend_safe_state_page_contains_range_and_confidence_components() -> None:
    html = (REPO_ROOT / "sports" / "nba" / "web" / "safe-state.html").read_text(encoding="utf-8")
    js = (REPO_ROOT / "sports" / "nba" / "web" / "safe-state.js").read_text(encoding="utf-8")

    assert "safe-state.js" in html
    assert "DistributionRangeBar" in js
    assert "ConfidenceBadge" in js
    assert "p10/p50/p90" in js
    assert "SHADOW" in js
    assert "simulation_credibility_gate.json" in js
    assert "research projection / uncalibrated" in js


def test_frozen_sample_builder_and_audit_use_pre_cutoff_inputs(tmp_path: Path) -> None:
    data_proc = _write_historical_logs(tmp_path)
    sample_dir = tmp_path / "frozen"
    report = build_frozen_preseason_backtest_sample(
        data_proc_dir=data_proc,
        output_dir=sample_dir,
        evaluated_season=2025,
        cutoff_date="2025-10-01",
    )
    state = pd.read_csv(sample_dir / "frozen_preseason_player_state_rows.csv")
    actuals = pd.read_csv(sample_dir / "frozen_preseason_actual_outcomes.csv")

    assert report["manifest"]["eligible_input_rows"] == 1
    assert pd.to_datetime(state["max_source_date"]).max() < pd.Timestamp("2025-10-01")
    assert "actual_pts" not in state.columns
    assert "actual_pts" in actuals.columns

    audit = audit_frozen_sample_leakage(
        frozen_sample_path=sample_dir / "frozen_preseason_player_state_rows.csv",
        actual_outcomes_path=sample_dir / "frozen_preseason_actual_outcomes.csv",
        output_dir=sample_dir,
        cutoff_date="2025-10-01",
    )
    assert audit["status"] == "LEAKAGE_AUDIT_PASSED"


def test_discovery_finds_pre_cutoff_logs_and_backfill_excludes_post_cutoff(tmp_path: Path) -> None:
    data_proc = _write_historical_logs(tmp_path)
    actuals = tmp_path / "actuals.csv"
    pd.DataFrame([{"player": "Sample Player", "player_id": "p1", "actual_pts": 14}]).to_csv(actuals, index=False)
    discovery = discover_pre_cutoff_player_logs(
        output_dir=tmp_path / "discovery",
        cutoff_date="2025-10-01",
        search_root=[data_proc],
    )
    assert discovery["report"]["usable_source_count"] >= 1

    manifest = backfill_pre_cutoff_player_state_logs(
        output_dir=tmp_path / "frozen",
        discovery_csv=Path(discovery["sources_csv"]),
        actual_outcomes=actuals,
        cutoff_date="2025-10-01",
        min_games=10,
    )
    state = pd.read_csv(tmp_path / "frozen" / "frozen_preseason_player_state_rows.csv")

    assert manifest["eligible_frozen_input_rows"] == 1
    assert pd.to_datetime(state["max_input_game_date"]).max() < pd.Timestamp("2025-10-01")
    assert "actual_pts" not in state.columns


def test_backfill_marks_insufficient_history_ineligible(tmp_path: Path) -> None:
    data_proc = _write_historical_logs(tmp_path)
    actuals = tmp_path / "actuals.csv"
    pd.DataFrame(
        [
            {"player": "Sample Player", "player_id": "p1", "actual_pts": 14},
            {"player": "Missing Player", "player_id": "p2", "actual_pts": 5},
        ]
    ).to_csv(actuals, index=False)
    discovery = discover_pre_cutoff_player_logs(output_dir=tmp_path / "discovery", cutoff_date="2025-10-01", search_root=[data_proc])
    manifest = backfill_pre_cutoff_player_state_logs(
        output_dir=tmp_path / "frozen",
        discovery_csv=Path(discovery["sources_csv"]),
        actual_outcomes=actuals,
        cutoff_date="2025-10-01",
        min_games=20,
    )
    ineligible = pd.read_csv(tmp_path / "frozen" / "frozen_preseason_ineligible_players.csv")

    assert manifest["eligible_frozen_input_rows"] == 0
    assert set(ineligible["player"]) == {"Sample Player", "Missing Player"}


def test_leakage_audit_fails_on_post_cutoff_input(tmp_path: Path) -> None:
    state = pd.DataFrame(
        [{"player": "Leaky Player", "max_source_date": "2025-10-02", "games_available_before_cutoff": 1}]
    )
    actuals = pd.DataFrame([{"player": "Leaky Player", "actual_pts": 10}])
    state_path = tmp_path / "state.csv"
    actual_path = tmp_path / "actual.csv"
    state.to_csv(state_path, index=False)
    actuals.to_csv(actual_path, index=False)

    try:
        audit_frozen_sample_leakage(
            frozen_sample_path=state_path,
            actual_outcomes_path=actual_path,
            output_dir=tmp_path,
            cutoff_date="2025-10-01",
        )
    except SystemExit as exc:
        assert "BACKTEST_FAILED_LEAKAGE" in str(exc)
    else:
        raise AssertionError("Expected leakage audit to fail")


def test_simulation_runs_from_frozen_sample_and_reports_calibration(tmp_path: Path) -> None:
    data_proc = _write_historical_logs(tmp_path)
    sample_dir = tmp_path / "frozen"
    build_frozen_preseason_backtest_sample(
        data_proc_dir=data_proc,
        output_dir=sample_dir,
        evaluated_season=2025,
        cutoff_date="2025-10-01",
    )
    report = simulate_next_season_player_states(
        data_proc_dir=data_proc,
        output_dir=tmp_path / "sim",
        cutoff_date="2025-10-01",
        cutoff_mode="preseason",
        input_frozen_sample=sample_dir / "frozen_preseason_player_state_rows.csv",
        actual_outcomes=sample_dir / "frozen_preseason_actual_outcomes.csv",
        backtest_season=2025,
        simulation_count=300,
        seed=7,
    )

    assert report["frozen_sample_mode"] is True
    assert report["backtest"]["joined_rows"] > 0
    assert "actual_within_p10_p90_rate" in report["backtest"]
    assert (tmp_path / "sim" / "simulation_backtest_2025_preseason_rows.csv").exists()


def test_credibility_gate_blocks_missing_leakage_and_undercoverage(tmp_path: Path) -> None:
    missing_gate = evaluate_simulation_credibility(backtest_report_path=None, output_dir=tmp_path / "missing")
    assert missing_gate["status"] == "BACKTEST_REQUIRED"

    leakage = tmp_path / "leakage.json"
    leakage.write_text('{"status":"BACKTEST_FAILED_LEAKAGE","failures":["input_Date_on_or_after_cutoff"]}', encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text('{"status":"BACKTEST_EVALUATED","joined_rows":200,"actual_within_p10_p90_rate":0.90,"confidence_tier_reliability":{"LOW_CONFIDENCE":{"p10_p90_coverage":0.8},"MEDIUM_CONFIDENCE":{"p10_p90_coverage":0.9}}}', encoding="utf-8")
    leak_gate = evaluate_simulation_credibility(backtest_report_path=backtest, leakage_audit_path=leakage, output_dir=tmp_path / "leak_gate")
    assert leak_gate["status"] == "BACKTEST_FAILED_LEAKAGE"

    weak = tmp_path / "weak.json"
    weak.write_text('{"status":"BACKTEST_EVALUATED","joined_rows":200,"actual_within_p10_p90_rate":0.50,"confidence_tier_reliability":{"LOW_CONFIDENCE":{"p10_p90_coverage":0.5},"MEDIUM_CONFIDENCE":{"p10_p90_coverage":0.6}}}', encoding="utf-8")
    weak_gate = evaluate_simulation_credibility(backtest_report_path=weak, output_dir=tmp_path / "weak_gate")
    assert weak_gate["status"] == "BACKTEST_FAILED_CALIBRATION"
