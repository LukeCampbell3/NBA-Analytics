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
