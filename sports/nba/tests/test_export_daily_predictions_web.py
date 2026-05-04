from __future__ import annotations

import sys
import json
import os
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from export_daily_predictions_web import (
    _evaluate_publication_candidate,
    apply_parlay_safety_gate,
    apply_adaptive_board_sizing,
    apply_variance_aware_reexpand,
    build_nba_precision_parlay_candidates,
    build_selector_pool_fallback,
    evaluate_parlay_safety_gate,
    enrich_selector_pool_candidates,
    find_latest_manifest,
    resolve_published_board,
)


def test_build_selector_pool_fallback_keeps_only_quality_rows() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Ayo Dosunmu",
                "game_key": "g1",
                "target": "TRB",
                "direction": "OVER",
                "expected_win_rate": 0.5143,
                "ev": 0.0184,
                "final_confidence": 0.0674,
                "abs_edge": 0.7018,
            },
            {
                "player": "Josh Hart",
                "game_key": "g2",
                "target": "TRB",
                "direction": "UNDER",
                "expected_win_rate": 0.5101,
                "ev": 0.0142,
                "final_confidence": 0.0682,
                "abs_edge": 0.5885,
            },
            {
                "player": "Nickeil Alexander-Walker",
                "game_key": "g2",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.5008,
                "ev": -0.0006,
                "final_confidence": 0.1616,
                "abs_edge": 1.2571,
            },
            {
                "player": "Jalen Duren",
                "game_key": "g3",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.5006,
                "ev": -0.0017,
                "final_confidence": 0.1525,
                "abs_edge": 1.3027,
            },
        ]
    )

    fallback = build_selector_pool_fallback(plays)

    assert fallback["player"].tolist() == ["Ayo Dosunmu", "Josh Hart"]


def test_build_selector_pool_fallback_returns_empty_when_slate_is_too_weak() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Weak One",
                "game_key": "g1",
                "expected_win_rate": 0.503,
                "ev": -0.001,
                "final_confidence": 0.06,
                "abs_edge": 0.8,
            },
            {
                "player": "Weak Two",
                "game_key": "g2",
                "expected_win_rate": 0.509,
                "ev": 0.003,
                "final_confidence": 0.03,
                "abs_edge": 0.7,
            },
        ]
    )

    fallback = build_selector_pool_fallback(plays)

    assert fallback.empty


def test_build_selector_pool_fallback_applies_adaptive_sizing_then_reexpands_to_three() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Core One",
                "game_key": "g1",
                "expected_win_rate": 0.5196,
                "ev": 0.0288,
                "final_confidence": 0.14,
                "abs_edge": 1.20,
                "uncertainty_sigma": 3.0,
                "history_rows": 70,
                "spike_probability": 0.20,
                "contradiction_score": 0.05,
                "recoverability_score": 0.80,
                "agreement_count": 3,
            },
            {
                "player": "Core Two",
                "game_key": "g2",
                "expected_win_rate": 0.5195,
                "ev": 0.0287,
                "final_confidence": 0.139,
                "abs_edge": 1.19,
                "uncertainty_sigma": 3.0,
                "history_rows": 69,
                "spike_probability": 0.21,
                "contradiction_score": 0.051,
                "recoverability_score": 0.79,
                "agreement_count": 3,
            },
            {
                "player": "Expand Three",
                "game_key": "g1",
                "expected_win_rate": 0.5194,
                "ev": 0.0286,
                "final_confidence": 0.138,
                "abs_edge": 1.18,
                "uncertainty_sigma": 3.0,
                "history_rows": 68,
                "spike_probability": 0.22,
                "contradiction_score": 0.052,
                "recoverability_score": 0.78,
                "agreement_count": 3,
            },
            {
                "player": "Expand Four",
                "game_key": "g2",
                "expected_win_rate": 0.5193,
                "ev": 0.0285,
                "final_confidence": 0.137,
                "abs_edge": 1.17,
                "uncertainty_sigma": 3.0,
                "history_rows": 67,
                "spike_probability": 0.23,
                "contradiction_score": 0.053,
                "recoverability_score": 0.77,
                "agreement_count": 3,
            },
        ]
    )

    fallback = build_selector_pool_fallback(plays)

    assert fallback["player"].tolist() == ["Core One", "Core Two", "Expand Three"]


def test_enrich_selector_pool_candidates_prefers_balanced_profile() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "High Prob Fragile",
                "game_key": "g1",
                "expected_win_rate": 0.515,
                "p_calibrated": 0.500,
                "ev": 0.001,
                "final_confidence": 0.05,
                "abs_edge": 0.50,
                "uncertainty_sigma": 9.0,
                "history_rows": 25,
                "spike_probability": 0.80,
                "contradiction_score": 0.30,
                "recoverability_score": 0.20,
                "agreement_count": 1,
            },
            {
                "player": "Balanced Strong",
                "game_key": "g2",
                "expected_win_rate": 0.511,
                "p_calibrated": 0.522,
                "ev": 0.010,
                "final_confidence": 0.15,
                "abs_edge": 1.20,
                "uncertainty_sigma": 3.0,
                "history_rows": 70,
                "spike_probability": 0.20,
                "contradiction_score": 0.05,
                "recoverability_score": 0.80,
                "agreement_count": 3,
            },
        ]
    )

    enriched = enrich_selector_pool_candidates(plays).sort_values("pool_selection_score", ascending=False).reset_index(drop=True)

    assert enriched.loc[0, "player"] == "Balanced Strong"


def test_apply_adaptive_board_sizing_trims_marginal_tail() -> None:
    plays = pd.DataFrame(
        [
            {"player": "Top One", "pool_selection_score": 2.60, "selection_confidence": 0.90},
            {"player": "Top Two", "pool_selection_score": 2.42, "selection_confidence": 0.86},
            {"player": "Marginal Three", "pool_selection_score": 2.05, "selection_confidence": 0.66},
            {"player": "Marginal Four", "pool_selection_score": 1.98, "selection_confidence": 0.60},
        ]
    )

    sized = apply_adaptive_board_sizing(plays)

    assert sized["player"].tolist() == ["Top One", "Top Two"]


def test_apply_variance_aware_reexpand_adds_third_play_on_low_conviction_two_leg_board() -> None:
    universe = pd.DataFrame(
        [
            {"player": "Top One", "selection_probability": 0.530, "selection_confidence": 0.11, "selection_ev": 0.02},
            {"player": "Top Two", "selection_probability": 0.519, "selection_confidence": 0.13, "selection_ev": 0.02},
            {"player": "Third Qualifier", "selection_probability": 0.513, "selection_confidence": 0.15, "selection_ev": 0.005},
        ]
    )
    trimmed = universe.iloc[:2].copy().reset_index(drop=True)

    expanded = apply_variance_aware_reexpand(
        trimmed,
        universe,
        probability_field="selection_probability",
        confidence_field="selection_confidence",
        ev_field="selection_ev",
        max_top2_avg_probability=0.526,
        min_third_probability=0.512,
        min_third_confidence=0.12,
        min_third_ev=0.0,
    )

    assert expanded["player"].tolist() == ["Top One", "Top Two", "Third Qualifier"]


def test_publication_candidate_blocks_heuristic_fallback_board() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Heuristic Weak",
                "expected_win_rate": 0.512,
                "final_confidence": 0.06,
                "recommendation": "pass",
            }
        ]
    )

    report = _evaluate_publication_candidate(
        plays,
        {
            "run_id": "artifact_free_heuristic",
            "history_mode": "heuristic_fallback_empty_history",
        },
        source_label="primary_selector_pool_fallback",
        accuracy_metrics={"available": False},
    )

    assert report["passes"] is False
    assert "artifact_free_model" in report["reasons"]
    assert "fallback_source_not_publishable" in report["reasons"]
    assert "accuracy_metrics_unavailable" in report["reasons"]


def test_publication_candidate_allows_validated_heuristic_fallback_board() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Validated One",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.612,
                "ev": 0.151,
                "final_confidence": 0.24,
                "selection_probability": 0.612,
                "selection_ev": 0.151,
                "selection_confidence": 0.24,
                "recommendation": "elite",
                "selected_rank": 1,
            },
            {
                "player": "Validated Two",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.601,
                "ev": 0.129,
                "final_confidence": 0.19,
                "selection_probability": 0.601,
                "selection_ev": 0.129,
                "selection_confidence": 0.19,
                "recommendation": "strong",
                "selected_rank": 2,
            },
        ]
    )

    report = _evaluate_publication_candidate(
        plays,
        {
            "run_id": "artifact_free_heuristic",
            "history_mode": "historical_backtest",
        },
        source_label="primary_selector_pool_fallback",
        accuracy_metrics={
            "available": True,
            "overall": {
                "graded_count": 24,
                "win_rate": 0.67,
                "roi_per_graded_play": 0.12,
            },
        },
    )

    assert report["passes"] is True
    assert report["publication_mode"] == "validated_heuristic_fallback"
    assert report["reasons"] == []


def test_resolve_published_board_prefers_shadow_final_board_over_selector_fallback(tmp_path: Path) -> None:
    primary_final_csv = tmp_path / "final.csv"
    primary_final_json = tmp_path / "final.json"
    primary_selector_csv = tmp_path / "selector.csv"
    shadow_final_csv = tmp_path / "shadow_final.csv"
    shadow_final_json = tmp_path / "shadow_final.json"
    manifest_path = tmp_path / "manifest.json"

    pd.DataFrame(columns=["player", "expected_win_rate"]).to_csv(primary_final_csv, index=False)
    primary_selector = pd.DataFrame(
        [
            {
                "player": "Fallback Only",
                "game_key": "g1",
                "expected_win_rate": 0.511,
                "ev": 0.006,
                "final_confidence": 0.06,
                "abs_edge": 0.9,
                "recommendation": "pass",
            }
        ]
    )
    primary_selector.to_csv(primary_selector_csv, index=False)
    shadow_final = pd.DataFrame(
        [
            {
                "player": "Shadow Strong",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.548,
                "final_confidence": 0.18,
                "recommendation": "strong",
                "selected_rank": 1,
            }
        ]
    )
    shadow_final.to_csv(shadow_final_csv, index=False)

    primary_payload = {
        "run_id": "artifact_free_heuristic",
        "policy_profile": "production_board_objective_b12",
        "history_mode": "heuristic_fallback_empty_history",
    }
    shadow_payload = {
        "run_id": "shadow_real_model",
        "policy_profile": "shadow_policy",
        "history_mode": "historical_backtest",
    }
    primary_final_json.write_text(json.dumps(primary_payload), encoding="utf-8")
    shadow_final_json.write_text(json.dumps(shadow_payload), encoding="utf-8")

    manifest = {
        "policy_profile": "production_board_objective_b12",
        "current_market_snapshot": str(tmp_path / "snapshot.parquet"),
        "final_csv": str(primary_final_csv),
        "final_json": str(primary_final_json),
        "selector_csv": str(primary_selector_csv),
        "shadow_runs": [
            {
                "policy_profile": "shadow_policy",
                "final_csv": str(shadow_final_csv),
                "final_json": str(shadow_final_json),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plays, final_payload, source, publication_gate, parlay_plays, parlay_source = resolve_published_board(
        manifest,
        manifest_path,
        accuracy_metrics={"available": True},
    )

    assert source == "shadow_final_board"
    assert plays["player"].tolist() == ["Shadow Strong"]
    assert final_payload["policy_profile"] == "shadow_policy"
    assert publication_gate["status"] == "ready"
    assert parlay_source == ""
    assert len(parlay_plays.index) == 0


def test_resolve_published_board_publishes_validated_primary_selector_fallback(tmp_path: Path) -> None:
    primary_final_csv = tmp_path / "final.csv"
    primary_final_json = tmp_path / "final.json"
    primary_selector_csv = tmp_path / "selector.csv"
    manifest_path = tmp_path / "manifest.json"

    pd.DataFrame(columns=["player", "expected_win_rate"]).to_csv(primary_final_csv, index=False)
    primary_selector = pd.DataFrame(
        [
            {
                "player": "Validated One",
                "game_key": "g1",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.612,
                "ev": 0.151,
                "final_confidence": 0.24,
                "abs_edge": 1.2,
            },
            {
                "player": "Validated Two",
                "game_key": "g2",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.601,
                "ev": 0.129,
                "final_confidence": 0.19,
                "abs_edge": 1.0,
            },
        ]
    )
    primary_selector.to_csv(primary_selector_csv, index=False)

    primary_payload = {
        "run_id": "artifact_free_heuristic",
        "policy_profile": "production_board_objective_b12",
        "history_mode": "historical_backtest",
    }
    primary_final_json.write_text(json.dumps(primary_payload), encoding="utf-8")

    manifest = {
        "policy_profile": "production_board_objective_b12",
        "current_market_snapshot": str(tmp_path / "snapshot.parquet"),
        "final_csv": str(primary_final_csv),
        "final_json": str(primary_final_json),
        "selector_csv": str(primary_selector_csv),
        "shadow_runs": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plays, final_payload, source, publication_gate, parlay_plays, parlay_source = resolve_published_board(
        manifest,
        manifest_path,
        accuracy_metrics={
            "available": True,
            "overall": {
                "graded_count": 24,
                "win_rate": 0.67,
                "roi_per_graded_play": 0.12,
            },
        },
    )

    assert source == "primary_selector_pool_fallback"
    assert final_payload["policy_profile"] == "production_board_objective_b12"
    assert publication_gate["status"] == "ready"
    assert publication_gate["publication_mode"] == "validated_heuristic_fallback"
    assert plays["selected_rank"].tolist() == [1, 2]
    assert set(plays["recommendation"].tolist()) == {"elite"}
    assert parlay_source == "primary_selector_pool_fallback"
    assert len(parlay_plays.index) == 2


def test_resolve_published_board_prefers_trained_shadow_final_board_over_validated_primary_fallback(tmp_path: Path) -> None:
    primary_final_csv = tmp_path / "final.csv"
    primary_final_json = tmp_path / "final.json"
    primary_selector_csv = tmp_path / "selector.csv"
    shadow_final_csv = tmp_path / "shadow_final.csv"
    shadow_final_json = tmp_path / "shadow_final.json"
    manifest_path = tmp_path / "manifest.json"

    pd.DataFrame(columns=["player", "expected_win_rate"]).to_csv(primary_final_csv, index=False)
    pd.DataFrame(
        [
            {
                "player": "Fallback One",
                "game_key": "g1",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.612,
                "ev": 0.151,
                "final_confidence": 0.24,
                "abs_edge": 1.2,
            },
            {
                "player": "Fallback Two",
                "game_key": "g2",
                "target": "AST",
                "direction": "OVER",
                "expected_win_rate": 0.601,
                "ev": 0.129,
                "final_confidence": 0.19,
                "abs_edge": 1.0,
            },
        ]
    ).to_csv(primary_selector_csv, index=False)
    pd.DataFrame(
        [
            {
                "player": "Shadow Anchor",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.552,
                "final_confidence": 0.21,
                "recommendation": "elite",
                "selected_rank": 1,
            },
            {
                "player": "Shadow Pair",
                "target": "TRB",
                "direction": "UNDER",
                "expected_win_rate": 0.545,
                "final_confidence": 0.18,
                "recommendation": "strong",
                "selected_rank": 2,
            },
        ]
    ).to_csv(shadow_final_csv, index=False)

    primary_payload = {
        "run_id": "artifact_free_heuristic",
        "policy_profile": "production_board_objective_b12",
        "history_mode": "historical_backtest",
    }
    shadow_payload = {
        "run_id": "surrogate_tabular_v1",
        "policy_profile": "shadow_policy",
        "history_mode": "historical_backtest",
    }
    primary_final_json.write_text(json.dumps(primary_payload), encoding="utf-8")
    shadow_final_json.write_text(json.dumps(shadow_payload), encoding="utf-8")

    manifest = {
        "policy_profile": "production_board_objective_b12",
        "current_market_snapshot": str(tmp_path / "snapshot.parquet"),
        "final_csv": str(primary_final_csv),
        "final_json": str(primary_final_json),
        "selector_csv": str(primary_selector_csv),
        "shadow_runs": [
            {
                "policy_profile": "shadow_policy",
                "final_csv": str(shadow_final_csv),
                "final_json": str(shadow_final_json),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plays, final_payload, source, publication_gate, parlay_plays, parlay_source = resolve_published_board(
        manifest,
        manifest_path,
        accuracy_metrics={
            "available": True,
            "overall": {
                "graded_count": 24,
                "win_rate": 0.67,
                "roi_per_graded_play": 0.12,
            },
        },
    )

    assert source == "shadow_final_board"
    assert final_payload["policy_profile"] == "shadow_policy"
    assert publication_gate["status"] == "ready"
    assert plays["player"].tolist() == ["Shadow Anchor", "Shadow Pair"]
    assert parlay_source == "shadow_final_board"
    assert len(parlay_plays.index) == 2


def test_find_latest_manifest_prefers_newer_run_stamp_over_mtime(tmp_path: Path) -> None:
    older_manifest = tmp_path / "20260427" / "daily_market_pipeline_manifest_20260427.json"
    newer_manifest = tmp_path / "20260428" / "daily_market_pipeline_manifest_20260428.json"
    older_manifest.parent.mkdir(parents=True, exist_ok=True)
    newer_manifest.parent.mkdir(parents=True, exist_ok=True)
    older_manifest.write_text("{}", encoding="utf-8")
    newer_manifest.write_text("{}", encoding="utf-8")

    os.utime(newer_manifest, (100, 100))
    os.utime(older_manifest, (200, 200))

    assert find_latest_manifest(tmp_path) == newer_manifest


def test_apply_parlay_safety_gate_blocks_unsupported_pairs() -> None:
    parlay_payload = {
        "plays": [
            {
                "player": "Alpha Guard",
                "parlay_tag": "parlay",
                "parlay_candidate": True,
                "parlay_pair_rank": 1,
                "parlay_score": 0.31,
                "parlay_projected_hit_rate": 0.56,
                "parlay_partner_key": "beta",
                "parlay_partner_name": "Beta Wing",
            },
            {
                "player": "Beta Wing",
                "parlay_tag": "parlay",
                "parlay_candidate": True,
                "parlay_pair_rank": 1,
                "parlay_score": 0.31,
                "parlay_projected_hit_rate": 0.56,
                "parlay_partner_key": "alpha",
                "parlay_partner_name": "Alpha Guard",
            },
        ],
        "pairs": [{"pair_rank": 1, "projected_probability": 0.56}],
        "summary": {
            "selection_mode": "strict",
            "selected_pair_count": 1,
            "tagged_play_count": 2,
            "avg_projected_pair_hit_rate": 0.56,
            "best_projected_pair_hit_rate": 0.56,
        },
    }
    validation = {
        "available": True,
        "sample_dates": 3,
        "selected": {
            "graded_pair_count": 3,
            "hit_pair_count": 1,
            "pair_hit_rate": 0.3333333333,
            "avg_projected_pair_hit_rate": 0.5459771766,
        },
        "baseline_all_pairs": {
            "pair_hit_rate": 0.3278168215,
        },
        "hit_rate_lift_vs_all_pairs": 0.0055,
    }

    gate = evaluate_parlay_safety_gate(validation)
    gated_payload, applied_gate = apply_parlay_safety_gate(parlay_payload, validation)

    assert gate["passed"] is False
    assert applied_gate["passed"] is False
    assert "insufficient_sample_dates" in gate["blockers"]
    assert "insufficient_pair_lift" in gate["blockers"]
    assert gated_payload["summary"]["selection_mode"] == "empirical_gate_blocked"
    assert gated_payload["summary"]["selected_pair_count"] == 0
    assert all(play["parlay_candidate"] is False for play in gated_payload["plays"])
    assert gated_payload["pairs"] == []


def test_evaluate_parlay_safety_gate_uses_empirical_lift_and_projection_control() -> None:
    validation = {
        "available": True,
        "sample_dates": 12,
        "selected": {
            "graded_pair_count": 12,
            "hit_pair_count": 9,
            "pair_hit_rate": 9 / 12,
            "avg_projected_pair_hit_rate": 0.77,
        },
        "baseline_all_pairs": {
            "pair_hit_rate": 0.70,
        },
        "hit_rate_lift_vs_all_pairs": 0.05,
    }

    gate = evaluate_parlay_safety_gate(validation)

    assert gate["passed"] is True
    assert gate["pair_hit_rate"] == 0.75
    assert gate["overprojection_gap"] <= gate["thresholds"]["max_overprojection_gap"]
    assert "min_pair_hit_rate" not in gate["thresholds"]


def test_build_nba_precision_parlay_candidates_keeps_only_supported_legs() -> None:
    plays = [
        {
            "player": "Trusted Under",
            "target": "PTS",
            "direction": "UNDER",
            "expected_win_rate": 0.611,
            "final_confidence": 0.18,
            "ev": 0.03,
        },
        {
            "player": "Weak Over",
            "target": "PTS",
            "direction": "OVER",
            "expected_win_rate": 0.624,
            "final_confidence": 0.19,
            "ev": 0.04,
        },
        {
            "player": "Low Confidence Under",
            "target": "AST",
            "direction": "UNDER",
            "expected_win_rate": 0.608,
            "final_confidence": 0.08,
            "ev": 0.02,
        },
        {
            "player": "Trusted Ast Under",
            "target": "AST",
            "direction": "UNDER",
            "expected_win_rate": 0.606,
            "final_confidence": 0.17,
            "ev": 0.01,
        },
    ]
    validation = {
        "available": True,
        "leg_segments": {
            "PTS|UNDER": {"rows": 87, "hit_rate": 0.718},
            "PTS|OVER": {"rows": 133, "hit_rate": 0.368},
            "AST|UNDER": {"rows": 91, "hit_rate": 0.813},
        },
    }

    prepared, summary = build_nba_precision_parlay_candidates(plays, validation)
    eligible = [play["player"] for play in prepared if play["parlay_precision_eligible"]]

    assert eligible == ["Trusted Under", "Trusted Ast Under"]
    assert summary["precision_pre_cap_candidate_count"] == 2
    assert summary["precision_eligible_play_count"] == 2


def test_build_nba_precision_parlay_candidates_allows_elite_near_miss_segment_support() -> None:
    plays = [
        {
            "player": "Elite Ast Under",
            "target": "AST",
            "direction": "UNDER",
            "expected_win_rate": 0.609,
            "final_confidence": 0.17,
            "ev": 0.02,
        },
        {
            "player": "Borderline Pts Under",
            "target": "PTS",
            "direction": "UNDER",
            "expected_win_rate": 0.608,
            "final_confidence": 0.17,
            "ev": 0.02,
        },
    ]
    validation = {
        "available": True,
        "leg_segments": {
            "AST|UNDER": {"rows": 64, "hit_rate": 0.789},
            "PTS|UNDER": {"rows": 60, "hit_rate": 0.679},
        },
    }

    prepared, summary = build_nba_precision_parlay_candidates(plays, validation)
    eligible = [play["player"] for play in prepared if play["parlay_precision_eligible"]]
    support_ok = {play["player"]: play["parlay_precision_segment_support_ok"] for play in prepared}

    assert eligible == ["Elite Ast Under"]
    assert support_ok["Elite Ast Under"] is True
    assert support_ok["Borderline Pts Under"] is False
    assert summary["precision_near_miss_segment_rows"] == 60
    assert summary["precision_elite_segment_hit_rate"] == 0.78
