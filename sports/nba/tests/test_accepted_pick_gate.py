from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

import decision_engine.accepted_pick_gate as accepted_pick_gate
import post_process_market_plays as ppm
from decision_engine.accepted_pick_gate import apply_accepted_pick_gate
from post_process_market_plays import compute_final_board


def _gate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Gate Anchor",
                "market_player_raw": "Gate Anchor",
                "target": "PTS",
                "direction": "UNDER",
                "prediction": 22.0,
                "market_line": 24.5,
                "edge": -2.5,
                "abs_edge": 2.5,
                "gap_percentile": 0.95,
                "recommendation": "strong",
                "expected_win_rate": 0.62,
                "expected_push_rate": 0.0,
                "confidence_score": 0.22,
                "belief_uncertainty": 0.82,
                "feasibility": 0.91,
                "market_books": 6.0,
                "baseline": 23.0,
                "baseline_edge": -1.5,
                "uncertainty_sigma": 3.6,
                "spike_probability": 0.15,
                "history_rows": 95,
                "market_date": "2026-05-01",
                "last_history_date": "2026-04-30",
                "market_event_id": "g1",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "game_key": "g1",
            },
            {
                "player": "Gate Tail",
                "market_player_raw": "Gate Tail",
                "target": "AST",
                "direction": "UNDER",
                "prediction": 5.1,
                "market_line": 6.0,
                "edge": -0.9,
                "abs_edge": 0.9,
                "gap_percentile": 0.89,
                "recommendation": "consider",
                "expected_win_rate": 0.58,
                "expected_push_rate": 0.0,
                "confidence_score": 0.18,
                "belief_uncertainty": 0.86,
                "feasibility": 0.88,
                "market_books": 6.0,
                "baseline": 5.4,
                "baseline_edge": -0.6,
                "uncertainty_sigma": 1.7,
                "spike_probability": 0.10,
                "history_rows": 88,
                "market_date": "2026-05-01",
                "last_history_date": "2026-04-30",
                "market_event_id": "g2",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "game_key": "g2",
            },
        ]
    )


def _install_gate_model_stubs(monkeypatch) -> None:
    monkeypatch.setattr(accepted_pick_gate, "load_gate_model_from_payload", lambda payload: {"payload": payload})

    def fake_predict(_model, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        keep_prob = np.where(frame["player"].eq("Gate Anchor"), 0.93, 0.18).astype("float64")
        harm = np.where(frame["player"].eq("Gate Anchor"), 0.05, 0.85).astype("float64")
        return keep_prob, harm

    monkeypatch.setattr(accepted_pick_gate, "predict_keep_harm_scores", fake_predict)

    def fake_shadow_policy(
        frame: pd.DataFrame,
        *,
        keep_prob_col: str,
        threshold: float,
        **_: object,
    ) -> pd.DataFrame:
        out = frame.copy()
        keep_prob = pd.to_numeric(out[keep_prob_col], errors="coerce").fillna(0.0)
        veto = keep_prob < float(threshold)
        out["gate_veto"] = veto
        out["gate_harm_score"] = pd.to_numeric(out.get("accepted_pick_gate_harm_score"), errors="coerce").fillna(0.0)
        out["gate_veto_reason"] = veto.map(lambda flagged: "shadow_veto" if flagged else "")
        return out

    monkeypatch.setattr(accepted_pick_gate, "apply_shadow_gate_policy", fake_shadow_policy)


def test_accepted_pick_gate_prefers_oof_control_and_falls_back_to_shadow(monkeypatch) -> None:
    _install_gate_model_stubs(monkeypatch)

    payload = {
        "threshold": 0.50,
        "shadow_only": False,
        "live_ready": True,
        "promotion_recommendation": {"pass": True, "failures": []},
        "oof_promotion_recommendation": {"pass": False, "failures": ["rolling_window_pass_rate_below_floor"]},
        "model": {"model_type": "unit_test_gate"},
    }

    scored, details = apply_accepted_pick_gate(_gate_frame(), payload, run_date_hint="2026-05-01", live=True)

    assert len(scored) == 2
    assert bool(details["live"]) is False
    assert str(details["live_guard_reason"]) == "adaptive_live_control_failed"
    assert str(details["adaptive_live_control"]["source"]) == "month_payload.oof_promotion_recommendation"
    assert details["adaptive_live_control"]["pass"] is False
    assert bool(scored["accepted_pick_gate_live"].any()) is False
    assert str(scored.loc[scored["player"] == "Gate Tail", "accepted_pick_gate_live_guard_reason"].iloc[0]) == "adaptive_live_control_failed"
    assert bool(scored.loc[scored["player"] == "Gate Tail", "accepted_pick_gate_veto"].iloc[0]) is True


def test_accepted_pick_gate_goes_live_when_adaptive_control_passes(monkeypatch) -> None:
    _install_gate_model_stubs(monkeypatch)

    payload = {
        "threshold": 0.50,
        "shadow_only": False,
        "live_ready": True,
        "adaptive_live_control": {"enabled": True, "source": "unit_test", "pass": True, "failures": []},
        "model": {"model_type": "unit_test_gate"},
    }

    scored, details = apply_accepted_pick_gate(_gate_frame(), payload, run_date_hint="2026-05-01", live=True)

    assert len(scored) == 1
    assert bool(details["live"]) is True
    assert int(details["drop_rows"]) == 1
    assert str(scored.iloc[0]["player"]) == "Gate Anchor"
    assert bool(scored.iloc[0]["accepted_pick_gate_live"]) is True
    assert float(scored.iloc[0]["accepted_pick_gate_live_control_pass"]) == 1.0


def test_compute_final_board_preserves_baseline_when_gate_auto_shadows(monkeypatch) -> None:
    _install_gate_model_stubs(monkeypatch)
    monkeypatch.setattr(ppm, "apply_accepted_pick_gate_fn", apply_accepted_pick_gate)

    plays = _gate_frame()
    payload = {
        "threshold": 0.50,
        "shadow_only": False,
        "live_ready": True,
        "oof_promotion_recommendation": {"pass": False, "failures": ["recent_profit_delta_below_floor"]},
        "model": {"model_type": "unit_test_gate"},
    }

    baseline = compute_final_board(
        plays,
        max_total_plays=2,
        max_plays_per_player=1,
        max_plays_per_game=1,
        max_plays_per_script_cluster=1,
        selection_mode="edge",
        ranking_mode="edge",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
    )
    guarded = compute_final_board(
        plays,
        max_total_plays=2,
        max_plays_per_player=1,
        max_plays_per_game=1,
        max_plays_per_script_cluster=1,
        selection_mode="edge",
        ranking_mode="edge",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
        accepted_pick_gate_payload=payload,
        accepted_pick_gate_enabled=True,
        accepted_pick_gate_live=True,
    )

    assert list(guarded["player"]) == list(baseline["player"])
    assert bool(guarded["accepted_pick_gate_live"].any()) is False
    assert "adaptive_live_control_failed" in set(guarded["accepted_pick_gate_live_guard_reason"])
    assert int(pd.to_numeric(guarded["accepted_pick_gate_drop_count"], errors="coerce").fillna(0).max()) == 0
