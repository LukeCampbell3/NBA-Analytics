from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.backfill_distribution_quantiles import backfill_distribution_quantiles
from research.safe_state.backfill_minutes_state import backfill_minutes_state
from research.safe_state.build_forecastability_gap import annotate_forecastability_gaps
from research.safe_state.build_similar_state_store import build_similar_state_features, build_similar_state_store
from research.safe_state.safe_state_classifier import annotate_safe_state


def _candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate::test",
        "game_id": "game_current",
        "market_date": "2026-05-20",
        "game_date": "2026-05-20",
        "player": "Test_Player",
        "player_name": "Test_Player",
        "target": "PTS",
        "market_type": "PTS_OVER",
        "side": "OVER",
        "direction": "OVER",
        "line": 20.5,
        "market_line": 20.5,
        "edge_defendability_tier": "EDGE_DEFENDABLE",
        "price_validity_status": "PRICE_VALID",
        "stress_edge": 0.06,
        "lcb_edge": 0.03,
        "stress_probability": 0.59,
        "lcb_probability": 0.55,
        "scenario_agreement": 0.80,
        "chaos_score": 0.20,
    }
    row.update(overrides)
    return row


def _write_logs(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    data_proc = tmp_path / "Data-Proc"
    player_dir = data_proc / "test_player"
    player_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(player_dir / "2026_processed_processed.csv", index=False)
    return data_proc


def _log_rows(minutes: list[float], points: list[float], *, include_current: bool = True) -> list[dict[str, object]]:
    dates = pd.date_range("2026-05-01", periods=len(minutes), freq="D")
    rows = [
        {
            "Date": date.strftime("%Y-%m-%d"),
            "Player": "Test_Player",
            "MP": mp,
            "PTS": pts,
            "TRB": 5,
            "AST": 4,
            "FGA": 12,
            "USG%": 22,
        }
        for date, mp, pts in zip(dates, minutes, points)
    ]
    if include_current:
        rows.append(
            {
                "Date": "2026-05-20",
                "Player": "Test_Player",
                "MP": 44,
                "PTS": 50,
                "TRB": 12,
                "AST": 10,
                "FGA": 30,
                "USG%": 40,
            }
        )
    return rows


def test_minutes_backfill_uses_only_games_before_market_date(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([30, 31, 32, 33, 34], [20, 21, 22, 23, 24]))
    out = backfill_minutes_state(pd.DataFrame([_candidate()]), data_proc_dir=data_proc)

    assert out.iloc[0]["minutes_state_sample_count"] == 5
    assert out.iloc[0]["minutes_ceiling_recent"] == 34


def test_wide_minutes_band_creates_minutes_state_gap(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([12, 34, 16, 36, 14], [20, 21, 22, 23, 24], include_current=False))
    out = backfill_minutes_state(pd.DataFrame([_candidate()]), data_proc_dir=data_proc)

    assert out.iloc[0]["minutes_state_gap_type"] == "FORECASTABILITY_GAP_MINUTES_STATE"


def test_stable_minutes_band_improves_score(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([32, 33, 34, 33, 32], [20, 21, 22, 23, 24], include_current=False))
    out = backfill_minutes_state(pd.DataFrame([_candidate()]), data_proc_dir=data_proc)

    assert out.iloc[0]["minutes_forecastability_score"] > 0.60


def test_sparse_minutes_sample_creates_insufficient_pre_event_data(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([32, 33], [20, 21], include_current=False))
    out = backfill_minutes_state(pd.DataFrame([_candidate()]), data_proc_dir=data_proc)

    assert out.iloc[0]["minutes_state_gap_type"] == "FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA"


def test_distribution_quantiles_produce_line_zone(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([32] * 8, [15, 16, 17, 18, 19, 20, 21, 22], include_current=False))
    out = backfill_distribution_quantiles(pd.DataFrame([_candidate(line=16.5, market_line=16.5)]), data_proc_dir=data_proc)

    assert out.iloc[0]["line_zone"] in {"BELOW_Q25", "NEAR_MEDIAN", "ABOVE_Q75", "EXTREME_TAIL"}
    assert pd.notna(out.iloc[0]["line_percentile_recent"])


def test_wide_distribution_creates_distribution_width_gap(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([32] * 8, [5, 35, 6, 34, 7, 33, 8, 32], include_current=False))
    out = backfill_distribution_quantiles(pd.DataFrame([_candidate()]), data_proc_dir=data_proc)

    assert out.iloc[0]["distribution_gap_type"] == "FORECASTABILITY_GAP_DISTRIBUTION_WIDTH"


def test_similar_state_store_excludes_current_game(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([32, 33, 34, 35, 36], [20, 21, 22, 23, 24], include_current=True))
    store = build_similar_state_store(pd.DataFrame([_candidate()]), data_proc_dir=data_proc)

    assert "2026-05-20" not in pd.to_datetime(store["game_date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist()


def test_sparse_similar_state_sample_creates_sample_gap(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([32, 33], [20, 21], include_current=False))
    out = build_similar_state_features(pd.DataFrame([_candidate()]), data_proc_dir=data_proc)

    assert out.iloc[0]["similar_state_gap_type"] == "FORECASTABILITY_GAP_SIMILAR_STATE_SAMPLE"


def test_scattered_similar_state_outcomes_create_scatter_gap(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path, _log_rows([32] * 8, [2, 40, 3, 39, 4, 38, 5, 37], include_current=False))
    out = build_similar_state_features(pd.DataFrame([_candidate()]), data_proc_dir=data_proc)

    assert out.iloc[0]["similar_state_gap_type"] == "FORECASTABILITY_GAP_SIMILAR_STATE_SCATTER"


def test_edge_defendable_true_unstable_state_becomes_unstable() -> None:
    row = _candidate(
        forecastability_tier="HIGH_FORECASTABILITY",
        forecastability_gap_primary="FORECASTABILITY_GAP_TRUE_UNSTABLE_STATE",
        forecastability_gap_fixability="TRUE_UNSTABLE_STATE",
        forecastability_gap_severity="CRITICAL",
        similar_state_reliability_tier="TIGHT",
        structural_mispricing_tier="STRUCTURAL_MISPRICE_STRONG",
    )
    out = annotate_safe_state(pd.DataFrame([row]))

    assert out.iloc[0]["safe_state_tier"] == "SAFE_STATE_UNSTABLE"


def test_edge_defendable_one_fixable_blocker_can_be_near_core() -> None:
    row = _candidate(
        forecastability_tier="HIGH_FORECASTABILITY",
        forecastability_gap_primary="FORECASTABILITY_GAP_TEAMMATE_CONTEXT",
        forecastability_gap_fixability="FIXABLE_WITH_NEW_PIPELINE_DATA",
        forecastability_gap_severity="LOW",
        similar_state_reliability_tier="TIGHT",
        structural_mispricing_tier="PRICE_ONLY_EDGE",
    )
    out = annotate_safe_state(pd.DataFrame([row]))

    assert out.iloc[0]["safe_state_tier"] == "SAFE_STATE_NEAR_CORE"


def test_missing_evidence_cannot_become_safe_state_core() -> None:
    row = _candidate(
        forecastability_tier="HIGH_FORECASTABILITY",
        forecastability_gap_primary="FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA",
        forecastability_gap_fixability="NEEDS_MORE_SAMPLE",
        forecastability_gap_severity="MEDIUM",
        similar_state_reliability_tier="INSUFFICIENT_SAMPLE",
        structural_mispricing_tier="STRUCTURAL_MISPRICE_STRONG",
    )
    out = annotate_safe_state(pd.DataFrame([row]))

    assert out.iloc[0]["safe_state_tier"] != "SAFE_STATE_CORE"
