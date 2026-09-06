import csv
import hashlib
import json
from pathlib import Path

from sports.mlb.scripts.run_sequential_pa_hitter_model import (
    SNAPSHOT_SCHEMA_VERSION,
    capture_pregame_feature_snapshot,
)


def _write_pool(path: Path) -> None:
    fields = [
        "Game_ID",
        "Player",
        "Player_Type",
        "Player_MLBAM_ID",
        "Team",
        "Opponent",
        "Opposing_Pitcher",
        "Opposing_Pitcher_ID",
        "Batting_Order",
        "Target",
        "Market_Line",
        "Market_Over_Price",
        "Market_Under_Price",
        "Market_Source",
        "Sequential_PA_Raw_Probability",
        "Game_Conditioned_Prior_Probability",
        "Game_Conditioned_Candidate_Probability",
        "Game_Conditioned_Production_Probability",
        "Game_Conditioned_Probability_LCB",
        "Game_Conditioned_Residual_Logit",
        "Game_Conditioned_Evidence_Strength",
        "Sequential_PA_Uncertainty",
        "Sequential_PA_Support",
        "Sequential_PA_Status",
        "Game_Conditioned_Authority",
        "Game_Conditioned_Expert_Weights",
        "Game_Conditioned_Expert_Signals",
        "Game_Conditioned_Expert_Activations",
        "Game_Conditioned_Expert_Contributions",
        "Sequential_PA_Diagnostics",
        # Deliberate leakage bait. Snapshot capture must ignore these.
        "Actual_H",
        "Actual_TB",
        "Settlement",
        "Realized_Return",
    ]
    diagnostics = {
        "game_conditioned": {
            "state": {
                "batter_k_rate": 0.18,
                "pitcher_k_rate": 0.31,
                "pitch_compatibility_score": 0.27,
                "batter_handedness": "L",
                "pitcher_handedness": "R",
                "temperature_f": 79.0,
            },
            "pitch_compatibility": {
                "score": 0.27,
                "matched_pitch_types": ["FF", "SL"],
            },
        }
    }
    row = {
        "Game_ID": "gid-1",
        "Player": "Test Hitter",
        "Player_Type": "hitter",
        "Player_MLBAM_ID": "111",
        "Team": "AAA",
        "Opponent": "BBB",
        "Opposing_Pitcher": "Test Pitcher",
        "Opposing_Pitcher_ID": "222",
        "Batting_Order": "2",
        "Target": "H",
        "Market_Line": "0.5",
        "Market_Over_Price": "-165",
        "Market_Under_Price": "+130",
        "Market_Source": "real",
        "Sequential_PA_Raw_Probability": "0.68",
        "Game_Conditioned_Prior_Probability": "0.64",
        "Game_Conditioned_Candidate_Probability": "0.66",
        "Game_Conditioned_Production_Probability": "0.64",
        "Game_Conditioned_Probability_LCB": "0.61",
        "Game_Conditioned_Residual_Logit": "0.08",
        "Game_Conditioned_Evidence_Strength": "0.84",
        "Sequential_PA_Uncertainty": "0.18",
        "Sequential_PA_Support": "0.90",
        "Sequential_PA_Status": "READY",
        "Game_Conditioned_Authority": "SHADOW_ONLY_NO_PRODUCTION_AUTHORITY",
        "Game_Conditioned_Expert_Weights": json.dumps({"strikeout_contact": 0.4, "contact_quality": 0.6}),
        "Game_Conditioned_Expert_Signals": json.dumps({"strikeout_contact": 0.3, "contact_quality": 0.2}),
        "Game_Conditioned_Expert_Activations": json.dumps({"strikeout_contact": 1.4, "contact_quality": 1.1}),
        "Game_Conditioned_Expert_Contributions": json.dumps({"strikeout_contact": 0.04, "contact_quality": 0.04}),
        "Sequential_PA_Diagnostics": json.dumps(diagnostics),
        "Actual_H": "1",
        "Actual_TB": "2",
        "Settlement": "WIN",
        "Realized_Return": "0.61",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        # Non-hitter rows must not enter this feature evidence stream.
        writer.writerow({**row, "Player": "Test Pitcher Prop", "Player_Type": "pitcher", "Target": "K"})


def _report() -> dict:
    return {
        "model_version": "game_conditioned_hitter_moe_v2",
        "structural_model_version": "sequential_pa_v1",
        "data_freshness_status": "FRESH",
        "advanced_manifest": {
            "effective_as_of_date": "2026-09-04",
            "sources": ["Baseball Savant / Statcast", "FanGraphs"],
        },
        "model_artifact": {
            "training_status": "FITTED_DIAGNOSTIC",
            "evidence_class": "ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION",
        },
    }


def test_snapshot_is_content_addressed_outcome_free_and_exact_state_preserving(tmp_path):
    pool = tmp_path / "pool.csv"
    root = tmp_path / "history"
    _write_pool(pool)

    path = capture_pregame_feature_snapshot(
        pool_csv=pool,
        report=_report(),
        run_date="2026-09-05",
        snapshot_root=root,
        captured_at_utc="2026-09-05T22:30:00+00:00",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert payload["evidence_class"] == "EXACT_PREGAME_FEATURE_SNAPSHOT_UNSETTLED"
    assert payload["outcomes_included"] is False
    assert payload["settlement_included"] is False
    assert payload["row_count"] == 1
    assert path.parent == root / "2026-09-05" / "game_conditioned_pregame"

    captured = payload["rows"][0]
    assert captured["player"] == "Test Hitter"
    assert captured["target"] == "H"
    assert captured["prior_probability"] == 0.64
    assert captured["game_state"]["pitch_compatibility_score"] == 0.27
    assert captured["game_state"]["batter_handedness"] == "L"
    assert captured["pitch_compatibility"]["matched_pitch_types"] == ["FF", "SL"]

    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in ("Actual_H", "Actual_TB", "Settlement", "Realized_Return"):
        assert forbidden not in serialized

    without_hash = dict(payload)
    expected_hash = without_hash.pop("snapshot_sha256")
    canonical = json.dumps(without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == expected_hash


def test_identical_capture_is_idempotent_and_never_overwrites(tmp_path):
    pool = tmp_path / "pool.csv"
    root = tmp_path / "history"
    _write_pool(pool)
    kwargs = {
        "pool_csv": pool,
        "report": _report(),
        "run_date": "2026-09-05",
        "snapshot_root": root,
        "captured_at_utc": "2026-09-05T22:30:00+00:00",
    }

    first = capture_pregame_feature_snapshot(**kwargs)
    first_bytes = first.read_bytes()
    second = capture_pregame_feature_snapshot(**kwargs)

    assert second == first
    assert second.read_bytes() == first_bytes
    assert len(list(first.parent.glob("*.json"))) == 1
