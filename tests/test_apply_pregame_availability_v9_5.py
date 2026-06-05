import json
import subprocess
from pathlib import Path

import pandas as pd


def test_apply_pregame_availability_uses_probability_weighted_deltas(tmp_path: Path):
    source = tmp_path / "v9_4"
    data = source / "data"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "player": ["Target_Player"],
            "date": ["2026-01-03"],
            "market": ["PTS"],
            "line": [12.5],
            "model_mean": [10.0],
            "v92_model_mean": [10.0],
            "sigma": [4.0],
            "v92_sigma": [4.0],
            "p_over_raw": [0.25],
        }
    ).to_csv(data / "prop_training_rows.csv", index=False)
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps({"model_version": "prop_engine_v9_4_lineup_delta_ready_distribution", "output": str(source)}),
        encoding="utf-8",
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pd.DataFrame(
        {
            "player": ["Target_Player"],
            "market": ["PTS"],
            "team": ["AAA"],
            "teammate": ["Teammate_A"],
            "teammate_id": ["2"],
            "baseline_rate": [12.0],
            "shrunk_delta": [4.0],
            "confidence": [0.5],
            "first_shared_date": ["2026-01-01"],
            "last_shared_date": ["2026-01-05"],
        }
    ).to_csv(artifacts / "player_teammate_out_deltas.csv", index=False)

    availability = tmp_path / "availability.csv"
    pd.DataFrame(
        {
            "snapshot_time": ["2026-01-03T18:00:00Z"],
            "game_start_time": ["2026-01-04T00:00:00Z"],
            "date": ["2026-01-03"],
            "team": ["AAA"],
            "player": ["Teammate_A"],
            "status": ["questionable"],
            "out_probability": [0.5],
            "availability_confidence": [0.8],
            "source": ["test"],
        }
    ).to_csv(availability, index=False)

    output = tmp_path / "v9_5"
    subprocess.run(
        [
            "python",
            "Player-Predictor/scripts/apply_pregame_availability_v9_5.py",
            "--source-manifest",
            str(manifest),
            "--availability-snapshots",
            str(availability),
            "--lineup-artifacts",
            str(artifacts),
            "--output",
            str(output),
        ],
        check=True,
    )
    rows = pd.read_csv(output / "data" / "prop_training_rows.csv")
    assert rows.loc[0, "pregame_lineup_adjustment"] == 1.6
    assert rows.loc[0, "pregame_teammate_out_expected_count"] == 0.4
    assert rows.loc[0, "p_over_raw"] > rows.loc[0, "p_over_raw_v94_safe"]
    manifest_out = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_out["status"] == "pregame_lineup_shadow_candidate"
    assert manifest_out["pregame_lineup_application"]["lineup_field_safety"]["oracle_fields_present"] is False
