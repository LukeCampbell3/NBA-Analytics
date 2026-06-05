import json
import subprocess
from pathlib import Path

import pandas as pd


def test_apply_lineup_deltas_writes_oracle_adjusted_rows(tmp_path: Path):
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

    logs = pd.DataFrame(
        [
            _log("G1", "2026-01-01", "1", "Target Player"),
            _log("G1", "2026-01-01", "2", "Teammate A"),
            _log("G2", "2026-01-03", "1", "Target Player"),
        ]
    )
    logs_path = tmp_path / "logs.csv"
    logs.to_csv(logs_path, index=False)

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pd.DataFrame(
        {
            "player": ["Target_Player"],
            "market": ["PTS"],
            "team": ["AAA"],
            "teammate": ["Teammate_A"],
            "teammate_id": ["2"],
            "shrunk_delta": [3.0],
            "confidence": [0.5],
            "first_shared_date": ["2026-01-01"],
            "last_shared_date": ["2026-01-03"],
        }
    ).to_csv(artifacts / "player_teammate_out_deltas.csv", index=False)

    output = tmp_path / "oracle"
    subprocess.run(
        [
            "python",
            "Player-Predictor/scripts/apply_lineup_deltas_v9_4.py",
            "--source-manifest",
            str(manifest),
            "--lineup-artifacts",
            str(artifacts),
            "--game-logs",
            str(logs_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    rows = pd.read_csv(output / "data" / "prop_training_rows.csv")
    assert rows.loc[0, "lineup_oracle_teammates_out_count"] == 1
    assert rows.loc[0, "lineup_oracle_adjustment"] == 1.5
    assert rows.loc[0, "p_over_raw"] > rows.loc[0, "p_over_raw_v93"]
    out_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert out_manifest["status"] == "research_only_oracle_lineup_not_promotable"


def _log(game_id, date, player_id, player):
    return {
        "GAME_ID": game_id,
        "GAME_DATE": date,
        "TEAM_ABBREVIATION": "AAA",
        "PLAYER_ID": player_id,
        "PLAYER_NAME": player,
        "MIN": 30,
        "AVAILABLE_FLAG": 1,
    }
