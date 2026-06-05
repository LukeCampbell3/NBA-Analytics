import json
import subprocess
from pathlib import Path

import pandas as pd


def test_lineup_delta_builder_uses_shrunk_teammate_out_splits(tmp_path: Path):
    model_dir = tmp_path / "model"
    data_dir = model_dir / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"date": ["2026-01-01"], "player": ["Target_Player"]}).to_csv(
        data_dir / "prop_training_rows.csv",
        index=False,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"output": str(model_dir)}), encoding="utf-8")

    logs = pd.DataFrame(
        [
            _log("G1", "2026-01-01", "1", "Target Player", 30, 10, 3, 4),
            _log("G1", "2026-01-01", "2", "Teammate A", 28, 12, 5, 2),
            _log("G2", "2026-01-03", "1", "Target Player", 32, 20, 4, 5),
            _log("G2", "2026-01-03", "3", "Teammate B", 20, 7, 4, 1),
            _log("G3", "2026-01-05", "1", "Target Player", 31, 14, 2, 6),
            _log("G3", "2026-01-05", "2", "Teammate A", 27, 11, 6, 3),
        ]
    )
    logs_path = tmp_path / "logs.csv"
    logs.to_csv(logs_path, index=False)

    output = tmp_path / "lineup"
    subprocess.run(
        [
            "python",
            "Player-Predictor/scripts/build_lineup_delta_artifacts.py",
            "--v9-manifest",
            str(manifest),
            "--game-logs",
            str(logs_path),
            "--output",
            str(output),
            "--shrink-k",
            "1",
            "--min-with-games",
            "2",
            "--min-without-games",
            "1",
            "--min-prior-games",
            "1",
        ],
        check=True,
    )

    report = json.loads((output / "lineup_delta_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "built_historical_lineup_delta_artifacts"
    assert report["teammate_out_delta_rows"] >= 3

    out_path = output / "player_teammate_out_deltas.parquet"
    if out_path.exists():
        out = pd.read_parquet(out_path)
    else:
        out = pd.read_csv(output / "player_teammate_out_deltas.csv")
    pts = out.loc[
        (out["player_id"].astype(str) == "1")
        & (out["teammate_id"].astype(str) == "2")
        & (out["market"] == "PTS")
    ].iloc[0]

    # With Teammate A: (10 + 14) / 2 = 12. Without: 20. Shrink with n=1,k=1 halves it.
    assert pts["raw_delta"] == 8
    assert pts["shrunk_delta"] == 4
    assert pts["confidence"] == 0.5


def _log(game_id, date, player_id, player, minutes, pts, reb, ast):
    return {
        "GAME_ID": game_id,
        "GAME_DATE": date,
        "TEAM_ABBREVIATION": "AAA",
        "PLAYER_ID": player_id,
        "PLAYER_NAME": player,
        "MIN": minutes,
        "PTS": pts,
        "REB": reb,
        "AST": ast,
        "FGA": 0,
        "FTA": 0,
        "TOV": 0,
        "AVAILABLE_FLAG": 1,
    }
