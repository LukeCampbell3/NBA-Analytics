from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import recover_historical_market_snapshots as recovery


def test_combine_snapshots_deduplicates_exact_book_offers() -> None:
    common = {
        "fetched_at_utc": "2026-05-03T17:26:01Z",
        "event_id": "game-1",
        "bookmaker_key": "draftkings",
        "market_key": "batter_total_bases",
        "player_name_norm": "example_player",
        "line": 1.5,
        "over_price": -110,
        "event_date_et": "2026-05-03",
    }
    older = pd.DataFrame([{**common, "history_origin": "git"}])
    current = pd.DataFrame([{**common, "history_origin": "working_tree"}])

    combined = recovery.combine_snapshots([older, current])

    assert len(combined) == 1
    assert combined.iloc[0]["history_origin"] == "working_tree"


def test_discover_snapshot_refs_finds_deleted_csv(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    snapshot_dir = tmp_path / recovery.DEFAULT_SNAPSHOT_ROOT
    snapshot_dir.mkdir(parents=True)
    snapshot = snapshot_dir / "player_props_long_20260503T172601Z.csv"
    snapshot.write_text("fetched_at_utc,event_id\n2026-05-03T17:26:01Z,game-1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add snapshot"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "rm", snapshot.relative_to(tmp_path).as_posix()], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "remove snapshot"], cwd=tmp_path, check=True, capture_output=True)

    refs = recovery.discover_snapshot_refs(tmp_path, recovery.DEFAULT_SNAPSHOT_ROOT)

    assert len(refs) == 1
    assert refs[0][1].endswith("player_props_long_20260503T172601Z.csv")
