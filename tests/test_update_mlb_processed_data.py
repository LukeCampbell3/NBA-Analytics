from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "Player-Predictor" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import update_mlb_processed_data as updater


def test_deduplicate_player_games_keeps_one_row_per_role_and_game() -> None:
    frame = pd.DataFrame(
        [
            {"Player": "Austin_Riley", "Player_Type": "hitter", "Game_ID": "824912", "Date": "2026-06-16", "H": 1},
            {"Player": "Austin_Riley", "Player_Type": "hitter", "Game_ID": "824912", "Date": "2026-06-16", "H": 1},
            {"Player": "Austin_Riley", "Player_Type": "hitter", "Game_ID": "824913", "Date": "2026-06-17", "H": 3},
        ]
    )

    result = updater.deduplicate_player_games(frame)

    assert len(result) == 2
    assert result["Game_ID"].tolist() == ["824912", "824913"]


def test_matchup_network_is_walk_forward_and_uses_stable_starter_id() -> None:
    rows = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-04-01"), "Game_Index": 0,
                "Player": "Test_Pitcher", "Player_MLBAM_ID": 42, "Player_Type": "pitcher",
                "Was_Starter": 1, "IP": 5.0, "BF": 22, "Pitches": 85, "K": 3,
                "ERA": 5.4, "FIP": 5.1, "H_allowed": 7, "HR_allowed": 1, "BB_allowed": 3,
            },
            {
                "Date": pd.Timestamp("2026-04-02"), "Game_Index": 0,
                "Player": "Test_Hitter", "Player_MLBAM_ID": 7, "Player_Type": "hitter",
                "Opp_Starter_ID": 42, "Opp_Starter_Player": "Test_Pitcher",
                "PA": 4, "SO": 0, "wOBA": 0.4, "ISO": 0.2, "HardHit%": 50,
                "Barrel%": 12, "Batting_Order": 3, "H": 2, "TB": 3, "R": 1, "HR": 0, "RBI": 1,
            },
            {
                "Date": pd.Timestamp("2026-04-08"), "Game_Index": 1,
                "Player": "Test_Hitter", "Player_MLBAM_ID": 7, "Player_Type": "hitter",
                "Opp_Starter_ID": 42, "Opp_Starter_Player": "Test_Pitcher",
                "PA": 4, "SO": 1, "wOBA": 0.3, "ISO": 0.1, "HardHit%": 35,
                "Barrel%": 6, "Batting_Order": 4, "H": 1, "TB": 1, "R": 0, "HR": 0, "RBI": 0,
            },
        ]
    )

    first = updater.attach_walk_forward_matchup_network(rows)
    changed = rows.copy()
    changed.loc[changed["Date"].eq(pd.Timestamp("2026-04-08")), "H"] = 10
    second = updater.attach_walk_forward_matchup_network(changed)

    later = first.loc[first["Date"].eq(pd.Timestamp("2026-04-08"))].iloc[0]
    later_changed = second.loc[second["Date"].eq(pd.Timestamp("2026-04-08"))].iloc[0]
    assert later["Batter_Vs_Starter_Games"] == 1
    assert later["Matchup_Network_H_Adjustment"] == later_changed["Matchup_Network_H_Adjustment"]


def test_matchup_network_cache_only_computes_new_hitter_rows(monkeypatch) -> None:
    rows = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-04-01"), "Game_Index": 0,
                "Player": "Test_Pitcher", "Player_MLBAM_ID": 42, "Player_Type": "pitcher", "Game_ID": "1",
                "Was_Starter": 1, "IP": 5.0, "BF": 22, "Pitches": 85, "K": 3,
                "ERA": 5.4, "FIP": 5.1, "H_allowed": 7, "HR_allowed": 1, "BB_allowed": 3,
            },
            {
                "Date": pd.Timestamp("2026-04-02"), "Game_Index": 0,
                "Player": "Test_Hitter", "Player_MLBAM_ID": 7, "Player_Type": "hitter", "Game_ID": "2",
                "Opp_Starter_ID": 42, "Opp_Starter_Player": "Test_Pitcher",
                "PA": 4, "SO": 0, "wOBA": 0.4, "ISO": 0.2, "HardHit%": 50,
                "Barrel%": 12, "Batting_Order": 3, "H": 2, "TB": 3, "R": 1, "HR": 0, "RBI": 1,
            },
            {
                "Date": pd.Timestamp("2026-04-08"), "Game_Index": 1,
                "Player": "Test_Hitter", "Player_MLBAM_ID": 7, "Player_Type": "hitter", "Game_ID": "3",
                "Opp_Starter_ID": 42, "Opp_Starter_Player": "Test_Pitcher",
                "PA": 4, "SO": 1, "wOBA": 0.3, "ISO": 0.1, "HardHit%": 35,
                "Barrel%": 6, "Batting_Order": 4, "H": 1, "TB": 1, "R": 0, "HR": 0, "RBI": 0,
            },
        ]
    )
    cached = updater.attach_walk_forward_matchup_network(rows)
    new_row = rows.iloc[-1].copy()
    new_row["Date"] = pd.Timestamp("2026-04-15")
    new_row["Game_Index"] = 2
    new_row["Game_ID"] = "4"
    expanded = pd.concat([rows, new_row.to_frame().T], ignore_index=True)

    calls = 0
    original = updater.build_matchup_network_signal

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(updater, "build_matchup_network_signal", counted)
    result = updater.attach_walk_forward_matchup_network(expanded, network_cache=cached)

    assert calls == 1
    cached_row = cached.loc[cached["Game_ID"].eq("3")].iloc[0]
    reused_row = result.loc[result["Game_ID"].eq("3")].iloc[0]
    assert reused_row["Matchup_Network_H_Adjustment"] == cached_row["Matchup_Network_H_Adjustment"]
    assert result.loc[result["Game_ID"].eq("4"), "Matchup_Network_Version"].item() == updater.MATCHUP_NETWORK_VERSION


def test_load_existing_processed_corpus_separates_raw_rows_and_network_cache(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "processed"
    player_dir = proc_root / "Test_Hitter"
    player_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Date": "2026-04-02", "Player": "Test_Hitter", "Player_Type": "hitter",
                "Game_ID": 2, "H": 1, "H_rolling_avg": 0.75,
                "Matchup_Network_Version": updater.MATCHUP_NETWORK_VERSION,
                "Matchup_Network_H_Adjustment": 0.03,
            }
        ]
    ).to_csv(player_dir / "2026_processed_processed.csv", index=False)
    monkeypatch.setattr(updater, "PROC_ROOT", proc_root)

    raw, cache = updater.load_existing_processed_corpus(2026)

    assert raw["Game_ID"].item() == "2"
    assert "H_rolling_avg" not in raw.columns
    assert cache["Matchup_Network_Version"].item() == updater.MATCHUP_NETWORK_VERSION
    assert cache["Matchup_Network_H_Adjustment"].item() == 0.03


def test_incremental_noop_preserves_existing_corpus(tmp_path, monkeypatch) -> None:
    proc_root = tmp_path / "processed"
    raw_root = tmp_path / "raw"
    player_dir = proc_root / "Test_Hitter"
    player_dir.mkdir(parents=True)
    player_path = player_dir / "2026_processed_processed.csv"
    pd.DataFrame(
        [
            {
                "Date": "2026-08-10", "Player": "Test_Hitter", "Player_Type": "hitter",
                "Game_ID": "10", "H": 1,
            }
        ]
    ).to_csv(player_path, index=False)
    original = player_path.read_bytes()
    (proc_root / "update_manifest_2026.json").write_text(
        json.dumps(
            {
                "processed_summary": {"players": 1, "rows": 1, "min_date": "2026-08-10", "max_date": "2026-08-10"},
                "written": {"Test_Hitter": {"path": str(player_path), "rows": 1}},
                "market_history_rows": 7,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(updater, "PROC_ROOT", proc_root)
    monkeypatch.setattr(updater, "RAW_ROOT", raw_root)
    monkeypatch.setattr(updater, "fetch_schedule", lambda **kwargs: [])
    monkeypatch.setattr(
        updater,
        "parse_args",
        lambda: Namespace(
            season=2026,
            start_date=None,
            through_date="2026-08-11",
            refresh_source=False,
            incremental=True,
            timeout_seconds=1.0,
            retries=1,
            sleep_seconds=0.0,
            min_rows=1,
            player_limit=None,
        ),
    )

    updater.main()

    manifest = json.loads((proc_root / "update_manifest_2026.json").read_text(encoding="utf-8"))
    assert player_path.read_bytes() == original
    assert manifest["incremental"] is True
    assert manifest["completed_games_this_run"] == 0
    assert manifest["processed_summary"]["max_date"] == "2026-08-10"
    assert manifest["market_history_rows"] == 7
