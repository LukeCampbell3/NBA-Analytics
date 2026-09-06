from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from sports.mlb.scripts.collect_historical_game_conditioned_outcomes import (
    EVIDENCE_CLASS,
    OutcomeConflictError,
    build_outcome_record,
    collect_outcomes,
    outcome_key,
    summarize,
    write_ledger,
)


def _row(**overrides):
    base = {
        "Date": "2026-06-10",
        "Player": "Test_Hitter",
        "Player_MLBAM_ID": "12345",
        "Player_Type": "hitter",
        "Team": "AAA",
        "Opponent": "BBB",
        "Season": "2026",
        "Game_ID": "game-1",
        "H": "2",
        "TB": "5",
        "HR": "1",
        "PA": "4",
        "AB": "4",
        "Did_Not_Play": "0",
        # Leakage bait: none of these may enter the outcome record.
        "Market_H": "0.5",
        "Market_H_over_price": "-180",
        "xwOBA": "0.410",
        "Matchup_Network_H_Score": "0.7",
        "H_market_gap": "0.3",
    }
    return {**base, **overrides}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_record_contains_only_realized_outcome_contract():
    record, reason = build_outcome_record(_row(), source_file="fixture.csv", source_row=2, season=2026)
    assert reason is None
    assert record is not None
    assert record["evidence_class"] == EVIDENCE_CLASS
    assert record["realized"] == {"H": 2, "TB": 5, "HR": 1, "PA": 4, "AB": 4}
    assert record["outcomes"] == {"H_OVER_0_5": 1, "TB_OVER_1_5": 1, "HR_OVER_0_5": 1}
    assert record["pregame_features_included"] is False
    assert record["market_data_included"] is False
    serialized = json.dumps(record, sort_keys=True)
    for forbidden in ("Market_H", "Market_H_over_price", "xwOBA", "Matchup_Network_H_Score", "H_market_gap"):
        assert forbidden not in serialized


def test_dnp_zero_pa_and_inconsistent_stats_are_rejected():
    record, reason = build_outcome_record(_row(Did_Not_Play="1"), source_file="x", source_row=2, season=2026)
    assert record is None and reason == "DID_NOT_PLAY"

    record, reason = build_outcome_record(_row(PA="0", AB="0", H="0", TB="0", HR="0"), source_file="x", source_row=2, season=2026)
    assert record is None and reason == "ZERO_PA"

    record, reason = build_outcome_record(_row(H="0", TB="4", HR="1"), source_file="x", source_row=2, season=2026)
    assert record is None and "HR_GT_H" in str(reason)

    record, reason = build_outcome_record(_row(H="2", TB="1", HR="0"), source_file="x", source_row=2, season=2026)
    assert record is None and "TB_LT_H" in str(reason)


def test_hash_is_stable_across_provenance_changes():
    a, _ = build_outcome_record(_row(), source_file="a.csv", source_row=2, season=2026)
    b, _ = build_outcome_record(_row(), source_file="b.csv", source_row=99, season=2026)
    assert a is not None and b is not None
    assert a["outcome_sha256"] == b["outcome_sha256"]
    assert a["source"] != b["source"]


def test_collection_deduplicates_identical_rows_and_rejects_conflicts(tmp_path):
    root = tmp_path / "players"
    _write_csv(root / "A" / "2026_processed_processed.csv", [_row()])
    _write_csv(root / "B" / "2026_processed_processed.csv", [_row()])

    rows, skipped = collect_outcomes(root, season=2026)
    assert len(rows) == 1
    assert skipped["EXACT_DUPLICATE"] == 1
    assert outcome_key(rows[0]) == "2026|game-1|12345"

    _write_csv(root / "B" / "2026_processed_processed.csv", [_row(H="1", TB="1", HR="0")])
    with pytest.raises(OutcomeConflictError):
        collect_outcomes(root, season=2026)


def test_summary_and_gzip_ledger_are_deterministic(tmp_path):
    rows = []
    for idx, payload in enumerate(
        [
            _row(Game_ID="g1", H="1", TB="1", HR="0"),
            _row(Game_ID="g2", H="0", TB="0", HR="0"),
            _row(Game_ID="g3", H="2", TB="5", HR="1"),
        ],
        start=2,
    ):
        record, reason = build_outcome_record(payload, source_file="fixture.csv", source_row=idx, season=2026)
        assert reason is None and record is not None
        rows.append(record)

    summary = summarize(rows, {}, season=2026)
    assert summary["hitter_games"] == 3
    assert summary["targets"]["H"]["clears"] == 2
    assert summary["targets"]["TB"]["clears"] == 1
    assert summary["targets"]["HR"]["clears"] == 1
    assert summary["contract"]["join_key"] == ["season", "game_id", "player_id"]

    path = tmp_path / "ledger.jsonl.gz"
    write_ledger(rows, path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        decoded = [json.loads(line) for line in handle if line.strip()]
    assert [row["outcome_sha256"] for row in decoded] == [row["outcome_sha256"] for row in rows]
