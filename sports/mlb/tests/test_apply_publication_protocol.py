from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import apply_publication_protocol as protocol  # noqa: E402


def _write(path: Path, players: list[str], stamp: str = "2026-09-03T15:30:00Z") -> None:
    path.write_text(json.dumps({"run_date": "2026-09-03", "generated_at_utc": stamp, "plays": [
        {"player": player, "game_id": str(i), "market": "hits", "side": "over", "line": 0.5, "sportsbook": "FanDuel", "odds": -120}
        for i, player in enumerate(players)
    ]}))


def test_discovery_captures_candidates_but_issues_nothing(tmp_path: Path) -> None:
    board, candidates, issued = tmp_path / "daily.json", tmp_path / "candidates.json", tmp_path / "issued"
    _write(board, ["Candidate"])
    result = protocol.apply_protocol(board, candidates, issued, "DISCOVER")
    assert result["candidate_count"] == 1
    assert result["issued_count"] == 0
    assert json.loads(board.read_text())["plays"] == []
    assert json.loads(candidates.read_text())["plays"][0]["player"] == "Candidate"


def test_refresh_cannot_delete_or_mutate_issued_pick(tmp_path: Path) -> None:
    board, candidates, issued = tmp_path / "daily.json", tmp_path / "candidates.json", tmp_path / "issued"
    _write(board, ["Official"])
    protocol.apply_protocol(board, candidates, issued, "PUBLISH")
    original = json.loads(board.read_text())["plays"][0]
    _write(board, ["Different"] , "2026-09-03T18:30:00Z")
    protocol.apply_protocol(board, candidates, issued, "REFRESH")
    assert json.loads(board.read_text())["plays"] == [original]


def test_late_add_appends_without_reissuing_duplicates(tmp_path: Path) -> None:
    board, candidates, issued = tmp_path / "daily.json", tmp_path / "candidates.json", tmp_path / "issued"
    _write(board, ["Official"])
    protocol.apply_protocol(board, candidates, issued, "PUBLISH")
    _write(board, ["Official", "Late"])
    protocol.apply_protocol(board, candidates, issued, "LATE_ADD")
    protocol.apply_protocol(board, candidates, issued, "LATE_ADD")
    plays = json.loads(board.read_text())["plays"]
    assert [row["player"] for row in plays] == ["Official", "Late"]
    assert plays[0]["issuance_board"] == "OFFICIAL"
    assert plays[1]["issuance_board"] == "LATE_ADD"


def test_deployment_migrates_prior_public_board(tmp_path: Path) -> None:
    board, prior, candidates, issued = tmp_path / "daily.json", tmp_path / "prior.json", tmp_path / "candidates.json", tmp_path / "issued"
    _write(prior, ["Already Public"])
    _write(board, ["New Candidate"])
    protocol.apply_protocol(board, candidates, issued, "DISCOVER", prior_board_path=prior)
    assert json.loads(board.read_text())["plays"][0]["player"] == "Already Public"
