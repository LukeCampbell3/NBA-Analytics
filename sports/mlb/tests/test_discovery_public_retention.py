from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import apply_publication_protocol as protocol  # noqa: E402


def _play(player: str, game_id: str) -> dict:
    return {
        "player": player,
        "game_id": game_id,
        "market": "hits",
        "side": "over",
        "line": 0.5,
        "sportsbook": "FanDuel",
        "odds": -120,
    }


def test_next_day_discovery_updates_candidates_without_advancing_public_board(tmp_path: Path) -> None:
    board = tmp_path / "daily.json"
    prior = tmp_path / "prior.json"
    candidates = tmp_path / "candidates.json"
    issued = tmp_path / "issued"

    prior_payload = {
        "run_date": "2026-09-06",
        "publication_status": "ready",
        "plays": [{**_play("Issued Today", "1"), "issuance_id": "MLB-20260906-1130-001"}],
        "publication_protocol": {
            "mode": "PUBLISH",
            "issued_count": 1,
            "public_pick_retention": 1.0,
            "issued_picks_immutable": True,
            "issued_source": "history/issued/2026-09-06.json",
        },
    }
    prior.write_text(json.dumps(prior_payload))
    board.write_text(json.dumps({
        "run_date": "2026-09-07",
        "publication_status": "ready",
        "plays": [_play("Tomorrow Candidate", "2")],
    }))

    result = protocol.apply_protocol(board, candidates, issued, "DISCOVER", prior_board_path=prior)

    visible = json.loads(board.read_text())
    candidate_payload = json.loads(candidates.read_text())
    assert visible == prior_payload
    assert visible["run_date"] == "2026-09-06"
    assert visible["plays"][0]["player"] == "Issued Today"
    assert candidate_payload["run_date"] == "2026-09-07"
    assert candidate_payload["plays"][0]["player"] == "Tomorrow Candidate"
    assert result["public_board_mutated"] is False
    assert result["public_board_retained_from"] == "2026-09-06"
    assert result["candidate_run_date"] == "2026-09-07"


def test_same_day_discovery_cannot_rewrite_managed_issued_board(tmp_path: Path) -> None:
    board = tmp_path / "daily.json"
    prior = tmp_path / "prior.json"
    candidates = tmp_path / "candidates.json"
    issued = tmp_path / "issued"

    prior_payload = {
        "run_date": "2026-09-06",
        "publication_status": "ready",
        "plays": [{**_play("Official", "1"), "issuance_id": "MLB-20260906-1130-001"}],
        "publication_protocol": {
            "mode": "PUBLISH",
            "issued_count": 1,
            "public_pick_retention": 1.0,
            "issued_picks_immutable": True,
        },
    }
    prior.write_text(json.dumps(prior_payload))
    board.write_text(json.dumps({
        "run_date": "2026-09-06",
        "publication_status": "ready",
        "plays": [_play("New Candidate", "2")],
    }))

    result = protocol.apply_protocol(board, candidates, issued, "DISCOVER", prior_board_path=prior)

    assert json.loads(board.read_text()) == prior_payload
    assert json.loads(candidates.read_text())["plays"][0]["player"] == "New Candidate"
    assert result["public_board_mutated"] is False
