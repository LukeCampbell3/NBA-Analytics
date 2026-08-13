from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
NBA_ROOT = REPO_ROOT / "sports" / "nba"
SCRIPT = NBA_ROOT / "pipeline" / "build_opening_night_pool.py"
SPEC = importlib.util.spec_from_file_location("build_opening_night_pool", SCRIPT)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_committed_opening_pool_is_market_gated_and_covers_all_three_games() -> None:
    payload = json.loads(
        (NBA_ROOT / "web" / "data" / "opening_night_pool.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "projection_pool_ready"
    assert payload["publication_status"] == "research_only"
    assert payload["market_status"] == "awaiting_lines"
    assert payload["game_count"] == 3
    assert payload["player_count"] == 19
    assert payload["projection_count"] == 76
    assert payload["target_counts"] == {"PTS": 19, "REB": 19, "AST": 19, "PRA": 19}
    assert {row["target"] for row in payload["pool"]} == {"PTS", "REB", "AST", "PRA"}
    assert all(row["market_line"] is None for row in payload["pool"])
    assert all(row["direction"] is None for row in payload["pool"])
    assert all(row["candidate_authorized"] is False for row in payload["pool"])
    assert payload["watchlist_policy"]["candidate_authorized"] is False
    assert len(payload["watchlists"]) == 3
    for watchlist in payload["watchlists"]:
        assert watchlist["candidate_authorized"] is False
        assert len(watchlist["legs"]) == 3
        assert len({leg["game_id"] for leg in watchlist["legs"]}) == 3


def test_offseason_team_overrides_and_frontend_contract() -> None:
    payload = json.loads(
        (NBA_ROOT / "web" / "data" / "opening_night_pool.json").read_text(encoding="utf-8")
    )
    player_teams = {row["player"]: row["team"] for row in payload["players"]}
    html = (NBA_ROOT / "web" / "predictions.html").read_text(encoding="utf-8")
    javascript = (NBA_ROOT / "web" / "predictions.js").read_text(encoding="utf-8")

    assert player_teams["Jaylen Brown"] == "PHI"
    assert player_teams["LeBron James"] == "PHI"
    assert 'id="openingProjectionPool"' in html
    assert 'id="openingTargetFilters"' in html
    assert 'id="openingWatchlists"' in html
    assert "data/opening_night_pool.json" in javascript
