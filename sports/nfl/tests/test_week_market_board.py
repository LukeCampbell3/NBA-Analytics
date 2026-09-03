from sports.nfl.scripts.build_nfl_week_market_board import build_board


BACKTEST = {
    "artifact_type": "nfl_week_market_policy_backtest",
    "policy_version": "test",
    "evidence_as_of_utc": "2025-12-31T00:00:00Z",
    "ranking_rule": "probability then edge",
    "policy": {
        "minimum_side_probability": 0.58,
        "minimum_no_vig_advantage": 0.10,
        "minimum_price": -130,
        "maximum_price": 130,
        "weekly_cap": 6,
    },
    "capabilities": {
        "passing": {"state": "BACKTEST_VALIDATED_SHADOW", "selection_authority": True},
        "receiving": {"state": "NO_RELIABLE_EDGE_FOUND", "selection_authority": False},
    },
}


def test_builds_position_and_same_game_shadow_pools() -> None:
    pool = {"season": 2026, "week": 1, "pool": [
        {"player":"A QB","player_id":"q","position":"QB","team":"A","opponent":"B","game_id":"g1","kickoff_utc":"2026-09-10T00:00:00Z","projection":260,"p10":180,"p90":340},
        {"player":"A WR","player_id":"w","position":"WR","team":"A","opponent":"B","game_id":"g1","kickoff_utc":"2026-09-10T00:00:00Z","projection":80,"p10":30,"p90":130},
        {"player":"C QB","player_id":"q2","position":"QB","team":"C","opponent":"D","game_id":"g2","kickoff_utc":"2026-09-11T00:00:00Z","projection":270,"p10":190,"p90":350},
    ]}
    def offer(player, target, market, line, event):
        team = {"A QB": "A", "A WR": "A", "C QB": "C"}[player]
        return {"player":player,"provider_team":team,"target":target,"market":market,"line":line,"over_price":100,"under_price":-120,"bookmaker":"draftkings","snapshot_time_utc":"2026-09-02T00:00:00Z","source":"rotowire_public_nfl_props","event_id":event}
    snapshot = {"audit":{"fetched_at_utc":"2026-09-02T00:00:00Z","source_url":"https://example.test","first_td_best_prices":[]},"observations":[
        offer("A QB","passing","player_pass_yds",240,"g1"),
        offer("A WR","receiving","player_reception_yds",70,"g1"),
        offer("C QB","passing","player_pass_yds",255,"g2"),
    ]}
    result = build_board(pool, snapshot, BACKTEST)
    assert result["candidate_count"] == 3
    assert len(result["pools"]["passer_parlay"]) == 2
    assert len(result["pools"]["same_game_parlay"]) == 2
    assert all(row["policy_eligible"] for row in result["pools"]["passer_parlay"])
    assert any(not row["policy_eligible"] for row in result["pools"]["same_game_parlay"])
    assert result["pool_status"]["receiver_parlay"] == "WITHHELD_NO_RELIABLE_EDGE"
    assert result["candidate_authorized"] is False
    assert "WITHHELD" in result["methodology"]["parlay_probability"]
    assert result["line_ladder_count"] == 0


def test_preserves_multi_book_lines_as_diagnostic_survival_curve() -> None:
    pool = {"season": 2026, "week": 1, "pool": [{
        "player":"A QB", "player_id":"q", "position":"QB", "team":"A",
        "opponent":"B", "game_id":"g1", "kickoff_utc":"2026-09-10T00:00:00Z",
        "projection":250, "p10":170, "p90":330,
    }]}
    def offer(book: str, line: float, over: float, under: float) -> dict:
        return {"player":"A QB", "provider_team":"A", "target":"passing",
                "market":"player_pass_yds", "line":line, "over_price":over,
                "under_price":under, "bookmaker":book,
                "snapshot_time_utc":"2026-09-02T00:00:00Z", "source":"rotowire"}
    snapshot = {"audit":{}, "observations":[
        offer("fanduel", 240.5, -110, -110),
        offer("draftkings", 250.5, 105, -125),
    ]}
    result = build_board(pool, snapshot, BACKTEST)
    assert result["line_ladder_count"] == 1
    ladder = result["line_ladders"][0]
    assert ladder["distinct_lines"] == [240.5, 250.5]
    assert ladder["selection_authority"] is False
    assert all("survival_probability_delta" in point for point in ladder["points"])


def test_rejects_name_match_when_provider_team_disagrees() -> None:
    pool = {"season": 2026, "week": 1, "pool": [{
        "player":"A Receiver","player_id":"w","position":"WR","team":"SF",
        "opponent":"LA","game_id":"g","kickoff_utc":"2026-09-10T00:00:00Z",
        "projection":60,"p10":20,"p90":100,
    }]}
    snapshot = {"audit":{},"observations":[{
        "player":"A Receiver","provider_team":"MIN","target":"receiving",
        "market":"player_reception_yds","line":50,"over_price":-110,
        "under_price":-110,"bookmaker":"fanduel","snapshot_time_utc":"2026-09-02T00:00:00Z",
        "source":"rotowire_public_nfl_props",
    }]}
    assert build_board(pool, snapshot, BACKTEST)["candidate_count"] == 0


def test_passing_pool_fails_closed_without_backtest_authority() -> None:
    pool = {"pool": [{
        "player": "A QB", "player_id": "q", "position": "QB", "team": "A",
        "opponent": "B", "game_id": "g", "kickoff_utc": "2026-09-10T00:00:00Z",
        "projection": 275, "p10": 190, "p90": 350,
    }]}
    snapshot = {"audit": {}, "observations": [{
        "player": "A QB", "provider_team": "A", "target": "passing",
        "market": "player_pass_yds", "line": 240.5, "over_price": 100,
        "under_price": -120, "bookmaker": "fanduel",
        "snapshot_time_utc": "2026-09-02T00:00:00Z", "source": "rotowire",
    }]}
    result = build_board(pool, snapshot)
    assert result["candidate_count"] == 1
    assert result["pools"]["passer_parlay"] == []
    assert result["best_available_singles"] == []
