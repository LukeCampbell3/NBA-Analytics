from sports.nfl.scripts.build_nfl_week_market_board import build_board


def test_builds_position_and_same_game_shadow_pools() -> None:
    pool = {"season": 2026, "week": 1, "pool": [
        {"player":"A QB","player_id":"q","position":"QB","team":"A","opponent":"B","game_id":"g1","kickoff_utc":"2026-09-10T00:00:00Z","projection":260,"p10":180,"p90":340},
        {"player":"A WR","player_id":"w","position":"WR","team":"A","opponent":"B","game_id":"g1","kickoff_utc":"2026-09-10T00:00:00Z","projection":80,"p10":30,"p90":130},
        {"player":"C QB","player_id":"q2","position":"QB","team":"C","opponent":"D","game_id":"g2","kickoff_utc":"2026-09-11T00:00:00Z","projection":270,"p10":190,"p90":350},
    ]}
    def offer(player, target, market, line, event):
        return {"player":player,"target":target,"market":market,"line":line,"over_price":100,"under_price":-120,"bookmaker":"draftkings","snapshot_time_utc":"2026-09-02T00:00:00Z","source":"rotowire_public_player_props","event_id":event}
    snapshot = {"audit":{"fetched_at_utc":"2026-09-02T00:00:00Z","source_url":"https://example.test","first_td_best_prices":[]},"observations":[
        offer("A QB","passing","player_pass_yds",250,"g1"),
        offer("A WR","receiving","player_reception_yds",70,"g1"),
        offer("C QB","passing","player_pass_yds",255,"g2"),
    ]}
    result = build_board(pool, snapshot)
    assert result["candidate_count"] == 3
    assert len(result["pools"]["passer_parlay"]) == 2
    assert len(result["pools"]["same_game_parlay"]) == 2
    assert result["candidate_authorized"] is False
    assert "WITHHELD" in result["methodology"]["parlay_probability"]
