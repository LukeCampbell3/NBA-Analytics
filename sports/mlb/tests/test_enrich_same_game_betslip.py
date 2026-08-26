from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
MLB_ODDS_PROVIDERS_ROOT = REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"
MLB_PARLAY_V2_ROOT = REPO_ROOT / "sports" / "mlb" / "parlay_v2"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))
sys.path.insert(0, str(MLB_ODDS_PROVIDERS_ROOT))
sys.path.insert(0, str(MLB_PARLAY_V2_ROOT))

import enrich_same_game_betslip as enrich  # noqa: E402


def _moneyline_row(home_full_name: str, away_full_name: str, home_deeplink: str, away_deeplink: str) -> dict:
    return {
        "home_team": home_full_name, "away_team": away_full_name, "target": "moneyline",
        "line": None, "home_moneyline_deeplink": home_deeplink, "away_moneyline_deeplink": away_deeplink,
    }


def _total_row(home_full_name: str, away_full_name: str, line: float, over_deeplink: str, under_deeplink: str) -> dict:
    return {
        "home_team": home_full_name, "away_team": away_full_name, "target": "game_total",
        "line": line, "over_deeplink": over_deeplink, "under_deeplink": under_deeplink,
    }


def _combo(home_team="DET", away_team="TB", ml_side="away", total_line=7.5, total_side="over") -> dict:
    return {
        "home_team": home_team, "away_team": away_team,
        "leg_a": {"market": "moneyline", "side": ml_side, "line": None, "sportsbook": "fanduel"},
        "leg_b": {"market": "game_total", "side": total_side, "line": total_line, "sportsbook": "fanduel"},
    }


def _payload_with_one_combo(**combo_kwargs) -> dict:
    return {"games": [{"game_id": "824234", "combo_candidates": [_combo(**combo_kwargs)]}]}


def test_enrich_combo_builds_real_multi_leg_url_when_both_legs_resolve():
    combo = _combo()
    index = {
        ("DET", "TB", "moneyline"): _moneyline_row("Detroit Tigers", "Tampa Bay Rays", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=11", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=12"),
        ("DET", "TB", "game_total"): _total_row("Detroit Tigers", "Tampa Bay Rays", 7.5, "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=21", "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=22"),
    }

    ready = enrich.enrich_combo(combo, index)

    assert ready is True
    assert combo["betslip"]["status"] == "ready"
    assert combo["betslip_url"].startswith("https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?")
    assert "marketId%5B0%5D=1" in combo["betslip_url"]
    assert "marketId%5B1%5D=2" in combo["betslip_url"]
    assert combo["leg_a"]["sportsbook_deeplink"].endswith("selectionId=12")  # away moneyline
    assert combo["leg_b"]["sportsbook_deeplink"].endswith("selectionId=21")  # over


def test_enrich_combo_marks_unavailable_when_one_leg_has_no_live_match():
    combo = _combo()
    index = {
        ("DET", "TB", "moneyline"): _moneyline_row("Detroit Tigers", "Tampa Bay Rays", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=11", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=12"),
    }

    ready = enrich.enrich_combo(combo, index)

    assert ready is False
    assert combo["betslip"]["status"] == "unavailable"
    assert combo["betslip"]["reason"] == "one_or_more_legs_have_no_live_fanduel_selection"
    assert "betslip_url" not in combo
    assert combo["leg_a"]["sportsbook_deeplink"]  # the resolvable leg still gets its own real deeplink
    assert "sportsbook_deeplink" not in combo["leg_b"]


def test_enrich_combo_refuses_a_line_that_has_moved_since_publication():
    """A live total line that no longer matches what was already
    published must never be silently relinked to a different real
    line -- the leg is left unlinked instead."""
    combo = _combo(total_line=7.5)
    index = {
        ("DET", "TB", "moneyline"): _moneyline_row("Detroit Tigers", "Tampa Bay Rays", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=11", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=12"),
        ("DET", "TB", "game_total"): _total_row("Detroit Tigers", "Tampa Bay Rays", 8.0, "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=21", "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=22"),
    }

    ready = enrich.enrich_combo(combo, index)

    assert ready is False
    assert "sportsbook_deeplink" not in combo["leg_b"]


def test_enrich_combo_noop_on_combo_with_no_leg_dicts():
    combo = {"leg_a": None, "leg_b": None}
    result = enrich.enrich_combo(combo, {})
    assert result is False
    assert "betslip" not in combo


def test_enrich_payload_enriches_every_combo_across_every_game():
    payload = {
        "games": [
            {"game_id": "824234", "combo_candidates": [_combo(home_team="DET", away_team="TB")]},
            {"game_id": "824235", "combo_candidates": [_combo(home_team="ATH", away_team="MIN", total_line=8.0)]},
        ],
    }
    rows = [
        _moneyline_row("Detroit Tigers", "Tampa Bay Rays", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=11", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=12"),
        _total_row("Detroit Tigers", "Tampa Bay Rays", 7.5, "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=21", "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=22"),
        _moneyline_row("Athletics", "Minnesota Twins", "https://sportsbook.fanduel.com/addToBetslip?marketId=3&selectionId=31", "https://sportsbook.fanduel.com/addToBetslip?marketId=3&selectionId=32"),
        _total_row("Athletics", "Minnesota Twins", 8.0, "https://sportsbook.fanduel.com/addToBetslip?marketId=4&selectionId=41", "https://sportsbook.fanduel.com/addToBetslip?marketId=4&selectionId=42"),
    ]

    enrich.enrich_payload(payload, odds_fetcher=lambda: {"status": "success", "odds": rows})

    for game in payload["games"]:
        assert game["combo_candidates"][0]["betslip"]["status"] == "ready"


def test_enrich_payload_no_op_when_odds_fetch_fails():
    payload = _payload_with_one_combo()

    enrich.enrich_payload(payload, odds_fetcher=lambda: {"status": "source_timeout"})

    combo = payload["games"][0]["combo_candidates"][0]
    assert "betslip" not in combo


def test_enrich_payload_handles_missing_games_key():
    assert enrich.enrich_payload({}) == {}


def test_enrich_file_round_trips_through_disk(tmp_path: Path):
    target = tmp_path / "same_game_predictions.json"
    target.write_text(json.dumps(_payload_with_one_combo()), encoding="utf-8")
    rows = [
        _moneyline_row("Detroit Tigers", "Tampa Bay Rays", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=11", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=12"),
        _total_row("Detroit Tigers", "Tampa Bay Rays", 7.5, "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=21", "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=22"),
    ]

    enrich.enrich_file(target, odds_fetcher=lambda: {"status": "success", "odds": rows})

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["games"][0]["combo_candidates"][0]["betslip"]["status"] == "ready"


def test_main_reports_ready_counts(tmp_path: Path, monkeypatch, capsys):
    target = tmp_path / "same_game_predictions.json"
    target.write_text(json.dumps(_payload_with_one_combo()), encoding="utf-8")
    rows = [
        _moneyline_row("Detroit Tigers", "Tampa Bay Rays", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=11", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=12"),
        _total_row("Detroit Tigers", "Tampa Bay Rays", 7.5, "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=21", "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=22"),
    ]
    monkeypatch.setattr(
        enrich.FanduelPublicMlbTeamMarketProvider, "collect_team_market_odds",
        lambda self: {"status": "success", "odds": rows},
    )
    monkeypatch.setattr(sys, "argv", ["enrich_same_game_betslip.py", "--same-game-predictions-path", str(target)])

    exit_code = enrich.main()

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["betslip_ready"][str(target)] == 1
