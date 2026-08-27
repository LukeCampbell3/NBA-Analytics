from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
MLB_ODDS_PROVIDERS_ROOT = REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))
sys.path.insert(0, str(MLB_ODDS_PROVIDERS_ROOT))

import enrich_parlay_leg_betslip as enrich  # noqa: E402


def _odds_row(player_name: str, market_type: str, line: float, side: str, market_id: str, selection_id: str) -> dict:
    return {
        "player_name": player_name,
        "market_type": market_type,
        "line": line,
        "side": side,
        "sportsbook_deeplink": f"https://sportsbook.fanduel.com/addToBetslip?marketId={market_id}&selectionId={selection_id}",
    }


def _pair_with_two_legs() -> dict:
    return {
        "leg_1": {"player": "Pete Alonso", "side": "OVER", "target": "R", "line": 0.5},
        "leg_2": {"player": "Pete Crow-Armstrong", "side": "OVER", "target": "TB", "line": 1.5},
    }


def test_build_odds_index_keys_on_normalized_player_market_line_side():
    rows = [_odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "111.1", "222")]
    index = enrich.build_odds_index(rows)
    assert index[("pete alonso", "batter_runs_scored", 0.5, "over")].endswith("marketId=111.1&selectionId=222")


def test_match_leg_to_deeplink_resolves_via_target_to_market_map():
    rows = [_odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "111.1", "222")]
    index = enrich.build_odds_index(rows)
    leg = {"player": "Pete Alonso", "side": "OVER", "target": "R", "line": 0.5}
    assert enrich.match_leg_to_deeplink(leg, index) is not None


def test_match_leg_to_deeplink_returns_none_for_unknown_target():
    leg = {"player": "Pete Alonso", "side": "OVER", "target": "NOT_A_REAL_TARGET", "line": 0.5}
    assert enrich.match_leg_to_deeplink(leg, {}) is None


def test_enrich_pair_builds_real_multi_leg_url_when_both_legs_resolve():
    pair = _pair_with_two_legs()
    rows = [
        _odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111"),
        _odds_row("Pete Crow-Armstrong", "batter_total_bases", 1.5, "over", "734.2", "222"),
    ]
    index = enrich.build_odds_index(rows)

    enrich.enrich_pair(pair, index)

    assert pair["betslip"]["status"] == "ready"
    assert pair["betslip_url"].startswith("https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?")
    assert "marketId%5B0%5D=734.1" in pair["betslip_url"]
    assert "marketId%5B1%5D=734.2" in pair["betslip_url"]
    assert pair["leg_1"]["sportsbook_deeplink"]
    assert pair["leg_2"]["sportsbook_deeplink"]


def test_enrich_pair_marks_unavailable_when_one_leg_has_no_live_match():
    pair = _pair_with_two_legs()
    rows = [_odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111")]
    index = enrich.build_odds_index(rows)

    enrich.enrich_pair(pair, index)

    assert pair["betslip"]["status"] == "unavailable"
    assert pair["betslip"]["reason"] == "one_or_more_legs_have_no_live_fanduel_selection"
    assert "betslip_url" not in pair


def test_enrich_pair_noop_on_pair_with_no_leg_dicts():
    pair = {"leg_1": None}
    enrich.enrich_pair(pair, {})
    assert "betslip" not in pair


def test_enrich_payload_enriches_both_selected_and_shadow_pairs():
    payload = {
        "parlays": {
            "selected_parlay": _pair_with_two_legs(),
            "shadow_candidate": _pair_with_two_legs(),
        }
    }
    rows = [
        _odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111"),
        _odds_row("Pete Crow-Armstrong", "batter_total_bases", 1.5, "over", "734.2", "222"),
    ]

    def fake_fetcher():
        return {"status": "success", "odds": rows}

    enrich.enrich_payload(payload, odds_fetcher=fake_fetcher)

    assert payload["parlays"]["selected_parlay"]["betslip"]["status"] == "ready"
    assert payload["parlays"]["shadow_candidate"]["betslip"]["status"] == "ready"


def test_enrich_payload_no_op_when_odds_fetch_fails():
    payload = {"parlays": {"shadow_candidate": _pair_with_two_legs()}}

    def failing_fetcher():
        return {"status": "source_timeout"}

    enrich.enrich_payload(payload, odds_fetcher=failing_fetcher)
    assert "betslip" not in payload["parlays"]["shadow_candidate"]


def test_enrich_payload_handles_missing_parlays_key():
    assert enrich.enrich_payload({}) == {}


def test_enrich_single_play_attaches_real_deeplink_from_main_board_fields():
    play = {"player_display_name": "Pete Alonso", "direction": "OVER", "target": "R", "market_line": 0.5}
    rows = [_odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111")]
    index = enrich.build_odds_index(rows)

    enrich.enrich_single_play(play, index)

    assert play["sportsbook_deeplink"] == "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=111"


def test_enrich_single_play_leaves_deeplink_absent_when_no_live_match():
    play = {"player_display_name": "Pete Alonso", "direction": "OVER", "target": "R", "market_line": 0.5}
    enrich.enrich_single_play(play, {})
    assert "sportsbook_deeplink" not in play


def test_enrich_payload_attaches_deeplinks_to_main_board_plays():
    payload = {
        "plays": [
            {"player_display_name": "Pete Alonso", "direction": "OVER", "target": "R", "market_line": 0.5},
        ],
    }
    rows = [_odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111")]

    enrich.enrich_payload(payload, odds_fetcher=lambda: {"status": "success", "odds": rows})

    assert payload["plays"][0]["sportsbook_deeplink"] == "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=111"


def test_enrich_file_round_trips_through_disk(tmp_path: Path):
    target = tmp_path / "daily_predictions.json"
    target.write_text(json.dumps({"parlays": {"shadow_candidate": _pair_with_two_legs()}}), encoding="utf-8")
    rows = [
        _odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111"),
        _odds_row("Pete Crow-Armstrong", "batter_total_bases", 1.5, "over", "734.2", "222"),
    ]

    enrich.enrich_file(target, odds_fetcher=lambda: {"status": "success", "odds": rows})

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["parlays"]["shadow_candidate"]["betslip"]["status"] == "ready"


def test_build_multi_region_odds_indexes_fetches_once_per_real_state():
    calls: list[str] = []

    def fake_factory(region: str):
        calls.append(region)
        rows = [_odds_row(f"Player {region}", "batter_runs_scored", 0.5, "over", f"{region}.1", f"{region}-1")]
        return type("FakeProvider", (), {"collect_player_props": lambda self: {"status": "success", "odds": rows}})()

    indexes = enrich.build_multi_region_odds_indexes(("NY", "PA"), provider_factory=fake_factory)

    assert calls == ["NY", "PA"]
    assert set(indexes) == {"NY", "PA"}
    assert indexes["NY"][("player ny", "batter_runs_scored", 0.5, "over")].endswith("marketId=NY.1&selectionId=NY-1")


def test_build_multi_region_odds_indexes_skips_a_state_whose_real_fetch_fails():
    def fake_factory(region: str):
        if region == "PA":
            return type("FakeProvider", (), {"collect_player_props": lambda self: {"status": "source_timeout"}})()
        rows = [_odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111")]
        return type("FakeProvider", (), {"collect_player_props": lambda self: {"status": "success", "odds": rows}})()

    indexes = enrich.build_multi_region_odds_indexes(("NY", "PA"), provider_factory=fake_factory)

    assert set(indexes) == {"NY"}  # PA's real fetch failed -- absent, never a guessed/empty stand-in


def test_match_leg_to_regions_only_includes_states_with_a_real_match():
    leg = {"player": "Pete Alonso", "side": "OVER", "target": "R", "line": 0.5}
    region_indexes = {
        "NY": enrich.build_odds_index([_odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "NY.1", "111")]),
        "PA": enrich.build_odds_index([]),  # no real PA match for this leg
    }

    deeplinks = enrich.match_leg_to_regions(leg, region_indexes)

    assert deeplinks == {"NY": "https://sportsbook.fanduel.com/addToBetslip?marketId=NY.1&selectionId=111"}


def test_enrich_pair_attaches_deeplinks_by_region_to_each_leg_when_region_indexes_given():
    pair = _pair_with_two_legs()
    rows = [
        _odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111"),
        _odds_row("Pete Crow-Armstrong", "batter_total_bases", 1.5, "over", "734.2", "222"),
    ]
    index = enrich.build_odds_index(rows)
    region_indexes = {"NY": index}

    enrich.enrich_pair(pair, index, region_indexes=region_indexes)

    assert pair["leg_1"]["deeplinks_by_region"]["NY"].endswith("marketId=734.1&selectionId=111")
    assert pair["leg_2"]["deeplinks_by_region"]["NY"].endswith("marketId=734.2&selectionId=222")
    # The original single-region field is untouched, for backward compat.
    assert pair["leg_1"]["sportsbook_deeplink"]


def test_enrich_pair_omits_deeplinks_by_region_when_no_region_indexes_given():
    pair = _pair_with_two_legs()
    rows = [
        _odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111"),
        _odds_row("Pete Crow-Armstrong", "batter_total_bases", 1.5, "over", "734.2", "222"),
    ]
    index = enrich.build_odds_index(rows)

    enrich.enrich_pair(pair, index)

    assert "deeplinks_by_region" not in pair["leg_1"]


def test_enrich_single_play_attaches_deeplinks_by_region():
    play = {"player_display_name": "Pete Alonso", "direction": "OVER", "target": "R", "market_line": 0.5}
    rows = [_odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111")]
    index = enrich.build_odds_index(rows)
    region_indexes = {"NY": index}

    enrich.enrich_single_play(play, index, region_indexes=region_indexes)

    assert play["deeplinks_by_region"] == {"NY": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=111"}


def test_main_reports_ready_counts(tmp_path: Path, monkeypatch, capsys):
    target = tmp_path / "daily_predictions.json"
    target.write_text(json.dumps({"parlays": {"shadow_candidate": _pair_with_two_legs()}}), encoding="utf-8")
    rows = [
        _odds_row("Pete Alonso", "batter_runs_scored", 0.5, "over", "734.1", "111"),
        _odds_row("Pete Crow-Armstrong", "batter_total_bases", 1.5, "over", "734.2", "222"),
    ]
    monkeypatch.setattr(
        enrich.FanduelPublicMlbProvider, "collect_player_props",
        lambda self: {"status": "success", "odds": rows},
    )
    monkeypatch.setattr(
        sys, "argv",
        ["enrich_parlay_leg_betslip.py", "--daily-predictions-path", str(target), "--disable-multi-region-betslip"],
    )

    exit_code = enrich.main()

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["betslip_ready"][str(target)] == 1
    assert out["region_coverage"] == {}
