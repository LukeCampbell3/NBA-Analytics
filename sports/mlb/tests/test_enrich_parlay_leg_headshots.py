from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import enrich_parlay_leg_headshots as enrich  # noqa: E402


def _payload_with_legs() -> dict:
    return {
        "parlays": {
            "shadow_candidate": {
                "leg_1": {"player": "Pete Alonso", "side": "OVER", "target": "R"},
                "leg_2": {"player": "Pete Crow-Armstrong", "side": "OVER", "target": "TB"},
            },
            "selected_parlay": None,
        }
    }


def test_find_leg_dicts_collects_named_legs_only():
    payload = _payload_with_legs()
    legs = enrich.find_leg_dicts(payload)
    assert [leg["player"] for leg in legs] == ["Pete Alonso", "Pete Crow-Armstrong"]


def test_find_leg_dicts_handles_missing_parlays_key():
    assert enrich.find_leg_dicts({}) == []
    assert enrich.find_leg_dicts({"parlays": {}}) == []
    assert enrich.find_leg_dicts({"parlays": {"selected_parlay": None}}) == []


def test_find_leg_dicts_also_collects_legacy_daily_parlay_ticket_legs():
    payload = {
        "daily_parlay": {
            "selected_ticket": {
                "legs": [
                    {"player": "CJ Abrams", "player_display_name": "CJ Abrams", "target": "H"},
                    {"player": "Alex Bregman", "player_display_name": "Alex Bregman", "target": "H"},
                ]
            }
        }
    }
    legs = enrich.find_leg_dicts(payload)
    assert [leg["player"] for leg in legs] == ["CJ Abrams", "Alex Bregman"]


def test_find_leg_dicts_handles_missing_daily_parlay_ticket():
    assert enrich.find_leg_dicts({"daily_parlay": {}}) == []
    assert enrich.find_leg_dicts({"daily_parlay": {"selected_ticket": None}}) == []


def test_enrich_payload_attaches_real_headshot_urls_for_resolved_players():
    payload = _payload_with_legs()

    def fake_resolver(name: str):
        return {"Pete Alonso": 624413, "Pete Crow-Armstrong": 691718}.get(name)

    enrich.enrich_payload(payload, person_id_resolver=fake_resolver)

    leg1 = payload["parlays"]["shadow_candidate"]["leg_1"]
    leg2 = payload["parlays"]["shadow_candidate"]["leg_2"]
    assert leg1["player_headshot_url"] == "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/624413/headshot/67/current"
    assert leg1["player_headshot_fallback_url"] == "https://midfield.mlbstatic.com/v1/people/624413/headshot/67/current"
    assert leg2["player_headshot_url"].endswith("/people/691718/headshot/67/current")


def test_enrich_payload_leaves_unresolved_players_without_headshot_fields():
    payload = _payload_with_legs()

    def fake_resolver(_name: str):
        return None

    enrich.enrich_payload(payload, person_id_resolver=fake_resolver)

    leg1 = payload["parlays"]["shadow_candidate"]["leg_1"]
    assert "player_headshot_url" not in leg1
    assert "player_headshot_fallback_url" not in leg1


def test_enrich_payload_calls_resolver_once_per_unique_player_name():
    payload = _payload_with_legs()
    payload["parlays"]["selected_parlay"] = {
        "leg_1": {"player": "Pete Alonso", "side": "OVER", "target": "R"},
    }
    calls: list[str] = []

    def counting_resolver(name: str):
        calls.append(name)
        return 624413

    enrich.enrich_payload(payload, person_id_resolver=counting_resolver)
    assert calls == ["Pete Alonso", "Pete Crow-Armstrong"]


def test_enrich_file_round_trips_through_disk(tmp_path: Path):
    target = tmp_path / "daily_predictions.json"
    target.write_text(json.dumps(_payload_with_legs()), encoding="utf-8")

    enrich.enrich_file(target, person_id_resolver=lambda name: {"Pete Alonso": 624413}.get(name))

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["parlays"]["shadow_candidate"]["leg_1"]["player_headshot_url"].endswith("/624413/headshot/67/current")
    assert "player_headshot_url" not in written["parlays"]["shadow_candidate"]["leg_2"]


def test_main_reports_enriched_counts(tmp_path: Path, monkeypatch, capsys):
    target = tmp_path / "daily_predictions.json"
    target.write_text(json.dumps(_payload_with_legs()), encoding="utf-8")
    monkeypatch.setattr(
        enrich.exporter, "search_person_id_by_name",
        lambda name: {"Pete Alonso": 624413, "Pete Crow-Armstrong": 691718}.get(name),
    )
    monkeypatch.setattr(sys, "argv", ["enrich_parlay_leg_headshots.py", "--daily-predictions-path", str(target)])

    exit_code = enrich.main()

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["enriched"][str(target)] == 2
