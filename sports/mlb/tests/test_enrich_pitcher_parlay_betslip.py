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

import enrich_pitcher_parlay_betslip as enrich  # noqa: E402


def _leg(pitcher_name="Kyle Leahy", line=2.5, side="over") -> dict:
    return {"pitcher_name": pitcher_name, "line": line, "side": side, "sportsbook": "fanduel"}


def _payload_with_parlay() -> dict:
    return {
        "parlay": {
            "leg_a": _leg("Kyle Leahy", 2.5, "over"),
            "leg_b": _leg("Mitch Bratt", 3.5, "under"),
            "max_hit_control": {
                "leg_a": _leg("Kyle Leahy", 2.5, "over"),
                "leg_b": _leg("Mitch Bratt", 3.5, "under"),
            },
        },
        "legs": [_leg("Kyle Leahy", 2.5, "over"), _leg("Max Fried", 4.5, "under")],
    }


def _region_index(*rows: tuple[str, float, str, str]) -> dict:
    """rows of (player, line, side, deeplink) -> the odds_index shape
    match_leg_to_regions expects, keyed exactly like build_odds_index
    (normalized player, market_type, line, side)."""
    import export_web_prediction_payload as exporter

    index = {}
    for player, line, side, deeplink in rows:
        index[(exporter.normalize_player_name(player), "pitcher_strikeouts", float(line), side)] = deeplink
    return index


def test_as_matchable_leg_maps_real_fields_to_the_generic_shape():
    leg = _leg("Kyle Leahy", 2.5, "over")
    matchable = enrich._as_matchable_leg(leg)
    assert matchable == {"player": "Kyle Leahy", "target": "K", "line": 2.5, "side": "over"}


def test_iter_leg_dicts_yields_parlay_control_and_flat_legs():
    payload = _payload_with_parlay()
    legs = list(enrich._iter_leg_dicts(payload))
    assert len(legs) == 6  # leg_a, leg_b, max_hit_control leg_a/leg_b, 2 flat legs


def test_enrich_payload_attaches_deeplinks_by_region_to_every_leg():
    payload = _payload_with_parlay()
    region_indexes = {
        "NY": _region_index(
            ("Kyle Leahy", 2.5, "over", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=11"),
            ("Mitch Bratt", 3.5, "under", "https://sportsbook.fanduel.com/addToBetslip?marketId=2&selectionId=22"),
        ),
    }

    enrich.enrich_payload(payload, region_indexes=region_indexes)

    assert payload["parlay"]["leg_a"]["deeplinks_by_region"]["NY"].endswith("selectionId=11")
    assert payload["parlay"]["leg_b"]["deeplinks_by_region"]["NY"].endswith("selectionId=22")
    assert payload["parlay"]["max_hit_control"]["leg_a"]["deeplinks_by_region"]["NY"].endswith("selectionId=11")
    assert payload["legs"][0]["deeplinks_by_region"]["NY"].endswith("selectionId=11")
    # Max Fried has no match in this region's index -- left empty, never guessed.
    assert payload["legs"][1]["deeplinks_by_region"] == {}


def test_enrich_payload_no_op_when_no_region_indexes():
    payload = _payload_with_parlay()

    result = enrich.enrich_payload(payload, region_indexes=None)

    assert result is payload
    assert "deeplinks_by_region" not in payload["parlay"]["leg_a"]


def test_enrich_payload_handles_missing_parlay_and_legs():
    assert enrich.enrich_payload({}, region_indexes={"NY": {}}) == {}


def test_enrich_file_round_trips_through_disk(tmp_path: Path):
    target = tmp_path / "pitcher_parlay_predictions.json"
    target.write_text(json.dumps(_payload_with_parlay()), encoding="utf-8")
    region_indexes = {
        "NY": _region_index(
            ("Kyle Leahy", 2.5, "over", "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=11"),
        ),
    }

    enrich.enrich_file(target, region_indexes=region_indexes)

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["parlay"]["leg_a"]["deeplinks_by_region"]["NY"].endswith("selectionId=11")


def test_main_reports_region_coverage(tmp_path: Path, monkeypatch, capsys):
    target = tmp_path / "pitcher_parlay_predictions.json"
    target.write_text(json.dumps(_payload_with_parlay()), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["enrich_pitcher_parlay_betslip.py", "--pitcher-parlay-predictions-path", str(target), "--disable-multi-region-betslip"],
    )

    exit_code = enrich.main()

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["region_coverage"] == {}
    assert out["legs_enriched"][str(target)] == 0
