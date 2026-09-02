"""Tests for the NFL RotoWire scraper.

Deliberately uses a small, real-shaped synthetic HTML fixture instead of
hitting the live rotowire page from a test -- CI must be able to run
this suite offline, matching every other real-scraper test in this
repository (see sports/nba/tests/test_rotowire_market_props.py for the
same posture).

The fixture is constructed from the exact JavaScript-embedded shape a
real live rotowire NFL page uses (per the docstring in
fetch_nfl_market_props_rotowire.py: `const dayNFL = "N"`,
`const settings = { data: [ ... ] }`, `const prop = "..."`), with three
real per-book fields per prop (`{book}_{prop}`, `{book}_{prop}Over`,
`{book}_{prop}Under`), so a future rotowire page change that alters
that shape immediately fails these tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "sports" / "nfl" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import fetch_nfl_market_props_rotowire as scraper  # noqa: E402


def _make_settings_block(prop: str, rows: list[dict]) -> str:
    return (
        "<script>"
        f'const prop = "{prop}";'
        "const settings = { chart: {}, data: "
        f"{json.dumps(rows)}"
        "};"
        "</script>"
    )


def _base_row(**overrides) -> dict:
    row = {
        "gameID": "2978648",
        "playerID": "15872",
        "firstName": "Malik",
        "lastName": "Willis",
        "name": "Malik Willis",
        "team": "MIA",
        "opp": "@LV",
    }
    row.update(overrides)
    return row


def _passyds_row(book: str, line: str, over: str, under: str, **overrides) -> dict:
    row = _base_row(**overrides)
    row[f"{book}_passyds"] = line
    row[f"{book}_passydsOver"] = over
    row[f"{book}_passydsUnder"] = under
    return row


def test_extract_page_payload_reads_week_and_props() -> None:
    html = (
        '<script>const dayNFL = "1";</script>'
        + _make_settings_block("passyds", [_passyds_row("fanduel", "225.5", "-113", "-108")])
    )
    week, bundles = scraper.extract_rotowire_nfl_page_payload(html)
    assert week == 1
    assert set(bundles) == {"passyds"}
    assert bundles["passyds"][0]["name"] == "Malik Willis"


def test_extract_page_payload_ignores_unsupported_prop_blocks() -> None:
    """rotowire NFL page carries 25+ real prop blocks; only three are
    currently mapped to a real downstream predictor. Everything else must
    be silently dropped, not emitted as observations the pipeline would
    then discard or -- worse -- try to project against without evidence."""
    html = (
        '<script>const dayNFL = "1";</script>'
        + _make_settings_block("firsttd", [_base_row()])
        + _make_settings_block("passtd", [_base_row()])
        + _make_settings_block("passyds", [_passyds_row("fanduel", "225.5", "-113", "-108")])
    )
    week, bundles = scraper.extract_rotowire_nfl_page_payload(html)
    assert week == 1
    assert set(bundles) == {"passyds"}


def test_build_observations_requires_both_sides_at_the_same_line() -> None:
    """A book with just a line and one side isn't a usable observation
    -- flatten_event_odds (the odds-api path this scraper feeds) uses
    the same two-sided-complete gate. Emitting one-sided rows would
    silently drift the two paths apart."""
    rows = [
        _passyds_row("fanduel", "225.5", "-113", "-108"),  # complete
        _passyds_row("caesars", "224.5", "-115", None),  # missing under
        _passyds_row("draftkings", "226.5", None, "-110"),  # missing over
        _passyds_row("betrivers", None, "-113", "-113"),  # missing line
    ]
    observations = scraper.build_observations(
        week=1, bundles={"passyds": rows}, fetched_at_utc="2026-09-02T00:00:00Z"
    )
    assert len(observations) == 1
    assert observations[0]["bookmaker"] == "fanduel"
    assert observations[0]["line"] == 225.5
    assert observations[0]["over_price"] == -113.0
    assert observations[0]["under_price"] == -108.0


def test_build_observations_maps_prop_keys_to_downstream_market_keys() -> None:
    rows_passyds = [_passyds_row("fanduel", "225.5", "-113", "-108")]
    rows_rushyds = [
        {
            "gameID": "1",
            "name": "Saquon Barkley",
            "team": "PHI",
            "opp": "@DAL",
            "fanduel_rushyds": "75.5",
            "fanduel_rushydsOver": "-115",
            "fanduel_rushydsUnder": "-105",
        }
    ]
    rows_recyds = [
        {
            "gameID": "2",
            "name": "CeeDee Lamb",
            "team": "DAL",
            "opp": "PHI",
            "fanduel_recyds": "82.5",
            "fanduel_recydsOver": "-113",
            "fanduel_recydsUnder": "-108",
        }
    ]
    observations = scraper.build_observations(
        week=1,
        bundles={"passyds": rows_passyds, "rushyds": rows_rushyds, "recyds": rows_recyds},
        fetched_at_utc="2026-09-02T00:00:00Z",
    )
    by_market = {obs["market"]: obs for obs in observations}
    assert by_market["player_pass_yds"]["target"] == "passing"
    assert by_market["player_rush_yds"]["target"] == "rushing"
    assert by_market["player_reception_yds"]["target"] == "receiving"


def test_build_observations_correctly_resolves_home_and_away_from_at_prefix() -> None:
    """Rotowire encodes the away side with a leading '@' on `opp`. A row
    where `team=MIA` and `opp=@LV` is Miami playing at Las Vegas -- the
    home is LV, not MIA. Mis-inverting this would flip every real matchup
    on the resulting board."""
    away_row = _passyds_row("fanduel", "225.5", "-113", "-108", team="MIA", opp="@LV")
    observations = scraper.build_observations(
        week=1, bundles={"passyds": [away_row]}, fetched_at_utc="2026-09-02T00:00:00Z"
    )
    assert observations[0]["home_team"] == "LV"
    assert observations[0]["away_team"] == "MIA"

    home_row = _passyds_row("fanduel", "225.5", "-113", "-108", team="DAL", opp="PHI")
    observations = scraper.build_observations(
        week=1, bundles={"passyds": [home_row]}, fetched_at_utc="2026-09-02T00:00:00Z"
    )
    assert observations[0]["home_team"] == "DAL"
    assert observations[0]["away_team"] == "PHI"


def test_build_snapshot_matches_load_fixture_slate_expected_shape() -> None:
    """The whole point of this scraper is that its output is a drop-in
    substitute for the odds-api path via run_nfl_daily_predictions.py's
    --market-input flag. Verify by round-tripping through the real
    live_market.load_fixture_slate reader that consumes the file in
    production."""
    sys.path.insert(0, str(REPO_ROOT / "sports" / "nfl" / "predictions"))
    from live_market import load_fixture_slate  # noqa: E402

    html = (
        '<script>const dayNFL = "1";</script>'
        + _make_settings_block("passyds", [_passyds_row("fanduel", "225.5", "-113", "-108")])
    )
    snapshot = scraper.build_snapshot(html, fetched_at_utc="2026-09-02T00:00:00Z")
    assert snapshot["schema_version"] == 1
    assert snapshot["audit"]["provider"] == "rotowire_public_nfl_props"
    assert snapshot["audit"]["complete_two_sided_rows"] == 1
    assert snapshot["audit"]["rotowire_week"] == 1

    tmp = REPO_ROOT / "sports" / "nfl" / "tests" / "_tmp_rotowire_snapshot.json"
    try:
        scraper.write_snapshot(tmp, snapshot)
        observations, audit = load_fixture_slate(tmp)
        assert len(observations) == 1
        assert audit["replayed_from_snapshot"] is True
        obs = observations[0]
        assert obs["market"] == "player_pass_yds"
        assert obs["target"] == "passing"
        assert obs["bookmaker"] == "fanduel"
        assert obs["over_price"] == -113.0
        assert obs["under_price"] == -108.0
    finally:
        tmp.unlink(missing_ok=True)


def test_prop_map_covers_only_currently_supported_downstream_markets() -> None:
    """A future edit that adds `firsttd` or `passtd` to ROTOWIRE_PROP_MAP
    without also wiring downstream predictor support would produce
    unbacked predictions. This test locks that invariant -- the map may
    only reference market keys the NFL live_market module actually
    understands (live_market.MARKET_KEYS)."""
    sys.path.insert(0, str(REPO_ROOT / "sports" / "nfl" / "predictions"))
    from live_market import MARKET_KEYS  # noqa: E402

    for _rw_prop, (market_key, _target) in scraper.ROTOWIRE_PROP_MAP.items():
        assert market_key in MARKET_KEYS, (
            f"{market_key!r} is not in live_market.MARKET_KEYS -- adding it here "
            f"without downstream support would produce unbacked predictions."
        )
