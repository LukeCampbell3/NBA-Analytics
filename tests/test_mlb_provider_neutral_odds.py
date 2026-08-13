from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ODDS_DIR = REPO_ROOT / "sports" / "mlb" / "predictions" / "odds"
PROVIDERS_DIR = ODDS_DIR / "providers"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mlb_odds" / "permitted_embedded_json.html"
sys.path.insert(0, str(ODDS_DIR))
sys.path.insert(0, str(PROVIDERS_DIR))

import provider_router
from odds_contract import CONTRACT_COLUMNS, ensure_contract, reconcile_observations, validate_contract
from fanduel_public_mlb_provider import FanduelPublicMlbProvider
from permitted_scrape_mlb_provider import PermittedScrapeMlbProvider
from the_odds_api_mlb_provider import TheOddsApiMlbProvider


NOW = datetime(2026, 8, 4, 15, 0, tzinfo=timezone.utc)


def observation(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "fixture",
        "source_market_id": "market-1",
        "sportsbook": "fanduel",
        "event_id": "event-1",
        "external_event_id": "external-event-1",
        "player_id": "aaronjudge",
        "external_player_id": "source-aaronjudge",
        "player_name": "Aaron Judge",
        "home_team": "New York Yankees",
        "away_team": "Texas Rangers",
        "game_start_utc": "2026-08-04T23:05:00Z",
        "league": "MLB",
        "market_type": "batter_total_bases",
        "side": "over",
        "line": 1.5,
        "price_american": -115,
        "observed_at_utc": "2026-08-04T14:55:00Z",
        "source_updated_at_utc": "2026-08-04T14:55:00Z",
        "source_url_or_endpoint": "https://example.test/mlb-odds",
        "acquisition_method": "fixture",
        "raw_record_hash": "a" * 64,
        "parser_version": "fixture-v1",
        "normalization_version": "mlb-odds-contract-v1",
        "validation_status": "UNVALIDATED",
    }
    row.update(overrides)
    return row


class HealthyScrape:
    def validate_config(self):
        return {"status": "ok"}

    def collect_player_props(self):
        return {"status": "success", "odds": [observation(source="scrape", acquisition_method="scrape")]}

    def normalize(self, rows):
        return ensure_contract(pd.DataFrame(rows), source="scrape", acquisition_method="scrape")


class HealthyApi:
    def validate_config(self):
        return {"status": "ok"}

    def collect_player_props(self):
        return {"status": "success", "odds": [observation(source="the_odds_api", acquisition_method="api")]}

    def normalize(self, rows):
        return ensure_contract(pd.DataFrame(rows), source="the_odds_api", acquisition_method="api")


class MissingCredentials:
    def validate_config(self):
        return {"status": "missing_credentials", "message": "key missing"}


class MissingSource:
    def validate_config(self):
        return {"status": "missing_source", "message": "source missing"}


class FailedPrimary:
    def validate_config(self):
        return {"status": "ok"}

    def collect_player_props(self):
        return {"status": "api_error", "message": "upstream unavailable"}


def isolated_router(monkeypatch, tmp_path, priority, classes):
    monkeypatch.setattr(provider_router, "SNAPSHOT_DIR", tmp_path / "empty-cache")
    return provider_router.MlbProviderRouter(
        provider_priority=priority,
        provider_classes=classes,
        now=NOW,
    )


def test_sportsgameodds_cannot_reenter_production_priority(monkeypatch, tmp_path) -> None:
    router = isolated_router(
        monkeypatch,
        tmp_path,
        ["scrape", "sportsgameodds"],
        {"scrape": HealthyScrape, "sportsgameodds": MissingCredentials},
    )
    frame, info = router.get_fresh_odds()
    assert frame is not None and not frame.empty
    assert info["terminal_status"] == "MLB_FRESH_ODDS_AVAILABLE"
    assert info["source_state"] == "MLB_ODDS_PRIMARY_HEALTHY"
    assert "sportsgameodds" not in info["provider_priority"]
    assert [result["provider_name"] for result in info["provider_results"]] == ["scrape"]


def test_legacy_environment_priority_cannot_disable_free_fanduel_provider(monkeypatch) -> None:
    monkeypatch.setenv("MLB_ODDS_PROVIDER_PRIORITY", "sportsgameodds,fresh_cache")

    router = provider_router.MlbProviderRouter(provider_classes={})

    assert router.provider_priority == ["fanduel_public", "fresh_cache"]


def test_missing_scraper_source_does_not_block_healthy_api(monkeypatch, tmp_path) -> None:
    router = isolated_router(
        monkeypatch,
        tmp_path,
        ["scrape", "the_odds_api"],
        {"scrape": MissingSource, "the_odds_api": HealthyApi},
    )
    frame, info = router.get_fresh_odds()
    assert frame is not None and not frame.empty
    assert info["source_state"] == "MLB_ODDS_FALLBACK_ACTIVE"
    assert info["successful_provider"] == "the_odds_api"


def test_missing_credentials_for_only_enabled_api_are_reported(monkeypatch, tmp_path) -> None:
    router = isolated_router(monkeypatch, tmp_path, ["the_odds_api"], {"the_odds_api": MissingCredentials})
    frame, info = router.get_fresh_odds()
    assert frame is None
    assert info["terminal_status"] == "MLB_WAITING_FOR_FRESH_PROPS"
    assert info["provider_results"][0]["provider_status"] == "missing_credentials"


def test_primary_failure_activates_secondary(monkeypatch, tmp_path) -> None:
    router = isolated_router(
        monkeypatch,
        tmp_path,
        ["scrape", "the_odds_api"],
        {"scrape": FailedPrimary, "the_odds_api": HealthyApi},
    )
    _, info = router.get_fresh_odds()
    assert info["source_state"] == "MLB_ODDS_FALLBACK_ACTIVE"
    assert [row["provider_status"] for row in info["provider_results"][:2]] == ["api_error", "success"]


def test_loading_page_and_changed_html_trigger_schema_drift(tmp_path) -> None:
    for name, body in [("loading.html", "<html>Loading...</html>"), ("changed.html", "<html><div>markets</div></html>")]:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        result = PermittedScrapeMlbProvider(fixture_path=path).collect_player_props()
        assert result["status"] == "schema_drift"
        assert "MLB_ODDS_SOURCE_SCHEMA_DRIFT" in result["message"]


def test_scraper_rejects_conflicting_duplicate_side(tmp_path) -> None:
    first = observation(source="scrape", acquisition_method="scrape")
    second = {**first, "price_american": -105, "raw_record_hash": "b" * 64}
    payload = {"source_timestamp": "2026-08-04T14:55:00Z", "records": [first, second]}
    path = tmp_path / "conflict.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = PermittedScrapeMlbProvider(fixture_path=path).collect_player_props()

    assert result["status"] == "schema_drift"
    assert "conflicting prices" in result["message"]


def test_embedded_json_fixture_normalizes_and_replays_deterministically() -> None:
    provider = PermittedScrapeMlbProvider(fixture_path=FIXTURE)
    first = provider.collect_player_props()
    second = provider.collect_player_props()
    assert first["status"] == second["status"] == "success"
    assert first["odds"] == second["odds"]
    normalized = provider.normalize(first["odds"])
    assert list(normalized["side"]) == ["over", "under"]
    assert set(CONTRACT_COLUMNS).issubset(normalized.columns)
    assert first["evidence"]["records_accepted"] == 2


def test_the_odds_api_fixture_uses_same_contract() -> None:
    payload = {
        "id": "event-1",
        "commence_time": "2026-08-04T23:05:00Z",
        "home_team": "New York Yankees",
        "away_team": "Texas Rangers",
        "bookmakers": [
            {
                "key": "fanduel",
                "last_update": "2026-08-04T14:55:00Z",
                "markets": [
                    {
                        "key": "batter_total_bases",
                        "last_update": "2026-08-04T14:55:00Z",
                        "outcomes": [{"name": "Over", "description": "Aaron Judge", "price": -115, "point": 1.5}],
                    }
                ],
            }
        ],
    }
    provider = TheOddsApiMlbProvider(fixture_payloads=[payload])
    result = provider.collect_player_props()
    normalized = provider.normalize(result["odds"])
    assert result["status"] == "success"
    assert set(CONTRACT_COLUMNS).issubset(normalized.columns)
    assert normalized.iloc[0]["price_decimal"] > 1.0
    scrape_columns = set(PermittedScrapeMlbProvider(fixture_path=FIXTURE).normalize(
        PermittedScrapeMlbProvider(fixture_path=FIXTURE).collect_player_props()["odds"]
    ).columns)
    assert set(normalized.columns) == scrape_columns


def test_fanduel_public_feed_maps_main_and_alternate_player_props_without_credentials() -> None:
    event_id = "35932482"
    event = {
        "eventId": int(event_id),
        "eventTypeId": 7511,
        "name": "Cincinnati Reds (A Abbott) @ Chicago White Sox (D Martin)",
        "openDate": "2026-08-04T23:05:00.000Z",
    }
    content = {"attachments": {"events": {event_id: event}}}
    event_attachment = {**event, "inPlay": False}

    def market(
        market_id: str,
        market_type: str,
        market_name: str,
        runners: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "marketId": market_id,
            "eventId": int(event_id),
            "marketType": market_type,
            "marketName": market_name,
            "marketStatus": "OPEN",
            "inPlay": False,
            "runners": runners,
        }

    def runner(selection_id: int, name: str, price: int, handicap: float = 0.0) -> dict[str, object]:
        return {
            "selectionId": selection_id,
            "runnerName": name,
            "runnerStatus": "ACTIVE",
            "handicap": handicap,
            "secondaryLogo": "https://assets.sportsbook.fanduel.com/images/team/mlb/cincinnati_reds.png",
            "winRunnerOdds": {"americanDisplayOdds": {"americanOddsInt": price}},
        }

    batter_payload = {
        "attachments": {
            "events": {event_id: event_attachment},
            "markets": {
                "734.100": market(
                    "734.100",
                    "PLAYER_TO_RECORD_A_HIT",
                    "To Record A Hit",
                    [runner(1001, "Elly De La Cruz", -250)],
                ),
                "734.101": market(
                    "734.101",
                    "PLAYER_TO_RECORD_2+_HITS",
                    "To Record 2+ Hits",
                    [runner(1001, "Elly De La Cruz", 195)],
                ),
            },
        }
    }
    pitcher_payload = {
        "attachments": {
            "events": {event_id: event_attachment},
            "markets": {
                "734.200": market(
                    "734.200",
                    "PITCHER_C_TOTAL_STRIKEOUTS",
                    "Andrew Abbott - Strikeouts",
                    [
                        runner(2001, "Andrew Abbott Over", -110, 5.5),
                        runner(2002, "Andrew Abbott Under", -116, 5.5),
                    ],
                ),
                "734.201": market(
                    "734.201",
                    "PITCHER_C_STRIKEOUTS",
                    "Andrew Abbott - Alt Strikeouts",
                    [runner(2003, "Andrew Abbott 7+ Strikeouts", 220)],
                ),
            },
        }
    }
    provider = FanduelPublicMlbProvider(
        content_payload=content,
        event_payloads={
            (event_id, "batter-props"): batter_payload,
            (event_id, "pitcher-props"): pitcher_payload,
        },
        now=NOW,
        sleep_fn=lambda _seconds: None,
    )

    result = provider.collect_player_props()
    normalized = provider.normalize(result["odds"])

    assert result["status"] == "success"
    assert result["cost_profile"] == "anonymous_public_read_only_no_subscription"
    assert len(normalized) == 5
    assert set(normalized["market_type"]) == {"batter_hits", "pitcher_strikeouts"}
    assert set(normalized.loc[normalized["player_name"].eq("Elly De La Cruz"), "line"]) == {0.5, 1.5}
    assert set(normalized.loc[normalized["player_name"].eq("Andrew Abbott"), "line"]) == {5.5, 6.5}
    assert set(normalized["source"]) == {"fanduel_public"}
    assert normalized["sportsbook_deeplink"].str.match(
        r"https://sportsbook\.fanduel\.com/addToBetslip\?marketId=734\.\d+&selectionId=\d+"
    ).all()
    assert normalized["source_market_id"].str.match(r"734\.\d+:\d+").all()


def test_duplicate_records_are_deduplicated_without_losing_sources() -> None:
    rows = [
        observation(source="scrape", raw_record_hash="a" * 64),
        observation(source="scrape", raw_record_hash="a" * 64),
        observation(source="the_odds_api", raw_record_hash="b" * 64),
    ]
    reconciled = reconcile_observations(pd.DataFrame(rows))
    assert len(reconciled) == 2
    assert set(reconciled["source"]) == {"scrape", "the_odds_api"}
    assert reconciled["source_count"].eq(2).all()


def test_unlike_lines_are_not_merged_and_disagreement_is_recorded() -> None:
    rows = [
        observation(source="scrape", line=1.5, price_american=-115),
        observation(source="the_odds_api", line=2.5, price_american=105, raw_record_hash="b" * 64),
    ]
    reconciled = reconcile_observations(pd.DataFrame(rows))
    assert len(reconciled) == 2
    assert set(reconciled["line"]) == {1.5, 2.5}
    assert reconciled["line_disagreement"].all()
    assert reconciled["source_count"].eq(1).all()


def test_cross_source_price_disagreement_selects_freshest() -> None:
    rows = [
        observation(source="scrape", observed_at_utc="2026-08-04T14:50:00Z", source_updated_at_utc="2026-08-04T14:50:00Z"),
        observation(
            source="the_odds_api", price_american=-105, raw_record_hash="b" * 64,
            observed_at_utc="2026-08-04T14:58:00Z", source_updated_at_utc="2026-08-04T14:58:00Z",
        ),
    ]
    reconciled = reconcile_observations(pd.DataFrame(rows))
    assert reconciled["price_disagreement"].all()
    selected = reconciled.loc[reconciled["canonical_selected"]].iloc[0]
    assert selected["source"] == "the_odds_api"
    assert selected["price_american"] == -105


def test_stale_scraped_and_api_odds_are_rejected_identically() -> None:
    for source, method in [("scrape", "scrape"), ("the_odds_api", "api")]:
        row = observation(
            source=source,
            acquisition_method=method,
            observed_at_utc="2026-08-04T12:00:00Z",
            source_updated_at_utc="2026-08-04T12:00:00Z",
        )
        valid, report = validate_contract(pd.DataFrame([row]), max_age_seconds=3600, now=NOW)
        assert valid.empty
        assert report["rejection_reasons"]["STALE_ODDS"] == 1


def test_price_format_conflict_and_unsupported_market_fail_validation() -> None:
    row = observation(price_decimal=9.99, market_type="unsupported_market")
    valid, report = validate_contract(pd.DataFrame([row]), max_age_seconds=3600, now=NOW)
    assert valid.empty
    assert report["rejection_reasons"]["PRICE_FORMAT_CONFLICT"] == 1
    assert report["rejection_reasons"]["UNSUPPORTED_MARKET"] == 1


def test_started_event_is_rejected_and_staking_has_no_fresh_source(monkeypatch, tmp_path) -> None:
    class StartedApi(HealthyApi):
        def collect_player_props(self):
            return {"status": "success", "odds": [observation(game_start_utc="2026-08-04T14:00:00Z")]}

    router = isolated_router(monkeypatch, tmp_path, ["the_odds_api"], {"the_odds_api": StartedApi})
    frame, info = router.get_fresh_odds()
    assert frame is None
    assert info["no_fresh_odds_available"] is True
    assert info["terminal_status"] == "MLB_WAITING_FOR_FRESH_PROPS"
    assert info["provider_results"][0]["validation"]["rejection_reasons"]["EVENT_STARTED"] == 1
