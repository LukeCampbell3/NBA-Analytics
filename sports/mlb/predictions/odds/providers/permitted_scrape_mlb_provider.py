#!/usr/bin/env python3
"""Permission-gated MLB scraper for public structured or embedded JSON."""
from __future__ import annotations

import html as html_module
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odds_contract import ensure_contract, stable_hash


SCRIPT_PATTERN = re.compile(
    r'<script[^>]+(?:id=["\']mlb-odds-data["\']|data-mlb-odds)[^>]*>(.*?)</script>',
    flags=re.IGNORECASE | re.DOTALL,
)
WORKSPACE = Path(__file__).resolve().parents[5]
CAPTURE_DIR = WORKSPACE / "sports" / "mlb" / "data" / "market_odds" / "production_shadow" / "raw" / "scrape"


class PermittedScrapeMlbProvider:
    """Read a documented JSON contract from an explicitly authorized source."""

    def __init__(self, source_url: str | None = None, fixture_path: Path | str | None = None):
        self.source_url = source_url if source_url is not None else os.environ.get("MLB_ODDS_SCRAPE_SOURCE", "")
        replay_path = os.environ.get("MLB_ODDS_SCRAPE_FIXTURE_PATH", "") if os.environ.get("MLB_ODDS_FIXTURE_REPLAY") == "1" else ""
        configured_fixture = fixture_path or replay_path
        self.fixture_path = Path(configured_fixture) if configured_fixture else None
        self.enabled = self.fixture_path is not None or os.environ.get("MLB_ODDS_SCRAPE_ENABLED", "0") == "1"
        self.authorized = self.fixture_path is not None or os.environ.get("MLB_ODDS_SCRAPE_AUTHORIZED", "0") == "1"
        self._evidence: dict[str, Any] = {}

    def validate_config(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "message": "MLB scrape adapter is disabled"}
        if not self.authorized:
            return {"status": "source_blocked", "message": "Scrape source lacks explicit automated-use authorization"}
        if self.fixture_path is None and not self.source_url:
            return {"status": "missing_source", "message": "MLB_ODDS_SCRAPE_SOURCE not set"}
        if self.fixture_path is not None and not self.fixture_path.is_file():
            return {"status": "missing_source", "message": f"Fixture not found: {self.fixture_path}"}
        return {"status": "ok"}

    def get_evidence(self) -> dict[str, Any]:
        return dict(self._evidence)

    def collect_player_props(self) -> dict[str, Any]:
        config = self.validate_config()
        if config["status"] != "ok":
            return config
        started = time.monotonic()
        requested_at = datetime.now(timezone.utc)
        response_status = 200
        final_url = str(self.fixture_path) if self.fixture_path else self.source_url
        try:
            if self.fixture_path is not None:
                body = self.fixture_path.read_text(encoding="utf-8")
            else:
                response = requests.get(
                    self.source_url,
                    timeout=float(os.environ.get("MLB_ODDS_SCRAPE_TIMEOUT_SECONDS", "20")),
                    headers={"User-Agent": "NBA-Analytics-MLB-Odds/1.0"},
                )
                response_status = response.status_code
                final_url = response.url
                if response.status_code in {401, 403}:
                    return self._failure("source_blocked", "SOURCE_BLOCKED", requested_at, started, response_status, final_url, response.text)
                if response.status_code == 429:
                    return self._failure("rate_limited", "SOURCE_RATE_LIMITED", requested_at, started, response_status, final_url, response.text)
                response.raise_for_status()
                body = response.text
            payload = self._parse_payload(body)
            records = payload.get("records")
            if not isinstance(records, list):
                return self._failure("schema_drift", "MLB_ODDS_SOURCE_SCHEMA_DRIFT", requested_at, started, response_status, final_url, body)
            if not records:
                status = "SOURCE_EMPTY" if payload.get("slate_empty") is True else "MLB_ODDS_SOURCE_SCHEMA_DRIFT"
                return self._failure("no_props" if status == "SOURCE_EMPTY" else "schema_drift", status, requested_at, started, response_status, final_url, body)
            min_records = int(os.environ.get("MLB_ODDS_SCRAPE_MIN_RECORDS", "1"))
            max_records = int(os.environ.get("MLB_ODDS_SCRAPE_MAX_RECORDS", "100000"))
            if len(records) < min_records or len(records) > max_records:
                return self._failure(
                    "schema_drift", "MLB_ODDS_SOURCE_SCHEMA_DRIFT: implausible record count",
                    requested_at, started, response_status, final_url, body,
                )
            contract_error = self._preflight_records(records)
            if contract_error:
                return self._failure(
                    "schema_drift", f"MLB_ODDS_SOURCE_SCHEMA_DRIFT: {contract_error}",
                    requested_at, started, response_status, final_url, body,
                )

            source_timestamp = payload.get("source_timestamp")
            accepted: list[dict[str, Any]] = []
            rejected = 0
            for record in records:
                if not isinstance(record, dict):
                    rejected += 1
                    continue
                row = dict(record)
                row.setdefault("source", "scrape")
                row.setdefault("source_url_or_endpoint", final_url)
                row.setdefault("acquisition_method", "scrape")
                row.setdefault("observed_at_utc", source_timestamp)
                row.setdefault("source_updated_at_utc", source_timestamp)
                row.setdefault("raw_record_hash", stable_hash(record))
                row.setdefault("parser_version", "permitted-json-scraper-v1")
                accepted.append(row)
            self._evidence = self._build_evidence(
                requested_at, started, response_status, final_url, body, len(records), len(accepted), rejected,
                source_timestamp, {} if not rejected else {"non_object_record": rejected},
            )
            self._persist_capture(body)
            if not accepted:
                return {"status": "source_invalid_data", "message": "SOURCE_INVALID_DATA", "evidence": self._evidence}
            return {"status": "success", "odds": accepted, "evidence": self._evidence}
        except requests.Timeout:
            return self._failure("source_timeout", "SOURCE_TIMEOUT", requested_at, started, response_status, final_url, "")
        except requests.RequestException as exc:
            return self._failure("api_error", str(exc)[:150], requested_at, started, response_status, final_url, "")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return self._failure("schema_drift", f"MLB_ODDS_SOURCE_SCHEMA_DRIFT: {str(exc)[:120]}", requested_at, started, response_status, final_url, "")

    def _parse_payload(self, body: str) -> dict[str, Any]:
        stripped = body.lstrip()
        if stripped.startswith("{"):
            payload = json.loads(stripped)
        else:
            match = SCRIPT_PATTERN.search(body)
            if not match:
                if re.search(r"loading|spinner|please wait", body, flags=re.IGNORECASE):
                    raise ValueError("loading-state page contained no market payload")
                raise ValueError("expected mlb-odds-data container missing")
            payload = json.loads(html_module.unescape(match.group(1)).strip())
        if not isinstance(payload, dict):
            raise ValueError("odds payload is not an object")
        return payload

    @staticmethod
    def _preflight_records(records: list[Any]) -> str:
        required = [
            "sportsbook", "event_id", "external_event_id", "player_name", "home_team", "away_team",
            "game_start_utc", "market_type", "side", "line", "price_american",
        ]
        seen_prices: dict[tuple[str, ...], float] = {}
        for record in records:
            if not isinstance(record, dict):
                return "non-object record"
            missing = [name for name in required if record.get(name) is None or record.get(name) == ""]
            if missing:
                return f"missing required fields: {','.join(missing)}"
            try:
                line = float(record["line"])
                price = float(record["price_american"])
            except (TypeError, ValueError):
                return "unknown line or price format"
            if -100.0 < price < 100.0:
                return "unknown American price format"
            identity = tuple(
                str(record.get(name, "")).strip().lower()
                for name in ["sportsbook", "external_event_id", "external_player_id", "player_name", "market_type", "side"]
            ) + (str(line),)
            prior = seen_prices.get(identity)
            if prior is not None and prior != price:
                return "duplicate side has conflicting prices"
            seen_prices[identity] = price
        return ""

    def _failure(
        self, status: str, message: str, requested_at: datetime, started: float,
        response_status: int, final_url: str, body: str,
    ) -> dict[str, Any]:
        self._evidence = self._build_evidence(
            requested_at, started, response_status, final_url, body, 0, 0, 0, None, {message: 1}
        )
        self._persist_capture(body)
        return {"status": status, "message": message, "evidence": self._evidence}

    def _persist_capture(self, body: str) -> None:
        if self.fixture_path is not None or os.environ.get("PYTEST_CURRENT_TEST"):
            return
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (CAPTURE_DIR / f"response_{stamp}.txt").write_text(body[:1_000_000], encoding="utf-8")
        (CAPTURE_DIR / f"evidence_{stamp}.json").write_text(json.dumps(self._evidence, indent=2), encoding="utf-8")

    @staticmethod
    def _build_evidence(
        requested_at: datetime, started: float, response_status: int, final_url: str, body: str,
        discovered: int, accepted: int, rejected: int, source_timestamp: Any,
        rejection_reasons: dict[str, int],
    ) -> dict[str, Any]:
        completed = datetime.now(timezone.utc)
        parsed_source = pd.to_datetime(source_timestamp, utc=True, errors="coerce")
        freshness = None if pd.isna(parsed_source) else max(0.0, (completed - parsed_source.to_pydatetime()).total_seconds())
        return {
            "requested_at_utc": requested_at.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "response_status": response_status,
            "final_url": final_url,
            "content_hash": stable_hash(body),
            "parser_version": "permitted-json-scraper-v1",
            "records_discovered": discovered,
            "records_accepted": accepted,
            "records_rejected": rejected,
            "rejection_reasons": rejection_reasons,
            "source_timestamp": source_timestamp,
            "freshness_seconds": freshness,
            "latency_seconds": time.monotonic() - started,
        }

    def normalize(self, raw_odds: list[dict[str, Any]]) -> pd.DataFrame:
        return ensure_contract(
            pd.DataFrame(raw_odds),
            source="scrape",
            acquisition_method="scrape",
            source_endpoint=self.source_url or str(self.fixture_path or ""),
            parser_version="permitted-json-scraper-v1",
        )
