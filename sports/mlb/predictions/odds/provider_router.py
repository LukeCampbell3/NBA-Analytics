#!/usr/bin/env python3
"""Provider-neutral MLB odds acquisition, validation, and reconciliation."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[4]
MLB_SHADOW_DIR = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"
ODDS_STATUS_PATH = MLB_SHADOW_DIR / "odds_source_status.json"
SNAPSHOT_DIR = WORKSPACE / "sports" / "mlb" / "data" / "market_odds" / "production_shadow" / "snapshots"
PROVIDER_CONFIG_PATH = MLB_SHADOW_DIR / "mlb_provider_config.json"
SOURCE_QUALITY_PATH = MLB_SHADOW_DIR / "odds_source_quality_history.jsonl"

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "providers"))

from odds_contract import ensure_contract, reconcile_observations, validate_contract


PROVIDERS_AVAILABLE: dict[str, Any] = {}

try:
    from fanduel_public_mlb_provider import FanduelPublicMlbProvider
    PROVIDERS_AVAILABLE["fanduel_public"] = FanduelPublicMlbProvider
except ImportError:
    pass
try:
    from permitted_scrape_mlb_provider import PermittedScrapeMlbProvider
    PROVIDERS_AVAILABLE["scrape"] = PermittedScrapeMlbProvider
except ImportError:
    pass
try:
    from the_odds_api_mlb_provider import TheOddsApiMlbProvider
    PROVIDERS_AVAILABLE["the_odds_api"] = TheOddsApiMlbProvider
except ImportError:
    pass
try:
    from existing_mlb_odds_provider import ExistingMlbOddsProvider
    PROVIDERS_AVAILABLE["existing_provider"] = ExistingMlbOddsProvider
except ImportError:
    pass


def validate_normalized_odds(
    df: pd.DataFrame,
    *,
    max_age_seconds: int = 3600,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper around the provider-neutral contract validator."""
    valid, report = validate_contract(df, max_age_seconds=max_age_seconds, now=now)
    report.update(
        {
            "valid_odds_rate": report.get("valid_record_rate", 0.0),
            "missing_required_fields": [],
            "invalid_odds_count": report.get("rejection_reasons", {}).get("INVALID_AMERICAN_PRICE", 0),
            "markets": sorted(valid["market_type"].dropna().astype(str).unique().tolist()) if not valid.empty else [],
            "books": sorted(valid["sportsbook"].dropna().astype(str).unique().tolist()) if not valid.empty else [],
            "players": int(valid["player_name"].nunique()) if not valid.empty else 0,
            "events": int(valid["event_id"].nunique()) if not valid.empty else 0,
        }
    )
    return report


def build_provider_result(
    provider_name: str,
    status: str,
    *,
    is_live: bool = False,
    is_cache: bool = False,
    cache_age_minutes: float = 0.0,
    failure_reason: str = "",
    rows_collected: int = 0,
    valid_rows: int = 0,
    markets_covered: list[str] | None = None,
    retry_after: str | None = None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "provider_name": provider_name,
        "provider_status": status,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "is_live": is_live,
        "is_cache": is_cache,
        "cache_age_minutes": cache_age_minutes,
        "failure_reason": failure_reason,
        "rows_collected": rows_collected,
        "valid_rows": valid_rows,
        "markets_covered": markets_covered or [],
        "validation": validation or {},
    }
    if retry_after is not None:
        result["retry_after"] = retry_after
    return result


class MlbProviderRouter:
    """Query configured adapters, preserve observations, and fail closed."""

    def __init__(
        self,
        max_cache_age_seconds: int = 3600,
        *,
        provider_priority: list[str] | None = None,
        provider_classes: dict[str, Any] | None = None,
        now: datetime | None = None,
    ):
        self.config = self._load_config()
        env_age = os.environ.get("MLB_ODDS_MAX_AGE_SECONDS")
        self.max_cache_age = int(env_age or self.config.get("freshness_limit_seconds", max_cache_age_seconds))
        self.min_valid_odds_rate = float(self.config.get("min_valid_odds_rate", 0.70))
        env_priority = [item.strip() for item in os.environ.get("MLB_ODDS_PROVIDER_PRIORITY", "").split(",") if item.strip()]
        configured = self.config.get("provider_priority") or self.config.get("fallback_order") or []
        default = ["fanduel_public", "the_odds_api", "scrape", "existing_provider", "fresh_cache"]
        self.provider_priority = provider_priority or env_priority or configured or default
        self.provider_priority = ["the_odds_api" if value == "odds_api_io" else value for value in self.provider_priority]
        self.provider_priority = [value for value in self.provider_priority if value != "sportsgameodds"]
        if provider_priority is None and "fanduel_public" not in self.provider_priority:
            self.provider_priority.insert(0, "fanduel_public")
        if "fresh_cache" not in self.provider_priority:
            self.provider_priority.append("fresh_cache")
        self.provider_classes = dict(PROVIDERS_AVAILABLE if provider_classes is None else provider_classes)
        self.now = now
        self.attempts: list[dict[str, Any]] = []
        self.provider_results: list[dict[str, Any]] = []

    @staticmethod
    def _load_config() -> dict[str, Any]:
        if PROVIDER_CONFIG_PATH.exists():
            try:
                return json.loads(PROVIDER_CONFIG_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "provider_priority": ["fanduel_public", "the_odds_api", "scrape", "existing_provider", "fresh_cache"],
            "freshness_limit_seconds": 3600,
            "min_valid_odds_rate": 0.70,
        }

    def get_fresh_odds(self) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        self.attempts = []
        self.provider_results = []
        info: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sport": "MLB",
            "provider_priority": self.provider_priority,
            "fallback_order": self.provider_priority,
            "providers_tried": [],
            "provider_results": [],
            "successful_provider": None,
            "successful_providers": [],
            "rows_obtained": 0,
            "no_fresh_odds_available": False,
            "valid_odds_rate": 0.0,
            "terminal_status": "",
            "source_state": "",
        }
        live_frames: list[pd.DataFrame] = []
        first_live_index: int | None = None

        for index, provider_name in enumerate(self.provider_priority):
            if provider_name == "fresh_cache" and live_frames:
                continue
            attempt = self._try_provider(provider_name)
            self.attempts.append(attempt)
            info["providers_tried"].append({k: v for k, v in attempt.items() if not k.startswith("_")})
            provider_result = build_provider_result(
                provider_name,
                attempt["status"],
                is_live=attempt["status"] == "success" and provider_name != "fresh_cache",
                is_cache=attempt["status"] == "success" and provider_name == "fresh_cache",
                cache_age_minutes=attempt.get("snapshot_age_seconds", 0) / 60.0,
                failure_reason=attempt.get("error", ""),
                rows_collected=attempt.get("rows_raw", 0),
                valid_rows=attempt.get("rows_normalized", 0),
                markets_covered=attempt.get("_markets", []),
                retry_after=attempt.get("retry_after"),
                validation=attempt.get("validation", {}),
            )
            self.provider_results.append(provider_result)
            info["provider_results"].append(provider_result)
            if attempt["status"] != "success":
                continue
            frame = attempt.get("_dataframe")
            if frame is None or frame.empty:
                continue
            if first_live_index is None:
                first_live_index = index
            live_frames.append(frame)
            info["successful_providers"].append(provider_name)

        if live_frames:
            reconciled = reconcile_observations(pd.concat(live_frames, ignore_index=True))
            info["successful_provider"] = info["successful_providers"][0]
            info["rows_obtained"] = int(len(reconciled))
            info["valid_odds_rate"] = float(sum(len(frame) for frame in live_frames) / max(1, sum(a.get("rows_raw", 0) for a in self.attempts)))
            info["terminal_status"] = "MLB_FRESH_ODDS_AVAILABLE"
            info["source_state"] = "MLB_ODDS_PRIMARY_HEALTHY" if first_live_index == 0 else "MLB_ODDS_FALLBACK_ACTIVE"
            reconciled["provider_status"] = "success"
            cache_only = info["successful_providers"] == ["fresh_cache"]
            reconciled["is_cache"] = cache_only
            reconciled["is_live"] = ~reconciled["is_cache"]
            if not reconciled["is_cache"].all():
                self._write_snapshot(reconciled, info["successful_provider"])
            info["source_quality"] = self._build_source_quality(reconciled)
            self._append_source_quality(info["source_quality"])
            self._save_status(info)
            return reconciled, info

        info["no_fresh_odds_available"] = True
        statuses = {attempt["status"] for attempt in self.attempts}
        if "schema_drift" in statuses:
            info["terminal_status"] = "MLB_ODDS_SOURCE_SCHEMA_DRIFT"
        elif "source_blocked" in statuses:
            info["terminal_status"] = "MLB_ODDS_SOURCE_BLOCKED"
        else:
            info["terminal_status"] = "MLB_WAITING_FOR_FRESH_PROPS"
        info["source_state"] = info["terminal_status"]
        info["error"] = "No configured provider produced fresh, valid, supported MLB prop markets"
        info["source_quality"] = self._build_source_quality(pd.DataFrame())
        self._append_source_quality(info["source_quality"])
        self._save_status(info)
        return None, info

    def _try_provider(self, provider_name: str) -> dict[str, Any]:
        attempt: dict[str, Any] = {
            "provider": provider_name,
            "status": "failed",
            "rows_raw": 0,
            "rows_normalized": 0,
            "valid_odds_rate": 0.0,
            "snapshot_age_seconds": 0,
            "error": "",
            "retry_after": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation": {},
            "_dataframe": None,
            "_markets": [],
        }
        if provider_name == "fresh_cache":
            return self._try_fresh_cache(attempt)
        provider_class = self.provider_classes.get(provider_name)
        if provider_class is None:
            attempt["status"] = "not_installed"
            attempt["error"] = f"Provider module not available: {provider_name}"
            return attempt

        try:
            provider = provider_class()
            config = provider.validate_config()
            if config.get("status") != "ok":
                attempt["status"] = config.get("status", "invalid_config")
                attempt["error"] = config.get("message", "Provider configuration is invalid")
                return attempt
            result = provider.collect_player_props()
            if result.get("status") != "success":
                attempt["status"] = result.get("status", "unexpected")
                attempt["error"] = result.get("message", "Provider acquisition failed")
                attempt["retry_after"] = result.get("retry_after")
                attempt["evidence"] = result.get("evidence", {})
                return attempt
            raw_odds = result.get("odds", [])
            attempt["rows_raw"] = len(raw_odds)
            if not raw_odds:
                attempt["status"] = "no_props"
                attempt["error"] = "Provider returned an empty successful response"
                return attempt
            normalized = provider.normalize(raw_odds)
            valid, validation = validate_contract(
                normalized,
                max_age_seconds=self.max_cache_age,
                now=self.now,
                reject_started=True,
            )
            attempt["validation"] = validation
            attempt["rows_normalized"] = len(valid)
            attempt["valid_odds_rate"] = validation.get("valid_record_rate", 0.0)
            attempt["_markets"] = sorted(valid["market_type"].unique().tolist()) if not valid.empty else []
            if valid.empty:
                reasons = validation.get("rejection_reasons", {})
                if "STALE_ODDS" in reasons:
                    attempt["status"] = "stale_odds"
                elif "UNSUPPORTED_MARKET" in reasons:
                    attempt["status"] = "no_supported_markets"
                else:
                    attempt["status"] = "source_invalid_data"
                attempt["error"] = f"No valid observations after validation: {reasons}"
                return attempt
            if attempt["valid_odds_rate"] < self.min_valid_odds_rate:
                attempt["status"] = "below_valid_odds_threshold"
                attempt["error"] = f"Valid record rate {attempt['valid_odds_rate']:.3f} < {self.min_valid_odds_rate:.3f}"
                return attempt
            attempt["status"] = "success"
            attempt["_dataframe"] = valid
            attempt["evidence"] = result.get("evidence", {})
            return attempt
        except Exception as exc:
            attempt["status"] = "exception"
            attempt["error"] = str(exc)[:200]
            return attempt

    def _try_fresh_cache(self, attempt: dict[str, Any]) -> dict[str, Any]:
        files = sorted(SNAPSHOT_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime) if SNAPSHOT_DIR.exists() else []
        if not files:
            attempt["status"] = "no_cache"
            attempt["error"] = "No MLB provider-neutral cache files found"
            return attempt
        latest = files[-1]
        try:
            frame = ensure_contract(pd.read_csv(latest), source="fresh_cache", acquisition_method="cache", source_endpoint=str(latest))
            valid, validation = validate_contract(frame, max_age_seconds=self.max_cache_age, now=self.now, reject_started=True)
            attempt["rows_raw"] = len(frame)
            attempt["rows_normalized"] = len(valid)
            attempt["validation"] = validation
            attempt["valid_odds_rate"] = validation.get("valid_record_rate", 0.0)
            if valid.empty:
                attempt["status"] = "stale_cache"
                attempt["error"] = f"No fresh cache observations: {validation.get('rejection_reasons', {})}"
                return attempt
            observed = pd.to_datetime(valid["observed_at_utc"], utc=True, errors="coerce").max()
            current = pd.Timestamp(self.now or datetime.now(timezone.utc))
            attempt["snapshot_age_seconds"] = int(max(0.0, (current - observed).total_seconds())) if pd.notna(observed) else 0
            attempt["_markets"] = sorted(valid["market_type"].unique().tolist())
            attempt["status"] = "success"
            attempt["_dataframe"] = valid
            return attempt
        except Exception as exc:
            attempt["status"] = "exception"
            attempt["error"] = str(exc)[:200]
            return attempt

    @staticmethod
    def _write_snapshot(df: pd.DataFrame, provider_name: str) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        df.to_csv(SNAPSHOT_DIR / f"{provider_name}_mlb_props_{stamp}.csv", index=False)

    def _build_source_quality(self, reconciled: pd.DataFrame) -> list[dict[str, Any]]:
        quality: list[dict[str, Any]] = []
        for attempt in self.attempts:
            provider = attempt["provider"]
            validation = attempt.get("validation", {})
            rows = int(validation.get("rows", attempt.get("rows_raw", 0)))
            stale = int(validation.get("rejection_reasons", {}).get("STALE_ODDS", 0))
            provider_rows = reconciled.loc[reconciled["source"].eq(provider)] if not reconciled.empty else pd.DataFrame()
            evidence = attempt.get("evidence", {})
            quality.append(
                {
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source": provider,
                    "request_success_rate": 1.0 if attempt["status"] == "success" else 0.0,
                    "parse_success_rate": 0.0 if attempt["status"] in {"schema_drift", "normalization_failed"} else 1.0,
                    "valid_record_rate": float(validation.get("valid_record_rate", 0.0)),
                    "event_match_rate": float(provider_rows["event_id"].astype(str).str.strip().ne("").mean()) if not provider_rows.empty else 0.0,
                    "player_match_rate": float(provider_rows["player_id"].astype(str).str.strip().ne("").mean()) if not provider_rows.empty else 0.0,
                    "stale_record_rate": float(stale / rows) if rows else 0.0,
                    "cross_source_agreement": float(provider_rows["source_count"].gt(1).mean()) if not provider_rows.empty else 0.0,
                    "schema_drift_count": 1 if attempt["status"] == "schema_drift" else 0,
                    "average_latency": evidence.get("latency_seconds"),
                    "availability": 1.0 if attempt["status"] == "success" else 0.0,
                }
            )
        return quality

    @staticmethod
    def _append_source_quality(rows: list[dict[str, Any]]) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST") or not rows:
            return
        SOURCE_QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SOURCE_QUALITY_PATH.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    @staticmethod
    def _save_status(info: dict[str, Any]) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        ODDS_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ODDS_STATUS_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
