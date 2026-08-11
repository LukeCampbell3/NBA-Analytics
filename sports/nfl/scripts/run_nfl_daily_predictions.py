#!/usr/bin/env python3
"""Build the current NFL live-shadow player-prop board."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from dotenv import load_dotenv


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.daily_policy import (  # noqa: E402
    MAXIMUM_AMERICAN_PRICE,
    MAXIMUM_WEEKLY_PICKS,
    MINIMUM_AMERICAN_PRICE,
    MINIMUM_BOOKS,
    MINIMUM_COMMON_BOOKS,
    MINIMUM_NO_VIG_ADVANTAGE,
    MINIMUM_SIDE_PROBABILITY,
    POLICY_VERSION,
    VALIDATED_TARGETS,
    build_shadow_parlay,
    score_market_offers,
    select_live_board,
)
from sports.nfl.predictions.live_market import (  # noqa: E402
    fetch_available_live_slate,
    load_fixture_slate,
    write_complete_slate,
)
from sports.nfl.predictions.live_scoring import (  # noqa: E402
    add_market_placeholders,
    attach_schedule_identity,
    build_live_scoring_frame,
)
from sports.nfl.predictions.pbp_stats import ROSTER_URL  # noqa: E402
from sports.nfl.predictions.pipeline import load_weekly_stats  # noqa: E402
from sports.nfl.scripts.fetch_historical_nfl_props import (  # noqa: E402
    SCHEDULE_URL,
    _kickoff_utc,
)


NFL_ROOT = REPO_ROOT / "sports" / "nfl"
EASTERN = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=None, help="Eastern date in YYYY-MM-DD format.")
    parser.add_argument("--as-of-utc", default=None, help="Frozen acquisition time for replay/tests.")
    parser.add_argument("--market-input", type=Path, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--regions", default="us")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument(
        "--stats",
        type=Path,
        default=NFL_ROOT / "data/raw/player_stats_deployment.parquet",
    )
    parser.add_argument("--schedule", type=Path, default=None)
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument(
        "--yardage-artifact",
        type=Path,
        default=NFL_ROOT / "model/nfl_yardage_latent_hybrid.joblib",
    )
    parser.add_argument(
        "--selector-artifact",
        type=Path,
        default=NFL_ROOT / "model/nfl_market_selector.joblib",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=NFL_ROOT / "data/evaluation/daily_policy_backtest.json",
    )
    parser.add_argument(
        "--output", type=Path, default=NFL_ROOT / "web/data/daily_predictions.json"
    )
    parser.add_argument(
        "--snapshot-output", type=Path, default=None
    )
    return parser.parse_args()


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_schedule(path: Path | None, seasons: set[int]) -> pd.DataFrame:
    schedule = pd.read_parquet(path or SCHEDULE_URL)
    schedule = schedule.loc[
        schedule["season"].isin(seasons) & schedule["game_type"].eq("REG")
    ].copy()
    schedule["commence_time_utc"] = _kickoff_utc(schedule)
    return schedule.dropna(subset=["commence_time_utc"])


def load_current_roster(path: Path | None, season: int) -> pd.DataFrame:
    if path is not None:
        return pd.read_parquet(path)
    cache = NFL_ROOT / f"data/raw/roster_weekly_{season}.parquet"
    try:
        roster = pd.read_parquet(ROSTER_URL.format(season=season))
        cache.parent.mkdir(parents=True, exist_ok=True)
        roster.to_parquet(cache, index=False)
        return roster
    except Exception:
        if cache.is_file():
            return pd.read_parquet(cache)
        return pd.DataFrame()


def withheld_payload(
    *,
    run_date: str,
    generated_at: str,
    reason: str,
    audit: dict,
    observations: int,
) -> dict:
    return {
        "schema_version": 2,
        "league": "NFL",
        "run_date": run_date,
        "generated_at_utc": generated_at,
        "publication_status": "withheld_current_pool",
        "publication_state": "withheld_current_pool",
        "mode": "live_shadow",
        "policy_profile": POLICY_VERSION,
        "plays": [],
        "daily_parlay": build_shadow_parlay([]),
        "selection": {
            "validated_targets": sorted(VALIDATED_TARGETS),
            "maximum_weekly_picks": MAXIMUM_WEEKLY_PICKS,
            "minimum_side_probability": MINIMUM_SIDE_PROBABILITY,
            "minimum_no_vig_advantage": MINIMUM_NO_VIG_ADVANTAGE,
            "american_price_range": [MINIMUM_AMERICAN_PRICE, MAXIMUM_AMERICAN_PRICE],
            "minimum_books": MINIMUM_BOOKS,
            "minimum_common_books": MINIMUM_COMMON_BOOKS,
        },
        "data_quality": {
            "status": "withheld",
            "reason": reason,
            "complete_market_observations": observations,
            "provider_audit": audit,
        },
        "policy_governance": {
            "policy_version": POLICY_VERSION,
            "publication_mode": "SHADOW_RESEARCH_ONLY",
            "candidate_authorization_enabled": False,
            "staking_enabled": False,
            "certificate_status": "PROSPECTIVE_CERTIFICATE_INACTIVE",
        },
    }


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=False)
    run_day = date.fromisoformat(args.run_date) if args.run_date else datetime.now(EASTERN).date()
    as_of = (
        datetime.fromisoformat(args.as_of_utc.replace("Z", "+00:00"))
        if args.as_of_utc
        else datetime.now(timezone.utc)
    )
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    start = datetime.combine(run_day, time.min, tzinfo=EASTERN).astimezone(timezone.utc)
    end = start + timedelta(days=max(1, int(args.window_days)))

    if args.market_input:
        observations, provider_audit = load_fixture_slate(args.market_input.resolve())
    else:
        provider_priority = tuple(
            value.strip()
            for value in os.getenv(
                "NFL_ODDS_PROVIDER_PRIORITY", "sportsgameodds,the_odds_api"
            ).split(",")
            if value.strip()
        )
        observations, provider_audit = fetch_available_live_slate(
            sportsgameodds_api_key=os.getenv("SPORTSGAMEODDS_API_KEY"),
            the_odds_api_key=(
                args.api_key
                or os.getenv("THE_ODDS_API_KEY")
                or os.getenv("ODDS_API_KEY")
            ),
            commence_from_utc=_iso(start),
            commence_to_utc=_iso(end),
            regions=args.regions,
            provider_priority=provider_priority,
        )

    snapshot_id = as_of.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = args.snapshot_output or (
        NFL_ROOT
        / f"data/production/snapshots/{run_day.isoformat()}/{snapshot_id}.json"
    )
    replaying_same_snapshot = (
        args.market_input is not None
        and args.market_input.resolve() == snapshot_path.resolve()
    )
    if not replaying_same_snapshot:
        write_complete_slate(snapshot_path, observations, provider_audit)
    if args.capture_only:
        print(
            json.dumps(
                {
                    "snapshot": str(snapshot_path),
                    "complete_market_observations": len(observations),
                },
                indent=2,
            )
        )
        return 0
    generated_at = _iso(as_of)
    if not observations:
        no_market_reason = (
            "No configured NFL odds provider is available; no sportsbook odds were validated."
            if provider_audit.get("status") == "missing_credentials"
            else "No complete two-sided regular-season NFL player-prop markets were available."
        )
        payload = withheld_payload(
            run_date=run_day.isoformat(),
            generated_at=generated_at,
            reason=no_market_reason,
            audit=provider_audit,
            observations=0,
        )
        write_payload(args.output, payload)
        print(json.dumps(payload["data_quality"], indent=2))
        return 0

    required = [args.stats, args.yardage_artifact, args.selector_artifact]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        payload = withheld_payload(
            run_date=run_day.isoformat(),
            generated_at=generated_at,
            reason=f"Required NFL scoring artifacts are unavailable: {missing}",
            audit=provider_audit,
            observations=len(observations),
        )
        write_payload(args.output, payload)
        return 0

    markets = pd.DataFrame(observations)
    schedule = load_schedule(args.schedule, {run_day.year, run_day.year + 1})
    markets = attach_schedule_identity(markets, schedule)
    markets = markets.dropna(subset=["season", "week"])
    stats = load_weekly_stats(args.stats, start_season=2018)
    current_roster = load_current_roster(args.roster, run_day.year)
    stats_with_placeholders, matched_markets, identity_audit = add_market_placeholders(
        stats, markets, current_roster=current_roster
    )
    yardage_artifact = joblib.load(args.yardage_artifact)
    selector_artifact = joblib.load(args.selector_artifact)
    live = build_live_scoring_frame(
        stats_with_placeholders,
        matched_markets,
        yardage_artifact=yardage_artifact,
        selector_artifact=selector_artifact,
    )
    scored = (
        score_market_offers(
            live,
            live["over_probability"].to_numpy(),
            now_utc=as_of,
        )
        if not live.empty
        else live
    )
    plays, selection_audit = select_live_board(scored)
    evidence = (
        json.loads(args.evidence.read_text(encoding="utf-8"))
        if args.evidence.is_file()
        else {}
    )
    payload = {
        "schema_version": 2,
        "league": "NFL",
        "run_date": run_day.isoformat(),
        "generated_at_utc": generated_at,
        "publication_status": "shadow_current_pool" if plays else "withheld_current_pool",
        "publication_state": "published_current_pool" if plays else "withheld_current_pool",
        "mode": "live_shadow",
        "policy_profile": POLICY_VERSION,
        "plays": plays,
        "daily_parlay": build_shadow_parlay(plays),
        "selection": {
            "validated_targets": sorted(VALIDATED_TARGETS),
            "maximum_weekly_picks": MAXIMUM_WEEKLY_PICKS,
            "minimum_side_probability": MINIMUM_SIDE_PROBABILITY,
            "minimum_no_vig_advantage": MINIMUM_NO_VIG_ADVANTAGE,
            "american_price_range": [MINIMUM_AMERICAN_PRICE, MAXIMUM_AMERICAN_PRICE],
            "minimum_books": MINIMUM_BOOKS,
            "minimum_common_books": MINIMUM_COMMON_BOOKS,
            **selection_audit,
        },
        "data_quality": {
            "status": "ready" if plays else "withheld",
            "reason": None if plays else "No offer survived model and execution gates.",
            "complete_market_observations": len(observations),
            "provider_audit": provider_audit,
            "identity_audit": identity_audit,
            "snapshot_path": str(snapshot_path),
        },
        "historical_evidence": evidence.get("locked_holdout", {}).get("singles", {}),
        "policy_governance": {
            "policy_version": POLICY_VERSION,
            "publication_mode": "SHADOW_RESEARCH_ONLY",
            "candidate_authorization_enabled": False,
            "staking_enabled": False,
            "certificate_status": "PROSPECTIVE_CERTIFICATE_INACTIVE",
        },
    }
    write_payload(args.output, payload)
    print(
        json.dumps(
            {
                "publication_status": payload["publication_status"],
                "plays": len(plays),
                "selection_audit": selection_audit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
