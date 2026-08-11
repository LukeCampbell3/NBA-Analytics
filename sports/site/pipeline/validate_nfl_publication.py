#!/usr/bin/env python3
"""Validate that the NFL frontend remains an honest research-only publication."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_PAYLOAD = Path("sports/nfl/web/data/daily_predictions.json")
VALIDATION_PAYLOAD = Path("sports/nfl/web/data/market_validation_summary.json")
DAILY_POLICY_EVIDENCE = Path("sports/nfl/data/evaluation/daily_policy_backtest.json")
LIVE_POLICY_VERSION = "nfl_passing_market_policy_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--run-date", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required NFL publication file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"NFL publication file must contain an object: {path}")
    return payload


def _validate_live_payload(payload: dict[str, Any], *, run_date: str | None) -> None:
    if payload.get("mode") != "live_shadow":
        raise ValueError("NFL schema v2 payload must identify live_shadow mode.")
    if payload.get("policy_profile") != LIVE_POLICY_VERSION:
        raise ValueError("NFL live payload has an unexpected policy version.")
    if run_date is not None and payload.get("run_date") != date.fromisoformat(run_date).isoformat():
        raise ValueError("NFL live payload is not for the requested run date.")
    if payload.get("publication_status") not in {
        "shadow_current_pool",
        "withheld_current_pool",
    }:
        raise ValueError("NFL live payload has an invalid publication status.")
    governance = payload.get("policy_governance") or {}
    if (
        governance.get("publication_mode") != "SHADOW_RESEARCH_ONLY"
        or bool(governance.get("candidate_authorization_enabled"))
        or bool(governance.get("staking_enabled"))
    ):
        raise ValueError("NFL live governance must remain unauthorized shadow research.")
    plays = payload.get("plays")
    if not isinstance(plays, list) or len(plays) > 12:
        raise ValueError("NFL live payload must contain at most 12 plays.")
    for index, play in enumerate(plays, start=1):
        if play.get("target") != "passing":
            raise ValueError(f"NFL live play {index} uses an unvalidated target.")
        if play.get("market_source") not in {
            "the_odds_api_live",
            "sportsgameodds_live",
        }:
            raise ValueError(f"NFL live play {index} lacks a true live market source.")
        if not bool(play.get("price_confirmed")):
            raise ValueError(f"NFL live play {index} lacks confirmed odds.")
        price = float(play.get("selected_side_price"))
        if not -150.0 <= price <= 130.0:
            raise ValueError(f"NFL live play {index} falls outside the executable price scope.")
        if int(play.get("market_books") or 0) < 2 or int(play.get("market_common_books") or 0) < 1:
            raise ValueError(f"NFL live play {index} lacks sportsbook coverage.")
        if bool(play.get("candidate_authorized")):
            raise ValueError(f"NFL live play {index} cannot be authorized without a certificate.")
    parlay = payload.get("daily_parlay") or {}
    if parlay.get("status") != "withheld" or parlay.get("validation_status") != "failed_locked_holdout":
        raise ValueError("NFL parlay must remain withheld after its failed locked holdout.")
    if bool(parlay.get("candidate_authorized")):
        raise ValueError("NFL parlay cannot be authorized.")


def validate_nfl_publication(
    *, repo_root: Path, output_dir: Path, run_date: str | None = None
) -> str:
    resolved_output = output_dir if output_dir.is_absolute() else repo_root / output_dir
    route = resolved_output / "nfl" / "predictions" / "index.html"
    if not route.is_file():
        raise FileNotFoundError(f"NFL prediction route is missing: {route}")

    source_daily = load_json(repo_root / DAILY_PAYLOAD)
    public_daily = load_json(resolved_output / "nfl/data/daily_predictions.json")
    source_validation = load_json(repo_root / VALIDATION_PAYLOAD)
    public_validation = load_json(
        resolved_output / "nfl/data/market_validation_summary.json"
    )

    if source_daily != public_daily:
        raise ValueError("NFL daily source and public payloads differ.")
    if source_validation != public_validation:
        raise ValueError("NFL validation source and public payloads differ.")

    schema_version = int(source_daily.get("schema_version") or 1)
    if schema_version >= 2:
        _validate_live_payload(source_daily, run_date=run_date)
        _validate_live_payload(public_daily, run_date=run_date)
    else:
        if source_daily.get("publication_status") != "research_only":
            raise ValueError("NFL legacy payload must remain research_only.")
        if source_daily.get("mode") != "historical_holdout":
            raise ValueError("NFL legacy payload must identify its historical_holdout mode.")
    if source_validation.get("publication_status") != "research_only_source_blocked":
        raise ValueError("NFL market evidence must remain source-blocked research.")

    deployment = (source_validation.get("gates") or {}).get("deployment") or {}
    if deployment.get("status") != "blocked":
        raise ValueError("NFL deployment gate must remain blocked without live-source evidence.")

    evidence = load_json(repo_root / DAILY_POLICY_EVIDENCE)
    if (evidence.get("gates") or {}).get("singles", {}).get("status") != "passed":
        raise ValueError("NFL frozen singles policy evidence has not passed.")
    if (evidence.get("gates") or {}).get("parlay", {}).get("status") != "failed":
        raise ValueError("NFL parlay evidence must remain failed and withheld.")

    run_date = str(source_daily.get("run_date") or "<missing>")
    targets = ",".join(source_validation.get("validated_targets") or []) or "none"
    return (
        f"NFL: status={source_daily.get('publication_status')}, "
        f"date={run_date}, plays={len(source_daily.get('plays') or [])}, "
        f"validated_targets={targets}"
    )


def main() -> None:
    args = parse_args()
    summary = validate_nfl_publication(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir,
        run_date=args.run_date,
    )
    print("NFL publication validation passed.")
    print(f"- {summary}")


if __name__ == "__main__":
    main()
