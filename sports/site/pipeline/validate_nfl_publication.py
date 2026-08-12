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
META_POLICY_EVIDENCE = Path("sports/nfl/data/evaluation/pick_meta_backtest.json")
LIVE_POLICY_VERSION = "nfl_passing_loss_aware_meta_policy_v2"
EXPECTED_META_POLICY = {
    "minimum_side_probability": 0.58,
    "minimum_no_vig_advantage": 0.1,
    "minimum_price": -130,
    "maximum_price": 130,
    "weekly_cap": 6,
}


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
    selection = payload.get("selection") or {}
    if selection.get("loss_aware_meta_policy") != EXPECTED_META_POLICY:
        raise ValueError("NFL live payload is missing the frozen loss-aware meta-policy.")
    calibration = selection.get("confidence_calibration") or {}
    if calibration.get("method") != "identity" or calibration.get("status") != "passed":
        raise ValueError("NFL live payload lacks validated confidence calibration.")
    governance = payload.get("policy_governance") or {}
    if (
        governance.get("publication_mode") != "SHADOW_RESEARCH_ONLY"
        or bool(governance.get("candidate_authorization_enabled"))
        or bool(governance.get("staking_enabled"))
    ):
        raise ValueError("NFL live governance must remain unauthorized shadow research.")
    plays = payload.get("plays")
    if not isinstance(plays, list) or len(plays) > EXPECTED_META_POLICY["weekly_cap"]:
        raise ValueError("NFL live payload exceeds the frozen six-play cap.")
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
        if not -130.0 <= price <= 130.0:
            raise ValueError(f"NFL live play {index} falls outside the executable price scope.")
        if float(play.get("model_hit_probability") or 0) < 0.58:
            raise ValueError(f"NFL live play {index} falls below the meta confidence gate.")
        if float(play.get("probability_advantage") or 0) < 0.10:
            raise ValueError(f"NFL live play {index} falls below the meta advantage gate.")
        if play.get("meta_policy_score") is None:
            raise ValueError(f"NFL live play {index} is missing its meta-policy score.")
        if play.get("raw_model_probability") is None or play.get("calibrated_hit_probability") is None:
            raise ValueError(f"NFL live play {index} is missing confidence provenance.")
        if not bool(play.get("confidence_in_support")):
            raise ValueError(f"NFL live play {index} falls outside calibration support.")
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
    meta_evidence = load_json(repo_root / META_POLICY_EVIDENCE)
    if meta_evidence.get("sport") != "NFL":
        raise ValueError("NFL meta-policy evidence has the wrong sport contract.")
    if (meta_evidence.get("locked_recent_validation") or {}).get("status") != "passed":
        raise ValueError("NFL recent locked meta-policy validation has not passed.")
    if (meta_evidence.get("confidence_calibration") or {}).get("status") != "passed":
        raise ValueError("NFL confidence calibration evidence has not passed.")
    if (meta_evidence.get("deployment") or {}).get("status") != "shadow_only":
        raise ValueError("NFL meta-policy must remain shadow-only.")

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
