#!/usr/bin/env python3
"""Immutable prospective evidence collection for BALANCED_RANKING_V3.

Snapshots contain only information available at decision time. Outcomes are
written later to a separate, hash-linked settlement file. Existing artifacts
are append-only: an identical rerun is idempotent and any conflicting rerun
fails closed.

Player/team/game identity fields from the decision-time pool are retained in
future snapshots so downstream V4 publication does not lose the exact identity
contract before the live validation overlay runs. Historical snapshots remain
valid and load unchanged because the extra fields are additive.

This is a shadow research path. It does not read or write the public payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import balanced_ranking_v3 as ranking  # noqa: E402
import select_high_precision_predictions as shp  # noqa: E402
from build_v11_eligible_training_set import DEFAULT_PROCESSED_ROOT, parse_v11_args  # noqa: E402
from validate_historical_final_pools import build_actual_lookup, grade_result, normalize_player_key  # noqa: E402


SCHEMA = "balanced_ranking_v3_prospective_evidence_v1"
DEFAULT_ROOT = SCRIPT_ROOT.parent / "data" / "predictions" / "balanced_ranking_v3_prospective"
SNAPSHOT_NAME = "snapshot.json"
SETTLEMENT_NAME = "settlement.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def artifact_path(root: Path, slate_date: str, filename: str) -> Path:
    date.fromisoformat(slate_date)
    return root / slate_date / filename


def _write_immutable(path: Path, payload: dict[str, Any], identity_fields: tuple[str, ...]) -> str:
    """Write once; accept only a byte-equivalent scientific identity later.

    Operational timestamps are intentionally excluded from identity_fields so
    a retried job is idempotent. The stored first-observed timestamp remains
    untouched.
    """
    identity = {field: payload[field] for field in identity_fields}
    identity_hash = _digest(identity)
    payload = dict(payload)
    payload["identity_sha256"] = identity_hash
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity_sha256") != identity_hash:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return "created"


def candidate_id(candidate: Any) -> str:
    raw = "|".join(
        (
            candidate.run_date.isoformat(),
            str(candidate.game_id),
            str(candidate.player),
            str(candidate.target),
            str(candidate.direction),
            f"{float(candidate.market_line):g}",
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _identity_field(candidate: Any, attribute: str, raw_key: str) -> str:
    """Read an identity field without making old/test Candidate objects fail.

    The selected Candidate exposes player_id/team directly while opponent and
    numeric team IDs currently remain in its original CSV row. Keeping this
    helper fail-soft preserves backward compatibility for historical tests and
    old snapshots; the separate live publication gate remains fail-closed.
    """
    value = getattr(candidate, attribute, None)
    if value not in (None, ""):
        return str(value)
    raw = getattr(candidate, "raw", None)
    if isinstance(raw, dict):
        return str(raw.get(raw_key, "") or "")
    return ""


def build_snapshot(pool_csv: Path, *, observed_at_utc: str, source_commit: str | None = None) -> dict[str, Any]:
    candidates, *_ = shp.prepare_candidates(parse_v11_args(pool_csv))
    rows: list[dict[str, Any]] = []
    slate_dates: set[str] = {candidate.run_date.isoformat() for candidate in candidates}
    for candidate in candidates:
        if not (
            candidate.target == ranking.TARGET
            and candidate.direction == ranking.DIRECTION
            and abs(float(candidate.market_line) - ranking.LINE) < 1e-9
            and candidate.market_source == "real"
            and bool(candidate.price_confirmed)
            and candidate.selected_side_price is not None
        ):
            continue
        balanced = float(candidate.final_hit_probability)
        market = float(candidate.market_implied_probability)
        price = float(candidate.selected_side_price)
        base_ev = balanced * ranking.american_to_decimal(price) - 1.0
        v19_eligible = balanced >= 0.60 and balanced >= market + 0.01 and base_ev >= 0.0
        rows.append(
            {
                "candidate_id": candidate_id(candidate),
                "game_id": str(candidate.game_id),
                "commence_time_utc": str(getattr(candidate, "commence_time_utc", "") or ""),
                "player": str(candidate.player),
                "player_id": _identity_field(candidate, "player_id", "Player_ID"),
                "team": _identity_field(candidate, "team", "Team"),
                "team_id": _identity_field(candidate, "team_id", "Team_ID"),
                "opponent": _identity_field(candidate, "opponent", "Opponent"),
                "opponent_id": _identity_field(candidate, "opponent_id", "Opponent_ID"),
                "is_home": _identity_field(candidate, "is_home", "Is_Home"),
                "target": candidate.target,
                "direction": candidate.direction,
                "line": float(candidate.market_line),
                "balanced_probability": balanced,
                "market_probability": market,
                "selected_side_price": price,
                "selected_sportsbook_key": str(getattr(candidate, "selected_sportsbook_key", "") or ""),
                "market_source": candidate.market_source,
                "price_confirmed": True,
                "base_ev": base_ev,
                "v19_eligible": v19_eligible,
                "v19_order_score": base_ev + (1000.0 if v19_eligible else 0.0),
            }
        )
    if len(slate_dates) != 1:
        raise ValueError(f"expected exactly one slate date, found {sorted(slate_dates)}")
    rows.sort(key=lambda row: row["candidate_id"])
    return {
        "schema": SCHEMA,
        "record_type": "pregame_snapshot",
        "v3_version": ranking.V3_VERSION,
        "preregistration_spec_hash": ranking.PREREGISTRATION_SPEC_HASH,
        "slate_date": next(iter(slate_dates)),
        "observed_at_utc": observed_at_utc,
        "source_commit": source_commit,
        "source_pool": pool_csv.name,
        "candidate_count": len(rows),
        "candidates": rows,
    }


def capture(pool_csv: Path, root: Path, *, observed_at_utc: str, source_commit: str | None = None) -> tuple[Path, str]:
    payload = build_snapshot(pool_csv, observed_at_utc=observed_at_utc, source_commit=source_commit)
    path = artifact_path(root, payload["slate_date"], SNAPSHOT_NAME)
    status = _write_immutable(
        path,
        payload,
        ("schema", "record_type", "v3_version", "preregistration_spec_hash", "slate_date", "source_commit", "source_pool", "candidate_count", "candidates"),
    )
    return path, status


def settle_snapshot(
    snapshot_path: Path,
    root: Path,
    *,
    settled_at_utc: str,
    as_of_date: date,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> tuple[Path, str] | None:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    slate_date = str(snapshot["slate_date"])
    if as_of_date <= date.fromisoformat(slate_date):
        raise ValueError("settlement must occur strictly after the slate date")
    lookup = build_actual_lookup(processed_root)
    completed_games = {
        (str(key[0]), str(key[3]))
        for key in lookup
        if isinstance(key, tuple) and len(key) == 4
    }
    results: list[dict[str, Any]] = []
    for row in snapshot["candidates"]:
        key = (slate_date, normalize_player_key(row["player"]), row["target"], str(row["game_id"]))
        actual = lookup.get(key)
        if actual is None:
            # Once this game is represented in the finalized results store,
            # an absent player/target observation is a non-participating bet
            # rather than evidence that the entire slate is still pending.
            # Preserve it as a void so the immutable candidate set remains
            # auditable, but never feed it to ranking inference.
            if (slate_date, str(row["game_id"])) not in completed_games:
                return None
            results.append({"candidate_id": row["candidate_id"], "result": "void", "win": None})
            continue
        result = grade_result(actual, float(row["line"]), row["direction"])
        if result not in {"win", "loss", "push"}:
            raise RuntimeError(f"unsupported grade {result!r} for {row['candidate_id']}")
        normalized_result = "void" if result == "push" else result
        results.append({
            "candidate_id": row["candidate_id"],
            "result": normalized_result,
            "win": None if normalized_result == "void" else int(normalized_result == "win"),
        })
    results.sort(key=lambda row: row["candidate_id"])
    payload = {
        "schema": SCHEMA,
        "record_type": "settlement",
        "slate_date": slate_date,
        "settled_at_utc": settled_at_utc,
        "snapshot_identity_sha256": snapshot["identity_sha256"],
        "candidate_count": len(results),
        "graded_count": sum(row["result"] in {"win", "loss"} for row in results),
        "void_count": sum(row["result"] == "void" for row in results),
        "results": results,
    }
    path = artifact_path(root, slate_date, SETTLEMENT_NAME)
    status = _write_immutable(
        path,
        payload,
        ("schema", "record_type", "slate_date", "snapshot_identity_sha256", "candidate_count", "graded_count", "void_count", "results"),
    )
    return path, status


def load_settled_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot_path in sorted(root.glob(f"*/{SNAPSHOT_NAME}")):
        settlement_path = snapshot_path.with_name(SETTLEMENT_NAME)
        if not settlement_path.exists():
            continue
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
        if settlement.get("snapshot_identity_sha256") != snapshot.get("identity_sha256"):
            raise RuntimeError(f"settlement does not reference snapshot: {settlement_path}")
        outcomes = {row["candidate_id"]: row for row in settlement["results"]}
        expected = {row["candidate_id"] for row in snapshot["candidates"]}
        if set(outcomes) != expected:
            raise RuntimeError(f"settlement candidate set mismatch: {settlement_path}")
        for candidate in snapshot["candidates"]:
            outcome = outcomes[candidate["candidate_id"]]
            if outcome.get("result") == "void":
                continue
            if outcome.get("result") not in {"win", "loss"} or outcome.get("win") not in {0, 1}:
                raise RuntimeError(f"invalid graded outcome in {settlement_path}")
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "date": snapshot["slate_date"],
                    "game_id": candidate["game_id"],
                    "win": int(outcome["win"]),
                    **{field: float(candidate[field]) for field in ranking.SCORE_FIELDS},
                }
            )
    return rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--pool-csv", type=Path, required=True)
    capture_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    capture_parser.add_argument("--source-commit")
    settle_parser = sub.add_parser("settle")
    settle_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    settle_parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    settle_parser.add_argument("--as-of-date", type=date.fromisoformat, default=date.today())
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.command == "capture":
        path, status = capture(
            args.pool_csv,
            args.root,
            observed_at_utc=_utc_now(),
            source_commit=args.source_commit or os.environ.get("GITHUB_SHA"),
        )
        print(f"{status}: {path}")
    elif args.command == "settle":
        for snapshot_path in sorted(args.root.glob(f"*/{SNAPSHOT_NAME}")):
            if snapshot_path.with_name(SETTLEMENT_NAME).exists():
                continue
            result = settle_snapshot(snapshot_path, args.root, settled_at_utc=_utc_now(), as_of_date=args.as_of_date, processed_root=args.processed_root)
            print(f"pending: {snapshot_path}" if result is None else f"{result[1]}: {result[0]}")
    else:
        rows = load_settled_rows(args.root)
        report = ranking.summarize(ranking.evaluate_rows(rows), phase="locked")
        report["prospective_only"] = True
        report["settled_independent_slates"] = len({row["date"] for row in rows})
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
