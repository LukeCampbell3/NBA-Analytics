#!/usr/bin/env python3
"""Capture an immutable MLB market universe and its as-of model feature pool."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.governance.policy_governance import load_policy_registry, sha256_file


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_PROVIDER_CSV = SPORT_ROOT / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io" / "latest_provider_observations.csv"
DEFAULT_POLICY_REGISTRY = SCRIPT_PATH.parent / "policies" / "mlb_policy_family_v1.json"
DEFAULT_EVIDENCE_INVENTORY = SCRIPT_PATH.parent / "evidence_inventory.json"
EASTERN = ZoneInfo("America/New_York")
SCHEMA_VERSION = "MLB_COMPLETE_SLATE_V1"
MARKET_TARGETS = {
    "batter_hits": "H",
    "batter_total_bases": "TB",
    "batter_runs_scored": "R",
    "batter_rbis": "RBI",
    "batter_home_runs": "HR",
    "pitcher_strikeouts": "K",
    "pitcher_earned_runs": "ER",
}
REQUIRED_PROVIDER_COLUMNS = {
    "event_id", "game_start_utc", "sportsbook", "player_name", "market_type", "side", "line",
    "price_american", "price_decimal", "observed_at_utc", "raw_record_hash", "parser_version",
    "validation_status",
}
LEDGER_COLUMNS = [
    "slate_id", "snapshot_id", "observed_at_utc", "event_id", "game_start_utc", "book", "player_id",
    "player_name", "market", "side", "line", "price", "price_decimal", "lineup_state", "player_state",
    "feature_cutoff", "feature_snapshot_id", "model_version", "model_score", "eligible_by_input_rules",
    "settlement", "realized_unit_return", "raw_source_hash", "parser_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture an immutable complete MLB slate snapshot.")
    parser.add_argument("--provider-csv", type=Path, default=DEFAULT_PROVIDER_CSV)
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-date", type=date.fromisoformat, required=True)
    parser.add_argument("--policy-registry", type=Path, default=DEFAULT_POLICY_REGISTRY)
    parser.add_argument("--evidence-inventory", type=Path, default=DEFAULT_EVIDENCE_INVENTORY)
    return parser.parse_args()


def normalize_player(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    return "_".join(re.findall(r"[a-z0-9]+", text))


def _gzip_bytes(content: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as handle:
        handle.write(content)
    return output.getvalue()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp(value: object) -> datetime | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _pool_lookup(pool: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pool.to_dict(orient="records"):
        player = normalize_player(row.get("Player_ID") or row.get("Player"))
        target = str(row.get("Target") or "").upper()
        if player and target:
            lookup[(player, target)] = row
    return lookup


def build_candidate_universe(
    provider: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    run_date: date,
    slate_id: str,
    snapshot_id: str,
    feature_snapshot_id: str,
) -> pd.DataFrame:
    lookup = _pool_lookup(pool)
    records: list[dict[str, Any]] = []
    for source in provider.to_dict(orient="records"):
        target = MARKET_TARGETS.get(str(source.get("market_type") or source.get("market") or "").lower(), "")
        player_id = normalize_player(source.get("player_name"))
        model_row = lookup.get((player_id, target), {})
        prediction = pd.to_numeric(model_row.get("Prediction"), errors="coerce")
        line = pd.to_numeric(source.get("line"), errors="coerce")
        side = str(source.get("side") or "").strip().upper()
        model_score = None
        if pd.notna(prediction) and pd.notna(line) and side in {"OVER", "UNDER"}:
            model_score = float(prediction - line) if side == "OVER" else float(line - prediction)
        observed = str(source.get("observed_at_utc") or source.get("snapshot_time_utc") or "")
        game_start = str(source.get("game_start_utc") or source.get("commence_time_utc") or "")
        observed_dt = _timestamp(observed)
        game_start_dt = _timestamp(game_start)
        pregame = bool(observed_dt and game_start_dt and observed_dt < game_start_dt)
        valid_price = pd.notna(pd.to_numeric(source.get("price_american"), errors="coerce"))
        valid_line = pd.notna(line)
        source_valid = str(source.get("validation_status") or "").upper() == "VALID"
        game_date_et = game_start_dt.astimezone(EASTERN).date() if game_start_dt else None
        model_version = "|".join(
            value for value in (
                str(model_row.get("Model_Selected") or ""),
                str(model_row.get("Matchup_Network_Version") or ""),
            ) if value
        )
        records.append(
            {
                "slate_id": slate_id,
                "snapshot_id": snapshot_id,
                "observed_at_utc": observed,
                "event_id": str(source.get("event_id") or ""),
                "game_start_utc": game_start,
                "book": str(source.get("sportsbook") or source.get("book") or "").lower(),
                "player_id": player_id,
                "player_name": str(source.get("player_name") or ""),
                "market": target or str(source.get("market_type") or ""),
                "side": side,
                "line": float(line) if pd.notna(line) else None,
                "price": int(float(source["price_american"])) if valid_price else None,
                "price_decimal": float(source["price_decimal"]) if pd.notna(pd.to_numeric(source.get("price_decimal"), errors="coerce")) else None,
                "lineup_state": "UNKNOWN_AT_CAPTURE",
                "player_state": str(model_row.get("Game_Status_Detail") or "UNKNOWN"),
                "feature_cutoff": str(model_row.get("Last_History_Date") or ""),
                "feature_snapshot_id": feature_snapshot_id,
                "model_version": model_version,
                "model_score": model_score,
                "eligible_by_input_rules": bool(
                    target and model_row and pregame and source_valid and valid_price and valid_line and game_date_et == run_date
                ),
                "settlement": "PENDING",
                "realized_unit_return": None,
                "raw_source_hash": str(source.get("raw_record_hash") or ""),
                "parser_version": str(source.get("parser_version") or ""),
            }
        )
    frame = pd.DataFrame.from_records(records, columns=LEDGER_COLUMNS)
    return frame.sort_values(["event_id", "book", "player_id", "market", "side", "line"], kind="stable")


def capture_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    provider_path = args.provider_csv.resolve()
    pool_path = args.pool_csv.resolve()
    missing_columns: list[str] = []
    provider = pd.read_csv(provider_path)
    pool = pd.read_csv(pool_path)
    missing_columns = sorted(REQUIRED_PROVIDER_COLUMNS - set(provider.columns))
    if missing_columns:
        raise ValueError(f"Provider observations are missing columns: {', '.join(missing_columns)}")
    if provider.empty:
        raise ValueError("Provider observation table is empty; complete-slate capture is impossible.")

    registry = load_policy_registry(args.policy_registry.resolve())
    evidence_inventory = json.loads(args.evidence_inventory.read_text(encoding="utf-8"))
    provider_hash = sha256_file(provider_path)
    pool_hash = sha256_file(pool_path)
    feature_snapshot_id = f"mlb_feature_{pool_hash[:16]}"
    observed_values = [_timestamp(value) for value in provider["observed_at_utc"]]
    observed_values = [value for value in observed_values if value is not None]
    if not observed_values:
        raise ValueError("Provider observations do not contain a valid acquisition timestamp.")
    observed_at = max(observed_values)
    slate_id = f"MLB_{args.run_date.strftime('%Y%m%d')}"
    identity_material = f"{slate_id}|{provider_hash}|{pool_hash}|{observed_at.isoformat()}".encode("utf-8")
    snapshot_hash = hashlib.sha256(identity_material).hexdigest()
    snapshot_id = f"mlb_{args.run_date.strftime('%Y%m%d')}_{observed_at.strftime('%H%M%SZ')}_{snapshot_hash[:12]}"
    universe = build_candidate_universe(
        provider,
        pool,
        run_date=args.run_date,
        slate_id=slate_id,
        snapshot_id=snapshot_id,
        feature_snapshot_id=feature_snapshot_id,
    )
    universe_csv = universe.to_csv(index=False, lineterminator="\n").encode("utf-8")
    pool_csv = pool.to_csv(index=False, lineterminator="\n").encode("utf-8")
    universe_digest = hashlib.sha256(universe_csv).hexdigest()
    feature_digest = hashlib.sha256(pool_csv).hexdigest()
    snapshot_dir = args.run_dir.resolve() / "governance" / "slates" / slate_id / snapshot_id
    manifest_path = snapshot_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "capture_label": "FULL_SLATE_SNAPSHOT",
        "full_policy_replay_available": True,
        "immutable": True,
        "slate_id": slate_id,
        "snapshot_id": snapshot_id,
        "observed_at_utc": observed_at.isoformat(),
        "run_date": args.run_date.isoformat(),
        "candidate_universe_rows": int(len(universe)),
        "all_available_provider_rows_retained": int(len(universe)) == int(len(provider)),
        "eligible_input_rows": int(universe["eligible_by_input_rules"].sum()),
        "events": int(universe["event_id"].nunique()),
        "books": int(universe["book"].nunique()),
        "markets": sorted(str(value) for value in universe["market"].dropna().unique()),
        "sides": sorted(str(value) for value in universe["side"].dropna().unique()),
        "providers": sorted(str(value) for value in provider.get("provider_name", provider.get("source", pd.Series(dtype=str))).dropna().unique()),
        "live_provider_rows": int(provider.get("is_live", pd.Series(False, index=provider.index)).fillna(False).astype(bool).sum()),
        "cached_provider_rows": int(provider.get("is_cache", pd.Series(False, index=provider.index)).fillna(False).astype(bool).sum()),
        "candidate_universe_sha256": universe_digest,
        "feature_pool_sha256": feature_digest,
        "raw_provider_file_sha256": provider_hash,
        "raw_pool_file_sha256": pool_hash,
        "feature_snapshot_id": feature_snapshot_id,
        "provider_source": str(provider_path),
        "pool_source": str(pool_path),
        "policy_registry": str(args.policy_registry.resolve()),
        "registered_policies": [
            {
                "policy_version": policy["policy_version"],
                "policy_stage": policy["policy_stage"],
                "policy_digest": policy["policy_digest"],
            }
            for policy in registry["policies"]
        ],
        "historical_evidence": evidence_inventory,
        "settlement_status": "UNRESOLVED",
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("candidate_universe_sha256") != universe_digest or existing.get("feature_pool_sha256") != feature_digest:
            raise ValueError(f"Immutable snapshot collision at {snapshot_dir}")
    else:
        _atomic_write(snapshot_dir / "candidate_universe.csv.gz", _gzip_bytes(universe_csv))
        _atomic_write(snapshot_dir / "feature_pool.csv.gz", _gzip_bytes(pool_csv))
        _atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))

    governance_status = {
        "schema_version": "MLB_POLICY_GOVERNANCE_STATUS_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": args.run_date.isoformat(),
        "slate_id": slate_id,
        "snapshot_id": snapshot_id,
        "capture_label": "FULL_SLATE_SNAPSHOT",
        "complete_slate_capture_passed": True,
        "candidate_authorization_enabled": False,
        "staking_enabled": False,
        "publication_mode": "SHADOW_RESEARCH_ONLY",
        "certificate_status": "NO_ACTIVE_PROSPECTIVE_CERTIFICATE",
        "authorization_reason": "Registered MLB policies remain in development and have no prospective certificate.",
        "registered_policies": manifest["registered_policies"],
        "historical_evidence_labels": [
            item["evidence_label"] for item in evidence_inventory.get("sources", [])
        ],
        "full_policy_replay_available_for_snapshot": True,
        "snapshot_manifest": str(manifest_path),
    }
    status_path = args.run_dir.resolve() / "governance" / "governance_status.json"
    _atomic_write(status_path, (json.dumps(governance_status, indent=2) + "\n").encode("utf-8"))
    return {"manifest": manifest, "governance_status": governance_status, "status_path": status_path}


def main() -> None:
    result = capture_snapshot(parse_args())
    manifest = result["manifest"]
    print(f"Complete MLB slate captured: {manifest['snapshot_id']}")
    print(
        f"Rows={manifest['candidate_universe_rows']}; eligible inputs={manifest['eligible_input_rows']}; "
        f"events={manifest['events']}; books={manifest['books']}"
    )
    print("Authorization: shadow research only; no prospective policy certificate exists.")


if __name__ == "__main__":
    main()
