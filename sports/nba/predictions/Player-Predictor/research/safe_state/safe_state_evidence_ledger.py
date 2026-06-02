from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.common import candidate_identity_columns


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return ""
    return value


def _entry_from_row(
    row: pd.Series,
    *,
    run_id: str,
    created_at: str,
    source_label: str,
    evidence_paths: dict[str, str],
) -> dict[str, Any]:
    recommended = str(row.get("recommended_action", row.get("recommended_next_action", row.get("recommended_repair", ""))) or "")
    queue_status = str(row.get("queue_status_after_recheck", row.get("queue_status", "")) or "")
    safe_tier = str(row.get("safe_state_tier", "") or "")
    fixability = str(row.get("gap_fixability", row.get("forecastability_gap_fixability", "")) or "")
    if recommended == "KEEP_UNSAFE_TRUE_VOLATILITY" or fixability == "TRUE_UNSTABLE_STATE":
        promotion_status = "REJECTED_UNSAFE"
    elif queue_status in {"NEEDS_MORE_SAMPLE", "READY_FOR_RECHECK"} or recommended == "NEEDS_MORE_SAMPLE":
        promotion_status = "NEEDS_MORE_SAMPLE"
    elif safe_tier == "SAFE_STATE_CORE" or queue_status == "PROMOTED_TO_SAFE_CORE":
        promotion_status = "SAFE_CORE_SHADOW_ONLY"
    else:
        promotion_status = "NOT_PROMOTION_ELIGIBLE"
    return {
        "run_id": run_id,
        "created_at": created_at,
        "candidate_id": _clean(row.get("candidate_id", "")),
        "player": _clean(row.get("player", row.get("player_name", ""))),
        "game_id": _clean(row.get("game_id", "")),
        "market_date": _clean(row.get("market_date", row.get("game_date", ""))),
        "target": _clean(row.get("target", "")),
        "side": _clean(row.get("side", row.get("direction", ""))),
        "line": _clean(row.get("line", row.get("market_line", ""))),
        "edge_defendability_tier": _clean(row.get("edge_defendability_tier", "")),
        "safe_state_tier": safe_tier,
        "forecastability_gap_primary": _clean(row.get("forecastability_gap_primary", "")),
        "gap_subtype": _clean(row.get("gap_subtype", "")),
        "root_cause_primary": _clean(row.get("root_cause_primary", "")),
        "fixability": fixability,
        "recommended_action": recommended,
        "queue_status": queue_status,
        "settlement_status": "PENDING",
        "later_recheck_status": _clean(row.get("recheck_status", "")),
        "promotion_status": promotion_status,
        "evidence_paths": evidence_paths,
        "notes": f"source={source_label}; shadow_only=true; no_production_behavior_change",
    }


def append_safe_state_evidence_ledger(
    *,
    ledger_path: Path,
    run_id: str,
    true_unstable_csv: Path | None = None,
    needs_more_sample_queue_csv: Path | None = None,
    recheck_csv: Path | None = None,
) -> dict[str, Any]:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    sources = [
        ("true_unstable_shadow_rejections", true_unstable_csv),
        ("needs_more_sample_queue", needs_more_sample_queue_csv),
        ("needs_more_sample_recheck", recheck_csv),
    ]
    entries: list[dict[str, Any]] = []
    evidence_paths = {
        "true_unstable_csv": str(true_unstable_csv) if true_unstable_csv else "",
        "needs_more_sample_queue_csv": str(needs_more_sample_queue_csv) if needs_more_sample_queue_csv else "",
        "recheck_csv": str(recheck_csv) if recheck_csv else "",
    }
    for label, path in sources:
        frame = _read_csv(path)
        if frame.empty:
            continue
        frame = candidate_identity_columns(frame)
        for _, row in frame.iterrows():
            entries.append(_entry_from_row(row, run_id=run_id, created_at=created_at, source_label=label, evidence_paths=evidence_paths))
    with ledger_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    report = {
        "ledger_path": str(ledger_path),
        "run_id": run_id,
        "entries_appended": int(len(entries)),
        "append_only": True,
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append safe-state shadow evidence rows to the lifecycle ledger.")
    parser.add_argument("--ledger-path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--true-unstable-csv", type=Path)
    parser.add_argument("--needs-more-sample-queue-csv", type=Path)
    parser.add_argument("--recheck-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = append_safe_state_evidence_ledger(
        ledger_path=args.ledger_path,
        run_id=args.run_id,
        true_unstable_csv=args.true_unstable_csv,
        needs_more_sample_queue_csv=args.needs_more_sample_queue_csv,
        recheck_csv=args.recheck_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
