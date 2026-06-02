from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.common import DEFAULT_STALE_MINUTES, normalize_timestamp
from research.market_quality.price_normalization import american_odds_to_break_even, american_odds_to_decimal
from research.market_quality.price_provenance_schema import annotate_price_provenance_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill break-even fields from timestamp-safe price sources.")
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stale-minutes-threshold", type=float, default=DEFAULT_STALE_MINUTES)
    return parser.parse_args()


def backfill_break_even_fields(audit_rows: pd.DataFrame, *, stale_minutes_threshold: float = DEFAULT_STALE_MINUTES) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = audit_rows.copy()
    if frame.empty:
        return frame, {"row_count": 0}

    frame["existing_market_side_price"] = pd.to_numeric(frame.get("existing_market_side_price", frame.get("market_side_price")), errors="coerce")
    frame["snapshot_market_side_price"] = pd.to_numeric(frame.get("snapshot_market_side_price"), errors="coerce")
    frame["snapshot_market_side_break_even"] = pd.to_numeric(frame.get("snapshot_market_side_break_even"), errors="coerce")
    frame["odds_snapshot_time"] = pd.to_datetime(
        frame.get("odds_snapshot_time", pd.Series(pd.NaT, index=frame.index)).map(normalize_timestamp),
        errors="coerce",
        utc=True,
    )
    frame["prediction_snapshot_time"] = pd.to_datetime(
        frame.get("prediction_snapshot_time", pd.Series(pd.NaT, index=frame.index)).map(normalize_timestamp),
        errors="coerce",
        utc=True,
    )
    frame["game_date"] = pd.to_datetime(frame.get("game_date"), errors="coerce", utc=True)
    frame["price_source_hint"] = frame.get("price_source_hint", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["snapshot_source"] = frame.get("snapshot_source", pd.Series("", index=frame.index)).fillna("").astype(str)

    existing_price = frame["existing_market_side_price"]
    snapshot_price = frame["snapshot_market_side_price"]
    existing_break_even = pd.to_numeric(
        frame.get(
            "existing_market_side_break_even",
            pd.Series(np.nan, index=frame.index, dtype="float64"),
        ),
        errors="coerce",
    )
    snapshot_break_even = frame["snapshot_market_side_break_even"]

    using_existing_price = existing_price.notna()
    using_snapshot_price = ~using_existing_price & snapshot_price.notna()

    preferred_price = existing_price.where(using_existing_price, snapshot_price)
    preferred_break_even = pd.Series(np.nan, index=frame.index, dtype="float64")
    preferred_break_even = preferred_break_even.where(
        ~using_existing_price,
        existing_break_even.where(existing_break_even.notna(), existing_price.map(american_odds_to_break_even)),
    )
    preferred_break_even = preferred_break_even.where(
        ~using_snapshot_price,
        snapshot_break_even.where(snapshot_break_even.notna(), snapshot_price.map(american_odds_to_break_even)),
    )
    preferred_break_even = preferred_break_even.where(
        preferred_break_even.notna(),
        preferred_price.map(american_odds_to_break_even),
    )

    frame["market_side_price"] = preferred_price
    frame["market_side_break_even"] = preferred_break_even.where(
        preferred_break_even.notna(),
        preferred_price.map(american_odds_to_break_even),
    )
    frame["market_side_decimal_odds"] = frame["market_side_price"].map(american_odds_to_decimal)
    frame["price_source"] = np.where(
        using_existing_price,
        "selector_embedded_unknown_time",
        np.where(using_snapshot_price, "current_market_snapshot_pre_event", ""),
    )
    frame["price_source_type"] = frame.get("price_source_type", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame = annotate_price_provenance_frame(
        frame,
        stale_seconds_threshold=float(stale_minutes_threshold) * 60.0,
    )
    correction_supported = (
        frame["timestamp_safe_flag"].astype(bool)
        & ~frame["diagnostic_only_flag"].astype(bool)
        & frame["market_side_price"].notna()
        & ~frame["price_validity_status"].isin(["INVALID_PRICE", "PRICE_SOURCE_UNKNOWN"])
    )
    frame["corrected_price"] = frame["market_side_price"].where(correction_supported)
    frame["corrected_break_even"] = frame["market_side_break_even"].where(correction_supported)
    frame["odds_decimal"] = frame["market_side_decimal_odds"]
    frame["break_even_probability"] = frame["market_side_break_even"]
    frame["price_snapshot_age_minutes"] = frame["minutes_between_odds_and_prediction"]

    manifest = {
        "row_count": int(len(frame)),
        "stale_minutes_threshold": float(stale_minutes_threshold),
        "valid_pre_event_rows": int(frame["price_validity_status"].eq("PRICE_VALID").sum()),
        "missing_price_rows": int(frame["price_validity_status"].eq("MISSING_PRICE").sum()),
        "invalid_price_rows": int(frame["price_validity_status"].eq("INVALID_PRICE").sum()),
        "diagnostic_only_rows": int(frame["price_validity_status"].eq("DIAGNOSTIC_ONLY").sum()),
        "price_source_unknown_rows": int(frame["price_validity_status"].eq("PRICE_SOURCE_UNKNOWN").sum()),
    }
    return frame, manifest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = pd.read_csv(args.input_audit, low_memory=False)
    backfilled_rows, manifest = backfill_break_even_fields(
        audit_rows,
        stale_minutes_threshold=float(args.stale_minutes_threshold),
    )
    output_csv = output_dir / "backfilled_price_rows.csv"
    manifest_json = output_dir / "backfill_manifest.json"
    backfilled_rows.to_csv(output_csv, index=False)
    manifest["output_path"] = str(output_csv)
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
