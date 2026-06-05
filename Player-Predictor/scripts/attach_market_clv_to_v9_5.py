#!/usr/bin/env python3
"""Attach true sportsbook no-vig/CLV snapshots to a v9.5 manifest."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

from attach_market_snapshots_v9_3 import attach_snapshots


def _resolve(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / text).resolve()


def _read(path: Path) -> pd.DataFrame:
    path = _resolve(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _copy_tree_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach market/CLV snapshots to v9.5 rows")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "model" / "props" / "v9_5_prelock_availability_w050" / "manifest.json")
    parser.add_argument("--market-snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_5_market_clv")
    parser.add_argument("--min-match-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = _resolve(args.source_manifest)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_output = _resolve(Path(source_manifest["output"]))
    rows = pd.read_csv(source_output / "data" / "prop_training_rows.csv")
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.date.astype(str)
    snapshots = _read(args.market_snapshots)
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce").dt.date.astype(str)
    merged, attachment = attach_snapshots(rows, snapshots)

    output = _resolve(args.output)
    (output / "data").mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "data" / "prop_training_rows.csv", index=False)
    _copy_tree_if_exists(source_output / "calibration", output / "calibration")
    _copy_tree_if_exists(source_output / "validation", output / "validation")

    matched = int(attachment.get("matched_rows", 0))
    real_market_rows = int(merged["book"].notna().sum()) if "book" in merged.columns else 0
    close_status_counts = merged["close_status"].dropna().value_counts().to_dict() if "close_status" in merged.columns else {}
    status = "market_clv_coverage_limited_candidate" if matched > 0 else "blocked_no_market_matches"
    manifest = {
        **source_manifest,
        "model_version": "prop_engine_v9_5_market_clv_validated_distribution",
        "status": status,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_v9_5_manifest": str(source_manifest_path.relative_to(REPO_ROOT)) if source_manifest_path.is_relative_to(REPO_ROOT) else str(source_manifest_path),
        "market_snapshots": str(_resolve(args.market_snapshots)),
        "output": str(output.relative_to(REPO_ROOT)) if output.is_relative_to(REPO_ROOT) else str(output),
        "market_attachment": {
            **attachment,
            "real_market_rows_after_attach": real_market_rows,
            "close_status_counts": close_status_counts,
            "clv_reliability": "limited_proxy_only" if close_status_counts else "unavailable",
            "note": "Rows with archived_historical_market_not_clv support true no-vig market Brier, not reliable CLV.",
        },
        "live_promotion_blockers": [
            "Market snapshot coverage is too low for full promotion unless match rate increases materially.",
            "CLV requires timestamped pre-lock and closing snapshots with exact game start times.",
            "Archived historical market rows are valid for no-vig baseline checks but not definitive CLV.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": status, "market_attachment": manifest["market_attachment"], "manifest": str(output / "manifest.json")}, indent=2, default=str))


if __name__ == "__main__":
    main()
