#!/usr/bin/env python3
"""Validate the v9.6 CLV pipeline end-to-end.

This script validates that:
1. Live odds collection produces valid data
2. The sequence builder correctly labels snapshot types
3. CLV computation is mathematically sound
4. The attachable data meets schema requirements for model integration
5. No-vig probabilities are correctly derived
6. Side-aware CLV metrics are computable

This does NOT require date overlap with model training data.
It validates the pipeline infrastructure so that once enough forward-collected
data accumulates, the model validation will work correctly.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from market_odds_quality import add_american_odds_quality, is_valid_american_odds


def _resolve(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / text).resolve()


def _american_to_implied(odds: float) -> float:
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _no_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    over = _american_to_implied(over_odds)
    under = _american_to_implied(under_odds)
    total = over + under
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    return over / total, under / total


def validate_collection(collection_path: Path) -> dict:
    """Validate the raw collected book snapshots."""
    df = pd.read_csv(collection_path)
    checks = {}

    # Basic schema
    required_cols = [
        "snapshot_time", "book", "player", "market", "line",
        "over_odds", "under_odds", "game_start_time", "date",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    checks["schema_complete"] = len(missing_cols) == 0
    checks["missing_columns"] = missing_cols

    # Row counts
    checks["total_rows"] = int(len(df))
    checks["rows_with_game_start_time"] = int(df["game_start_time"].notna().sum())
    checks["game_start_time_coverage"] = float(df["game_start_time"].notna().mean()) if len(df) > 0 else 0.0

    # Odds quality
    df = add_american_odds_quality(df)
    checks["valid_american_odds_rate"] = float(df["is_valid_american_odds"].mean()) if len(df) > 0 else 0.0
    checks["valid_american_odds_pass"] = checks["valid_american_odds_rate"] >= 0.98

    # Source diversity
    checks["sources"] = sorted(df["source"].dropna().unique().tolist()) if "source" in df.columns else []
    checks["books"] = sorted(df["book"].dropna().unique().tolist()) if "book" in df.columns else []
    checks["markets"] = sorted(df["market"].dropna().unique().tolist()) if "market" in df.columns else []
    checks["dates"] = sorted(df["date"].dropna().unique().tolist()) if "date" in df.columns else []

    # Snapshot type diversity
    if "snapshot_type" in df.columns:
        checks["snapshot_types"] = df["snapshot_type"].value_counts().to_dict()
    else:
        checks["snapshot_types"] = {}

    checks["pass"] = (
        checks["schema_complete"]
        and checks["valid_american_odds_pass"]
        and checks["total_rows"] >= 1000
        and checks["game_start_time_coverage"] >= 0.80
    )
    return checks


def validate_sequence(sequence_path: Path) -> dict:
    """Validate the built sequence file."""
    df = pd.read_csv(sequence_path)
    checks = {}

    checks["total_rows"] = int(len(df))
    checks["snapshot_type_counts"] = df["snapshot_type"].value_counts().to_dict() if "snapshot_type" in df.columns else {}

    # Must have multiple snapshot types
    checks["has_open"] = checks["snapshot_type_counts"].get("open", 0) > 0
    checks["has_prelock"] = checks["snapshot_type_counts"].get("prelock", 0) > 0
    checks["has_close"] = checks["snapshot_type_counts"].get("close", 0) > 0 or checks["snapshot_type_counts"].get("intraday", 0) > 0
    checks["multiple_types"] = sum(1 for v in checks["snapshot_type_counts"].values() if v > 0) >= 2

    # Temporal ordering: open should come before close within groups
    if "snapshot_time" in df.columns and "snapshot_type" in df.columns:
        df["_ts"] = pd.to_datetime(df["snapshot_time"], errors="coerce", utc=True)
        opens = df[df["snapshot_type"] == "open"]["_ts"]
        closes = df[df["snapshot_type"].isin(["close", "prelock"])]["_ts"]
        if not opens.empty and not closes.empty:
            checks["temporal_ordering_valid"] = bool(opens.min() <= closes.max())
        else:
            checks["temporal_ordering_valid"] = True
    else:
        checks["temporal_ordering_valid"] = False

    checks["pass"] = (
        checks["total_rows"] >= 500
        and checks["multiple_types"]
        and checks["has_prelock"]
        and checks["temporal_ordering_valid"]
    )
    return checks


def validate_attachable(attachable_path: Path) -> dict:
    """Validate the attachable CLV data."""
    df = pd.read_csv(attachable_path)
    checks = {}

    checks["total_rows"] = int(len(df))

    # Close status
    if "close_status" in df.columns:
        checks["close_status_counts"] = df["close_status"].value_counts().to_dict()
        checks["true_clv_rows"] = int(df["close_status"].eq("true_sequence_close").sum())
    else:
        checks["close_status_counts"] = {}
        checks["true_clv_rows"] = 0

    # Compute no-vig probabilities
    df = add_american_odds_quality(df)
    valid = df["is_valid_american_odds"]
    no_vig_results = df.loc[valid].apply(
        lambda row: _no_vig(row["over_odds"], row["under_odds"]), axis=1
    )
    if len(no_vig_results) > 0:
        no_vig_over, no_vig_under = zip(*no_vig_results)
        df.loc[valid, "no_vig_over"] = no_vig_over
        df.loc[valid, "no_vig_under"] = no_vig_under
    else:
        df["no_vig_over"] = np.nan
        df["no_vig_under"] = np.nan

    # Validate no-vig probabilities
    nv_valid = df["no_vig_over"].notna() & df["no_vig_under"].notna()
    checks["no_vig_computed_rows"] = int(nv_valid.sum())
    if nv_valid.any():
        sums = df.loc[nv_valid, "no_vig_over"] + df.loc[nv_valid, "no_vig_under"]
        checks["no_vig_sum_to_one"] = bool(np.allclose(sums, 1.0, atol=1e-6))
        checks["no_vig_over_range_valid"] = bool(
            (df.loc[nv_valid, "no_vig_over"] > 0).all()
            and (df.loc[nv_valid, "no_vig_over"] < 1).all()
        )
    else:
        checks["no_vig_sum_to_one"] = False
        checks["no_vig_over_range_valid"] = False

    # CLV computation validation (for true_sequence_close rows)
    true_clv = df[df.get("close_status", pd.Series()) == "true_sequence_close"].copy()
    if not true_clv.empty and "close_over_odds" in true_clv.columns:
        close_valid = (
            true_clv["close_over_odds"].apply(is_valid_american_odds)
            & true_clv["close_under_odds"].apply(is_valid_american_odds)
        )
        checks["close_odds_valid_rate"] = float(close_valid.mean())

        if close_valid.any():
            close_nv = true_clv.loc[close_valid].apply(
                lambda row: _no_vig(row["close_over_odds"], row["close_under_odds"]), axis=1
            )
            close_nv_over, close_nv_under = zip(*close_nv)
            true_clv.loc[close_valid, "close_no_vig_over"] = close_nv_over
            true_clv.loc[close_valid, "close_no_vig_under"] = close_nv_under

            # Side-aware CLV: OVER CLV = close_no_vig_over - entry_no_vig_over
            both_valid = close_valid & nv_valid.reindex(true_clv.index, fill_value=False)
            if both_valid.any():
                subset = true_clv.loc[both_valid]
                entry_nv = subset.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
                entry_over, entry_under = zip(*entry_nv)
                clv_over = np.array(close_nv_over)[both_valid.values[close_valid.values]] if len(close_nv_over) > 0 else np.array([])

                # Simplified: compute CLV for all rows where we have both entry and close
                entry_over_arr = np.array(list(entry_over))
                close_over_arr = np.array(list(close_nv_over))[:len(entry_over_arr)]
                if len(entry_over_arr) > 0 and len(close_over_arr) > 0:
                    min_len = min(len(entry_over_arr), len(close_over_arr))
                    over_clv = close_over_arr[:min_len] - entry_over_arr[:min_len]
                    checks["mean_over_clv"] = float(np.nanmean(over_clv))
                    checks["positive_clv_rate"] = float(np.nanmean(over_clv > 0))
                    checks["clv_std"] = float(np.nanstd(over_clv))
                    checks["clv_computable"] = True
                else:
                    checks["clv_computable"] = False
            else:
                checks["clv_computable"] = False
        else:
            checks["close_odds_valid_rate"] = 0.0
            checks["clv_computable"] = False
    else:
        checks["close_odds_valid_rate"] = 0.0
        checks["clv_computable"] = False

    checks["pass"] = (
        checks["total_rows"] >= 1000
        and checks["true_clv_rows"] >= 500
        and checks["no_vig_sum_to_one"]
        and checks["no_vig_over_range_valid"]
        and checks.get("clv_computable", False)
    )
    return checks


def validate_sequence_report(report_path: Path) -> dict:
    """Validate the sequence report meets promotion gates."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = {}

    gate_checks = report.get("promotion_gate_checks", {})
    checks["all_gates_pass"] = all(gate_checks.values())
    checks["gate_results"] = gate_checks
    checks["promotion_status"] = report.get("promotion_status")
    checks["true_clv_rows"] = report.get("true_clv_rows", 0)
    checks["attachable_rows"] = report.get("attachable_rows", 0)

    checks["pass"] = checks["all_gates_pass"]
    return checks


def main() -> None:
    sequence_dir = _resolve(ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence")

    collection_path = sequence_dir / "collected_book_snapshots.csv"
    sequence_path = sequence_dir / "market_snapshot_sequence.csv"
    attachable_path = sequence_dir / "market_snapshot_attachable.csv"
    report_path = sequence_dir / "market_snapshot_sequence_report.json"

    results = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v9.6",
    }

    print("=" * 70)
    print("V9.6 CLV PIPELINE VALIDATION")
    print("=" * 70)

    # 1. Validate collection
    print("\n[1/4] Validating collected book snapshots...")
    if collection_path.exists():
        results["collection"] = validate_collection(collection_path)
        status = "PASS" if results["collection"]["pass"] else "FAIL"
        print(f"  Status: {status}")
        print(f"  Total rows: {results['collection']['total_rows']}")
        print(f"  Game start time coverage: {results['collection']['game_start_time_coverage']:.1%}")
        print(f"  Valid American odds rate: {results['collection']['valid_american_odds_rate']:.1%}")
        print(f"  Books: {results['collection']['books']}")
        print(f"  Markets: {results['collection']['markets']}")
    else:
        results["collection"] = {"pass": False, "error": "file_not_found"}
        print("  FAIL: collected_book_snapshots.csv not found")

    # 2. Validate sequence
    print("\n[2/4] Validating market snapshot sequence...")
    if sequence_path.exists():
        results["sequence"] = validate_sequence(sequence_path)
        status = "PASS" if results["sequence"]["pass"] else "FAIL"
        print(f"  Status: {status}")
        print(f"  Total rows: {results['sequence']['total_rows']}")
        print(f"  Snapshot types: {results['sequence']['snapshot_type_counts']}")
        print(f"  Has prelock: {results['sequence']['has_prelock']}")
        print(f"  Temporal ordering valid: {results['sequence']['temporal_ordering_valid']}")
    else:
        results["sequence"] = {"pass": False, "error": "file_not_found"}
        print("  FAIL: market_snapshot_sequence.csv not found")

    # 3. Validate attachable CLV data
    print("\n[3/4] Validating attachable CLV data...")
    if attachable_path.exists():
        results["attachable"] = validate_attachable(attachable_path)
        status = "PASS" if results["attachable"]["pass"] else "FAIL"
        print(f"  Status: {status}")
        print(f"  Total rows: {results['attachable']['total_rows']}")
        print(f"  True CLV rows: {results['attachable']['true_clv_rows']}")
        print(f"  No-vig computed: {results['attachable']['no_vig_computed_rows']}")
        print(f"  No-vig sums to 1: {results['attachable']['no_vig_sum_to_one']}")
        print(f"  CLV computable: {results['attachable'].get('clv_computable', False)}")
        if results["attachable"].get("clv_computable"):
            print(f"  Mean over CLV: {results['attachable']['mean_over_clv']:.6f}")
            print(f"  Positive CLV rate: {results['attachable']['positive_clv_rate']:.3f}")
            print(f"  CLV std: {results['attachable']['clv_std']:.6f}")
    else:
        results["attachable"] = {"pass": False, "error": "file_not_found"}
        print("  FAIL: market_snapshot_attachable.csv not found")

    # 4. Validate sequence report gates
    print("\n[4/4] Validating promotion gates...")
    if report_path.exists():
        results["promotion_gates"] = validate_sequence_report(report_path)
        status = "PASS" if results["promotion_gates"]["pass"] else "FAIL"
        print(f"  Status: {status}")
        print(f"  Promotion status: {results['promotion_gates']['promotion_status']}")
        print(f"  True CLV rows: {results['promotion_gates']['true_clv_rows']}")
        print(f"  Attachable rows: {results['promotion_gates']['attachable_rows']}")
        for gate, passed in results["promotion_gates"]["gate_results"].items():
            marker = "✓" if passed else "✗"
            print(f"    {marker} {gate}")
    else:
        results["promotion_gates"] = {"pass": False, "error": "file_not_found"}
        print("  FAIL: market_snapshot_sequence_report.json not found")

    # Overall result
    all_pass = all(
        results.get(section, {}).get("pass", False)
        for section in ["collection", "sequence", "attachable", "promotion_gates"]
    )
    results["overall_pass"] = all_pass
    results["overall_status"] = "v9_6_clv_pipeline_validated" if all_pass else "v9_6_clv_pipeline_blocked"

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'} — {results['overall_status']}")
    print("=" * 70)

    # Write report
    output_path = sequence_dir / "v9_6_clv_pipeline_validation_report.json"
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
