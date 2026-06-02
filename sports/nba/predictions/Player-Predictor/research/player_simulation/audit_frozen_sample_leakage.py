from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ACTUAL_PREFIXES = ("actual_",)
ACTUAL_COLUMNS_ALLOWED = {"actual_available"}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def audit_frozen_sample_leakage(
    *,
    frozen_sample_path: Path,
    actual_outcomes_path: Path,
    output_dir: Path,
    cutoff_date: str = "2025-10-01",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.to_datetime(cutoff_date, errors="raise")
    state = _read_table(frozen_sample_path)
    actuals = _read_table(actual_outcomes_path)
    failures: list[str] = []

    if state.empty:
        failures.append("frozen_input_rows_empty")
    if actuals.empty:
        failures.append("actual_outcome_rows_empty")

    max_source_date = None
    for source_date_col in ["max_source_date", "max_input_game_date"]:
        if source_date_col not in state.columns:
            continue
        dates = pd.to_datetime(state[source_date_col], errors="coerce").dropna()
        if not dates.empty:
            max_source_date = max(max_source_date or "", dates.max().strftime("%Y-%m-%d"))
            if dates.ge(cutoff).any():
                failures.append(f"input_{source_date_col}_on_or_after_cutoff")
    if "game_date" in state.columns:
        dates = pd.to_datetime(state["game_date"], errors="coerce").dropna()
        if dates.ge(cutoff).any():
            failures.append("input_game_date_on_or_after_cutoff")
    if "Date" in state.columns:
        dates = pd.to_datetime(state["Date"], errors="coerce").dropna()
        if dates.ge(cutoff).any():
            failures.append("input_Date_on_or_after_cutoff")

    leaked_actual_cols = [
        col for col in state.columns if col.startswith(ACTUAL_PREFIXES) and col not in ACTUAL_COLUMNS_ALLOWED
    ]
    if leaked_actual_cols:
        failures.append("actual_outcome_columns_present_in_input_rows")

    overlap_actual_cols = sorted(set(state.columns).intersection({"actual_pts", "actual_reb", "actual_ast", "actual_pra", "actual_mpg"}))
    if overlap_actual_cols:
        failures.append("target_outcome_fields_joined_to_input")

    status = "BACKTEST_FAILED_LEAKAGE" if any(
        failure
        for failure in failures
        if failure
        not in {
            "frozen_input_rows_empty",
            "actual_outcome_rows_empty",
        }
    ) else "LEAKAGE_AUDIT_PASSED"
    report = {
        "status": status,
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "frozen_sample_path": str(frozen_sample_path),
        "actual_outcomes_path": str(actual_outcomes_path),
        "input_rows": int(len(state)),
        "actual_outcome_rows": int(len(actuals)),
        "max_input_source_date": max_source_date,
        "actual_outcomes_isolated": not leaked_actual_cols and not overlap_actual_cols,
        "failures": failures,
        "production_behavior_changed": False,
        "promotion_ready": False,
    }
    (output_dir / "frozen_preseason_leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "frozen_preseason_leakage_audit.md").write_text(_format_md(report), encoding="utf-8")
    if status == "BACKTEST_FAILED_LEAKAGE":
        raise SystemExit(json.dumps(report, indent=2))
    return report


def _format_md(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen Preseason Leakage Audit",
        "",
        f"- status: {report.get('status')}",
        f"- cutoff_date: {report.get('cutoff_date')}",
        f"- input_rows: {report.get('input_rows')}",
        f"- actual_outcome_rows: {report.get('actual_outcome_rows')}",
        f"- max_input_source_date: {report.get('max_input_source_date')}",
        f"- actual_outcomes_isolated: {report.get('actual_outcomes_isolated')}",
    ]
    failures = report.get("failures", [])
    if failures:
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen preseason sample for cutoff leakage.")
    parser.add_argument("--frozen-sample", type=Path, required=True)
    parser.add_argument("--actual-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-date", default="2025-10-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_frozen_sample_leakage(
        frozen_sample_path=args.frozen_sample,
        actual_outcomes_path=args.actual_outcomes,
        output_dir=args.output_dir,
        cutoff_date=str(args.cutoff_date),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
