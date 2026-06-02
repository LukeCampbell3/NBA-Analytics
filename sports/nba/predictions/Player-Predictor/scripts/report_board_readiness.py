#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decision_engine.board_readiness import annotate_board_readiness


def _render_review_rows(frame: pd.DataFrame) -> str:
    cols = [col for col in ["player", "target", "direction", "board_readiness_status", "board_readiness_risk_score", "board_readiness_reasons"] if col in frame.columns]
    if not cols:
        return ""
    preview = frame[cols].head(10).copy()
    preview["board_readiness_risk_score"] = pd.to_numeric(preview["board_readiness_risk_score"], errors="coerce").round(3)
    header = " | ".join(cols)
    divider = " | ".join(["---"] * len(cols))
    rows = [
        " | ".join(str(preview.iloc[idx][col]) for col in cols)
        for idx in range(len(preview))
    ]
    return "\n".join([header, divider, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final-board readiness and fragility without changing selection.")
    parser.add_argument("--board-csv", type=Path, required=True, help="Final board CSV to audit.")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "board_readiness",
        help="Output prefix path without extension.",
    )
    return parser.parse_args()


def build_board_readiness_report(board: pd.DataFrame, out_prefix: Path) -> dict[str, Any]:
    annotated, summary = annotate_board_readiness(board)
    out_prefix = out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows_csv = out_prefix.with_name(f"{out_prefix.name}_rows.csv")
    summary_json = out_prefix.with_name(f"{out_prefix.name}_summary.json")
    summary_md = out_prefix.with_name(f"{out_prefix.name}_summary.md")

    annotated.to_csv(rows_csv, index=False)
    payload = {
        "input_rows": int(len(board)),
        "rows_csv": str(rows_csv),
        "summary": summary,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Board Readiness Report",
        "",
        "## Executive Summary",
        f"- rows: `{summary['row_count']}`",
        f"- board_readiness_status: `{summary['board_readiness_status']}`",
        f"- board_readiness_score: `{summary['board_readiness_score']:.3f}`",
        f"- production_readiness_clear: `{summary['production_readiness_clear']}`",
        f"- recommended_action: `{summary['recommended_action']}`",
        "",
        "## Blocked Reasons",
    ]
    blocked = summary.get("blocked_reasons", [])
    if blocked:
        lines.extend([f"- `{reason}`" for reason in blocked])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Warning Counts",
            f"- high_uncertainty_rows: `{summary['high_uncertainty_rows']}`",
            f"- line_fragility_rows: `{summary['line_fragility_rows']}`",
            f"- line_instability_rows: `{summary['line_instability_rows']}`",
            f"- low_quality_rows: `{summary['low_quality_rows']}`",
            f"- low_recency_rows: `{summary['low_recency_rows']}`",
            f"- price_untrusted_rows: `{summary['price_untrusted_rows']}`",
            f"- timestamp_safe_price_rows: `{summary['timestamp_safe_price_rows']}`",
            f"- same_game_max_share: `{summary['same_game_max_share']:.3f}`",
            f"- same_script_cluster_max_share: `{summary['same_script_cluster_max_share']:.3f}`",
        ]
    )
    top_review = annotated.loc[annotated["board_readiness_review_required"]].copy()
    if not top_review.empty:
        lines.extend(["", "## Review Rows", _render_review_rows(top_review)])
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "rows_csv": str(rows_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    board_path = args.board_csv.resolve()
    if not board_path.exists():
        raise FileNotFoundError(f"Board CSV not found: {board_path}")
    board = pd.read_csv(board_path)
    outputs = build_board_readiness_report(board, args.out_prefix)
    print(f"Rows CSV:     {outputs['rows_csv']}")
    print(f"Summary JSON: {outputs['summary_json']}")
    print(f"Summary MD:   {outputs['summary_md']}")
    summary = outputs["summary"]
    print(f"Status:       {summary['board_readiness_status']}")
    print(f"Score:        {summary['board_readiness_score']:.3f}")
    print(f"Blocked:      {summary['blocked_reasons']}")


if __name__ == "__main__":
    main()
