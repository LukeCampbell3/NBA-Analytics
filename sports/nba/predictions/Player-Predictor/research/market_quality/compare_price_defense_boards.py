from __future__ import annotations

import argparse
import json
import sys
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return candidate_identity_columns(frame.copy())


def _identity_series(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype="object")
    working = rows.copy()
    if "candidate_id" in working.columns and working["candidate_id"].notna().any():
        return working["candidate_id"].astype(str).fillna("")
    keys = (
        working.get("player", pd.Series("", index=working.index)).astype(str).fillna("")
        + "|"
        + working.get("market_date", pd.Series("", index=working.index)).astype(str).fillna("")
        + "|"
        + working.get("target", pd.Series("", index=working.index)).astype(str).fillna("")
        + "|"
        + working.get("direction", pd.Series("", index=working.index)).astype(str).fillna("")
        + "|"
        + working.get("market_side_price", pd.Series("", index=working.index)).astype(str).fillna("")
    )
    return keys


def _stable_row_signature(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype="object")
    return (
        rows.get("player", pd.Series("", index=rows.index)).astype(str).fillna("")
        + "|"
        + rows.get("market_date", pd.Series("", index=rows.index)).astype(str).fillna("")
        + "|"
        + rows.get("target", pd.Series("", index=rows.index)).astype(str).fillna("")
        + "|"
        + rows.get("direction", pd.Series("", index=rows.index)).astype(str).fillna("")
        + "|"
        + rows.get("market_side_price", pd.Series("", index=rows.index)).astype(str).fillna("")
    )


def _numeric_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.empty:
        return None
    mean_value = float(values.mean())
    return mean_value


def compare_price_defense_boards(
    *,
    output_dir: Path,
    production_board_csv: Path | None = None,
    shadow_board_csv: Path | None = None,
) -> dict[str, Any]:
    output_dir.resolve().mkdir(parents=True, exist_ok=True)

    production_board = _normalize_frame(_read_csv(production_board_csv))
    shadow_board = _normalize_frame(_read_csv(shadow_board_csv))

    production_keys = _identity_series(production_board)
    shadow_keys = _identity_series(shadow_board)
    production_signatures = _stable_row_signature(production_board)
    shadow_signatures = _stable_row_signature(shadow_board)

    production_set = set(production_keys.dropna().astype(str).tolist())
    shadow_set = set(shadow_keys.dropna().astype(str).tolist())
    production_signature_set = set(production_signatures.dropna().astype(str).tolist())
    shadow_signature_set = set(shadow_signatures.dropna().astype(str).tolist())

    matched_production = 0
    for identity, signature in zip(production_keys.astype(str).tolist(), production_signatures.astype(str).tolist()):
        if identity in shadow_set or signature in shadow_signature_set:
            matched_production += 1
    matched_shadow = 0
    for identity, signature in zip(shadow_keys.astype(str).tolist(), shadow_signatures.astype(str).tolist()):
        if identity in production_set or signature in production_signature_set:
            matched_shadow += 1

    production_only = max(0, int(len(production_board) - matched_production))
    shadow_only = max(0, int(len(shadow_board) - matched_shadow))
    common_rows = min(matched_production, matched_shadow)
    shadow_added_rows = int(shadow_board.get("append_shadow_added", pd.Series(False, index=shadow_board.index)).fillna(False).astype(bool).sum())

    price_mismatches: list[dict[str, Any]] = []
    if not production_board.empty and not shadow_board.empty and "candidate_id" in production_board.columns and "candidate_id" in shadow_board.columns:
        produced = production_board.set_index("candidate_id", drop=False)
        shadowed = shadow_board.set_index("candidate_id", drop=False)
        shared_ids = set(produced.index) & set(shadowed.index)
        for candidate_id in sorted(shared_ids)[:20]:
            production_row = produced.loc[candidate_id]
            shadow_row = shadowed.loc[candidate_id]
            production_price = production_row.get("market_side_price")
            shadow_price = shadow_row.get("market_side_price")
            if pd.isna(production_price) and pd.isna(shadow_price):
                continue
            if str(production_price) != str(shadow_price):
                price_mismatches.append(
                    {
                        "candidate_id": candidate_id,
                        "production_price": production_price,
                        "shadow_price": shadow_price,
                        "production_line": production_row.get("line", production_row.get("market_line")),
                        "shadow_line": shadow_row.get("line", shadow_row.get("market_line")),
                    }
                )

    report = {
        "input_paths": {
            "production_board_csv": str(production_board_csv) if production_board_csv else "",
            "shadow_board_csv": str(shadow_board_csv) if shadow_board_csv else "",
        },
        "output_paths": {
            "comparison_report_json": str(output_dir / "price_defense_shadow_comparison.json"),
            "comparison_report_md": str(output_dir / "price_defense_shadow_comparison.md"),
        },
        "production_rows": int(len(production_board)),
        "shadow_rows": int(len(shadow_board)),
        "common_rows": int(common_rows),
        "production_only_rows": int(production_only),
        "shadow_only_rows": int(shadow_only),
        "shadow_added_rows": shadow_added_rows,
        "production_average_edge": _numeric_mean(production_board, "edge"),
        "shadow_average_edge": _numeric_mean(shadow_board, "edge"),
        "production_average_confidence": _numeric_mean(production_board, "final_confidence"),
        "shadow_average_confidence": _numeric_mean(shadow_board, "final_confidence"),
        "price_mismatch_rows": len(price_mismatches),
        "price_mismatch_examples": price_mismatches,
        "production_behavior_changed": False,
    }

    _write_json(output_dir / "price_defense_shadow_comparison.json", report)

    md_lines = [
        "# Price Defense Shadow Board Comparison",
        "",
        f"- Production final board rows: {report['production_rows']}",
        f"- Shadow board rows: {report['shadow_rows']}",
        f"- Rows present in both boards: {report['common_rows']}",
        f"- Production-only rows: {report['production_only_rows']}",
        f"- Shadow-only rows: {report['shadow_only_rows']}",
        f"- Shadow-added rows: {report['shadow_added_rows']}",
        f"- Production average edge: {report['production_average_edge']}",
        f"- Shadow average edge: {report['shadow_average_edge']}",
        f"- Price mismatches observed: {report['price_mismatch_rows']}",
        "",
        "This comparison is audit-only and does not change production behavior.",
    ]
    _write_markdown(output_dir / "price_defense_shadow_comparison.md", md_lines)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a production final board against a shadow price-defense board.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--production-board-csv", type=Path, required=True)
    parser.add_argument("--shadow-board-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_price_defense_boards(
        output_dir=args.output_dir,
        production_board_csv=args.production_board_csv,
        shadow_board_csv=args.shadow_board_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
