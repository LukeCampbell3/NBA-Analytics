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

from research.safe_state.evaluate_safe_state_shadow_results import BOARD_VARIANTS, evaluate_safe_state_shadow_results


def _read_csv_or_json(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return pd.DataFrame()
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ["rows", "actuals", "data"]:
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _norm_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna("").astype(str).str.strip().str.lower()
    return pd.Series("", index=frame.index, dtype="object")


def _norm_line(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame.get("line", frame.get("market_line", pd.Series(pd.NA, index=frame.index))), errors="coerce")
    return values.round(3).fillna(-9999.0).astype(str)


def _signature(frame: pd.DataFrame) -> pd.Series:
    market_date = _norm_text(frame, "market_date").where(_norm_text(frame, "market_date").ne(""), _norm_text(frame, "game_date"))
    player = _norm_text(frame, "player").where(_norm_text(frame, "player").ne(""), _norm_text(frame, "player_name"))
    side = _norm_text(frame, "side").where(_norm_text(frame, "side").ne(""), _norm_text(frame, "direction"))
    target = _norm_text(frame, "target")
    return market_date + "::" + player + "::" + target + "::" + side + "::" + _norm_line(frame)


def _actual_columns(actuals: pd.DataFrame) -> list[str]:
    preferred = [
        "candidate_id",
        "actual_stat",
        "actual_result",
        "result",
        "settled_result",
        "settlement_status",
        "void_reason",
        "game_id",
        "market_date",
        "game_date",
        "player",
        "player_name",
        "target",
        "side",
        "direction",
        "line",
        "market_line",
    ]
    return [column for column in preferred if column in actuals.columns]


def _merge_actuals(board: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if board.empty or actuals.empty:
        return board.copy()
    out = board.copy()
    actuals = actuals.copy()
    actual_cols = _actual_columns(actuals)
    if "candidate_id" in out.columns and "candidate_id" in actuals.columns:
        joined = out.merge(actuals[actual_cols].drop_duplicates("candidate_id"), on="candidate_id", how="left", suffixes=("", "_actuals"))
    else:
        out["_settlement_signature"] = _signature(out)
        actuals["_settlement_signature"] = _signature(actuals)
        joined = out.merge(actuals[["_settlement_signature", *[c for c in actual_cols if c != "candidate_id"]]].drop_duplicates("_settlement_signature"), on="_settlement_signature", how="left", suffixes=("", "_actuals"))
    for column in ["actual_stat", "actual_result", "result", "settled_result", "settlement_status"]:
        actual_col = f"{column}_actuals"
        if actual_col in joined.columns:
            if column not in joined.columns:
                joined[column] = joined[actual_col]
            else:
                base = joined[column].astype("object")
                addition = joined[actual_col].astype("object")
                joined[column] = base.where(base.notna() & base.astype(str).str.strip().ne(""), addition)
    return joined.drop(columns=[c for c in joined.columns if c.endswith("_actuals") or c == "_settlement_signature"], errors="ignore")


def update_safe_state_shadow_settlement(
    *,
    run_dir: Path,
    actuals_source: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    joined_dir = output_dir / "settlement_joined_boards"
    joined_dir.mkdir(parents=True, exist_ok=True)
    actuals = _read_csv_or_json(actuals_source)

    joined_paths: dict[str, str] = {}
    for variant in BOARD_VARIANTS:
        board = _read_csv(run_dir / f"{variant}.csv")
        joined = _merge_actuals(board, actuals)
        path = joined_dir / f"{variant}.csv"
        joined.to_csv(path, index=False)
        joined_paths[variant] = str(path)

    evaluation = evaluate_safe_state_shadow_results(board_dir=joined_dir, output_dir=output_dir)
    report = {
        "run_dir": str(run_dir),
        "actuals_source": str(actuals_source) if actuals_source else "",
        "output_dir": str(output_dir),
        "joined_board_dir": str(joined_dir),
        "joined_board_paths": joined_paths,
        "actual_rows": int(len(actuals)),
        "output_paths": evaluation.get("output_paths", {}),
        "settlement_status": "SETTLEMENT_EVALUATED" if int(pd.read_csv(output_dir / "safe_state_shadow_settlement_metrics.csv")["resolved_rows"].sum()) > 0 else "WAITING_FOR_SETTLEMENT",
        "production_behavior_changed": False,
        "promotion_claim": False,
    }
    (output_dir / "safe_state_shadow_settlement_update.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join actual outcomes to production-shadow safe-state board variants.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--actuals-source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = update_safe_state_shadow_settlement(run_dir=args.run_dir, actuals_source=args.actuals_source, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
