from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BOARD_VARIANTS = [
    "production_board_as_is",
    "price_defense_only_board",
    "forecastable_price_board",
    "structural_misprice_board",
    "safe_state_core_board",
    "safe_state_near_core_board",
    "safe_state_expanded_board",
    "true_unstable_shadow_rejections",
    "needs_more_sample_queue",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _identity(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    if "candidate_id" in frame.columns:
        return frame["candidate_id"].fillna("").astype(str)
    pieces = []
    for col in ["game_date", "player", "player_name", "target", "market_type", "side", "direction", "line", "market_line"]:
        if col in frame.columns:
            pieces.append(frame[col].fillna("").astype(str))
    if not pieces:
        return pd.Series([str(i) for i in frame.index], index=frame.index)
    out = pieces[0]
    for piece in pieces[1:]:
        out = out + "::" + piece
    return out


def _result_labels(frame: pd.DataFrame) -> pd.Series:
    status = _settlement_status(frame)["settlement_status"] if not frame.empty else pd.Series(dtype="object")
    labels = pd.Series(np.nan, index=frame.index, dtype="float64")
    labels = labels.mask(status.eq("SETTLED_WIN"), 1.0)
    labels = labels.mask(status.eq("SETTLED_LOSS"), 0.0)
    return labels


def _settlement_status(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "settlement_status",
                "is_resolved",
                "is_pending",
                "resolution_reason",
            ],
            index=frame.index,
        )

    status = pd.Series("PENDING", index=frame.index, dtype="object")
    reason = pd.Series("no_settled_result_or_actual_stat", index=frame.index, dtype="object")

    result_found = pd.Series(False, index=frame.index, dtype="bool")
    for col in ["settlement_status", "actual_result", "result", "settled_result"]:
        if col not in frame.columns:
            continue
        raw = frame[col].fillna("").astype(str).str.strip().str.upper()
        has_value = raw.ne("") & ~raw.isin({"NAN", "NONE", "NULL", "PENDING"})
        status = status.mask(has_value & raw.str.contains("VOID|CANCEL|DNP|NO ACTION", regex=True), "VOID")
        status = status.mask(has_value & raw.str.contains("PUSH|TIE", regex=True), "SETTLED_PUSH")
        status = status.mask(has_value & raw.str.contains("WIN|HIT|WON", regex=True), "SETTLED_WIN")
        status = status.mask(has_value & raw.str.contains("LOSS|MISS|LOST", regex=True), "SETTLED_LOSS")
        matched = has_value & status.isin({"SETTLED_WIN", "SETTLED_LOSS", "SETTLED_PUSH", "VOID"})
        result_found = result_found | matched
        reason = reason.mask(matched, f"explicit_{col}")

    stat_cols = ["actual_stat", "actual", "actual_value", "settled_stat"]
    actual = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in stat_cols:
        if col in frame.columns:
            actual = actual.where(actual.notna(), pd.to_numeric(frame[col], errors="coerce"))
    line = pd.to_numeric(frame.get("line", frame.get("market_line", pd.Series(np.nan, index=frame.index))), errors="coerce")
    side = frame.get("side", frame.get("direction", pd.Series("", index=frame.index))).fillna("").astype(str).str.upper()
    market_type = frame.get("market_type", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    side = side.where(side.isin({"OVER", "UNDER"}), np.where(market_type.str.endswith("_UNDER"), "UNDER", "OVER"))
    has_actual_and_line = actual.notna() & line.notna()
    unresolved = ~result_found
    push_mask = unresolved & has_actual_and_line & actual.eq(line)
    win_mask = unresolved & has_actual_and_line & (
        (side.eq("OVER") & actual.gt(line))
        | (side.eq("UNDER") & actual.lt(line))
    )
    loss_mask = unresolved & has_actual_and_line & (
        (side.eq("OVER") & actual.lt(line))
        | (side.eq("UNDER") & actual.gt(line))
    )
    status = status.mask(push_mask, "SETTLED_PUSH")
    status = status.mask(win_mask, "SETTLED_WIN")
    status = status.mask(loss_mask, "SETTLED_LOSS")
    reason = reason.mask(push_mask | win_mask | loss_mask, "actual_stat_vs_line")

    unknown_result = pd.Series(False, index=frame.index, dtype="bool")
    for col in ["actual_result", "result", "settled_result"]:
        if col not in frame.columns:
            continue
        raw = frame[col].fillna("").astype(str).str.strip()
        has_unknown = raw.ne("") & ~raw.str.lower().isin({"nan", "none", "null", "pending"}) & ~result_found
        unknown_result = unknown_result | has_unknown
    status = status.mask(unknown_result & status.eq("PENDING"), "UNMATCHED_OUTCOME")
    reason = reason.mask(unknown_result & reason.eq("no_settled_result_or_actual_stat"), "unmatched_result_label")

    is_resolved = status.isin({"SETTLED_WIN", "SETTLED_LOSS", "SETTLED_PUSH"})
    is_pending = status.isin({"PENDING", "UNMATCHED_OUTCOME"})
    return pd.DataFrame(
        {
            "settlement_status": status,
            "is_resolved": is_resolved,
            "is_pending": is_pending,
            "resolution_reason": reason,
        },
        index=frame.index,
    )


def _brier(frame: pd.DataFrame, labels: pd.Series) -> float | None:
    prob_cols = ["stress_probability", "model_probability", "expected_win_rate"]
    probs = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in prob_cols:
        if col in frame.columns:
            probs = probs.where(probs.notna(), pd.to_numeric(frame[col], errors="coerce"))
    status = _settlement_status(frame)["settlement_status"] if not frame.empty else pd.Series(dtype="object")
    mask = labels.notna() & probs.notna() & status.isin({"SETTLED_WIN", "SETTLED_LOSS"})
    if not mask.any():
        return None
    return float(((probs.loc[mask] - labels.loc[mask]) ** 2).mean())


def _ece(frame: pd.DataFrame, labels: pd.Series, bins: int = 5) -> float | None:
    prob_cols = ["stress_probability", "model_probability", "expected_win_rate"]
    probs = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in prob_cols:
        if col in frame.columns:
            probs = probs.where(probs.notna(), pd.to_numeric(frame[col], errors="coerce"))
    status = _settlement_status(frame)["settlement_status"] if not frame.empty else pd.Series(dtype="object")
    mask = labels.notna() & probs.notna() & status.isin({"SETTLED_WIN", "SETTLED_LOSS"})
    if not mask.any():
        return None
    probs = probs.loc[mask].clip(0, 1)
    labels = labels.loc[mask]
    edges = np.linspace(0, 1, bins + 1)
    total = len(probs)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        bucket = probs.ge(low) & probs.lt(high if high < 1 else high + 1e-9)
        if bucket.any():
            ece += float(bucket.sum() / total) * abs(float(probs.loc[bucket].mean()) - float(labels.loc[bucket].mean()))
    return float(ece)


def _roi(frame: pd.DataFrame, labels: pd.Series) -> tuple[float | None, float]:
    status = _settlement_status(frame)["settlement_status"] if not frame.empty else pd.Series(dtype="object")
    resolved = status.isin({"SETTLED_WIN", "SETTLED_LOSS"})
    if not resolved.any():
        return None, 0.0
    odds = pd.to_numeric(frame.get("market_side_decimal_odds", pd.Series(1.9091, index=frame.index)), errors="coerce").fillna(1.9091)
    profit = labels.where(labels != 1.0, odds - 1.0).where(labels == 1.0, -1.0)
    profit = profit.where(labels.notna(), 0.0)
    units = float(profit.loc[resolved].sum())
    return float(units / resolved.sum()), units


def _metrics(name: str, frame: pd.DataFrame, production: pd.DataFrame) -> dict[str, Any]:
    status = _settlement_status(frame)
    labels = _result_labels(frame)
    resolved = status["is_resolved"] if not status.empty else pd.Series(dtype=bool)
    pending = status["is_pending"] if not status.empty else pd.Series(dtype=bool)
    wins = int(labels.eq(1.0).sum())
    losses = int(labels.eq(0.0).sum())
    pushes = int(status["settlement_status"].eq("SETTLED_PUSH").sum()) if not status.empty else 0
    voids = int(status["settlement_status"].eq("VOID").sum()) if not status.empty else 0
    unmatched = int(status["settlement_status"].eq("UNMATCHED_OUTCOME").sum()) if not status.empty else 0
    roi, profit = _roi(frame, labels)
    prod_ids = set(_identity(production).tolist())
    frame_ids = set(_identity(frame).tolist())
    removed = production.loc[~_identity(production).isin(frame_ids)].copy() if not production.empty else pd.DataFrame()
    added = frame.loc[~_identity(frame).isin(prod_ids)].copy() if not frame.empty else pd.DataFrame()
    removed_labels = _result_labels(removed)
    added_labels = _result_labels(added)
    return {
        "variant": name,
        "rows": int(len(frame)),
        "pending_rows": int(pending.sum()) if not pending.empty else 0,
        "resolved_rows": int(resolved.sum()),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "voids": voids,
        "unresolved_rows": int(pending.sum() + voids) if not pending.empty else int(voids),
        "unmatched_rows": unmatched,
        "hit_rate": None if (wins + losses) == 0 else float(wins / (wins + losses)),
        "roi": roi,
        "profit_units": profit,
        "brier": _brier(frame, labels),
        "ece": _ece(frame, labels),
        "calibration_gap": None,
        "production_rows_removed": int(len(removed)),
        "production_wins_removed": int(removed_labels.eq(1.0).sum()),
        "production_losses_removed": int(removed_labels.eq(0.0).sum()),
        "shadow_rows_added": int(len(added)),
        "shadow_added_wins": int(added_labels.eq(1.0).sum()),
        "shadow_added_losses": int(added_labels.eq(0.0).sum()),
    }


def evaluate_safe_state_shadow_results(*, board_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or board_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    boards = {name: _read_csv(board_dir / f"{name}.csv") for name in BOARD_VARIANTS}
    production = boards.get("production_board_as_is", pd.DataFrame())
    rows = [_metrics(name, board, production) for name, board in boards.items()]
    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / "safe_state_shadow_settlement_metrics.csv"
    evaluation_rows_path = output_dir / "safe_state_settlement_rows.csv"
    status_audit_path = output_dir / "safe_state_settlement_status_audit.csv"
    status_audit_json_path = output_dir / "safe_state_settlement_status_audit.json"
    metrics_df.to_csv(metrics_path, index=False)
    settlement_rows = []
    audit_rows = []
    for name, board in boards.items():
        if board.empty:
            continue
        working = board.copy()
        working["board_variant"] = name
        status_frame = _settlement_status(working)
        for col in status_frame.columns:
            working[col] = status_frame[col]
        working["settlement_label"] = _result_labels(working)
        settlement_rows.append(working)
        audit = pd.DataFrame(
            {
                "board_variant": name,
                "candidate_id": working.get("candidate_id", pd.Series("", index=working.index)),
                "player": working.get("player", working.get("player_name", pd.Series("", index=working.index))),
                "market_date": working.get("market_date", working.get("game_date", pd.Series("", index=working.index))),
                "target": working.get("target", pd.Series("", index=working.index)),
                "side": working.get("side", working.get("direction", pd.Series("", index=working.index))),
                "line": working.get("line", working.get("market_line", pd.Series(np.nan, index=working.index))),
                "actual_stat": working.get("actual_stat", working.get("actual", working.get("actual_value", pd.Series(np.nan, index=working.index)))),
                "settlement_status": status_frame["settlement_status"],
                "is_resolved": status_frame["is_resolved"],
                "is_pending": status_frame["is_pending"],
                "resolution_reason": status_frame["resolution_reason"],
            }
        )
        audit_rows.append(audit)
    pd.concat(settlement_rows, ignore_index=True).to_csv(evaluation_rows_path, index=False) if settlement_rows else pd.DataFrame().to_csv(evaluation_rows_path, index=False)
    audit_df = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    audit_df.to_csv(status_audit_path, index=False)
    audit_summary = {
        "rows": int(len(audit_df)),
        "settlement_status_counts": audit_df.get("settlement_status", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict() if not audit_df.empty else {},
        "pending_rows": int(audit_df.get("is_pending", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not audit_df.empty else 0,
        "resolved_rows": int(audit_df.get("is_resolved", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not audit_df.empty else 0,
        "production_behavior_changed": False,
        "promotion_claim": False,
    }
    status_audit_json_path.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")

    unique_dates = set()
    for board in boards.values():
        if "game_date" in board.columns:
            unique_dates.update(board["game_date"].dropna().astype(str).tolist())
        elif "market_date" in board.columns:
            unique_dates.update(board["market_date"].dropna().astype(str).tolist())
    one_slate = len(unique_dates) <= 1
    report = {
        "board_dir": str(board_dir),
        "output_paths": {
            "metrics_csv": str(metrics_path),
            "rows_csv": str(evaluation_rows_path),
            "status_audit_csv": str(status_audit_path),
            "status_audit_json": str(status_audit_json_path),
            "report_json": str(output_dir / "safe_state_shadow_settlement_report.json"),
            "evaluation_json": str(output_dir / "safe_state_settlement_evaluation.json"),
            "evaluation_markdown": str(output_dir / "safe_state_settlement_evaluation.md"),
        },
        "variant_metrics": rows,
        "critical_questions": {
            "does_price_defense_alone_help": "requires_settlement" if audit_summary["pending_rows"] > 0 else "requires_settled_multi_window_replay",
            "does_forecastability_improve_price_defense": "requires_settlement" if audit_summary["pending_rows"] > 0 else "requires_settled_multi_window_replay",
            "does_structural_mispricing_improve_price_defense": "requires_settlement" if audit_summary["pending_rows"] > 0 else "requires_settled_multi_window_replay",
            "does_safe_state_core_outperform_production": "requires_settlement" if audit_summary["pending_rows"] > 0 else "requires_settled_multi_window_replay",
            "did_true_volatility_rejections_mostly_lose": "requires_settlement" if audit_summary["pending_rows"] > 0 else "requires_multi_slate_replay",
            "did_needs_more_sample_rows_scatter": "requires_settlement" if audit_summary["pending_rows"] > 0 else "requires_multi_slate_replay",
            "did_zero_safe_state_core_correctly_imply_abstention": "tracked_shadow_only",
        },
        "promotion_ready": False,
        "promotion_claim": False,
        "status": "NEEDS_MORE_SAMPLE" if one_slate else "SHADOW_ONLY_NEEDS_REPLAY",
        "blocked_reasons": ["single_slate_or_insufficient_windows"] if one_slate else ["shadow_only_no_promotion_gate"],
    }
    (output_dir / "safe_state_shadow_settlement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "safe_state_settlement_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "safe_state_shadow_settlement_report.md").write_text(
        "\n".join(
            [
                "# Safe-State Shadow Settlement Report",
                "",
                f"- Status: {report['status']}",
                "- Promotion ready: false",
                "- Promotion claim: false",
                f"- Pending rows: {audit_summary['pending_rows']}",
                f"- Resolved rows: {audit_summary['resolved_rows']}",
                "",
                "This evaluator is settlement-ready but cannot approve promotion from one slate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "safe_state_settlement_evaluation.md").write_text(
        "\n".join(
            [
                "# Safe-State Settlement Evaluation",
                "",
                f"- Status: {report['status']}",
                "- Promotion ready: false",
                "- Promotion claim: false",
                f"- Pending rows: {audit_summary['pending_rows']}",
                f"- Resolved rows: {audit_summary['resolved_rows']}",
                "",
                "Boards compared: production, price-defense-only, SAFE_STATE_CORE, SAFE_STATE_NEAR_CORE, true-unstable rejections, and needs-more-sample queue.",
                "",
                "A single slate cannot promote anything. Winning true-unstable rows do not automatically invalidate the risk label.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate settled safe-state shadow board variants.")
    parser.add_argument("--board-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_safe_state_shadow_results(board_dir=args.board_dir, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
