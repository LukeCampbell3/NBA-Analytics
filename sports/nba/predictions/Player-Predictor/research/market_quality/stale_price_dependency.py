from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.market_quality.common import DEFAULT_BREAK_EVEN_PROB, american_odds_to_break_even
from research.common import safe_float, series_numeric, series_text


DEFAULT_PRICE_DEPENDENT_MARGIN = 0.015


def annotate_stale_price_dependency_rows(rows: pd.DataFrame, *, price_dependent_margin: float = DEFAULT_PRICE_DEPENDENT_MARGIN) -> pd.DataFrame:
    frame = rows.copy()
    if frame.empty:
        return frame

    stress_probability = pd.to_numeric(
        frame.get("stress_probability", frame.get("expected_win_rate")),
        errors="coerce",
    ).fillna(pd.to_numeric(frame.get("expected_win_rate"), errors="coerce"))
    original_break_even = pd.to_numeric(
        frame.get("existing_market_side_break_even", frame.get("market_side_break_even")),
        errors="coerce",
    ).fillna(
        pd.to_numeric(frame.get("break_even_prob"), errors="coerce")
    ).fillna(DEFAULT_BREAK_EVEN_PROB)
    original_price = pd.to_numeric(
        frame.get("existing_market_side_price", frame.get("market_side_price")),
        errors="coerce",
    )
    corrected_price = pd.to_numeric(frame.get("corrected_price", frame.get("market_side_price")), errors="coerce")
    corrected_break_even = pd.to_numeric(frame.get("corrected_break_even", frame.get("market_side_break_even")), errors="coerce")
    corrected_break_even = corrected_break_even.where(corrected_break_even.notna(), corrected_price.map(american_odds_to_break_even))

    original_edge = stress_probability - original_break_even
    corrected_edge = stress_probability - corrected_break_even
    edge_decay = corrected_edge - original_edge

    validity = series_text(frame, "price_validity_status")
    source_type = series_text(frame, "price_source_type").str.upper().str.strip()
    stale_flag = series_text(frame, "stale_price_flag").map(lambda value: str(value).strip().lower() in {"true", "1"})
    line_move = pd.to_numeric(frame.get("line_moved_since_prediction"), errors="coerce").fillna(0.0)
    odds_move = pd.to_numeric(frame.get("odds_moved_since_prediction"), errors="coerce").fillna(0.0)
    selected_rank = pd.to_numeric(frame.get("selected_rank", pd.Series(np.nan, index=frame.index)), errors="coerce").fillna(np.inf)
    market_books = pd.to_numeric(frame.get("market_books", frame.get("snapshot_market_books", pd.Series(np.nan, index=frame.index))), errors="coerce").fillna(np.inf)
    selected_or_near = (
        series_text(frame, "record_scope").eq("selected")
        | series_text(frame, "recommendation").isin(["consider", "strong", "elite", "balanced_playable", "boundary_playable"])
        | selected_rank.le(12.0)
    )
    price_reliant = selected_or_near & (
        market_books.le(2.0)
        | (original_edge.ge(0.0) & original_edge.le(0.02))
    )

    close_only_mask = validity.eq("DIAGNOSTIC_ONLY") & source_type.eq("CLOSE_ONLY_DIAGNOSTIC")
    synthetic_mask = validity.eq("DIAGNOSTIC_ONLY") & source_type.eq("SYNTHETIC_DIAGNOSTIC")
    unknown_mask = validity.eq("PRICE_SOURCE_UNKNOWN") | (
        validity.eq("DIAGNOSTIC_ONLY") & source_type.eq("UNKNOWN")
    )
    subregion = pd.Series("", index=frame.index, dtype="object")
    subregion = subregion.mask(price_reliant & validity.eq("MISSING_PRICE"), "MISSING_PRICE_EDGE_UNTRUSTED")
    subregion = subregion.mask(price_reliant & validity.eq("INVALID_PRICE"), "INVALID_PRICE_EDGE_UNTRUSTED")
    subregion = subregion.mask(price_reliant & close_only_mask, "CLOSE_ONLY_PRICE_CONTAMINATION")
    subregion = subregion.mask(price_reliant & synthetic_mask, "CLOSE_ONLY_PRICE_CONTAMINATION")
    subregion = subregion.mask(price_reliant & unknown_mask, "PRICE_SOURCE_UNKNOWN_DIAGNOSTIC_ONLY")
    subregion = subregion.mask(price_reliant & (subregion.eq("")) & stale_flag & odds_move.abs().gt(0.0), "PRICE_MOVED_EDGE_DECAY")
    subregion = subregion.mask(price_reliant & (subregion.eq("")) & stale_flag & line_move.abs().gt(0.0), "LINE_MOVED_EDGE_DECAY")
    subregion = subregion.mask(price_reliant & (subregion.eq("")) & stale_flag, "STALE_PRICE_DEPENDENCY")

    proposed_decision = pd.Series("KEEP", index=frame.index, dtype="object")
    proposed_decision = proposed_decision.mask(subregion.isin(["PRICE_SOURCE_UNKNOWN_DIAGNOSTIC_ONLY", "CLOSE_ONLY_PRICE_CONTAMINATION"]), "DIAGNOSTIC_ONLY")
    proposed_decision = proposed_decision.mask(subregion.isin(["MISSING_PRICE_EDGE_UNTRUSTED", "INVALID_PRICE_EDGE_UNTRUSTED"]), "PASS_AT_PRICE")
    proposed_decision = proposed_decision.mask((subregion.ne("")) & corrected_break_even.notna() & (corrected_edge <= 0.0), "PASS_AT_PRICE")
    proposed_decision = proposed_decision.mask(
        (subregion.ne("")) & corrected_break_even.notna() & (corrected_edge > 0.0) & (corrected_edge <= float(price_dependent_margin)),
        "PRICE_DEPENDENT",
    )
    proposed_decision = proposed_decision.mask(subregion.isin(["PRICE_SOURCE_UNKNOWN_DIAGNOSTIC_ONLY", "CLOSE_ONLY_PRICE_CONTAMINATION"]), "DIAGNOSTIC_ONLY")

    price_fix_supported = corrected_break_even.notna()
    would_change_decision = price_reliant & subregion.ne("") & proposed_decision.ne("KEEP") & price_fix_supported

    frame["stale_price_subregion"] = subregion
    frame["original_price"] = original_price
    frame["corrected_price"] = corrected_price
    frame["original_break_even"] = original_break_even
    frame["corrected_break_even"] = corrected_break_even
    frame["original_edge"] = original_edge
    frame["corrected_edge"] = corrected_edge
    frame["edge_decay"] = edge_decay
    frame["would_change_decision"] = would_change_decision
    frame["proposed_decision_after_price_fix"] = proposed_decision
    return frame


def summarize_stale_price_dependency(rows: pd.DataFrame) -> dict[str, Any]:
    frame = rows.copy()
    if frame.empty:
        return {
            "row_count": 0,
            "subregion_counts": [],
        }
    flagged = frame.loc[series_text(frame, "stale_price_subregion").ne("")].copy()
    summary = {
        "row_count": int(len(frame)),
        "flagged_row_count": int(len(flagged)),
        "selected_row_count": int(series_text(frame, "record_scope").eq("selected").sum()),
        "would_change_decision_count": int(series_text(frame, "would_change_decision").map(lambda value: str(value).strip().lower() in {"true", "1"}).sum()),
        "subregion_counts": flagged.groupby("stale_price_subregion", dropna=False).agg(
            row_count=("candidate_id", "count"),
            selected_count=("record_scope", lambda s: int(pd.Series(s).astype(str).eq("selected").sum())),
            resolved_count=("result", lambda s: int(pd.Series(s).astype(str).str.lower().isin(["win", "loss"]).sum())),
            losses=("result", lambda s: int(pd.Series(s).astype(str).str.lower().eq("loss").sum())),
            wins=("result", lambda s: int(pd.Series(s).astype(str).str.lower().eq("win").sum())),
            would_change_decision_count=("would_change_decision", lambda s: int(pd.Series(s).astype(bool).sum())),
            avg_edge_decay=("edge_decay", "mean"),
        ).reset_index().to_dict(orient="records") if not flagged.empty else [],
    }
    return summary


def write_stale_price_outputs(
    rows: pd.DataFrame,
    *,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir.resolve().mkdir(parents=True, exist_ok=True)
    annotated = annotate_stale_price_dependency_rows(rows)
    summary = summarize_stale_price_dependency(annotated)
    annotated.to_csv(output_dir / "stale_price_dependency_rows.csv", index=False)
    (output_dir / "stale_price_dependency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return annotated, summary
