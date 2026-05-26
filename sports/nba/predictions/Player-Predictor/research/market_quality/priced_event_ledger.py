from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.common import build_candidate_id
from .edge_defendability import annotate_edge_defendability
from .price_normalization import american_odds_to_decimal, american_odds_to_break_even
from .price_provenance_schema import annotate_price_provenance_frame


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            out = out.fillna(pd.to_numeric(frame[column], errors="coerce"))
    if np.isnan(default):
        return out
    return out.fillna(float(default))


def _coalesce_text(frame: pd.DataFrame, columns: list[str], default: str = "") -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str)
        out = pd.Series(out, index=frame.index, dtype="object")
        out = out.where(out.astype(str).str.strip().ne(""), values)
    return out.fillna(default).astype(str)


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def _derive_uncertainty_penalty(frame: pd.DataFrame, stress_probability: pd.Series) -> pd.Series:
    explicit = _coalesce_numeric(frame, ["uncertainty_penalty"], default=np.nan)
    if explicit.notna().any():
        return explicit.fillna(0.0).clip(lower=0.0, upper=0.15)
    posterior_ci_low = _coalesce_numeric(frame, ["posterior_ci_low", "lcb_probability"], default=np.nan)
    penalty = pd.Series(np.nan, index=frame.index, dtype="float64")
    penalty = penalty.where(posterior_ci_low.isna(), (stress_probability - posterior_ci_low).clip(lower=0.0, upper=0.15))
    posterior_variance = _coalesce_numeric(frame, ["posterior_variance"], default=np.nan)
    variance_penalty = np.sqrt(posterior_variance.clip(lower=0.0)).clip(lower=0.0, upper=1.0) * 0.35
    penalty = penalty.fillna(variance_penalty.clip(lower=0.0, upper=0.12))
    belief_unc = _coalesce_numeric(frame, ["belief_uncertainty_normalized"], default=np.nan)
    belief_unc = belief_unc.fillna(_coalesce_numeric(frame, ["belief_uncertainty"], default=1.0).clip(lower=0.0, upper=1.0))
    penalty = penalty.fillna((belief_unc * 0.06).clip(lower=0.0, upper=0.10))
    return penalty.fillna(0.0).clip(lower=0.0, upper=0.15)


def _expected_value(probability: pd.Series, decimal_odds: pd.Series, p_push: pd.Series) -> pd.Series:
    prob = _clip01(probability)
    push = _clip01(p_push)
    odds = pd.to_numeric(decimal_odds, errors="coerce")
    payout = odds - 1.0
    loss_prob = (1.0 - prob - push).clip(lower=0.0, upper=1.0)
    ev = prob * payout - loss_prob
    return ev.where(odds.notna(), np.nan)


def build_priced_event_ledger_frame(
    rows: pd.DataFrame,
    *,
    record_scope: str | None = None,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    working = rows.copy()
    if "candidate_id" not in working.columns and {"target", "direction"}.issubset(set(working.columns)):
        working["candidate_id"] = build_candidate_id(working)
    working = annotate_price_provenance_frame(working)

    if record_scope is not None:
        working["record_scope"] = str(record_scope)
    elif "record_scope" not in working.columns:
        working["record_scope"] = "candidate"
    working["selected_on_board"] = _coalesce_numeric(working, ["selected_on_board"], default=np.nan).fillna(
        working["record_scope"].astype(str).eq("selected").astype(float)
    ).astype(bool)

    working["game_id"] = _coalesce_text(working, ["game_id", "market_event_id"], default="")
    working["game_date"] = _coalesce_text(working, ["game_date", "market_date", "run_date"], default="")
    working["player_name"] = _coalesce_text(working, ["player_name", "player", "market_player_raw"], default="")
    working["market_type"] = _coalesce_text(working, ["market_type", "market_id"], default="")
    if working["market_type"].eq("").any() and {"target", "direction"}.issubset(set(working.columns)):
        missing = working["market_type"].eq("")
        working.loc[missing, "market_type"] = (
            _coalesce_text(working.loc[missing], ["target"]).str.upper().str.strip()
            + "_"
            + _coalesce_text(working.loc[missing], ["direction", "side"]).str.upper().str.strip()
        )
    working["side"] = _coalesce_text(working, ["side", "direction"], default="").str.upper().str.strip()
    working["line"] = _coalesce_numeric(working, ["line", "market_line"], default=np.nan)
    working["market_side_decimal_odds"] = _coalesce_numeric(
        working,
        ["market_side_decimal_odds"],
        default=np.nan,
    ).where(
        _coalesce_numeric(working, ["market_side_decimal_odds"], default=np.nan).notna(),
        _coalesce_numeric(working, ["market_side_price"], default=np.nan).map(american_odds_to_decimal),
    )
    working["market_side_break_even"] = _coalesce_numeric(
        working,
        ["market_side_break_even", "break_even_probability", "implied_probability"],
        default=np.nan,
    ).where(
        _coalesce_numeric(working, ["market_side_break_even", "break_even_probability", "implied_probability"], default=np.nan).notna(),
        _coalesce_numeric(working, ["market_side_price"], default=np.nan).map(american_odds_to_break_even),
    )
    working["market_side_implied_probability"] = _coalesce_numeric(
        working,
        ["market_side_implied_probability", "implied_probability"],
        default=np.nan,
    ).fillna(working["market_side_break_even"])

    working["model_probability"] = _clip01(
        _coalesce_numeric(
            working,
            [
                "model_probability",
                "predicted_probability",
                "raw_expected_win_rate",
                "bayesian_expected_win_rate",
                "selected_board_prob_raw",
                "p_base",
                "expected_win_rate",
            ],
            default=0.50,
        )
    )
    working["stress_probability"] = _clip01(
        _coalesce_numeric(
            working,
            [
                "stress_probability",
                "expected_win_rate",
                "board_play_win_prob",
                "p_final",
                "selector_expected_win_rate",
                "model_probability",
            ],
            default=0.50,
        )
    )
    working["p_push"] = _clip01(
        _coalesce_numeric(working, ["p_push", "expected_push_rate"], default=0.0)
    )
    uncertainty_penalty = _derive_uncertainty_penalty(working, working["stress_probability"])
    existing_lcb = _coalesce_numeric(working, ["lcb_probability", "posterior_ci_low"], default=np.nan)
    working["lcb_probability"] = _clip01(
        existing_lcb.where(existing_lcb.notna(), working["stress_probability"] - uncertainty_penalty)
    )
    working["forecastability_score"] = _coalesce_numeric(working, ["forecastability_score"], default=np.nan)
    working["scenario_agreement"] = _coalesce_numeric(working, ["scenario_agreement"], default=np.nan)
    working["chaos_score"] = _coalesce_numeric(working, ["chaos_score"], default=np.nan)

    working["raw_edge"] = _coalesce_numeric(working, ["raw_edge"], default=np.nan).where(
        _coalesce_numeric(working, ["raw_edge"], default=np.nan).notna(),
        working["model_probability"] - working["market_side_break_even"],
    )
    working["stress_edge"] = _coalesce_numeric(working, ["stress_edge"], default=np.nan).where(
        _coalesce_numeric(working, ["stress_edge"], default=np.nan).notna(),
        working["stress_probability"] - working["market_side_break_even"],
    )
    working["lcb_edge"] = _coalesce_numeric(working, ["lcb_edge"], default=np.nan).where(
        _coalesce_numeric(working, ["lcb_edge"], default=np.nan).notna(),
        working["lcb_probability"] - working["market_side_break_even"],
    )
    working["raw_ev"] = _expected_value(working["model_probability"], working["market_side_decimal_odds"], working["p_push"])
    working["stress_ev"] = _expected_value(working["stress_probability"], working["market_side_decimal_odds"], working["p_push"])
    working["lcb_ev"] = _expected_value(working["lcb_probability"], working["market_side_decimal_odds"], working["p_push"])

    working = annotate_edge_defendability(working)
    return working


def summarize_priced_event_ledger(rows: pd.DataFrame) -> dict[str, Any]:
    frame = rows.copy()
    if frame.empty:
        return {"row_count": 0}
    defendability = frame.get("edge_defendability_tier", pd.Series("", index=frame.index)).astype(str)
    return {
        "row_count": int(len(frame)),
        "selected_rows": int(frame.get("record_scope", pd.Series("", index=frame.index)).astype(str).eq("selected").sum()),
        "timestamp_safe_price_rows": int(frame.get("timestamp_safe_flag", pd.Series(False, index=frame.index)).astype(bool).sum()),
        "price_valid_rows": int(frame.get("price_validity_status", pd.Series("", index=frame.index)).astype(str).eq("PRICE_VALID").sum()),
        "edge_defendability_counts": {
            str(key): int(value)
            for key, value in defendability.value_counts(dropna=False).to_dict().items()
        },
    }


def write_priced_event_ledger(
    rows: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    record_scope: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = build_priced_event_ledger_frame(rows, record_scope=record_scope)
    output_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_csv, index=False)
    summary = summarize_priced_event_ledger(ledger)
    summary["output_csv"] = str(output_csv.resolve())
    if summary_json is not None:
        summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ledger, summary
