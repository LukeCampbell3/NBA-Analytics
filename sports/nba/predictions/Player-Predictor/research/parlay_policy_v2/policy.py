"""Leakage-safe two-leg parlay policy utilities (NBA).

This module deliberately separates:
  * probability / hit-rate eligibility,
  * uncertainty and shift control,
  * sportsbook-price EV,
  * ranking among already eligible parlays.

It is a research/backtest utility, not a live selector: nothing here places
wagers or reads a production board. Candidate rows are plain mappings so it
can be driven from any adapter (a CSV, a DataFrame, a synthetic fixture) as
long as the required fields below are present.

Scope note: this module implements and unit-tests the *gating mechanism*
(the order of operations, the penalties, the reject/accept reasons). It does
NOT ship a historical NBA hit-rate backtest, because this repository does not
currently contain settled two-leg NBA parlay candidates carrying the full
field set the policy needs (joint_sigma, joint_lcb, an actual sportsbook SGP
quote, injury/role/support state, shared-failure risk). See REPORT.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParlayPolicy:
    # Hit-rate / probability gates
    min_joint_probability: float = 0.50
    min_joint_lcb: float = 0.45
    min_leg_probability: float = 0.68
    uncertainty_lambda: float = 1.0

    # Structural reliability gates
    max_shared_failure_risk: float = 0.35
    min_compatible_state_score: float = 0.60
    max_shift_risk: float = 0.35
    max_joint_uncertainty: float = 0.12

    # Price gate. Must use the actual sportsbook parlay quote, never a
    # synthetic product of the two legs' individual decimal prices.
    min_actual_quote_ev: float = 0.00

    # Execution-state gates
    require_lineup_confirmed: bool = True
    require_role_stable: bool = True
    require_no_material_injury_uncertainty: bool = True
    require_in_support: bool = True
    require_joint_model_reliable: bool = True
    leg_count: int = 2


REQUIRED_SELECTION_FIELDS = {
    "leg_count",
    "min_leg_probability",
    "min_leg_sigma",
    "joint_probability",
    "joint_sigma",
    "joint_lcb",
    "actual_quote_decimal",
    "shared_failure_risk",
    "compatible_state_score",
    "shift_risk",
    "lineup_confirmed",
    "role_stable",
    "material_injury_uncertainty",
    "all_legs_in_support",
    "joint_model_reliable",
}

# Explicitly excluded from this optimizer because the prior cross-game path
# representation (raw "turn"/"accel_ratio" cross-event ordering) was
# structurally invalidated elsewhere in this repo's research history. A
# future within-event market path representation can be added under a new,
# timestamp-safe field name, but never under these legacy names.
FORBIDDEN_LEGACY_FIELDS = {"turn", "accel_ratio", "path_efficiency"}


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def decimal_to_break_even(decimal_odds: float) -> float:
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError("Decimal odds must be > 1")
    return 1.0 / d


def actual_quote_ev(joint_probability: float, actual_quote_decimal: float) -> float:
    """Expected return per 1 unit risked using the actual parlay quote."""
    p = float(joint_probability)
    d = float(actual_quote_decimal)
    if not 0 <= p <= 1:
        raise ValueError("Probability must be in [0,1]")
    if d <= 1:
        raise ValueError("Decimal odds must be > 1")
    return p * d - 1.0


def usable_probability(p: float, sigma: float, lam: float) -> float:
    """Conservative point probability p - lambda*sigma, clipped to [0,1]."""
    return float(np.clip(float(p) - float(lam) * max(0.0, float(sigma)), 0.0, 1.0))


def naive_joint_probability(leg_probabilities: Sequence[float]) -> float:
    p = 1.0
    for x in leg_probabilities:
        x = float(x)
        if not 0 <= x <= 1:
            raise ValueError("Leg probabilities must be in [0,1]")
        p *= x
    return p


def conservative_joint_probability(
    joint_probability: float,
    joint_sigma: float,
    uncertainty_lambda: float,
    dependency_penalty: float = 0.0,
) -> float:
    """Apply uncertainty and dependency penalties to a modeled joint probability."""
    p = usable_probability(joint_probability, joint_sigma, uncertainty_lambda)
    return float(np.clip(p - max(0.0, float(dependency_penalty)), 0.0, 1.0))


def validate_schema(candidate: Mapping[str, Any]) -> list[str]:
    missing = sorted(REQUIRED_SELECTION_FIELDS - set(candidate))
    forbidden = sorted(FORBIDDEN_LEGACY_FIELDS & set(candidate))
    errors = []
    if missing:
        errors.append("missing:" + ",".join(missing))
    if forbidden:
        errors.append("legacy_path_fields_forbidden:" + ",".join(forbidden))
    return errors


def evaluate_candidate(candidate: Mapping[str, Any], policy: ParlayPolicy) -> dict[str, Any]:
    """Return an auditable eligibility decision and derived metrics."""
    schema_errors = validate_schema(candidate)
    if schema_errors:
        return {"eligible": False, "reasons": schema_errors}

    min_leg_usable = usable_probability(
        candidate["min_leg_probability"], candidate["min_leg_sigma"], policy.uncertainty_lambda
    )
    p_joint_usable = conservative_joint_probability(
        candidate["joint_probability"], candidate["joint_sigma"], policy.uncertainty_lambda,
        candidate.get("dependency_penalty", 0.0),
    )
    p_joint_lcb = float(candidate["joint_lcb"])
    quote = float(candidate["actual_quote_decimal"])
    ev = actual_quote_ev(p_joint_usable, quote)
    break_even = decimal_to_break_even(quote)

    reasons: list[str] = []
    if int(candidate["leg_count"]) != policy.leg_count:
        reasons.append("LEG_COUNT")
    if min_leg_usable < policy.min_leg_probability:
        reasons.append("LEG_PROBABILITY")
    if p_joint_usable < policy.min_joint_probability:
        reasons.append("JOINT_PROBABILITY")
    if p_joint_lcb < policy.min_joint_lcb:
        reasons.append("JOINT_LCB")
    if float(candidate["joint_sigma"]) > policy.max_joint_uncertainty:
        reasons.append("JOINT_UNCERTAINTY")
    if float(candidate["shared_failure_risk"]) > policy.max_shared_failure_risk:
        reasons.append("SHARED_FAILURE")
    if float(candidate["compatible_state_score"]) < policy.min_compatible_state_score:
        reasons.append("STATE_COMPATIBILITY")
    if float(candidate["shift_risk"]) > policy.max_shift_risk:
        reasons.append("SHIFT_RISK")
    if policy.require_lineup_confirmed and not bool(candidate["lineup_confirmed"]):
        reasons.append("LINEUP")
    if policy.require_role_stable and not bool(candidate["role_stable"]):
        reasons.append("ROLE")
    if policy.require_no_material_injury_uncertainty and bool(candidate["material_injury_uncertainty"]):
        reasons.append("INJURY_UNCERTAINTY")
    if policy.require_in_support and not bool(candidate["all_legs_in_support"]):
        reasons.append("OUT_OF_SUPPORT")
    if policy.require_joint_model_reliable and not bool(candidate["joint_model_reliable"]):
        reasons.append("JOINT_MODEL_UNRELIABLE")
    if ev <= policy.min_actual_quote_ev:
        reasons.append("ACTUAL_QUOTE_EV")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "min_leg_usable": min_leg_usable,
        "p_joint_usable": p_joint_usable,
        "p_joint_lcb": p_joint_lcb,
        "break_even_probability": break_even,
        "actual_quote_ev": ev,
    }


def wilson_interval(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = wins / n
    den = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt((phat * (1 - phat) / n) + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def brier_score(y: Sequence[int], p: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    return float(np.mean((pp - yy) ** 2))


def expected_calibration_error(y: Sequence[int], p: Sequence[float], bins: int = 8) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    if len(yy) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pp >= lo) & ((pp < hi) if hi < 1 else (pp <= hi))
        if m.any():
            ece += m.mean() * abs(pp[m].mean() - yy[m].mean())
    return float(ece)


def evaluate_frame(df: pd.DataFrame, selected_col: str = "selected") -> dict[str, Any]:
    x = df[df[selected_col]].copy() if selected_col in df else df.copy()
    n = len(x)
    wins = int(x["won"].sum()) if n else 0
    lo, hi = wilson_interval(wins, n)
    out = {
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "hit_rate": wins / n if n else float("nan"),
        "wilson95_low": lo,
        "wilson95_high": hi,
        "coverage": n / len(df) if len(df) else float("nan"),
    }
    if n and "actual_quote_ev_realized" in x:
        out["mean_realized_return"] = float(x["actual_quote_ev_realized"].mean())
    if n and "p_joint_usable" in x:
        out["mean_predicted_joint"] = float(x["p_joint_usable"].mean())
        out["brier"] = brier_score(x["won"], x["p_joint_usable"])
        out["ece"] = expected_calibration_error(x["won"], x["p_joint_usable"])
    return out


def apply_policy_frame(df: pd.DataFrame, policy: ParlayPolicy) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        decision = evaluate_candidate(r.to_dict(), policy)
        row = r.to_dict()
        row.update(decision)
        row["selected"] = decision["eligible"]
        rows.append(row)
    return pd.DataFrame(rows)


def rank_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Rank only after eligibility: highest actual-quote EV, then joint LCB/probability."""
    x = df[df["eligible"]].copy()
    if x.empty:
        return x
    return x.sort_values(
        ["actual_quote_ev", "p_joint_lcb", "p_joint_usable"],
        ascending=[False, False, False],
    )


def rolling_regime_gate(
    outcomes: Sequence[int],
    window: int = 5,
    min_history: int = 5,
    min_recent_hit_rate: float = 0.50,
) -> list[bool]:
    """Strictly pre-outcome health gate. A slate's own outcome never decides itself.

    Shadow outcomes can continue to update the gate even while execution is
    suspended -- only ``hist`` (strictly prior outcomes) feeds the decision
    for the current index.
    """
    hist: list[int] = []
    active: list[bool] = []
    for y in outcomes:
        recent = hist[-window:]
        healthy = True
        if len(recent) >= min_history:
            healthy = float(np.mean(recent)) >= min_recent_hit_rate
        active.append(healthy)
        hist.append(int(y))
    return active


def _policy_from_params(params: Mapping[str, Any]) -> ParlayPolicy:
    return ParlayPolicy(**params)


def optimize_policy_grid(
    development: pd.DataFrame,
    param_grid: Iterable[Mapping[str, Any]],
    min_selected: int = 20,
    min_coverage: float = 0.10,
) -> tuple[ParlayPolicy, pd.DataFrame]:
    """Tune only on a development block.

    Objective is lexicographic and conservative:
      1) maximize the Wilson lower bound of hit rate,
      2) maximize realized mean return if present,
      3) maximize coverage.

    A grid cell that cannot meet sample/coverage constraints is ineligible,
    so a small, lucky, 100%-hit-rate cell cannot outrank a large, calibrated
    one on raw hit rate alone.
    """
    records = []
    for params in param_grid:
        policy = _policy_from_params(params)
        applied = apply_policy_frame(development, policy)
        metrics = evaluate_frame(applied)
        eligible = metrics["n"] >= min_selected and metrics["coverage"] >= min_coverage
        records.append({**params, **metrics, "grid_eligible": eligible})
    table = pd.DataFrame(records)
    ok = table[table["grid_eligible"]].copy()
    if ok.empty:
        raise ValueError("No parameter setting meets minimum sample/coverage constraints")
    if "mean_realized_return" not in ok:
        ok["mean_realized_return"] = 0.0
    ok = ok.sort_values(
        ["wilson95_low", "mean_realized_return", "coverage"],
        ascending=[False, False, False],
    )
    best_row = ok.iloc[0]
    fields = ParlayPolicy.__dataclass_fields__.keys()
    best_params = {k: best_row[k] for k in fields if k in best_row.index}
    return ParlayPolicy(**best_params), table


def date_blocked_walk_forward(
    df: pd.DataFrame,
    param_grid: Iterable[Mapping[str, Any]],
    min_train_rows: int = 50,
    min_selected: int = 10,
    min_coverage: float = 0.10,
) -> pd.DataFrame:
    """Retune using strictly earlier dates, then score every row on the current date."""
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    out = []
    for date in sorted(x["date"].unique()):
        train = x[x["date"] < date]
        test = x[x["date"] == date]
        if len(train) < min_train_rows:
            continue
        try:
            policy, _ = optimize_policy_grid(train, param_grid, min_selected, min_coverage)
        except ValueError:
            continue
        scored = apply_policy_frame(test, policy)
        for _, r in scored.iterrows():
            d = r.to_dict()
            d["policy"] = asdict(policy)
            d["train_end"] = str(train["date"].max().date())
            out.append(d)
    return pd.DataFrame(out)
