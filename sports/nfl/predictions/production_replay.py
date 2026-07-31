"""Production-style replay and effectiveness gates for the NFL betting board."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from .market_selector import prune_weekly_pool, summarize_market_rows


REQUIRED_POOL_COLUMNS = {
    "season",
    "week",
    "player_id",
    "player_display_name",
    "target",
    "side",
    "line",
    "over_price",
    "under_price",
    "selected_price",
    "estimated_side_probability",
    "no_vig_side_probability",
    "probability_advantage",
    "actual",
    "current_prediction",
    "result",
    "profit_units",
    "selected_architecture",
    "target_final_validation_status",
    "bookmaker",
    "source",
    "snapshot_time_utc",
    "commence_time_utc",
}

SELECTION_FINGERPRINT_COLUMNS = [
    "season",
    "week",
    "player_id",
    "target",
    "bookmaker",
    "side",
    "line",
    "selected_price",
    "estimated_side_probability",
    "no_vig_side_probability",
    "probability_advantage",
    "selected_architecture",
]


def _american_profit(price: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(price, errors="coerce")
    return pd.Series(
        np.where(numeric.gt(0), numeric / 100.0, 100.0 / numeric.abs()),
        index=price.index,
    )


def grade_sides(frame: pd.DataFrame, side: pd.Series | np.ndarray) -> pd.DataFrame:
    """Recompute results and units without trusting stored grading fields."""

    graded = frame.copy()
    side_values = pd.Series(side, index=graded.index).astype(str).str.lower()
    if not side_values.isin(["over", "under"]).all():
        raise ValueError("Every replay side must be over or under.")
    graded["side"] = side_values
    over = graded["side"].eq("over")
    push = graded["actual"].eq(graded["line"])
    win = np.where(
        over,
        graded["actual"].gt(graded["line"]),
        graded["actual"].lt(graded["line"]),
    )
    graded["result"] = np.where(push, "push", np.where(win, "win", "loss"))
    graded["pick_validation"] = graded["result"].map(
        {"win": "pass", "loss": "fail", "push": "push"}
    )
    graded["selected_price"] = np.where(over, graded["over_price"], graded["under_price"])
    win_profit = _american_profit(graded["selected_price"])
    graded["profit_units"] = np.where(
        graded["result"].eq("push"),
        0.0,
        np.where(graded["result"].eq("win"), win_profit, -1.0),
    )
    return graded


def selection_fingerprint(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["season", "week", "target", "player_id", "bookmaker"]
    ).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(
        ordered[SELECTION_FINGERPRINT_COLUMNS], index=False
    ).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def validate_pool_contract(pool: pd.DataFrame, policy_report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    missing = sorted(REQUIRED_POOL_COLUMNS.difference(pool.columns))
    if missing:
        errors.append(f"missing_columns:{','.join(missing)}")
        return {"status": "failed", "errors": errors}
    if pool.empty:
        return {"status": "failed", "errors": ["empty_pool"]}

    design = policy_report["design"]
    expected_season = int(design["final_test_season"])
    seasons = sorted(int(value) for value in pool["season"].dropna().unique())
    if seasons != [expected_season]:
        errors.append(f"unexpected_seasons:{seasons}")
    allowed_targets = set(policy_report["validated_targets"])
    unexpected_targets = sorted(set(pool["target"].astype(str)).difference(allowed_targets))
    if unexpected_targets:
        errors.append(f"unvalidated_targets:{','.join(unexpected_targets)}")
    if not pool["target_final_validation_status"].eq("passed").all():
        errors.append("failed_target_rows_present")
    if not pool["side"].isin(["over", "under"]).all():
        errors.append("invalid_side")
    numeric_checks = {
        "line": pool["line"].gt(0),
        "over_price": pool["over_price"].abs().ge(100),
        "under_price": pool["under_price"].abs().ge(100),
        "estimated_side_probability": pool["estimated_side_probability"].between(0, 1),
        "no_vig_side_probability": pool["no_vig_side_probability"].between(0, 1),
    }
    for name, valid in numeric_checks.items():
        if not valid.fillna(False).all():
            errors.append(f"invalid_{name}")
    if not pool["estimated_side_probability"].ge(
        float(design["minimum_side_probability"])
    ).all():
        errors.append("side_probability_below_policy")
    if not pool["probability_advantage"].ge(
        float(design["minimum_no_vig_advantage"])
    ).all():
        errors.append("probability_advantage_below_policy")
    if not np.allclose(
        pool["probability_advantage"].astype(float),
        pool["estimated_side_probability"].astype(float)
        - pool["no_vig_side_probability"].astype(float),
        atol=1e-9,
    ):
        errors.append("probability_advantage_mismatch")
    expected_price = np.where(
        pool["side"].eq("over"), pool["over_price"], pool["under_price"]
    )
    if not np.allclose(
        pool["selected_price"].astype(float), expected_price, equal_nan=False
    ):
        errors.append("selected_price_side_mismatch")
    expected_architectures = {
        item["target"]: item["selected_architecture"]
        for item in policy_report.get("targets", [])
    }
    architecture_valid = pool.apply(
        lambda row: row["selected_architecture"]
        == expected_architectures.get(str(row["target"])),
        axis=1,
    )
    if not architecture_valid.all():
        errors.append("selected_architecture_mismatch")
    if "evaluation_split" in pool and not pool["evaluation_split"].eq("final_test").all():
        errors.append("unexpected_evaluation_split")
    duplicate_keys = ["season", "week", "player_id", "target", "bookmaker"]
    duplicate_count = int(pool.duplicated(duplicate_keys).sum())
    if duplicate_count:
        errors.append(f"duplicate_props:{duplicate_count}")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "rows": int(len(pool)),
        "season": expected_season,
        "validated_targets": sorted(allowed_targets),
        "duplicate_props": duplicate_count,
    }


def apply_locked_policy(pool: pd.DataFrame, policy_report: dict[str, Any]) -> pd.DataFrame:
    allowed = set(policy_report["validated_targets"])
    eligible = pool.loc[
        pool["target"].isin(allowed)
        & pool["estimated_side_probability"].ge(
            float(policy_report["design"]["minimum_side_probability"])
        )
        & pool["probability_advantage"].ge(
            float(policy_report["design"]["minimum_no_vig_advantage"])
        )
    ].copy()
    return prune_weekly_pool(
        eligible,
        top_n=int(policy_report["weekly_cap_policy"]["selected_top_n"]),
    )


def _one_sided_binomial_p_value(wins: int, decisions: int, null_rate: float = 0.5) -> float:
    if decisions <= 0:
        return math.nan
    probability = 0.0
    for value in range(wins, decisions + 1):
        probability += (
            math.comb(decisions, value)
            * (null_rate**value)
            * ((1.0 - null_rate) ** (decisions - value))
        )
    return float(min(1.0, probability))


def week_cluster_bootstrap(
    graded: pd.DataFrame,
    *,
    samples: int = 10_000,
    random_state: int = 42,
) -> dict[str, Any]:
    grouped = (
        graded.assign(
            win=graded["result"].eq("win").astype(float),
            decision=graded["result"].isin(["win", "loss"]).astype(float),
            bet=1.0,
        )
        .groupby("week")
        .agg(
            wins=("win", "sum"),
            decisions=("decision", "sum"),
            bets=("bet", "sum"),
            profit=("profit_units", "sum"),
        )
    )
    if grouped.empty:
        raise ValueError("Cannot bootstrap an empty replay.")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(grouped), size=(int(samples), len(grouped)))
    decisions = grouped["decisions"].to_numpy()[indices].sum(axis=1)
    hit_rate = grouped["wins"].to_numpy()[indices].sum(axis=1) / decisions
    bets = grouped["bets"].to_numpy()[indices].sum(axis=1)
    roi = grouped["profit"].to_numpy()[indices].sum(axis=1) / bets
    return {
        "method": "resample whole season-weeks with replacement",
        "samples": int(samples),
        "random_state": int(random_state),
        "hit_rate_95": [round(float(value), 4) for value in np.quantile(hit_rate, [0.025, 0.975])],
        "roi_95": [round(float(value), 4) for value in np.quantile(roi, [0.025, 0.975])],
        "probability_hit_rate_above_50_percent": round(float(np.mean(hit_rate > 0.5)), 4),
        "probability_roi_above_zero": round(float(np.mean(roi > 0.0)), 4),
    }


def _period_summary(graded: pd.DataFrame, start: int, end: int) -> dict[str, Any]:
    part = graded.loc[graded["week"].between(start, end)]
    return {"weeks": [start, end], **summarize_market_rows(part)}


def _maximum_weekly_drawdown(graded: pd.DataFrame) -> dict[str, float]:
    weekly = graded.groupby("week", sort=True)["profit_units"].sum()
    cumulative = np.r_[0.0, weekly.cumsum().to_numpy(dtype=float)]
    peak = np.maximum.accumulate(cumulative)
    return {
        "final_profit_units": round(float(cumulative[-1]), 4),
        "maximum_drawdown_units": round(float((cumulative - peak).min()), 4),
    }


def _paired_side_comparison(model: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    model_win = model["result"].eq("win").to_numpy()
    baseline_win = baseline["result"].eq("win").to_numpy()
    model_only = int(np.sum(model_win & ~baseline_win))
    baseline_only = int(np.sum(~model_win & baseline_win))
    discordant = model_only + baseline_only
    return {
        "model_only_wins": model_only,
        "baseline_only_wins": baseline_only,
        "discordant_decisions": discordant,
        "one_sided_exact_p_value": (
            round(_one_sided_binomial_p_value(model_only, discordant), 6)
            if discordant
            else None
        ),
        "status": (
            "statistically_superior"
            if discordant
            and model_only > baseline_only
            and _one_sided_binomial_p_value(model_only, discordant) < 0.05
            else "directionally_better_not_proven"
        ),
    }


def build_weekly_ledger(graded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week, part in graded.groupby("week", sort=True):
        rows.append({"season": int(part["season"].iloc[0]), "week": int(week), **summarize_market_rows(part)})
    return pd.DataFrame(rows)


def run_production_replay(
    pool: pd.DataFrame,
    policy_report: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Replay a frozen board, producing operational and statistical gates."""

    contract = validate_pool_contract(pool, policy_report)
    if contract["status"] != "passed":
        return (
            {"status": "failed_closed", "contract_gate": contract},
            pd.DataFrame(),
            pd.DataFrame(),
        )

    selected = apply_locked_policy(pool, policy_report)
    selected_again = apply_locked_policy(pool.sample(frac=1.0, random_state=9), policy_report)
    deterministic = selection_fingerprint(selected) == selection_fingerprint(selected_again)
    outcome_mutated = pool.copy()
    outcome_mutated["actual"] = outcome_mutated["actual"].iloc[::-1].to_numpy()
    outcome_mutated["result"] = "mutated"
    outcome_independent = selection_fingerprint(selected) == selection_fingerprint(
        apply_locked_policy(outcome_mutated, policy_report)
    )

    stored_result = selected["result"].astype(str).copy()
    stored_profit = selected["profit_units"].astype(float).copy()
    graded = grade_sides(selected, selected["side"])
    grading_matches = bool(
        stored_result.reset_index(drop=True).equals(graded["result"].reset_index(drop=True))
        and np.allclose(stored_profit.to_numpy(), graded["profit_units"].to_numpy())
    )
    summary = summarize_market_rows(graded)
    bootstrap = week_cluster_bootstrap(
        graded, samples=bootstrap_samples, random_state=random_state
    )
    decisions = int(summary["graded_decisions"])
    exact_p = _one_sided_binomial_p_value(int(summary["wins"]), decisions)

    always_under = grade_sides(graded, np.repeat("under", len(graded)))
    projection_side = np.where(
        graded["current_prediction"].gt(graded["line"]), "over", "under"
    )
    point_projection = grade_sides(graded, projection_side)
    baselines = {
        "always_under_same_cohort": summarize_market_rows(always_under),
        "point_projection_side_same_cohort": summarize_market_rows(point_projection),
        "model_vs_always_under_paired": _paired_side_comparison(graded, always_under),
        "model_vs_point_projection_paired": _paired_side_comparison(graded, point_projection),
    }
    halves = [_period_summary(graded, 1, 9), _period_summary(graded, 10, 18)]
    thirds = [
        _period_summary(graded, 1, 6),
        _period_summary(graded, 7, 12),
        _period_summary(graded, 13, 18),
    ]
    sensitivity: list[dict[str, Any]] = []
    for top_n in (8, 10, 12):
        part = grade_sides(prune_weekly_pool(pool, top_n=top_n), prune_weekly_pool(pool, top_n=top_n)["side"])
        sensitivity.append({"top_n": top_n, **summarize_market_rows(part)})

    top_n = int(policy_report["weekly_cap_policy"]["selected_top_n"])
    operational_passed = bool(
        deterministic
        and outcome_independent
        and grading_matches
        and graded.groupby("week").size().max() <= top_n
        and not graded.duplicated(["season", "week", "player_id", "target", "bookmaker"]).any()
    )
    statistical_passed = bool(
        decisions >= 200
        and summary["hit_rate"] >= 0.58
        and summary["hit_rate_wilson_95"][0] > 0.5
        and summary["roi"] > 0.0
        and exact_p < 0.05
        and bootstrap["hit_rate_95"][0] > 0.5
        and bootstrap["roi_95"][0] > 0.0
    )
    stability_passed = bool(
        all(item["hit_rate"] >= 0.55 and item["roi"] > 0 for item in halves)
        and all(item["hit_rate"] >= 0.55 and item["roi"] > 0 for item in thirds)
        and all(item["hit_rate"] >= 0.57 and item["roi"] > 0.05 for item in sensitivity)
    )
    source_timestamps_verified = bool(
        graded["snapshot_time_utc"].notna().all()
        and graded["commence_time_utc"].notna().all()
        and pd.to_datetime(graded["snapshot_time_utc"], utc=True, errors="coerce").lt(
            pd.to_datetime(graded["commence_time_utc"], utc=True, errors="coerce")
        ).all()
    )
    effectiveness_passed = operational_passed and statistical_passed and stability_passed
    deployment_passed = bool(effectiveness_passed and source_timestamps_verified)
    report = {
        "schema_version": 1,
        "status": (
            "production_ready"
            if deployment_passed
            else "effectiveness_proven_source_blocked"
            if effectiveness_passed
            else "effectiveness_not_proven"
        ),
        "locked_policy": {
            "season": int(graded["season"].iloc[0]),
            "targets": policy_report["validated_targets"],
            "minimum_side_probability": policy_report["design"]["minimum_side_probability"],
            "minimum_no_vig_advantage": policy_report["design"]["minimum_no_vig_advantage"],
            "weekly_top_n": top_n,
            "selection_fingerprint_sha256": selection_fingerprint(graded),
        },
        "contract_gate": contract,
        "operational_gate": {
            "status": "passed" if operational_passed else "failed",
            "deterministic_under_input_shuffle": deterministic,
            "selection_independent_of_outcomes": outcome_independent,
            "stored_grading_matches_recomputed_grading": grading_matches,
            "maximum_weekly_picks": int(graded.groupby("week").size().max()),
        },
        "effectiveness": {
            "status": "passed" if statistical_passed else "failed",
            "result": summary,
            "one_sided_exact_p_value_vs_50_percent": round(exact_p, 6),
            "week_cluster_bootstrap": bootstrap,
        },
        "baseline_comparison": baselines,
        "stability_gate": {
            "status": "passed" if stability_passed else "failed",
            "halves": halves,
            "thirds": thirds,
            "weekly_cap_sensitivity": sensitivity,
            "drawdown": _maximum_weekly_drawdown(graded),
        },
        "source_provenance_gate": {
            "status": "passed" if source_timestamps_verified else "failed",
            "all_snapshots_verified_before_kickoff": source_timestamps_verified,
        },
        "deployment_gate": {
            "status": "passed" if deployment_passed else "blocked",
            "reason": (
                "All operational, effectiveness, stability, and source gates passed."
                if deployment_passed
                else "Effectiveness is proven, but source timestamps remain unauthenticated."
                if effectiveness_passed
                else "One or more operational or effectiveness gates failed."
            ),
        },
    }
    return report, graded, build_weekly_ledger(graded)
