"""PAR-F v0.7 empirical atom persistence validation.

This module freezes the seven-season v0.6 evidence and runs a strict
walk-forward challenger test. It does not change PAR weights or v0.6 constants.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .config import MODEL_CONFIG


PARF_V07_VERSION = "parf_v0_7_empirical_atom_persistence"
CHAMPION = "CURRENT_PAR_BASELINE"
V06_STATUS = "rejected_incremental_forecast_value"
ATOM_FIELDS = {
    "scoring_volume_above_replacement": ("SCORING", "scoring_par"),
    "passing_creation": ("CREATION", "creation_par"),
    "negative_turnover_value": ("BALL_SECURITY", "ball_security_par"),
    "steals": ("PERIMETER_DISRUPTION", "perimeter_disruption_par"),
}
MINUTES_BINS = [
    ("500-999", 500, 1000),
    ("1000-1499", 1000, 1500),
    ("1500-1999", 1500, 2000),
    ("2000+", 2000, math.inf),
]
DECAY_CANDIDATES: list[tuple[str, float | None]] = [
    ("current_season_only", None),
    ("2_season_equal_weight", None),
    ("3_season_equal_weight", None),
    ("exponential_decay_season_0_50", 0.50),
    ("exponential_decay_season_0_70", 0.70),
    ("exponential_decay_season_0_85", 0.85),
    ("sample_weighted_exponential_0_70", 0.70),
]
SHRINKAGE_K = [500, 1500]


@dataclass
class LinearFit:
    slope: float
    intercept: float

    def predict(self, value: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
        return self.intercept + self.slope * value


def finite_pair(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    return df[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()


def corr_metrics(df: pd.DataFrame, pred: str, actual: str) -> dict[str, Any]:
    sample = finite_pair(df, pred, actual)
    if len(sample) < 3:
        return {"n": int(len(sample)), "pearson": None, "spearman": None}
    return {
        "n": int(len(sample)),
        "pearson": round(float(pearsonr(sample[pred], sample[actual]).statistic), 6),
        "spearman": round(float(spearmanr(sample[pred], sample[actual]).statistic), 6),
    }


def model_metrics(df: pd.DataFrame, pred: str, actual: str = "next_par") -> dict[str, Any]:
    sample = finite_pair(df, pred, actual)
    if sample.empty:
        return {"n": 0, "pearson": None, "spearman": None, "mae": None, "rmse": None, "tier_accuracy": None}
    err = sample[pred] - sample[actual]
    return {
        **corr_metrics(sample, pred, actual),
        "mae": round(float(err.abs().mean()), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(err)))), 6),
        "tier_accuracy": round(float(tier_accuracy(sample[pred], sample[actual])), 6) if len(sample) >= 9 else None,
    }


def tier_accuracy(pred: pd.Series, actual: pd.Series) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    frame["pred_tier"] = pd.qcut(frame["pred"].rank(method="first"), 3, labels=False)
    frame["actual_tier"] = pd.qcut(frame["actual"].rank(method="first"), 3, labels=False)
    return float((frame["pred_tier"] == frame["actual_tier"]).mean())


def fit_linear(x: pd.Series, y: pd.Series) -> LinearFit:
    sample = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 3 or float(sample["x"].var()) == 0.0:
        return LinearFit(0.0, float(sample["y"].mean()) if len(sample) else 0.0)
    slope, intercept = np.polyfit(sample["x"].astype(float), sample["y"].astype(float), 1)
    return LinearFit(float(slope), float(intercept))


def quantiles(series: pd.Series) -> dict[str, float]:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    qs = s.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "mean": round(float(s.mean()), 6),
        "median": round(float(s.median()), 6),
        "std": round(float(s.std()), 6),
        "p05": round(float(qs.loc[0.05]), 6),
        "p10": round(float(qs.loc[0.10]), 6),
        "p25": round(float(qs.loc[0.25]), 6),
        "p50": round(float(qs.loc[0.50]), 6),
        "p75": round(float(qs.loc[0.75]), 6),
        "p90": round(float(qs.loc[0.90]), 6),
        "p95": round(float(qs.loc[0.95]), 6),
        "minimum": round(float(s.min()), 6),
        "maximum": round(float(s.max()), 6),
    }


def load_rows(validation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = pd.read_csv(validation_dir / "par_seven_season_player_rows.csv")
    eligible = pd.read_csv(validation_dir / "par_seven_season_eligible_rows.csv")
    report = json.loads((validation_dir / "par_seven_season_validation_report.json").read_text(encoding="utf-8"))
    return rows, eligible, report


def add_rates(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["par_rate"] = np.where(rows["minutes"] > 0, rows["par"] / rows["minutes"] * 1000, np.nan)
    for atom, (_, field) in ATOM_FIELDS.items():
        rows[f"{atom}_rate"] = np.where(rows["minutes"] > 0, rows[field] / rows["minutes"] * 1000, np.nan)
    return rows


def build_forward_pairs(rows: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    future_cols = ["player_key", "bref_year", "season", "role", "team", "minutes", "par", *[field for _, field in ATOM_FIELDS.values()]]
    future = rows[future_cols].copy()
    future["bref_year"] = future["bref_year"] - 1
    future = future.rename(
        columns={
            "season": "season_t1",
            "role": "role_t1",
            "team": "team_t1",
            "minutes": "minutes_t1",
            "par": "next_par",
            **{field: f"{field}_t1" for _, field in ATOM_FIELDS.values()},
        }
    )
    pairs = eligible.merge(future, on=["player_key", "bref_year"], how="inner")
    pairs["season_t"] = pairs["season"]
    pairs["same_role"] = pairs["role"] == pairs["role_t1"]
    pairs["same_team"] = pairs["team"] == pairs["team_t1"]
    pairs["current_par_error"] = pairs["par"] - pairs["next_par"]
    pairs["parf_v06_error"] = pairs["projected_parf"] - pairs["next_par"]
    return pairs


def build_atom_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in pairs.iterrows():
        for atom, (category, field) in ATOM_FIELDS.items():
            atom_t = float(row[field])
            atom_t1 = float(row[f"{field}_t1"])
            records.append(
                {
                    "player_id": row["player_key"],
                    "season_t": row["season_t"],
                    "season_t1": row["season_t1"],
                    "atom_type": atom,
                    "category": category,
                    "atom_par_t": atom_t,
                    "atom_par_t1": atom_t1,
                    "atom_rate_t": atom_t / row["minutes"] * 1000 if row["minutes"] else None,
                    "atom_rate_t1": atom_t1 / row["minutes_t1"] * 1000 if row["minutes_t1"] else None,
                    "minutes_t": float(row["minutes"]),
                    "minutes_t1": float(row["minutes_t1"]),
                    "opportunities_t": None,
                    "opportunities_t1": None,
                    "opportunity_status": "atom_specific_denominator_unavailable_in_frozen_bref_rows",
                    "role_t": row["role"],
                    "role_t1": row["role_t1"],
                    "age_t": None,
                    "age_t1": None,
                    "team_t": row["team"],
                    "team_t1": row["team_t1"],
                    "source_tier_t": "TIER_A_DIRECT",
                    "reliability_t": 1.0,
                    "evidence_coverage_t": 1.0,
                }
            )
    return pd.DataFrame(records)


def persistence_stats(atom_pairs: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for atom, group in atom_pairs.groupby("atom_type"):
        raw_fit = fit_linear(group["atom_par_t"], group["atom_par_t1"])
        rate_fit = fit_linear(group["atom_rate_t"], group["atom_rate_t1"])
        err = group["atom_par_t"] - group["atom_par_t1"]
        player_means = group.groupby("player_id")["atom_par_t"].mean()
        within = group.groupby("player_id")["atom_par_t"].var().dropna()
        strata = {}
        for label, lo, hi in MINUTES_BINS:
            sg = group[(group["minutes_t"] >= lo) & (group["minutes_t"] < hi)]
            strata[label] = {
                "n": int(len(sg)),
                "raw_par": corr_metrics(sg, "atom_par_t", "atom_par_t1"),
                "rate": corr_metrics(sg, "atom_rate_t", "atom_rate_t1"),
            }
        report[atom] = {
            "n": int(len(group)),
            "raw_atom_par": {
                **corr_metrics(group, "atom_par_t", "atom_par_t1"),
                "ols_slope": round(raw_fit.slope, 6),
                "ols_intercept": round(raw_fit.intercept, 6),
                "mae": round(float(err.abs().mean()), 6),
                "rmse": round(float(np.sqrt(np.mean(np.square(err)))), 6),
            },
            "atom_par_rate": {
                **corr_metrics(group, "atom_rate_t", "atom_rate_t1"),
                "ols_slope": round(rate_fit.slope, 6),
                "ols_intercept": round(rate_fit.intercept, 6),
            },
            "opportunity_rate": {"status": "blocked", "reason": "atom-specific opportunities unavailable"},
            "year_to_year_variance": round(float((group["atom_par_t1"] - group["atom_par_t"]).var()), 6),
            "within_player_variance": round(float(within.mean()), 6) if len(within) else None,
            "between_player_variance": round(float(player_means.var()), 6),
            "minute_strata": strata,
        }
    return report


def history_feature(
    history: pd.DataFrame,
    player_key: str,
    current_year: int,
    rate_col: str,
    decay_type: str,
    decay_factor: float | None,
    shrink_k: int,
    prior: float,
) -> float:
    hist = history[(history["player_key"] == player_key) & (history["bref_year"] <= current_year)].sort_values("bref_year")
    if hist.empty:
        return prior
    if decay_type == "current_season_only":
        hist = hist.tail(1)
        weights = np.ones(len(hist))
    elif decay_type == "2_season_equal_weight":
        hist = hist.tail(2)
        weights = np.ones(len(hist))
    elif decay_type == "3_season_equal_weight":
        hist = hist.tail(3)
        weights = np.ones(len(hist))
    else:
        ages = current_year - hist["bref_year"].to_numpy()
        factor = float(decay_factor or 0.70)
        weights = factor**ages
        if decay_type.startswith("sample_weighted"):
            weights = weights * hist["minutes"].to_numpy()
    rates = hist[rate_col].to_numpy(dtype=float)
    minutes = hist["minutes"].to_numpy(dtype=float)
    shrunk = []
    for rate, sample in zip(rates, minutes):
        w = sample / (sample + shrink_k) if sample > 0 else 0.0
        shrunk.append(w * rate + (1 - w) * prior)
    return float(np.average(shrunk, weights=weights))


def candidate_features(
    rows: pd.DataFrame,
    pairs: pd.DataFrame,
    atom: str,
    decay_type: str,
    decay_factor: float | None,
    shrink_k: int,
    role_priors: dict[str, float],
    global_prior: float,
) -> pd.Series:
    rate_col = f"{atom}_rate"
    feature_rows = rows[["player_key", "bref_year", "role", "minutes", rate_col]].copy()
    feature_rows["prior"] = feature_rows["role"].map(role_priors).fillna(global_prior)
    sample_weight = feature_rows["minutes"] / (feature_rows["minutes"] + shrink_k)
    feature_rows["shrunk_rate"] = sample_weight * feature_rows[rate_col] + (1 - sample_weight) * feature_rows["prior"]
    values: list[pd.DataFrame] = []
    for _, group in feature_rows.sort_values(["player_key", "bref_year"]).groupby("player_key"):
        group = group.copy()
        if decay_type == "current_season_only":
            group["history_feature"] = group["shrunk_rate"]
        elif decay_type == "2_season_equal_weight":
            group["history_feature"] = group["shrunk_rate"].rolling(2, min_periods=1).mean()
        elif decay_type == "3_season_equal_weight":
            group["history_feature"] = group["shrunk_rate"].rolling(3, min_periods=1).mean()
        else:
            out = []
            years = group["bref_year"].to_numpy(dtype=int)
            rates = group["shrunk_rate"].to_numpy(dtype=float)
            minutes = group["minutes"].to_numpy(dtype=float)
            factor = float(decay_factor or 0.70)
            for idx, year in enumerate(years):
                ages = year - years[: idx + 1]
                weights = factor**ages
                if decay_type.startswith("sample_weighted"):
                    weights = weights * minutes[: idx + 1]
                out.append(float(np.average(rates[: idx + 1], weights=weights)))
            group["history_feature"] = out
        values.append(group[["player_key", "bref_year", "history_feature"]])
    feature_frame = pd.concat(values, ignore_index=True)
    merged = pairs[["player_key", "bref_year"]].merge(feature_frame, on=["player_key", "bref_year"], how="left")
    return merged["history_feature"].fillna(global_prior)


def choose_atom_model(train_pairs: pd.DataFrame, rows: pd.DataFrame, atom: str, field: str) -> dict[str, Any]:
    rate_col = f"{atom}_rate"
    target_rate = f"{field}_t1"
    role_priors = rows[rows["bref_year"].isin(train_pairs["bref_year"])].groupby("role")[rate_col].mean().to_dict()
    global_prior = float(rows[rows["bref_year"].isin(train_pairs["bref_year"])][rate_col].mean())
    candidates = []
    for decay_type, decay_factor in DECAY_CANDIDATES:
        for k in SHRINKAGE_K:
            features = candidate_features(rows, train_pairs, atom, decay_type, decay_factor, k, role_priors, global_prior)
            actual_rate = np.where(train_pairs["minutes_t1"] > 0, train_pairs[target_rate] / train_pairs["minutes_t1"] * 1000, np.nan)
            fit = fit_linear(features, pd.Series(actual_rate))
            pred = fit.predict(features)
            mae = float((pd.Series(pred) - pd.Series(actual_rate)).abs().mean())
            candidates.append((mae, decay_type, decay_factor, k, fit, global_prior, role_priors))
    candidates.sort(key=lambda item: item[0])
    mae, decay_type, decay_factor, k, fit, global_prior, role_priors = candidates[0]
    return {
        "atom_type": atom,
        "decay_type": decay_type,
        "decay_factor": decay_factor,
        "shrinkage_k_minutes": k,
        "training_rate_mae": round(mae, 6),
        "rate_fit": fit,
        "global_prior": global_prior,
        "role_priors": role_priors,
    }


def walk_forward(rows: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    predictions = []
    decay_report: dict[str, Any] = {}
    stabilization_report: dict[str, Any] = {}
    for predict_year in sorted(pairs["bref_year"].unique()):
        train = pairs[pairs["bref_year"] < predict_year]
        test = pairs[pairs["bref_year"] == predict_year]
        if train.empty or test.empty:
            continue
        minute_fit = fit_linear(train["minutes"], train["minutes_t1"])
        atom_models = {}
        for atom, (_, field) in ATOM_FIELDS.items():
            model = choose_atom_model(train, rows, atom, field)
            atom_models[atom] = model
            decay_report.setdefault(atom, []).append(
                {
                    "fold_predict_season": str(test["season_t1"].iloc[0]),
                    "decay_type": model["decay_type"],
                    "decay_factor": model["decay_factor"],
                    "shrinkage_k_minutes": model["shrinkage_k_minutes"],
                    "training_rate_mae": model["training_rate_mae"],
                }
            )
        for _, row in test.iterrows():
            projected_minutes = max(0.0, float(minute_fit.predict(float(row["minutes"]))))
            atom_forecasts = {}
            for atom, (category, field) in ATOM_FIELDS.items():
                model = atom_models[atom]
                prior = float(model["role_priors"].get(row["role"], model["global_prior"]))
                feature = history_feature(
                    rows,
                    row["player_key"],
                    int(row["bref_year"]),
                    f"{atom}_rate",
                    model["decay_type"],
                    model["decay_factor"],
                    int(model["shrinkage_k_minutes"]),
                    prior,
                )
                projected_rate = float(model["rate_fit"].predict(feature))
                atom_forecasts[atom] = {
                    "category": category,
                    "projected_rate": projected_rate,
                    "projected_par": projected_rate * projected_minutes / 1000,
                    "skill_persistence_effect": projected_rate - float(row[f"{atom}_rate"]),
                    "opportunity_effect": projected_minutes - float(row["minutes"]),
                    "role_effect": 0.0,
                    "minutes_effect": projected_minutes - float(row["minutes"]),
                }
            projected_par = sum(item["projected_par"] for item in atom_forecasts.values())
            predictions.append(
                {
                    "player_id": row["player_key"],
                    "player": row["player"],
                    "season_from": row["season_t"],
                    "season_to": row["season_t1"],
                    "role": row["role"],
                    "current_par": row["par"],
                    "next_par": row["next_par"],
                    "parf_v06": row["projected_parf"],
                    "parf_v07": projected_par,
                    "projected_minutes": projected_minutes,
                    "projected_atom_par": {k: round(v["projected_par"], 6) for k, v in atom_forecasts.items()},
                    "projected_category_par": category_projection(atom_forecasts),
                    "skill_persistence_effect": round(sum(v["skill_persistence_effect"] for v in atom_forecasts.values()), 6),
                    "opportunity_effect": round(sum(v["opportunity_effect"] for v in atom_forecasts.values()), 6),
                    "role_effect": 0.0,
                    "minutes_effect": round(projected_minutes - float(row["minutes"]), 6),
                    "age_effect": None,
                    "confidence_interval_low": None,
                    "confidence_interval_high": None,
                    "continuation_score": None,
                    "role_portability_score": None,
                    "model_version": PARF_V07_VERSION,
                    "reconciliation_delta": 0.0,
                }
            )
    pred_df = pd.DataFrame(predictions)
    for atom in ATOM_FIELDS:
        folds = decay_report.get(atom, [])
        stabilization_report[atom] = {
            "selected_shrinkage_k_by_fold": [f["shrinkage_k_minutes"] for f in folds],
            "median_selected_k": float(np.median([f["shrinkage_k_minutes"] for f in folds])) if folds else None,
            "opportunity_stabilization": "blocked_atom_specific_opportunities_unavailable",
        }
    return pred_df, decay_report, stabilization_report


def category_projection(atom_forecasts: dict[str, dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for forecast in atom_forecasts.values():
        totals[forecast["category"]] = totals.get(forecast["category"], 0.0) + forecast["projected_par"]
    return {k: round(v, 6) for k, v in totals.items()}


def bootstrap_delta(df: pd.DataFrame, challenger: str, n: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(17)
    mae_delta = []
    spearman_delta = []
    base = df[["par", challenger, "next_par"]].dropna().reset_index(drop=True)
    for _ in range(n):
        sample = base.iloc[rng.integers(0, len(base), len(base))]
        base_mae = float((sample["par"] - sample["next_par"]).abs().mean())
        ch_mae = float((sample[challenger] - sample["next_par"]).abs().mean())
        mae_delta.append(base_mae - ch_mae)
        base_s = float(spearmanr(sample["par"], sample["next_par"]).statistic)
        ch_s = float(spearmanr(sample[challenger], sample["next_par"]).statistic)
        spearman_delta.append(ch_s - base_s)
    return {
        "challenger": challenger,
        "mae_improvement_ci": [round(float(np.quantile(mae_delta, q)), 6) for q in [0.025, 0.5, 0.975]],
        "spearman_delta_ci": [round(float(np.quantile(spearman_delta, q)), 6) for q in [0.025, 0.5, 0.975]],
        "statistically_distinguishable_from_zero": bool(np.quantile(spearman_delta, 0.025) > 0 and np.quantile(mae_delta, 0.025) > 0),
    }


def champion_challenger(pred: pd.DataFrame) -> dict[str, Any]:
    models = {
        CHAMPION: "par",
        "PARF_V0_6": "parf_v06",
        "PARF_V0_7": "parf_v07",
    }
    metrics = {name: model_metrics(pred, col) for name, col in models.items()}
    champion = metrics[CHAMPION]
    for name, vals in metrics.items():
        if name == CHAMPION:
            vals["delta_vs_champion"] = "baseline"
        else:
            vals["delta_vs_champion"] = {
                "pearson": None if vals["pearson"] is None else round(vals["pearson"] - champion["pearson"], 6),
                "spearman": None if vals["spearman"] is None else round(vals["spearman"] - champion["spearman"], 6),
                "mae": None if vals["mae"] is None else round(champion["mae"] - vals["mae"], 6),
                "rmse": None if vals["rmse"] is None else round(champion["rmse"] - vals["rmse"], 6),
                "tier_accuracy": None if vals["tier_accuracy"] is None else round(vals["tier_accuracy"] - champion["tier_accuracy"], 6),
            }
    v07_bootstrap = bootstrap_delta(pred, "parf_v07")
    point_gate_passed = bool(metrics["PARF_V0_7"]["spearman"] > champion["spearman"] and metrics["PARF_V0_7"]["mae"] < champion["mae"])
    product_promoted = bool(point_gate_passed and v07_bootstrap["statistically_distinguishable_from_zero"])
    return {
        "models": metrics,
        "paired_bootstrap": {
            "PARF_V0_6": bootstrap_delta(pred, "parf_v06"),
            "PARF_V0_7": v07_bootstrap,
        },
        "acceptance_rule": "PAR-F v0.7 must beat current PAR on Spearman and MAE out of sample.",
        "parf_v0_7_point_gate_passed": point_gate_passed,
        "parf_v0_7_product_promoted": product_promoted,
    }


def role_validation(pred: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for role, group in pred.groupby("role"):
        output[role] = {
            CHAMPION: model_metrics(group, "par"),
            "PARF_V0_6": model_metrics(group, "parf_v06"),
            "PARF_V0_7": model_metrics(group, "parf_v07"),
        }
    return output


def composition_bucket(row: pd.Series) -> str:
    values = {
        "scoring": abs(float(row["scoring_par"])),
        "creation": abs(float(row["creation_par"])),
        "ball_security": abs(float(row["ball_security_par"])),
        "perimeter": abs(float(row["perimeter_disruption_par"])),
    }
    total = sum(values.values())
    if total == 0:
        return "balanced"
    top, top_value = max(values.items(), key=lambda item: item[1])
    if top_value / total >= 0.50:
        if top == "perimeter":
            return "defense_dominant"
        return f"{top}_dominant"
    return "balanced"


def composition_validation(pred: pd.DataFrame) -> dict[str, Any]:
    frame = pred.copy()
    frame["composition_bucket"] = frame.apply(composition_bucket, axis=1)
    buckets = {}
    for bucket, group in frame.groupby("composition_bucket"):
        buckets[bucket] = {
            CHAMPION: model_metrics(group, "par"),
            "PARF_V0_6": model_metrics(group, "parf_v06"),
            "PARF_V0_7": model_metrics(group, "parf_v07"),
        }
    for missing in ["rebounding_dominant", "proxy_heavy", "residual_heavy"]:
        buckets.setdefault(missing, {"status": "not_observed_in_direct_atom_validation"})
    return buckets


def matched_composition_test(pred: pd.DataFrame, tolerance: float = 50.0, min_distance: float = 0.55) -> dict[str, Any]:
    wins = {"current": 0, "v07": 0, "ties": 0, "n": 0}
    atom_cols = [field for _, field in ATOM_FIELDS.values()]
    for _, group in pred.groupby("season_from"):
        group = group.reset_index(drop=True)
        comp = group[atom_cols].abs().div(group[atom_cols].abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if abs(group.loc[i, "par"] - group.loc[j, "par"]) > tolerance:
                    continue
                dist = float(np.abs(comp.iloc[i] - comp.iloc[j]).sum() / 2)
                if dist < min_distance:
                    continue
                actual = np.sign(group.loc[i, "next_par"] - group.loc[j, "next_par"])
                if actual == 0:
                    continue
                current_rank = np.sign(group.loc[i, "par"] - group.loc[j, "par"])
                v07_rank = np.sign(group.loc[i, "parf_v07"] - group.loc[j, "parf_v07"])
                wins["n"] += 1
                cur_hit = current_rank == actual
                v07_hit = v07_rank == actual
                if cur_hit and not v07_hit:
                    wins["current"] += 1
                elif v07_hit and not cur_hit:
                    wins["v07"] += 1
                else:
                    wins["ties"] += 1
    wins["current_only_rate"] = round(wins["current"] / wins["n"], 6) if wins["n"] else None
    wins["v07_only_rate"] = round(wins["v07"] / wins["n"], 6) if wins["n"] else None
    return wins


def residual_validation(pred: pd.DataFrame) -> dict[str, Any]:
    preds = []
    features = ["box_score_index", "BPM", "VORP", "WS"]
    for year in sorted(pred["bref_year"].unique()):
        train = pred[pred["bref_year"] < year].dropna(subset=features + ["par"])
        test = pred[pred["bref_year"] == year].dropna(subset=features + ["par"])
        if len(train) < 20 or test.empty:
            continue
        x = np.column_stack([np.ones(len(train)), train[features].to_numpy(dtype=float)])
        beta = np.linalg.lstsq(x, train["par"].to_numpy(dtype=float), rcond=None)[0]
        xt = np.column_stack([np.ones(len(test)), test[features].to_numpy(dtype=float)])
        out = test.copy()
        out["expected_par_from_box_metrics"] = xt @ beta
        out["par_residual"] = out["par"] - out["expected_par_from_box_metrics"]
        preds.append(out)
    frame = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if frame.empty:
        return {"status": "blocked", "reason": "no temporal residual folds"}
    category_corr = {
        field: corr_metrics(frame.assign(abs_field=frame[field].abs()), "par_residual", field)
        for _, field in ATOM_FIELDS.values()
    }
    frame["role_continuity"] = frame["role"] == frame["role_t1"]
    return {
        "status": "complete",
        "sample_size": int(len(frame)),
        "residual_to_next_par": corr_metrics(frame, "par_residual", "next_par"),
        "residual_to_next_minutes": corr_metrics(frame, "par_residual", "minutes_t1"),
        "residual_role_continuity_mean_by_sign": {
            "positive_residual": round(float(frame[frame["par_residual"] >= 0]["role_continuity"].mean()), 6),
            "negative_residual": round(float(frame[frame["par_residual"] < 0]["role_continuity"].mean()), 6),
        },
        "category_explanation": category_corr,
        "interpretation": "Residual framework complete; current direct atoms show limited incremental-value evidence beyond box baselines.",
    }


def target_scale_audit(pairs: pd.DataFrame) -> dict[str, Any]:
    pairs = pairs.copy()
    pairs["current_par_baseline_error"] = pairs["par"] - pairs["next_par"]
    gate_error = {
        "status": "validation_contract_error",
        "historical_gate": "MAE < 35 PAR",
        "finding": "Gate is scale-incompatible with season-total PAR on the seven-season 500 MP population.",
        "observed_current_par_mae": round(float(pairs["current_par_baseline_error"].abs().mean()), 6),
        "versioned_corrected_gate": {
            "name": "PARF_TOTAL_SEASON_500MP_GATE_V1",
            "primary": "outperform CURRENT_PAR_BASELINE on Spearman and MAE",
            "secondary_reference": "MAE should be interpreted on season-total PAR scale, not possession/rate scale.",
        },
    }
    return {
        "next_season_par": quantiles(pairs["next_par"]),
        "current_par": quantiles(pairs["par"]),
        "parf_v06_error": quantiles(pairs["parf_v06_error"]),
        "current_par_baseline_error": quantiles(pairs["current_par_baseline_error"]),
        "current_par_to_next_par_mae": round(float(pairs["current_par_baseline_error"].abs().mean()), 6),
        "current_par_to_next_par_rmse": round(float(np.sqrt(np.mean(np.square(pairs["current_par_baseline_error"])))), 6),
        "gate_audit": gate_error,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, title: str, report: dict[str, Any]) -> None:
    lines = [f"# {title}", "", f"- PAR model: `{MODEL_CONFIG.par_model_version}`", f"- PAR-F v0.7 model: `{PARF_V07_VERSION}`"]
    if "models" in report:
        lines.append("")
        lines.append("| model | pearson | spearman | MAE | RMSE | tier accuracy |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model, metrics in report["models"].items():
            lines.append(
                f"| {model} | {metrics.get('pearson')} | {metrics.get('spearman')} | {metrics.get('mae')} | {metrics.get('rmse')} | {metrics.get('tier_accuracy')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(validation_dir: Path, out: Path) -> dict[str, Any]:
    rows, eligible, seven_report = load_rows(validation_dir)
    rows = add_rates(rows)
    eligible = add_rates(eligible)
    pairs = build_forward_pairs(rows, eligible)
    atom_pairs = build_atom_pairs(pairs)
    persistence = persistence_stats(atom_pairs)
    pred, decay, stabilization = walk_forward(rows, pairs)
    pred = pred.merge(
        pairs[
            [
                "player_key",
                "season_t",
                "bref_year",
                "role_t1",
                "minutes_t1",
                "box_score_index",
                "BPM",
                "VORP",
                "WS",
                *[field for _, field in ATOM_FIELDS.values()],
            ]
        ],
        left_on=["player_id", "season_from"],
        right_on=["player_key", "season_t"],
        how="left",
    )
    pred["par"] = pred["current_par"]
    pred["parf_v06"] = pred["parf_v06"]
    pred["parf_v07_error"] = pred["parf_v07"] - pred["next_par"]
    target_audit = target_scale_audit(pairs)
    champion = champion_challenger(pred)
    role = role_validation(pred)
    composition = composition_validation(pred)
    matched = matched_composition_test(pred)
    residual = residual_validation(pred)
    source_manifest = {
        "frozen_milestones": [
            "PAR_SEVEN_SEASON_BOX_DOMINANCE_CONFIRMED",
            "PARF_V0_6_INCREMENTAL_FORECAST_VALUE_REJECTED",
        ],
        "seasons": seven_report["seasons"],
        "player_seasons": seven_report["player_seasons"],
        "eligible_player_seasons": seven_report["eligible_player_seasons"],
        "forward_validation_pairs": int(len(pairs)),
        "par_model_version": MODEL_CONFIG.par_model_version,
        "parf_v06_model_version": MODEL_CONFIG.parf_model_version,
        "parf_v07_model_version": PARF_V07_VERSION,
        "source": seven_report["source"],
        "source_urls": seven_report["source_urls"],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "par_seven_season_frozen_manifest.json", source_manifest)
    write_json(out / "par_validation_milestones.json", source_manifest)
    write_json(out / "parf_v0_7_atom_persistence_report.json", persistence)
    write_json(out / "parf_v0_7_stabilization_report.json", stabilization)
    write_json(out / "parf_v0_7_decay_report.json", decay)
    write_json(out / "parf_v0_7_walk_forward_report.json", {"target_scale_audit": target_audit, "predictions": pred.to_dict(orient="records")})
    write_json(out / "parf_v0_7_role_validation.json", role)
    write_json(out / "parf_v0_7_composition_validation.json", {**composition, "matched_composition_test": matched})
    write_json(out / "parf_v0_7_champion_challenger_report.json", champion)
    write_json(out / "par_incremental_residual_validation.json", residual)
    atom_pairs.to_csv(out / "parf_v0_7_longitudinal_atom_pairs.csv", index=False)
    pred.to_csv(out / "parf_v0_7_walk_forward_predictions.csv", index=False)
    write_markdown(out / "parf_v0_7_champion_challenger_summary.md", "PAR-F v0.7 Champion/Challenger", champion)
    compact = {
        "milestone": "PARF_V0_7_EMPIRICAL_ATOM_PERSISTENCE_VALIDATION",
        "source_manifest": source_manifest,
        "target_scale_audit": target_audit,
        "champion_challenger": champion,
        "atom_persistence_summary": persistence,
        "atom_stabilization_summary": stabilization,
        "selected_decay_behavior_by_atom": decay,
        "role_stratified_results": role,
        "composition_stratified_results": composition,
        "matched_composition_test": matched,
        "par_residual_incremental_validity": residual,
        "product_status": {
            "par_page": "analytical_preview_available_when_accounting_proof_passes",
            "parf_v0_6_validation_status": V06_STATUS,
            "forecast_display": "Next-Season PAR Baseline unless an accepted challenger beats CURRENT_PAR_BASELINE",
        },
    }
    write_json(out / "parf_v0_7_validation_summary.json", compact)
    return compact


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run PAR-F v0.7 empirical atom persistence validation.")
    parser.add_argument("--validation-dir", default="out/par_validation_7y")
    parser.add_argument("--out", default="out/parf_v0_7_validation")
    args = parser.parse_args(argv)
    result = run(Path(args.validation_dir), Path(args.out))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
