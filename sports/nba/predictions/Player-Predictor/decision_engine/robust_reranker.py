from __future__ import annotations

<<<<<<< HEAD:Player-Predictor/decision_engine/robust_reranker.py
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRanker

from .uncertainty import belief_confidence_factor, normalize_belief_uncertainty


TARGETS = ("PTS", "TRB", "AST")
TARGET_THRESHOLDS = {
    "PTS": {"consider_pct": 0.80, "strong_pct": 0.90, "elite_pct": 0.95},
    "TRB": {"consider_pct": 0.90, "strong_pct": 0.95, "elite_pct": 0.975},
    "AST": {"consider_pct": 0.90, "strong_pct": 0.95, "elite_pct": 0.975},
}
RECOMMENDATION_VALUES = {"pass": 0.0, "consider": 0.35, "strong": 0.7, "elite": 1.0}
MIN_RECOMMENDATION_VALUES = {"pass": 0.0, "consider": 0.35, "strong": 0.7, "elite": 1.0}
BASE_FEATURE_COLUMNS = [
    "expected_win_rate",
    "raw_expected_win_rate",
    "gap_percentile",
    "abs_edge",
    "edge_over_line",
    "edge_vs_baseline",
    "final_confidence_proxy",
    "belief_uncertainty",
    "feasibility",
    "quality_score",
    "fallback_blend",
    "market_books",
    "history_rows",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "rec_num",
    "is_under",
    "is_pts",
    "is_trb",
    "is_ast",
] + [f"{column}_rank_pct" for column in BASE_FEATURE_COLUMNS] + [f"{column}_z_within" for column in BASE_FEATURE_COLUMNS] + [
    "slate_rows",
    "target_rows",
    "target_share",
    "direction_rows",
    "direction_share",
]


@dataclass
class RobustRerankerArtifacts:
    rank_model: XGBRanker
    point_model: HistGradientBoostingClassifier
    point_calibrator: IsotonicRegression | None
    blend_calibrator: IsotonicRegression | None
    feature_columns: list[str]
    train_rows: int
    train_dates: int
    holdout_rows: int
    holdout_dates: int
    positive_rate: float
    blend_weight: float
    cutoff_date: str


def _safe_numeric(values, default=0.0):
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").fillna(default)
    numeric = pd.to_numeric(pd.Series([values]), errors="coerce").fillna(default).iloc[0]
    return float(numeric)


def _percentile_of_gap(gaps_sorted: np.ndarray, gap: float) -> float:
    if gaps_sorted.size == 0:
        return 0.0
    rank = np.searchsorted(gaps_sorted, float(gap), side="right")
    return float(rank / gaps_sorted.size)


def _classify_play(target: str, percentile: float) -> str:
    thresholds = TARGET_THRESHOLDS[target]
    if percentile >= thresholds["elite_pct"]:
        return "elite"
    if percentile >= thresholds["strong_pct"]:
        return "strong"
    if percentile >= thresholds["consider_pct"]:
        return "consider"
    return "pass"


def _tail_rate_curve(prior_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    if prior_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float), 0.5
    percentile_points = np.array([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99], dtype=float)
    percentile_thresholds = np.quantile(prior_df["abs_edge"].to_numpy(dtype=float), percentile_points)
    min_tail_rows = max(25, int(len(prior_df) * 0.01))
    all_rate = float(prior_df["label"].mean())
    running_rate = all_rate
    percentile_rates: list[float] = []
    for threshold in percentile_thresholds:
        subset = prior_df.loc[prior_df["abs_edge"] >= float(threshold)]
        if len(subset) >= min_tail_rows:
            running_rate = max(running_rate, float(subset["label"].mean()))
        percentile_rates.append(float(running_rate))
    return percentile_points, np.asarray(percentile_rates, dtype=float), all_rate


def _expected_rate_for(percentile: float, prior_df: pd.DataFrame) -> float:
    percentile_points, percentile_rates, all_rate = _tail_rate_curve(prior_df)
    if percentile_points.size and percentile_rates.size:
        pct = float(np.clip(percentile, 0.0, 1.0))
        if pct <= float(percentile_points[0]):
            lower_anchor = float(percentile_points[0])
            blend = 0.0 if lower_anchor <= 0.0 else pct / lower_anchor
            return float(all_rate + blend * (float(percentile_rates[0]) - all_rate))
        return float(np.interp(pct, percentile_points, percentile_rates))
    return all_rate


def _shrink_expected_win_rate(raw_rate: pd.Series, shrink_factor: float) -> pd.Series:
    raw = pd.to_numeric(raw_rate, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    shrink = float(np.clip(shrink_factor, 0.0, 1.0))
    return (0.5 + shrink * (raw - 0.5)).clip(lower=0.0, upper=1.0)


def _add_group_rank_features(df: pd.DataFrame, group_key: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["_group_key"] = group_key
    parts: list[pd.DataFrame] = []
    for _, group in out.groupby("_group_key", sort=False):
        group_out = group.copy()
        slate_size = float(len(group_out))
        group_out["slate_rows"] = slate_size
        for column in BASE_FEATURE_COLUMNS:
            values = _safe_numeric(group_out[column], default=0.0)
            std = float(values.std(ddof=0))
            group_out[column] = values
            group_out[f"{column}_rank_pct"] = values.rank(pct=True, method="average")
            group_out[f"{column}_z_within"] = (values - float(values.mean())) / (std if std > 0 else 1.0)
        target_rows = group_out.groupby("target")["target"].transform("size").astype(float)
        direction_rows = group_out.groupby(["target", "direction"])["direction"].transform("size").astype(float)
        group_out["target_rows"] = target_rows
        group_out["target_share"] = target_rows / max(slate_size, 1.0)
        group_out["direction_rows"] = direction_rows
        group_out["direction_share"] = direction_rows / target_rows.clip(lower=1.0)
        parts.append(group_out)
    combined = pd.concat(parts, ignore_index=True) if parts else out
    return combined.drop(columns=["_group_key"], errors="ignore")


def add_reranker_features(selector_df: pd.DataFrame) -> pd.DataFrame:
    out = selector_df.copy()
    out["recommendation"] = out.get("recommendation", "pass").astype(str)
    out["direction"] = out.get("direction", "PUSH").astype(str)
    out["target"] = out.get("target", "").astype(str)

    for column in [
        "expected_win_rate",
        "raw_expected_win_rate",
        "gap_percentile",
        "abs_edge",
        "belief_uncertainty",
        "feasibility",
        "fallback_blend",
        "market_books",
        "history_rows",
        "market_line",
        "baseline_edge",
    ]:
        if column in out.columns:
            out[column] = _safe_numeric(out[column], default=0.0)
        else:
            out[column] = 0.0

    belief_raw = _safe_numeric(out.get("belief_uncertainty"), default=1.0)
    out["belief_uncertainty_raw"] = belief_raw
    out["belief_uncertainty"] = normalize_belief_uncertainty(belief_raw, default=1.0)
    out["belief_confidence_factor"] = belief_confidence_factor(belief_raw, default=1.0)
    out["quality_score"] = out["belief_confidence_factor"] * np.clip(out["feasibility"], 0.0, 1.0)
    out["edge_over_line"] = out["abs_edge"] / out["market_line"].abs().clip(lower=1.0)
    out["edge_vs_baseline"] = out["abs_edge"] - _safe_numeric(out.get("baseline_edge"), default=0.0).abs()
    out["final_confidence_proxy"] = out["gap_percentile"] * out["belief_confidence_factor"] * np.clip(out["feasibility"], 0.0, None)
    out["rec_num"] = out["recommendation"].map(RECOMMENDATION_VALUES).fillna(0.0)
    out["is_under"] = (out["direction"] == "UNDER").astype(float)
    out["is_pts"] = (out["target"] == "PTS").astype(float)
    out["is_trb"] = (out["target"] == "TRB").astype(float)
    out["is_ast"] = (out["target"] == "AST").astype(float)

    group_key = pd.to_datetime(out.get("market_date"), errors="coerce").dt.normalize()
    return _add_group_rank_features(out, group_key=group_key)


def build_historical_reranker_table(
    history_df: pd.DataFrame,
    probability_shrink_factor: float,
    elite_pct: float,
    min_prior_rows_per_target: int = 120,
) -> pd.DataFrame:
    rows: list[dict] = []
    working = history_df.copy()
    working["date"] = pd.to_datetime(working.get("date"), errors="coerce").dt.normalize()
    working = working.loc[working["date"].notna()].copy()

    for target in TARGETS:
        market_col = f"market_{target}"
        pred_col = f"pred_{target}"
        correct_col = f"directional_correct_{target}"
        edge_col = f"pred_minus_market_{target}"
        baseline_edge_col = f"baseline_minus_market_{target}"
        market_books_col = f"market_books_{target}"

        target_df = pd.DataFrame(
            {
                "market_date": working["date"],
                "player": working.get("player"),
                "target": target,
                "prediction": _safe_numeric(working.get(pred_col)),
                "market_line": _safe_numeric(working.get(market_col), default=np.nan),
                "market_books": _safe_numeric(working.get(market_books_col), default=0.0),
                "edge": _safe_numeric(working.get(edge_col)),
                "baseline_edge": _safe_numeric(working.get(baseline_edge_col)),
                "belief_uncertainty": _safe_numeric(working.get("belief_uncertainty"), default=1.0),
                "feasibility": _safe_numeric(working.get("feasibility"), default=0.0),
                "fallback_blend": _safe_numeric(working.get("fallback_blend"), default=0.0),
                "history_rows": _safe_numeric(working.get("history_rows"), default=0.0),
                "label": _safe_numeric(working.get(correct_col), default=np.nan),
            }
        )
        target_df = target_df.loc[target_df["market_line"].notna() & target_df["label"].notna()].copy()
        target_df["direction"] = np.where(target_df["edge"] > 0.0, "OVER", np.where(target_df["edge"] < 0.0, "UNDER", "PUSH"))
        target_df = target_df.loc[target_df["direction"] != "PUSH"].copy()
        target_df["abs_edge"] = target_df["edge"].abs()
        target_df = target_df.sort_values(["market_date", "player"]).reset_index(drop=True)

        prior_rows: list[dict] = []
        for _, row in target_df.iterrows():
            if len(prior_rows) >= int(min_prior_rows_per_target):
                prior_df = pd.DataFrame.from_records(prior_rows)
                gap_percentile = _percentile_of_gap(np.sort(prior_df["abs_edge"].to_numpy(dtype=float)), float(row["abs_edge"]))
                raw_expected_win_rate = _expected_rate_for(gap_percentile, prior_df)
                expected_win_rate = float(_shrink_expected_win_rate(pd.Series([raw_expected_win_rate]), probability_shrink_factor).iloc[0])
                recommendation = _classify_play(target, gap_percentile)
                if gap_percentile >= float(elite_pct):
                    recommendation = "elite"
                rows.append(
                    {
                        "market_date": row["market_date"],
                        "player": row["player"],
                        "target": target,
                        "direction": row["direction"],
                        "prediction": float(row["prediction"]),
                        "market_line": float(row["market_line"]),
                        "market_books": float(max(0.0, row["market_books"])),
                        "abs_edge": float(row["abs_edge"]),
                        "baseline_edge": float(row["baseline_edge"]),
                        "gap_percentile": float(gap_percentile),
                        "raw_expected_win_rate": float(raw_expected_win_rate),
                        "expected_win_rate": float(expected_win_rate),
                        "belief_uncertainty": float(np.clip(row["belief_uncertainty"], 0.0, 1.5)),
                        "feasibility": float(np.clip(row["feasibility"], 0.0, 1.0)),
                        "fallback_blend": float(np.clip(row["fallback_blend"], 0.0, 1.0)),
                        "history_rows": float(max(0.0, row["history_rows"])),
                        "recommendation": recommendation,
                        "label": int(row["label"]),
                    }
                )
            prior_rows.append({"abs_edge": float(row["abs_edge"]), "label": int(row["label"])})

    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    return add_reranker_features(out)


def _filter_candidate_rows(
    table: pd.DataFrame,
    min_candidate_expected_win_rate: float,
    min_candidate_final_confidence: float,
    min_candidate_recommendation: str,
) -> pd.DataFrame:
    min_rec_value = float(MIN_RECOMMENDATION_VALUES[min_candidate_recommendation])
    out = table.copy()
    out["expected_win_rate"] = _safe_numeric(out.get("expected_win_rate"), default=0.5)
    out["final_confidence_proxy"] = _safe_numeric(out.get("final_confidence_proxy"), default=0.0)
    out["rec_num"] = _safe_numeric(out.get("rec_num"), default=0.0)
    out = out.loc[out["expected_win_rate"] >= float(min_candidate_expected_win_rate)].copy()
    out = out.loc[out["final_confidence_proxy"] >= float(min_candidate_final_confidence)].copy()
    out = out.loc[out["rec_num"] >= min_rec_value].copy()
    return out


def _fit_rank_model(train_df: pd.DataFrame, num_pair_per_sample: int) -> XGBRanker:
    ordered = train_df.sort_values(["market_date", "player", "target"]).reset_index(drop=True)
    qid = pd.factorize(ordered["market_date"], sort=True)[0]
    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@8",
        tree_method="hist",
        learning_rate=0.05,
        n_estimators=300,
        max_depth=4,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=max(1, int(num_pair_per_sample)),
        random_state=42,
    )
    model.fit(ordered[FEATURE_COLUMNS], ordered["label"].astype(float), qid=qid)
    return model


def _fit_point_model(train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_depth=3,
        min_samples_leaf=50,
        l2_regularization=0.2,
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"].astype(int))
    return model


def _score_rank_within_group(df: pd.DataFrame, raw_scores: np.ndarray) -> pd.Series:
    scored = df[["market_date"]].copy()
    scored["_raw_rank_score"] = raw_scores
    parts: list[pd.Series] = []
    for _, group in scored.groupby("market_date", sort=False):
        values = pd.Series(group["_raw_rank_score"].to_numpy(dtype=float), index=group.index)
        parts.append(values.rank(pct=True, method="average"))
    if not parts:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return pd.concat(parts).sort_index().astype(float)


def _fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression | None:
    if len(scores) < 50:
        return None
    if np.unique(labels).size < 2:
        return None
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(scores, labels)
    return calibrator


def _apply_calibrator(scores: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    if calibrator is None:
        return np.clip(scores, 0.0, 1.0)
    return np.clip(calibrator.predict(scores), 0.0, 1.0)


def _precision_at_k(subset: pd.DataFrame, score_column: str, k: int) -> float:
    if subset.empty:
        return 0.0
    selected = subset.sort_values([score_column, "expected_win_rate", "abs_edge"], ascending=[False, False, False]).head(k)
    return float(selected["label"].mean()) if len(selected) else 0.0


def _weighted_topk_precision(valid_df: pd.DataFrame, score_column: str) -> float:
    if valid_df.empty:
        return 0.0
    weights = [(1, 0.50), (3, 0.30), (5, 0.20)]
    totals: list[float] = []
    for _, group in valid_df.groupby("market_date", sort=False):
        if group.empty:
            continue
        score = 0.0
        for k, weight in weights:
            score += weight * _precision_at_k(group, score_column=score_column, k=min(k, len(group)))
        totals.append(score)
    return float(np.mean(totals)) if totals else 0.0


def fit_robust_reranker_for_cutoff(
    history_table: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    min_train_rows: int = 4000,
    holdout_days: int = 45,
    min_holdout_rows: int = 250,
    num_pair_per_sample: int = 12,
    min_candidate_expected_win_rate: float = 0.55,
    min_candidate_final_confidence: float = 0.03,
    min_candidate_recommendation: str = "consider",
) -> RobustRerankerArtifacts | None:
    if history_table.empty:
        return None
    cutoff_ts = pd.Timestamp(cutoff_date).normalize()
    eligible = history_table.loc[history_table["market_date"] < cutoff_ts].copy()
    eligible = _filter_candidate_rows(
        eligible,
        min_candidate_expected_win_rate=float(min_candidate_expected_win_rate),
        min_candidate_final_confidence=float(min_candidate_final_confidence),
        min_candidate_recommendation=str(min_candidate_recommendation),
    )
    if len(eligible) < int(min_train_rows):
        return None
    if eligible["label"].nunique() < 2:
        return None

    holdout_start = cutoff_ts - pd.Timedelta(days=int(holdout_days))
    valid_df = eligible.loc[eligible["market_date"] >= holdout_start].copy()
    train_df = eligible.loc[eligible["market_date"] < holdout_start].copy()
    if len(valid_df) < int(min_holdout_rows):
        split_idx = int(len(eligible) * 0.8)
        train_df = eligible.iloc[:split_idx].copy()
        valid_df = eligible.iloc[split_idx:].copy()
    if len(train_df) < int(min_train_rows) or len(valid_df) < int(min_holdout_rows):
        return None
    if train_df["label"].nunique() < 2 or valid_df["label"].nunique() < 2:
        return None

    rank_model = _fit_rank_model(train_df, num_pair_per_sample=int(num_pair_per_sample))
    point_model = _fit_point_model(train_df)

    valid_rank_raw = rank_model.predict(valid_df[FEATURE_COLUMNS])
    valid_rank_pct = _score_rank_within_group(valid_df, valid_rank_raw).to_numpy(dtype=float)
    valid_point_raw = point_model.predict_proba(valid_df[FEATURE_COLUMNS])[:, 1]
    point_calibrator = _fit_isotonic(valid_point_raw, valid_df["label"].to_numpy(dtype=float))
    valid_point_prob = _apply_calibrator(valid_point_raw, point_calibrator)

    best_weight = 0.5
    best_score = float("-inf")
    best_blend_raw = None
    for weight in np.linspace(0.0, 1.0, 11):
        blend_raw = weight * valid_point_prob + (1.0 - weight) * valid_rank_pct
        eval_df = valid_df.copy()
        eval_df["blend_raw"] = blend_raw
        score = _weighted_topk_precision(eval_df, "blend_raw")
        if score > best_score:
            best_score = score
            best_weight = float(weight)
            best_blend_raw = blend_raw

    if best_blend_raw is None:
        best_blend_raw = 0.5 * valid_point_prob + 0.5 * valid_rank_pct

    blend_calibrator = _fit_isotonic(best_blend_raw, valid_df["label"].to_numpy(dtype=float))

    final_rank_model = _fit_rank_model(eligible, num_pair_per_sample=int(num_pair_per_sample))
    final_point_model = _fit_point_model(eligible)
    return RobustRerankerArtifacts(
        rank_model=final_rank_model,
        point_model=final_point_model,
        point_calibrator=point_calibrator,
        blend_calibrator=blend_calibrator,
        feature_columns=list(FEATURE_COLUMNS),
        train_rows=int(len(eligible)),
        train_dates=int(eligible["market_date"].nunique()),
        holdout_rows=int(len(valid_df)),
        holdout_dates=int(valid_df["market_date"].nunique()),
        positive_rate=float(eligible["label"].mean()),
        blend_weight=float(best_weight),
        cutoff_date=str(cutoff_ts.date()),
    )
=======
from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path

from .accepted_pick_gate import LogisticGateConfig, RegularizedLogisticGate, build_meta_cohort_columns


TARGETS = {"PTS", "TRB", "AST"}
PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PLAYER_PREDICTOR_ROOT / "model" / "analysis"
SHARED_VALIDATION_ROOT = Path(__file__).resolve().parents[4] / "validation"
HISTORY_SOURCE_PATTERNS: tuple[tuple[str, int], ...] = (
    ("validation_recent_pool_selector", 500),
    ("selector_replay_rows_rebuilt", 420),
    ("selector_replay_rows", 360),
    ("validation_current_prod_hitrate_rows", 300),
    ("board_size_history_rows", 260),
    ("latest_market_comparison_strict_rows", 180),
)
NUMERIC_FEATURES = [
    "meta_prob",
    "meta_ev",
    "meta_confidence",
    "meta_abs_edge",
    "meta_edge_to_sigma",
    "meta_history_rows",
    "meta_uncertainty_sigma",
    "meta_spike_probability",
    "meta_market_line",
]
CATEGORICAL_FEATURES = [
    "target",
    "direction",
    "meta_seg",
    "meta_seg_ew",
    "meta_seg_ev",
    "meta_seg_line",
]


def _numeric_series(frame: pd.DataFrame, columns: tuple[str, ...], default: float = np.nan) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _string_series(frame: pd.DataFrame, columns: tuple[str, ...], default: str = "") -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _coerce_event_datetime(values: Any) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    out = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    ymd_mask = numeric.between(19000101, 21001231, inclusive="both")
    if bool(ymd_mask.any()):
        ymd_text = numeric.round().astype("Int64").astype(str)
        parsed_ymd = pd.to_datetime(ymd_text.where(ymd_mask), format="%Y%m%d", errors="coerce")
        out = out.where(~ymd_mask, parsed_ymd)
    return out


def _recommendation_rank(values: pd.Series) -> pd.Series:
    order = {"elite": 3, "strong": 2, "consider": 1, "pass": 0}
    return values.fillna("").astype(str).str.strip().str.lower().map(order).fillna(0).astype("float64")


def _compute_edge_to_sigma(abs_edge: pd.Series, uncertainty_sigma: pd.Series) -> pd.Series:
    sigma = pd.to_numeric(uncertainty_sigma, errors="coerce").abs()
    edge = pd.to_numeric(abs_edge, errors="coerce").abs()
    return edge / sigma.where(sigma > 1e-6, np.nan)


def _safe_brier(prob: np.ndarray, label: np.ndarray) -> float | None:
    if prob.size == 0 or label.size == 0:
        return None
    return float(np.mean((prob - label) ** 2))


def _safe_log_loss(prob: np.ndarray, label: np.ndarray) -> float | None:
    if prob.size == 0 or label.size == 0:
        return None
    clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(label * np.log(clipped) + (1.0 - label) * np.log(1.0 - clipped)))


def _resolve_holdout_split(
    prepared_history: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp | pd.NaT,
    min_train_rows: int,
    holdout_days: int,
    min_holdout_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = prepared_history.copy()
    holdout = prepared_history.iloc[0:0].copy()
    meta = {"mode": "disabled"}
    if prepared_history.empty:
        return train, holdout, meta

    if int(max(0, holdout_days)) > 0 and pd.notna(cutoff_date):
        holdout_cutoff = pd.Timestamp(cutoff_date) - pd.Timedelta(days=int(max(0, holdout_days)))
        maybe_holdout = prepared_history.loc[prepared_history["event_date"] >= holdout_cutoff].copy()
        maybe_train = prepared_history.loc[prepared_history["event_date"] < holdout_cutoff].copy()
        if len(maybe_train) >= int(max(1, min_train_rows)) and len(maybe_holdout) >= int(max(1, min_holdout_rows)):
            return maybe_train, maybe_holdout, {
                "mode": "calendar_days",
                "holdout_days": int(max(0, holdout_days)),
            }

    unique_days = sorted(pd.to_datetime(prepared_history["event_date"], errors="coerce").dt.normalize().dropna().unique().tolist())
    if len(unique_days) >= 6:
        holdout_day_count = max(2, int(np.ceil(float(len(unique_days)) * 0.20)))
        holdout_days_set = set(unique_days[-holdout_day_count:])
        mask = pd.to_datetime(prepared_history["event_date"], errors="coerce").dt.normalize().isin(holdout_days_set)
        maybe_holdout = prepared_history.loc[mask].copy()
        maybe_train = prepared_history.loc[~mask].copy()
        relaxed_train_floor = max(1000, int(max(1, min_train_rows) * 0.60))
        if len(maybe_train) >= int(relaxed_train_floor) and len(maybe_holdout) >= int(max(100, min_holdout_rows)):
            return maybe_train, maybe_holdout, {
                "mode": "tail_unique_days",
                "holdout_day_count": int(holdout_day_count),
                "relaxed_train_floor": int(relaxed_train_floor),
            }

    return train, holdout, meta


def _resolve_effective_shrink(
    requested_shrink: float,
    *,
    holdout_prob: np.ndarray | None,
    base_prob: np.ndarray | None,
    label: np.ndarray | None,
) -> tuple[float, dict[str, Any]]:
    requested = float(np.clip(requested_shrink, 0.0, 1.0))
    if holdout_prob is None or base_prob is None or label is None or len(holdout_prob) == 0:
        return max(0.25, requested * 0.60), {"mode": "no_holdout", "requested": requested}

    model_brier = _safe_brier(holdout_prob, label)
    base_brier = _safe_brier(base_prob, label)
    model_log_loss = _safe_log_loss(holdout_prob, label)
    base_log_loss = _safe_log_loss(base_prob, label)
    brier_gain = float(base_brier - model_brier) if model_brier is not None and base_brier is not None else 0.0
    log_loss_gain = float(base_log_loss - model_log_loss) if model_log_loss is not None and base_log_loss is not None else 0.0

    if brier_gain > 0.002 and log_loss_gain > 0.002:
        effective = requested
        mode = "full"
    elif brier_gain > 0.0 or log_loss_gain > 0.0:
        effective = max(0.35, requested * 0.75)
        mode = "partial"
    else:
        effective = min(0.25, requested * 0.35)
        mode = "degraded"

    return float(np.clip(effective, 0.0, 1.0)), {
        "mode": mode,
        "requested": requested,
        "effective": float(np.clip(effective, 0.0, 1.0)),
        "brier_gain": float(brier_gain),
        "log_loss_gain": float(log_loss_gain),
        "model_brier": model_brier,
        "base_brier": base_brier,
        "model_log_loss": model_log_loss,
        "base_log_loss": base_log_loss,
    }


def _resolve_cutoff_date(selector_df: pd.DataFrame) -> pd.Timestamp | pd.NaT:
    for column in ("market_date", "run_date", "target_date"):
        if column not in selector_df.columns:
            continue
        parsed = _coerce_event_datetime(selector_df[column]).dropna()
        if not parsed.empty:
            return pd.Timestamp(parsed.min()).normalize()
    return pd.NaT


def _path_pattern_score(path: Path) -> int:
    name = path.name.lower()
    for pattern, score in HISTORY_SOURCE_PATTERNS:
        if pattern in name:
            return int(score)
    return 100


def _history_summary(prepared: pd.DataFrame) -> dict[str, Any]:
    event_dates = pd.to_datetime(prepared.get("event_date"), errors="coerce").dropna() if "event_date" in prepared.columns else pd.Series(dtype="datetime64[ns]")
    unique_days = int(event_dates.dt.normalize().nunique()) if not event_dates.empty else 0
    latest_date = str(event_dates.max().date()) if not event_dates.empty else ""
    return {
        "rows": int(len(prepared)),
        "unique_days": int(unique_days),
        "latest_date": latest_date,
    }


def _discover_history_csv_paths() -> list[Path]:
    candidates: list[Path] = []
    for root in (ANALYSIS_ROOT, SHARED_VALIDATION_ROOT):
        if not root.exists():
            continue
        for pattern, _ in HISTORY_SOURCE_PATTERNS:
            candidates.extend(root.rglob(f"{pattern}*.csv"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except Exception:
            resolved = str(candidate)
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _choose_history_frame(
    history_df: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp | pd.NaT,
    min_train_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared_current = _prepare_history_rows(history_df, cutoff_date)
    current_summary = _history_summary(prepared_current)
    current_meta = {
        "source": "in_memory_history_df",
        "path": "",
        **current_summary,
    }

    best_frame = prepared_current
    best_meta = current_meta
    best_score = (0, current_summary["unique_days"], current_summary["rows"], current_summary["latest_date"])

    discovered_rows: list[dict[str, Any]] = []
    for path in _discover_history_csv_paths():
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        prepared = _prepare_history_rows(raw, cutoff_date)
        if prepared.empty:
            continue
        summary = _history_summary(prepared)
        name_score = int(_path_pattern_score(path))
        score = (name_score, summary["unique_days"], summary["rows"], summary["latest_date"])
        discovered_rows.append(
            {
                "path": str(path.resolve()),
                "rows": int(summary["rows"]),
                "unique_days": int(summary["unique_days"]),
                "latest_date": str(summary["latest_date"]),
                "name_score": int(name_score),
            }
        )
        if score > best_score:
            best_score = score
            best_frame = prepared
            best_meta = {
                "source": path.name,
                "path": str(path.resolve()),
                **summary,
            }

    current_is_rich_enough = (
        int(current_summary["rows"]) >= int(max(1, min_train_rows))
        and int(current_summary["rows"]) >= int(best_meta.get("rows", 0))
        and int(current_summary["unique_days"]) >= int(best_meta.get("unique_days", 0))
    )
    if current_is_rich_enough:
        best_frame = prepared_current
        best_meta = current_meta

    best_meta["discovered_candidates"] = discovered_rows
    return best_frame, best_meta


def _prepare_model_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if source == "history":
        prob = _numeric_series(frame, ("estimated_win_rate", "expected_win_rate", "p_calibrated", "board_play_win_prob"), default=0.5)
        ev = _numeric_series(frame, ("estimated_ev", "ev", "ev_adjusted"), default=0.0)
        confidence = _numeric_series(frame, ("selection_confidence", "final_confidence", "confidence_score"), default=0.0)
        market_line = _numeric_series(frame, ("market_line",), default=np.nan)
    else:
        prob = _numeric_series(frame, ("expected_win_rate", "p_calibrated", "board_play_win_prob"), default=0.5)
        ev = _numeric_series(frame, ("ev", "ev_adjusted", "estimated_ev"), default=0.0)
        confidence = _numeric_series(frame, ("final_confidence", "confidence_score", "selection_confidence"), default=0.0)
        market_line = _numeric_series(frame, ("market_line",), default=np.nan)

    out = pd.DataFrame(index=frame.index)
    out["target"] = _string_series(frame, ("target",)).str.upper().str.strip()
    out["direction"] = _string_series(frame, ("direction",)).str.upper().str.strip()
    out["meta_prob"] = prob.fillna(0.5).clip(lower=0.0, upper=1.0)
    out["meta_ev"] = ev.fillna(0.0)
    out["meta_confidence"] = confidence.fillna(0.0).clip(lower=0.0, upper=1.0)
    out["meta_abs_edge"] = _numeric_series(frame, ("abs_edge",), default=0.0).fillna(0.0).abs()
    out["meta_history_rows"] = _numeric_series(frame, ("history_rows",), default=0.0).fillna(0.0).clip(lower=0.0)
    out["meta_uncertainty_sigma"] = _numeric_series(frame, ("uncertainty_sigma",), default=np.nan).fillna(0.0).clip(lower=0.0)
    out["meta_spike_probability"] = _numeric_series(frame, ("spike_probability",), default=0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["meta_market_line"] = market_line.fillna(0.0)
    edge_to_sigma = _numeric_series(frame, ("edge_to_sigma", "sigma_ratio"), default=np.nan)
    edge_to_sigma = edge_to_sigma.where(edge_to_sigma.notna(), _compute_edge_to_sigma(out["meta_abs_edge"], out["meta_uncertainty_sigma"]))
    out["meta_edge_to_sigma"] = edge_to_sigma.fillna(0.0).clip(lower=0.0)
    out["meta_recommendation_rank"] = _recommendation_rank(_string_series(frame, ("recommendation",), default=""))

    cohort_frame = pd.DataFrame(
        {
            "target": out["target"],
            "direction": out["direction"],
            "expected_win_rate": out["meta_prob"],
            "ev": out["meta_ev"],
            "market_line": out["meta_market_line"],
        },
        index=out.index,
    )
    cohorts = build_meta_cohort_columns(cohort_frame, target_col="target", direction_col="direction")
    for column in ("meta_seg", "meta_seg_ew", "meta_seg_ev", "meta_seg_line"):
        out[column] = cohorts.get(column, pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    return out


def _prepare_history_rows(history_df: pd.DataFrame, cutoff_date: pd.Timestamp | pd.NaT) -> pd.DataFrame:
    if history_df.empty:
        return history_df.iloc[0:0].copy()
    working = history_df.copy()
    working["target"] = _string_series(working, ("target",)).str.upper().str.strip()
    working["direction"] = _string_series(working, ("direction",)).str.upper().str.strip()
    working = working.loc[working["target"].isin(TARGETS) & working["direction"].isin({"OVER", "UNDER"})].copy()
    result = _string_series(working, ("result",)).str.strip().str.lower()
    working = working.loc[result.isin({"win", "loss"})].copy()
    if working.empty:
        return working
    working["label"] = np.where(result.loc[working.index].eq("win"), 1.0, 0.0)
    working["event_date"] = _coerce_event_datetime(_string_series(working, ("market_date", "run_date", "target_date")))
    working = working.loc[working["event_date"].notna()].copy()
    if pd.notna(cutoff_date):
        working = working.loc[working["event_date"] < pd.Timestamp(cutoff_date)].copy()
    if working.empty:
        return working
    prepared = _prepare_model_frame(working, source="history")
    prepared["label"] = working["label"].astype("float64")
    prepared["event_date"] = working["event_date"]
    prepared["base_prob"] = prepared["meta_prob"]
    return prepared


def _normalize_blend_component(values: pd.Series, center: float, scale: float) -> pd.Series:
    return (0.5 + 0.5 * np.tanh((pd.to_numeric(values, errors="coerce").fillna(center) - center) / max(scale, 1e-6))).clip(0.0, 1.0)


def _soft_floor_strength(values: pd.Series, floor: float, span: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(float(floor))
    return ((numeric - float(floor)) / max(float(span), 1e-6) + 0.5).clip(0.0, 1.0)
>>>>>>> bab5e1cbad04c51851618e595a872ba22b138292:sports/nba/predictions/Player-Predictor/decision_engine/robust_reranker.py


def score_selector_with_robust_reranker(
    selector_df: pd.DataFrame,
    history_df: pd.DataFrame,
<<<<<<< HEAD:Player-Predictor/decision_engine/robust_reranker.py
    probability_shrink_factor: float,
    elite_pct: float,
=======
    *,
    probability_shrink_factor: float = 0.75,
    elite_pct: float = 0.95,
>>>>>>> bab5e1cbad04c51851618e595a872ba22b138292:sports/nba/predictions/Player-Predictor/decision_engine/robust_reranker.py
    min_train_rows: int = 4000,
    holdout_days: int = 45,
    min_holdout_rows: int = 250,
    num_pair_per_sample: int = 12,
    min_candidate_expected_win_rate: float = 0.55,
    min_candidate_final_confidence: float = 0.03,
    min_candidate_recommendation: str = "consider",
<<<<<<< HEAD:Player-Predictor/decision_engine/robust_reranker.py
) -> tuple[pd.DataFrame, dict]:
    if selector_df.empty:
        return selector_df.copy(), {"enabled": False, "reason": "empty_selector"}
    if history_df.empty:
        out = selector_df.copy()
        out["robust_reranker_rank_score"] = np.nan
        out["robust_reranker_point_prob"] = np.nan
        out["robust_reranker_blend_raw"] = np.nan
        out["robust_reranker_prob"] = np.nan
        out["robust_reranker_enabled"] = False
        return out, {"enabled": False, "reason": "empty_history"}

    history_table = build_historical_reranker_table(
        history_df,
        probability_shrink_factor=float(probability_shrink_factor),
        elite_pct=float(elite_pct),
    )
    if history_table.empty:
        out = selector_df.copy()
        out["robust_reranker_rank_score"] = np.nan
        out["robust_reranker_point_prob"] = np.nan
        out["robust_reranker_blend_raw"] = np.nan
        out["robust_reranker_prob"] = np.nan
        out["robust_reranker_enabled"] = False
        return out, {"enabled": False, "reason": "empty_history_table"}

    scored_parts: list[pd.DataFrame] = []
    summary: dict[str, dict] = {}
    for market_date, group in selector_df.groupby("market_date", sort=False):
        group_out = add_reranker_features(group)
        cutoff_ts = pd.Timestamp(market_date).normalize()
        artifacts = fit_robust_reranker_for_cutoff(
            history_table,
            cutoff_date=cutoff_ts,
            min_train_rows=min_train_rows,
            holdout_days=holdout_days,
            min_holdout_rows=min_holdout_rows,
            num_pair_per_sample=num_pair_per_sample,
            min_candidate_expected_win_rate=min_candidate_expected_win_rate,
            min_candidate_final_confidence=min_candidate_final_confidence,
            min_candidate_recommendation=min_candidate_recommendation,
        )
        if artifacts is None:
            group_out["robust_reranker_rank_score"] = np.nan
            group_out["robust_reranker_point_prob"] = np.nan
            group_out["robust_reranker_blend_raw"] = np.nan
            group_out["robust_reranker_prob"] = np.nan
            group_out["robust_reranker_enabled"] = False
            summary[str(market_date)] = {"enabled": False, "reason": "insufficient_training_history"}
        else:
            rank_raw = artifacts.rank_model.predict(group_out[artifacts.feature_columns])
            rank_pct = _score_rank_within_group(group_out, rank_raw).to_numpy(dtype=float)
            point_raw = artifacts.point_model.predict_proba(group_out[artifacts.feature_columns])[:, 1]
            point_prob = _apply_calibrator(point_raw, artifacts.point_calibrator)
            blend_raw = artifacts.blend_weight * point_prob + (1.0 - artifacts.blend_weight) * rank_pct
            blend_prob = _apply_calibrator(blend_raw, artifacts.blend_calibrator)
            group_out["robust_reranker_rank_score"] = rank_pct
            group_out["robust_reranker_point_prob"] = point_prob
            group_out["robust_reranker_blend_raw"] = blend_raw
            group_out["robust_reranker_prob"] = blend_prob
            group_out["robust_reranker_enabled"] = True
            summary[str(market_date)] = {
                "enabled": True,
                "train_rows": int(artifacts.train_rows),
                "train_dates": int(artifacts.train_dates),
                "holdout_rows": int(artifacts.holdout_rows),
                "holdout_dates": int(artifacts.holdout_dates),
                "positive_rate": float(artifacts.positive_rate),
                "blend_weight": float(artifacts.blend_weight),
                "selector_rows": int(len(group_out)),
            }
        scored_parts.append(group_out)

    out = pd.concat(scored_parts, ignore_index=True) if scored_parts else selector_df.copy()
    return out, {"enabled": True, "by_market_date": summary}
=======
) -> tuple[pd.DataFrame, dict[str, Any]]:
    del elite_pct
    del num_pair_per_sample

    out = selector_df.copy()
    out["robust_reranker_prob"] = np.nan
    out["robust_reranker_blend_raw"] = np.nan
    out["robust_reranker_enabled"] = False
    out["robust_reranker_candidate_eligible"] = False
    out["robust_reranker_train_rows"] = 0
    out["robust_reranker_holdout_rows"] = 0
    out["robust_reranker_source"] = "disabled"

    cutoff_date = _resolve_cutoff_date(out)
    prepared_history, history_meta = _choose_history_frame(
        history_df,
        cutoff_date=cutoff_date,
        min_train_rows=int(max(1, min_train_rows)),
    )
    if len(prepared_history) < int(max(1, min_train_rows)):
        return out, {
            "enabled": False,
            "reason": "insufficient_train_rows",
            "train_rows": int(len(prepared_history)),
            "min_train_rows": int(max(1, min_train_rows)),
            "cutoff_date": str(cutoff_date.date()) if pd.notna(cutoff_date) else "",
            "history_source": history_meta,
        }

    prepared_selector = _prepare_model_frame(out, source="selector")
    if prepared_selector.empty:
        return out, {
            "enabled": False,
            "reason": "empty_selector",
            "train_rows": int(len(prepared_history)),
        }

    train, holdout, holdout_meta = _resolve_holdout_split(
        prepared_history,
        cutoff_date=cutoff_date,
        min_train_rows=int(max(1, min_train_rows)),
        holdout_days=int(max(0, holdout_days)),
        min_holdout_rows=int(max(1, min_holdout_rows)),
    )

    gate = RegularizedLogisticGate(
        config=LogisticGateConfig(
            learning_rate=0.05,
            l2_strength=2.5,
            max_iter=3000,
            tolerance=1e-7,
            class_weight_positive=1.0,
            class_weight_negative=1.0,
        )
    )
    try:
        gate.fit_dataframe(
            train,
            label_col="label",
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
    except Exception as exc:
        return out, {
            "enabled": False,
            "reason": "fit_error",
            "error": f"{type(exc).__name__}: {exc}",
            "train_rows": int(len(train)),
            "cutoff_date": str(cutoff_date.date()) if pd.notna(cutoff_date) else "",
            "history_source": history_meta,
        }

    holdout_prob = None
    base_holdout_prob = None
    holdout_label = None
    if not holdout.empty:
        holdout_prob = gate.predict_proba_dataframe(holdout)
        base_holdout_prob = holdout["base_prob"].to_numpy(dtype="float64", copy=False)
        holdout_label = holdout["label"].to_numpy(dtype="float64", copy=False)

    effective_shrink, shrink_meta = _resolve_effective_shrink(
        float(np.clip(probability_shrink_factor, 0.0, 1.0)),
        holdout_prob=holdout_prob,
        base_prob=base_holdout_prob,
        label=holdout_label,
    )

    model_prob = gate.predict_proba_dataframe(prepared_selector)
    base_prob = prepared_selector["meta_prob"].to_numpy(dtype="float64", copy=False)
    confidence = prepared_selector["meta_confidence"].to_numpy(dtype="float64", copy=False)
    edge_component = _normalize_blend_component(prepared_selector["meta_edge_to_sigma"], center=0.20, scale=0.35).to_numpy(
        dtype="float64",
        copy=False,
    )
    history_component = _normalize_blend_component(prepared_selector["meta_history_rows"], center=90.0, scale=45.0).to_numpy(
        dtype="float64",
        copy=False,
    )
    fallback_blend = 0.55 * base_prob + 0.20 * confidence + 0.15 * edge_component + 0.10 * history_component
    shrink = float(np.clip(effective_shrink, 0.0, 1.0))
    reranker_prob = np.clip(shrink * model_prob + (1.0 - shrink) * fallback_blend, 0.0, 1.0)
    blend_raw = np.clip(0.70 * reranker_prob + 0.20 * fallback_blend + 0.10 * edge_component, 0.0, 1.0)

    min_reco_rank = {"elite": 3, "strong": 2, "consider": 1, "pass": 0}.get(str(min_candidate_recommendation).strip().lower(), 1)
    candidate_eligible = (
        prepared_selector["meta_prob"].ge(float(min_candidate_expected_win_rate))
        & prepared_selector["meta_confidence"].ge(float(min_candidate_final_confidence))
        & prepared_selector["meta_recommendation_rank"].ge(float(min_reco_rank))
    )
    prob_strength = _soft_floor_strength(prepared_selector["meta_prob"], float(min_candidate_expected_win_rate), 0.08)
    confidence_strength = _soft_floor_strength(prepared_selector["meta_confidence"], float(min_candidate_final_confidence), 0.10)
    reco_strength = ((prepared_selector["meta_recommendation_rank"] - float(min_reco_rank)) / 2.0 + 0.5).clip(0.0, 1.0)
    candidate_strength = (0.45 * prob_strength + 0.35 * confidence_strength + 0.20 * reco_strength).clip(0.0, 1.0)
    eligible_bonus = np.where(candidate_eligible.to_numpy(dtype=bool, copy=False), 0.06, -0.03)
    reranker_prob = np.clip(0.84 * reranker_prob + 0.16 * candidate_strength.to_numpy(dtype="float64", copy=False) + eligible_bonus, 0.0, 1.0)
    blend_raw = np.clip(0.78 * blend_raw + 0.22 * candidate_strength.to_numpy(dtype="float64", copy=False) + eligible_bonus, 0.0, 1.0)

    out["robust_reranker_prob"] = pd.Series(reranker_prob, index=out.index, dtype="float64")
    out["robust_reranker_blend_raw"] = pd.Series(blend_raw, index=out.index, dtype="float64")
    out["robust_reranker_enabled"] = True
    out["robust_reranker_candidate_eligible"] = candidate_eligible.astype(bool)
    out["robust_reranker_candidate_strength"] = candidate_strength.astype("float64")
    out["robust_reranker_train_rows"] = int(len(prepared_history))
    out["robust_reranker_holdout_rows"] = int(len(holdout))
    out["robust_reranker_source"] = str(history_meta.get("source") or "walk_forward_logistic")

    summary: dict[str, Any] = {
        "enabled": True,
        "model_type": "walk_forward_logistic_meta_selector_v1",
        "cutoff_date": str(cutoff_date.date()) if pd.notna(cutoff_date) else "",
        "pre_cutoff_rows": int(len(prepared_history)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "selector_rows": int(len(out)),
        "candidate_eligible_rows": int(candidate_eligible.sum()),
        "candidate_strength_mean": float(candidate_strength.mean()) if len(candidate_strength) else 0.0,
        "feature_count": int(len(gate.feature_names())),
        "source": "walk_forward_logistic",
        "history_source": history_meta,
        "holdout_split": holdout_meta,
        "shrink": shrink_meta,
    }

    if not holdout.empty:
        summary["holdout"] = {
            "rows": int(len(holdout)),
            "mean_prob": float(np.mean(holdout_prob)),
            "mean_label": float(np.mean(holdout_label)),
            "brier": _safe_brier(holdout_prob, holdout_label),
            "log_loss": _safe_log_loss(holdout_prob, holdout_label),
            "baseline_brier": _safe_brier(base_holdout_prob, holdout_label),
            "baseline_log_loss": _safe_log_loss(base_holdout_prob, holdout_label),
        }

    return out, summary
>>>>>>> bab5e1cbad04c51851618e595a872ba22b138292:sports/nba/predictions/Player-Predictor/decision_engine/robust_reranker.py
