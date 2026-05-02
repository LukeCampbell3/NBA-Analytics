from __future__ import annotations

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


def score_selector_with_robust_reranker(
    selector_df: pd.DataFrame,
    history_df: pd.DataFrame,
    probability_shrink_factor: float,
    elite_pct: float,
    min_train_rows: int = 4000,
    holdout_days: int = 45,
    min_holdout_rows: int = 250,
    num_pair_per_sample: int = 12,
    min_candidate_expected_win_rate: float = 0.55,
    min_candidate_final_confidence: float = 0.03,
    min_candidate_recommendation: str = "consider",
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
