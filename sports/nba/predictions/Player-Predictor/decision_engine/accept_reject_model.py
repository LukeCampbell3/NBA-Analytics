from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .uncertainty import belief_confidence_factor, normalize_belief_uncertainty


TARGETS = ("PTS", "TRB", "AST")
TARGET_THRESHOLDS = {
    "PTS": {"consider_pct": 0.80, "strong_pct": 0.90, "elite_pct": 0.95},
    "TRB": {"consider_pct": 0.90, "strong_pct": 0.95, "elite_pct": 0.975},
    "AST": {"consider_pct": 0.90, "strong_pct": 0.95, "elite_pct": 0.975},
}
RECOMMENDATION_VALUES = {"pass": 0.0, "consider": 0.35, "strong": 0.7, "elite": 1.0}
FEATURE_COLUMNS = [
    "expected_win_rate",
    "raw_expected_win_rate",
    "gap_percentile",
    "abs_edge",
    "edge_over_line",
    "belief_uncertainty",
    "feasibility",
    "quality_score",
    "market_books",
    "fallback_blend",
    "history_rows",
    "rec_num",
    "is_under",
    "is_pts",
    "is_trb",
    "is_ast",
]


@dataclass
class AcceptorArtifacts:
    model: HistGradientBoostingClassifier
    feature_columns: list[str]
    train_rows: int
    train_dates: int
    holdout_rows: int
    holdout_dates: int
    learned_threshold: float
    threshold_floor: float
    threshold: float
    holdout_accept_rows: int
    holdout_accept_win_rate: float | None
    positive_rate: float
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


def add_acceptor_features(selector_df: pd.DataFrame) -> pd.DataFrame:
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
        "market_books",
        "fallback_blend",
        "history_rows",
        "market_line",
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
    out["rec_num"] = out["recommendation"].map(RECOMMENDATION_VALUES).fillna(0.0)
    out["is_under"] = (out["direction"] == "UNDER").astype(float)
    out["is_pts"] = (out["target"] == "PTS").astype(float)
    out["is_trb"] = (out["target"] == "TRB").astype(float)
    out["is_ast"] = (out["target"] == "AST").astype(float)
    return out


def build_historical_acceptor_table(
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
    return add_acceptor_features(out)


def _fit_classifier(train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
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


def _choose_threshold(valid_df: pd.DataFrame, probs: np.ndarray, min_accept_rows: int) -> tuple[float, int, float | None]:
    if len(valid_df) == 0:
        return 0.5, 0, None
    candidate_thresholds = sorted({float(np.quantile(probs, q)) for q in np.linspace(0.50, 0.95, 19)})
    candidate_thresholds.extend([0.50, 0.55, 0.60, 0.65, 0.70])
    best: tuple[float, int, float | None, float] | None = None
    for threshold in sorted(set(candidate_thresholds)):
        accepted = valid_df.loc[probs >= threshold]
        if len(accepted) < int(min_accept_rows):
            continue
        win_rate = float(accepted["label"].mean()) if len(accepted) else None
        avg_prob = float(probs[probs >= threshold].mean()) if len(accepted) else 0.0
        candidate = (float(threshold), int(len(accepted)), win_rate, avg_prob)
        if best is None:
            best = candidate
            continue
        if (candidate[2] or -1.0) > (best[2] or -1.0):
            best = candidate
        elif candidate[2] == best[2] and candidate[1] > best[1]:
            best = candidate
        elif candidate[2] == best[2] and candidate[1] == best[1] and candidate[3] > best[3]:
            best = candidate
    if best is None:
        return 1.1, 0, None
    return best[0], best[1], best[2]


def fit_acceptor_for_cutoff(
    history_table: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    min_train_rows: int = 3000,
    holdout_days: int = 45,
    min_holdout_rows: int = 250,
    min_accept_rate: float = 0.05,
    threshold_floor: float = 0.0,
) -> AcceptorArtifacts | None:
    if history_table.empty:
        return None
    cutoff_ts = pd.Timestamp(cutoff_date).normalize()
    eligible = history_table.loc[history_table["market_date"] < cutoff_ts].copy()
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

    selection_model = _fit_classifier(train_df)
    valid_probs = selection_model.predict_proba(valid_df[FEATURE_COLUMNS])[:, 1]
    min_accept_rows = max(25, int(len(valid_df) * float(min_accept_rate)))
    learned_threshold, _, _ = _choose_threshold(valid_df, valid_probs, min_accept_rows=min_accept_rows)
    applied_threshold = max(float(learned_threshold), float(threshold_floor))
    accepted = valid_df.loc[valid_probs >= applied_threshold]
    accept_rows = int(len(accepted))
    accept_win_rate = float(accepted["label"].mean()) if len(accepted) else None

    final_model = _fit_classifier(eligible)
    return AcceptorArtifacts(
        model=final_model,
        feature_columns=list(FEATURE_COLUMNS),
        train_rows=int(len(eligible)),
        train_dates=int(eligible["market_date"].nunique()),
        holdout_rows=int(len(valid_df)),
        holdout_dates=int(valid_df["market_date"].nunique()),
        learned_threshold=float(learned_threshold),
        threshold_floor=float(threshold_floor),
        threshold=float(applied_threshold),
        holdout_accept_rows=int(accept_rows),
        holdout_accept_win_rate=accept_win_rate,
        positive_rate=float(eligible["label"].mean()),
        cutoff_date=str(cutoff_ts.date()),
    )


def apply_acceptor_to_selector(
    selector_df: pd.DataFrame,
    history_df: pd.DataFrame,
    probability_shrink_factor: float,
    elite_pct: float,
    min_train_rows: int = 3000,
    holdout_days: int = 45,
    min_holdout_rows: int = 250,
    min_accept_rate: float = 0.05,
    threshold_floor: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    if selector_df.empty:
        return selector_df.copy(), {"enabled": False, "reason": "empty_selector"}
    if history_df.empty:
        out = selector_df.copy()
        out["accept_reject_score"] = np.nan
        out["accept_reject_threshold"] = np.nan
        out["accept_reject_accept"] = False
        return out, {"enabled": False, "reason": "empty_history"}

    history_table = build_historical_acceptor_table(
        history_df,
        probability_shrink_factor=float(probability_shrink_factor),
        elite_pct=float(elite_pct),
    )
    if history_table.empty:
        out = selector_df.copy()
        out["accept_reject_score"] = np.nan
        out["accept_reject_threshold"] = np.nan
        out["accept_reject_accept"] = False
        return out, {"enabled": False, "reason": "empty_history_table"}

    scored_parts: list[pd.DataFrame] = []
    summary: dict[str, dict] = {}
    for market_date, group in selector_df.groupby("market_date", sort=False):
        cutoff_ts = pd.Timestamp(market_date).normalize()
        artifacts = fit_acceptor_for_cutoff(
            history_table,
            cutoff_date=cutoff_ts,
            min_train_rows=min_train_rows,
            holdout_days=holdout_days,
            min_holdout_rows=min_holdout_rows,
            min_accept_rate=min_accept_rate,
            threshold_floor=threshold_floor,
        )
        group_out = add_acceptor_features(group)
        if artifacts is None:
            group_out["accept_reject_score"] = np.nan
            group_out["accept_reject_threshold"] = np.nan
            group_out["accept_reject_accept"] = True
            summary[str(market_date)] = {"enabled": False, "reason": "insufficient_training_history"}
        else:
            probs = artifacts.model.predict_proba(group_out[artifacts.feature_columns])[:, 1]
            group_out["accept_reject_score"] = probs
            group_out["accept_reject_threshold"] = float(artifacts.threshold)
            group_out["accept_reject_accept"] = probs >= float(artifacts.threshold)
            summary[str(market_date)] = {
                "enabled": True,
                "train_rows": int(artifacts.train_rows),
                "train_dates": int(artifacts.train_dates),
                "holdout_rows": int(artifacts.holdout_rows),
                "holdout_dates": int(artifacts.holdout_dates),
                "learned_threshold": float(artifacts.learned_threshold),
                "threshold_floor": float(artifacts.threshold_floor),
                "threshold": float(artifacts.threshold),
                "holdout_accept_rows": int(artifacts.holdout_accept_rows),
                "holdout_accept_win_rate": artifacts.holdout_accept_win_rate,
                "positive_rate": float(artifacts.positive_rate),
                "accepted_rows": int(group_out["accept_reject_accept"].sum()),
                "selector_rows": int(len(group_out)),
            }
        scored_parts.append(group_out)

    out = pd.concat(scored_parts, ignore_index=True) if scored_parts else selector_df.copy()
    if "accept_reject_accept" in out.columns:
        accepted = out.loc[out["accept_reject_accept"].fillna(False)].copy()
        if not accepted.empty:
            out = accepted
    return out, {"enabled": True, "by_market_date": summary}
