from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from .uncertainty import belief_confidence_factor, normalize_belief_uncertainty


TARGETS = ("PTS", "TRB", "AST")
TARGET_THRESHOLDS = {
    "PTS": {"consider_pct": 0.75, "strong_pct": 0.90, "elite_pct": 0.95},
    "TRB": {"consider_pct": 0.85, "strong_pct": 0.95, "elite_pct": 0.975},
    "AST": {"consider_pct": 0.85, "strong_pct": 0.95, "elite_pct": 0.975},
}
RECOMMENDATION_VALUES = {"pass": 0.0, "consider": 0.35, "strong": 0.7, "elite": 1.0}
BASE_FEATURE_COLUMNS = [
    "raw_expected_win_rate",
    "gap_percentile",
    "abs_edge",
    "edge_over_line",
    "edge_vs_baseline",
    "belief_uncertainty",
    "feasibility",
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
class XgbLtrArtifacts:
    model: XGBRanker
    feature_columns: list[str]
    train_rows: int
    train_dates: int
    cutoff_date: str
    positive_rate: float


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


def _expected_rate_for(target: str, percentile: float, prior_df: pd.DataFrame) -> float:
    if prior_df.empty:
        return 0.5
    thresholds = TARGET_THRESHOLDS[target]
    abs_gap = prior_df["abs_edge"]
    quartile_cut = float(abs_gap.quantile(0.75))
    decile_cut = float(abs_gap.quantile(0.90))
    if percentile >= thresholds["strong_pct"]:
        subset = prior_df.loc[prior_df["abs_edge"] >= decile_cut]
        if not subset.empty:
            return float(subset["label"].mean())
    if percentile >= thresholds["consider_pct"]:
        subset = prior_df.loc[prior_df["abs_edge"] >= quartile_cut]
        if not subset.empty:
            return float(subset["label"].mean())
    return float(prior_df["label"].mean())


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


def add_selector_features(selector_df: pd.DataFrame) -> pd.DataFrame:
    out = selector_df.copy()
    out["recommendation"] = out.get("recommendation", "pass").astype(str)
    out["direction"] = out.get("direction", "PUSH").astype(str)
    out["target"] = out.get("target", "").astype(str)

    for column in [
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
    out["edge_over_line"] = out["abs_edge"] / out["market_line"].abs().clip(lower=1.0)
    out["edge_vs_baseline"] = out["abs_edge"] - _safe_numeric(out.get("baseline_edge"), default=0.0).abs()
    out["rec_num"] = out["recommendation"].map(RECOMMENDATION_VALUES).fillna(0.0)
    out["is_under"] = (out["direction"] == "UNDER").astype(float)
    out["is_pts"] = (out["target"] == "PTS").astype(float)
    out["is_trb"] = (out["target"] == "TRB").astype(float)
    out["is_ast"] = (out["target"] == "AST").astype(float)

    group_key = pd.to_datetime(out.get("market_date"), errors="coerce").dt.normalize()
    return _add_group_rank_features(out, group_key=group_key)


def build_historical_ltr_table(history_df: pd.DataFrame, min_prior_rows_per_target: int = 120) -> pd.DataFrame:
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
                rows.append(
                    {
                        "market_date": row["market_date"],
                        "player": row["player"],
                        "target": target,
                        "direction": row["direction"],
                        "prediction": float(row["prediction"]),
                        "market_line": float(row["market_line"]),
                        "abs_edge": float(row["abs_edge"]),
                        "baseline_edge": float(row["baseline_edge"]),
                        "belief_uncertainty": float(np.clip(row["belief_uncertainty"], 0.0, 1.5)),
                        "feasibility": float(np.clip(row["feasibility"], 0.0, 1.0)),
                        "fallback_blend": float(np.clip(row["fallback_blend"], 0.0, 1.0)),
                        "market_books": float(max(0.0, row["market_books"])),
                        "history_rows": float(max(0.0, row["history_rows"])),
                        "gap_percentile": gap_percentile,
                        "raw_expected_win_rate": _expected_rate_for(target, gap_percentile, prior_df),
                        "recommendation": _classify_play(target, gap_percentile),
                        "label": int(row["label"]),
                    }
                )
            prior_rows.append({"abs_edge": float(row["abs_edge"]), "label": int(row["label"])})

    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    out["edge_over_line"] = out["abs_edge"] / out["market_line"].abs().clip(lower=1.0)
    out["edge_vs_baseline"] = out["abs_edge"] - _safe_numeric(out.get("baseline_edge"), default=0.0).abs()
    out["rec_num"] = out["recommendation"].map(RECOMMENDATION_VALUES).fillna(0.0)
    out["is_under"] = (out["direction"] == "UNDER").astype(float)
    out["is_pts"] = (out["target"] == "PTS").astype(float)
    out["is_trb"] = (out["target"] == "TRB").astype(float)
    out["is_ast"] = (out["target"] == "AST").astype(float)
    return _add_group_rank_features(out, group_key=out["market_date"])


def _fit_xgb_ltr_from_table(
    history_table: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    min_train_rows: int = 4000,
    num_pair_per_sample: int = 12,
) -> XgbLtrArtifacts | None:
    if history_table.empty:
        return None
    cutoff_ts = pd.Timestamp(cutoff_date).normalize()
    train_df = history_table.loc[history_table["market_date"] < cutoff_ts].copy()
    if len(train_df) < int(min_train_rows):
        return None
    if train_df["label"].nunique() < 2:
        return None

    train_df = train_df.sort_values(["market_date", "player", "target"]).reset_index(drop=True)
    qid = pd.factorize(train_df["market_date"], sort=True)[0]
    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"].to_numpy(dtype=float)

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
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
    model.fit(X, y, qid=qid)
    return XgbLtrArtifacts(
        model=model,
        feature_columns=list(FEATURE_COLUMNS),
        train_rows=int(len(train_df)),
        train_dates=int(train_df["market_date"].nunique()),
        cutoff_date=str(cutoff_ts.date()),
        positive_rate=float(train_df["label"].mean()),
    )


def fit_xgb_ltr_reranker(
    history_df: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    min_train_rows: int = 4000,
    num_pair_per_sample: int = 12,
) -> XgbLtrArtifacts | None:
    history_table = build_historical_ltr_table(history_df)
    return _fit_xgb_ltr_from_table(
        history_table,
        cutoff_date=cutoff_date,
        min_train_rows=min_train_rows,
        num_pair_per_sample=num_pair_per_sample,
    )


def score_selector_with_xgb_ltr(
    selector_df: pd.DataFrame,
    history_df: pd.DataFrame,
    min_train_rows: int = 4000,
    num_pair_per_sample: int = 12,
) -> tuple[pd.DataFrame, dict]:
    if selector_df.empty:
        return selector_df.copy(), {"enabled": False, "reason": "empty_selector"}
    if history_df.empty:
        out = selector_df.copy()
        out["xgb_ltr_score"] = np.nan
        out["xgb_ltr_enabled"] = False
        return out, {"enabled": False, "reason": "empty_history"}

    history_table = build_historical_ltr_table(history_df)
    if history_table.empty:
        out = selector_df.copy()
        out["xgb_ltr_score"] = np.nan
        out["xgb_ltr_enabled"] = False
        return out, {"enabled": False, "reason": "empty_history_table"}

    scored_parts: list[pd.DataFrame] = []
    summary: dict[str, dict] = {}
    for market_date, group in selector_df.groupby("market_date", sort=False):
        cutoff_ts = pd.Timestamp(market_date).normalize()
        train_df = history_table.loc[history_table["market_date"] < cutoff_ts].copy()
        if len(train_df) < int(min_train_rows) or train_df["label"].nunique() < 2:
            group_out = add_selector_features(group)
            group_out["xgb_ltr_score"] = np.nan
            group_out["xgb_ltr_enabled"] = False
            summary[str(market_date)] = {"enabled": False, "train_rows": int(len(train_df)), "train_dates": int(train_df["market_date"].nunique()) if not train_df.empty else 0}
            scored_parts.append(group_out)
            continue

        artifacts = _fit_xgb_ltr_from_table(
            history_table,
            cutoff_date=cutoff_ts,
            min_train_rows=min_train_rows,
            num_pair_per_sample=num_pair_per_sample,
        )
        group_out = add_selector_features(group)
        if artifacts is None:
            group_out["xgb_ltr_score"] = np.nan
            group_out["xgb_ltr_enabled"] = False
            summary[str(market_date)] = {"enabled": False, "train_rows": int(len(train_df)), "train_dates": int(train_df["market_date"].nunique())}
        else:
            group_out["xgb_ltr_score"] = artifacts.model.predict(group_out[artifacts.feature_columns])
            group_out["xgb_ltr_enabled"] = True
            summary[str(market_date)] = {
                "enabled": True,
                "train_rows": int(artifacts.train_rows),
                "train_dates": int(artifacts.train_dates),
                "positive_rate": float(artifacts.positive_rate),
                "cutoff_date": artifacts.cutoff_date,
            }
        scored_parts.append(group_out)

    out = pd.concat(scored_parts, ignore_index=True) if scored_parts else selector_df.copy()
    return out, {"enabled": True, "by_market_date": summary}
