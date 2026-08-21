from __future__ import annotations

"""Records every ranker variant tried during development, including the
ones that did not make the frozen cut -- run once, output committed to
reports/ for the audit trail. DEVELOPMENT_STAMPS only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .chronological_cv import expanding_day_folds, split
from .data_windows import DEVELOPMENT_STAMPS, verify_against_disk
from .eligibility import eligible_rows_for_stamps
from .ranker import FEATURE_COLUMNS, build_features

OUTPUT_DIR = Path(__file__).resolve().parent / "reports"
MIN_TRAIN_DAYS = 6


def _fit_score(train, val, feats, c):
    train_x = train[list(feats)].to_numpy(dtype=float)
    val_x = val[list(feats)].to_numpy(dtype=float)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-9] = 1.0
    train_x_std = (train_x - mean) / std
    val_x_std = (val_x - mean) / std
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(train_x_std, train["win"])
    probs = model.predict_proba(val_x_std)[:, 1]
    return probs, model


def evaluate_variant(frame, folds, feats, c):
    briers, top1_hits, top2_rates = [], [], []
    for fold in folds:
        train, val = split(frame, fold)
        if len(train) < 20 or val.empty:
            continue
        probs, _ = _fit_score(train, val, feats, c)
        val = val.copy()
        val["p"] = probs
        briers.append(float(np.mean((probs - val["win"].to_numpy(dtype=float)) ** 2)))
        ranked = val.sort_values(["p", "rmse", "player"], ascending=[False, True, True])
        top1_hits.append(int(ranked["win"].iloc[0]))
        top2_rates.append(float(ranked["win"].iloc[:2].mean()))
    return {
        "features": ",".join(feats),
        "n_features": len(feats),
        "c": c,
        "n_folds": len(top1_hits),
        "mean_oof_brier": float(np.mean(briers)) if briers else float("nan"),
        "top1_hit_rate": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "top2_hit_rate": float(np.mean(top2_rates)) if top2_rates else float("nan"),
    }


def main() -> pd.DataFrame:
    verify_against_disk()
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    dates = sorted(rows["date"].unique())
    folds = expanding_day_folds(dates, min_train_days=MIN_TRAIN_DAYS)
    featured = build_features(rows)

    results = []

    # single-feature ablations (which single signal carries the most weight alone)
    all_candidate_feats = (
        "corrected_edge", "rmse", "q_edge_over_rmse", "log1p_history_rows", "is_real_market",
    )
    for feat in all_candidate_feats:
        r = evaluate_variant(featured, folds, (feat,), c=1.0)
        r["variant"] = f"single_feature[{feat}]"
        results.append(r)

    # regularization sweep on the full candidate feature set (incl. edge^2, rejected below)
    full_with_quad = all_candidate_feats + ("corrected_edge_sq",)
    featured_with_quad = featured.copy()
    featured_with_quad["corrected_edge_sq"] = featured_with_quad["corrected_edge"] ** 2
    for c in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
        r = evaluate_variant(featured_with_quad, folds, full_with_quad, c=c)
        r["variant"] = f"full_with_edge_sq[C={c}]"
        results.append(r)

    # quadratic-term ablation at the frozen C
    r_with = evaluate_variant(featured_with_quad, folds, full_with_quad, c=0.1)
    r_with["variant"] = "with_edge_sq[C=0.1]"
    results.append(r_with)
    r_without = evaluate_variant(featured, folds, FEATURE_COLUMNS, c=0.1)
    r_without["variant"] = "frozen_H_OVER_RANKER_V1[C=0.1, no edge_sq]"
    results.append(r_without)

    table = pd.DataFrame(results)[
        ["variant", "features", "n_features", "c", "n_folds", "mean_oof_brier", "top1_hit_rate", "top2_hit_rate"]
    ]
    print(table.to_string(index=False))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "all_tried_variants.csv", index=False)
    return table


if __name__ == "__main__":
    main()
