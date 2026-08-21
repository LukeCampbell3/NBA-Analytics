from __future__ import annotations

"""Run the full H_OVER_RANKER_V1 development protocol on DERIVE+SELECT only.

Never imports or reads TEST_STAMPS. See test_h_over_ranker.py::
test_run_development_never_touches_test_stamps for the static-analysis
guard on that.
"""

import json
from pathlib import Path

import pandas as pd

from .chronological_cv import expanding_day_folds
from .data_windows import DEVELOPMENT_STAMPS, verify_against_disk
from .edge_shape import evaluate_edge_shapes, summarize_shape_comparison
from .eligibility import eligible_rows_for_stamps
from .baselines import add_baseline_scores
from .evaluate import evaluate_score_chronologically
from .ranker import fit_predict_walkforward

MIN_TRAIN_DAYS = 6
OUTPUT_DIR = Path(__file__).resolve().parent / "reports"


def main() -> dict:
    verify_against_disk()
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    dates_with_rows = sorted(rows["date"].unique())
    print(f"development rows: {len(rows)} across {len(dates_with_rows)} dates: {dates_with_rows}")

    folds = expanding_day_folds(dates_with_rows, min_train_days=MIN_TRAIN_DAYS)
    print(f"walk-forward folds: {len(folds)} (min_train_days={MIN_TRAIN_DAYS})")
    for f in folds:
        print(f"  fold {f.index}: train={len(f.train_dates)} dates, val={f.val_date}")

    # 1. edge-shape test, disciplined out-of-fold
    shape_results = evaluate_edge_shapes(rows, folds)
    shape_summary = summarize_shape_comparison(shape_results)
    print("\n=== edge-shape comparison (out-of-fold Brier, lower is better) ===")
    print(shape_summary)

    # 2. baselines
    scored = add_baseline_scores(rows)
    baseline_reports = {}
    for col, name in [
        ("score_probability", "current_probability_ranking"),
        ("score_raw_edge", "raw_bias_corrected_edge"),
        ("score_q_edge_over_rmse", "q_edge_over_rmse"),
    ]:
        baseline_reports[name] = evaluate_score_chronologically(scored, col, folds, score_name=name)

    # 3. fitted H_OVER_RANKER_V1 candidate
    ranker_scored = fit_predict_walkforward(rows, folds)
    ranker_report = evaluate_score_chronologically(ranker_scored, "score_ranker_v1", folds, score_name="H_OVER_RANKER_V1_candidate")

    print("\n=== pooled results across all candidate scores ===")
    pooled_rows = []
    for name, report in {**baseline_reports, "H_OVER_RANKER_V1_candidate": ranker_report}.items():
        pooled = report.pooled()
        ci = report.day_clustered_bootstrap_ci("top1")
        pooled["top1_day_clustered_95_one_sided_lb"] = ci["one_sided_95_lower"]
        pooled_rows.append(pooled)
    pooled_df = pd.DataFrame(pooled_rows).set_index("score_name")
    print(pooled_df.to_string())

    print("\n=== per-fold detail, current_probability_ranking vs H_OVER_RANKER_V1_candidate ===")
    for name, report in [("current_probability_ranking", baseline_reports["current_probability_ranking"]),
                          ("H_OVER_RANKER_V1_candidate", ranker_report)]:
        print(f"-- {name} --")
        for s in report.per_fold:
            print(f"  {s.date}: n={s.n} pool={s.pool_hit_rate:.3f} top1={s.top1_hit} top2={s.top2_hit_rate}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled_df.to_csv(OUTPUT_DIR / "pooled_candidate_comparison.csv")
    shape_summary.to_csv(OUTPUT_DIR / "edge_shape_comparison.csv")
    with open(OUTPUT_DIR / "fold_models_ranker_v1_candidate.json", "w") as f:
        json.dump(ranker_scored.attrs.get("fold_models", []), f, indent=2)

    return {
        "shape_summary": shape_summary,
        "pooled": pooled_df,
        "baseline_reports": baseline_reports,
        "ranker_report": ranker_report,
        "ranker_scored": ranker_scored,
    }


if __name__ == "__main__":
    main()
