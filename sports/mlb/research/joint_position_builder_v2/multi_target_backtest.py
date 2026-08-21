from __future__ import annotations

"""Multi-target (R/TB/HR) pair backtest -- tests the mission's explicit
"do not assume H-OVER only" instruction rather than assuming it.

Reuses pairs.enumerate_candidate_pairs and the same walk-forward
calibration-warmup discipline as ablation.run_variant, UNCHANGED, applied to
multi_target_universe.build_multi_target_universe instead of
observation_universe.build_observation_universe. DEVELOPMENT_STAMPS only;
TEST_STAMPS stays retired and unread by this module.

Critical methodological point this script exists to make explicit (see
STATE.md): `joint_ev` / `joint_ev_lcb` as computed by pairs.py are MODEL-
CONFIDENCE figures (p_joint, from the frozen marginal model under an
independence assumption, times the real market price). They are NOT the
same thing as REALIZED backtest return. A high mean `joint_ev` only tells
you the model believes it has an edge -- it says nothing about whether that
belief is calibrated. This script computes both, side by side, specifically
so the gap between them is visible rather than silently assumed away.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sports.mlb.conditional_chain.outcome_worlds import conformal_aps_threshold
from sports.mlb.research.h_over_ranker.data_windows import DEVELOPMENT_STAMPS, verify_against_disk

from .ablation import MIN_CALIBRATION_PAIRS, TARGET_MISCOVERAGE, _PAIR_RECORD_COLUMNS, _filter_pairs, _pair_to_record
from .multi_target_universe import PRICED_TARGETS, action_universe, build_multi_target_universe
from .pairs import enumerate_candidate_pairs

OUTPUT_DIR = Path(__file__).resolve().parent / "reports"


def run_multi_target_backtest(mode: str = "broad", targets: tuple[str, ...] = PRICED_TARGETS) -> pd.DataFrame:
    verify_against_disk()
    universe = build_multi_target_universe(DEVELOPMENT_STAMPS, targets=targets, mode=mode)
    action = action_universe(universe)
    dates = sorted(action["date"].unique())

    calibration_scores: list[float] = []
    calibration_days_seen: set[str] = set()
    pair_records: list[dict] = []

    for date in dates:
        day_rows = action[action["date"] == date].reset_index(drop=True)
        if len(day_rows) < 2:
            continue
        if len(calibration_scores) < MIN_CALIBRATION_PAIRS:
            threshold = 1.0  # warm-up: retain everything, diagnostics only, never actioned
        else:
            threshold = conformal_aps_threshold(calibration_scores, target_miscoverage=TARGET_MISCOVERAGE)

        day_pairs = enumerate_candidate_pairs(day_rows, aps_threshold=threshold, calibration_slates=len(calibration_scores))
        filtered = _filter_pairs(day_pairs, "all_classes")
        for pair in filtered:
            pair_records.append(_pair_to_record(pair, len(calibration_scores), len(calibration_days_seen)))
            calibration_scores.append(pair.aps_score_true_world)
        calibration_days_seen.add(date)

    all_pairs = pd.DataFrame(pair_records, columns=list(_PAIR_RECORD_COLUMNS))
    return all_pairs


def day_clustered_bootstrap_ci(values_by_date: pd.DataFrame, value_col: str, n_boot: int = 3000, seed: int = 0) -> tuple[float, float, float]:
    """Resamples DATES (not rows) -- rows within a date are correlated
    (a day's pairs share legs), so row-level bootstrap would understate the
    true uncertainty. Returns (mean, ci_low_5, ci_high_95)."""
    rng = np.random.default_rng(seed)
    dates = values_by_date["date"].unique()
    if len(dates) == 0:
        return float("nan"), float("nan"), float("nan")
    boot = np.empty(n_boot)
    grouped = {d: values_by_date.loc[values_by_date["date"] == d, value_col] for d in dates}
    for b in range(n_boot):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        boot[b] = pd.concat([grouped[d] for d in sample_dates]).mean()
    return float(boot.mean()), float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


def summarize(all_pairs: pd.DataFrame) -> dict:
    priced = all_pairs[all_pairs["evaluated"] & all_pairs["d_s"].notna()].copy()
    priced["realized_return"] = np.where(priced["both_win"], priced["d_s"] - 1.0, -1.0)
    priced["market_implied_p"] = 1.0 / priced["d_s"]

    summary: dict = {
        "n_evaluated_priced_pairs": int(len(priced)),
        "n_days": int(priced["date"].nunique()) if len(priced) else 0,
        "overall": {},
        "by_pair_class": {},
        "value_subset_p_joint_gt_market_implied": {},
    }

    def block(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"n": 0}
        mean_ret, lo, hi = day_clustered_bootstrap_ci(frame, "realized_return")
        return {
            "n": int(len(frame)),
            "n_days": int(frame["date"].nunique()),
            "hit_rate_both_win": float(frame["both_win"].mean()),
            "mean_p_joint_model": float(frame["p_joint"].mean()),
            "mean_market_implied_p": float(frame["market_implied_p"].mean()),
            "model_overconfidence_vs_actual": float(frame["p_joint"].mean() - frame["both_win"].mean()),
            "model_overconfidence_vs_market": float(frame["p_joint"].mean() - frame["market_implied_p"].mean()),
            "mean_joint_ev_model_confidence_NOT_realized": float(frame["joint_ev"].mean()),
            "mean_realized_return_direct": float(frame["realized_return"].mean()),
            "day_clustered_bootstrap_mean_realized_return": mean_ret,
            "day_clustered_bootstrap_90pct_ci": [lo, hi],
        }

    summary["overall"] = block(priced)
    for cls, part in priced.groupby("pair_class"):
        summary["by_pair_class"][cls] = block(part)
    value_subset = priced[priced["p_joint"] > priced["market_implied_p"]]
    summary["value_subset_p_joint_gt_market_implied"] = block(value_subset)
    summary["value_subset_p_joint_gt_market_implied"]["caveat"] = (
        "This filter (p_joint > market-implied price) was constructed POST-HOC during "
        "exploratory analysis of this exact DEVELOPMENT-window result, not predeclared "
        "before looking. It is a HYPOTHESIS for a future frozen/SELECT-confirmed rule, "
        "not itself confirmed evidence -- reporting it without this caveat would violate "
        "the mission's multiplicity-control requirement."
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = run_multi_target_backtest(mode="broad")
    all_pairs.to_csv(OUTPUT_DIR / "multi_target_broad_pairs.csv", index=False)
    summary = summarize(all_pairs)
    with open(OUTPUT_DIR / "multi_target_broad_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
