from __future__ import annotations

"""Chronological day-grouped evaluation harness.

Primary endpoint: does the top of a ranking concentrate wins above the
eligible-pool base rate? AUC/log-loss are reported only as secondary
diagnostics -- a ranker that improves AUC without concentrating wins at
rank 1-2 is not considered promising here.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import beta

from .chronological_cv import Fold, split


@dataclass(frozen=True)
class DailySlice:
    date: str
    n: int
    pool_hit_rate: float
    top1_hit: int | None
    top2_hit_rate: float | None
    top_decile_hit_rate: float | None
    top_quintile_hit_rate: float | None


def _rank_day(day: pd.DataFrame, score_col: str) -> pd.DataFrame:
    # Deterministic tie-break: lower rmse first, then player name, so ties
    # never depend on row insertion order.
    return day.sort_values([score_col, "rmse", "player"], ascending=[False, True, True]).reset_index(drop=True)


def score_one_day(day: pd.DataFrame, score_col: str) -> DailySlice:
    ranked = _rank_day(day, score_col)
    n = len(ranked)
    pool_hit_rate = float(ranked["win"].mean())
    top1_hit = int(ranked["win"].iloc[0]) if n >= 1 else None
    top2_hit_rate = float(ranked["win"].iloc[:2].mean()) if n >= 2 else None
    decile_k = max(1, n // 10)
    quintile_k = max(1, n // 5)
    top_decile_hit_rate = float(ranked["win"].iloc[:decile_k].mean()) if n >= 10 else None
    top_quintile_hit_rate = float(ranked["win"].iloc[:quintile_k].mean()) if n >= 5 else None
    return DailySlice(
        date=str(ranked["date"].iloc[0]) if n else "",
        n=n,
        pool_hit_rate=pool_hit_rate,
        top1_hit=top1_hit,
        top2_hit_rate=top2_hit_rate,
        top_decile_hit_rate=top_decile_hit_rate,
        top_quintile_hit_rate=top_quintile_hit_rate,
    )


@dataclass
class ChronologicalReport:
    score_name: str
    per_fold: list[DailySlice] = field(default_factory=list)

    @property
    def n_dates(self) -> int:
        return len(self.per_fold)

    def pooled(self) -> dict:
        top1 = [s.top1_hit for s in self.per_fold if s.top1_hit is not None]
        top2 = [s.top2_hit_rate for s in self.per_fold if s.top2_hit_rate is not None]
        decile = [s.top_decile_hit_rate for s in self.per_fold if s.top_decile_hit_rate is not None]
        quintile = [s.top_quintile_hit_rate for s in self.per_fold if s.top_quintile_hit_rate is not None]
        pool = [s.pool_hit_rate for s in self.per_fold]
        n_total = sum(s.n for s in self.per_fold)
        return {
            "score_name": self.score_name,
            "n_dates": self.n_dates,
            "n_rows_total": n_total,
            "pool_hit_rate_mean_of_days": float(np.mean(pool)) if pool else float("nan"),
            "top1_hit_rate": float(np.mean(top1)) if top1 else float("nan"),
            "top1_n_dates": len(top1),
            "top2_hit_rate": float(np.mean(top2)) if top2 else float("nan"),
            "top2_n_dates": len(top2),
            "top_decile_hit_rate": float(np.mean(decile)) if decile else float("nan"),
            "top_decile_n_dates": len(decile),
            "top_quintile_hit_rate": float(np.mean(quintile)) if quintile else float("nan"),
            "top_quintile_n_dates": len(quintile),
            "top1_lift_vs_pool": (
                float(np.mean(top1) - np.mean(pool)) if top1 and pool else float("nan")
            ),
            "top2_lift_vs_pool": (
                float(np.mean(top2) - np.mean(pool)) if top2 and pool else float("nan")
            ),
        }

    def day_clustered_bootstrap_ci(self, metric: str = "top1", n_boot: int = 20000, seed: int = 20260821):
        """95% CI via resampling DATES (the near-independent unit), not rows."""
        values = [s.top1_hit for s in self.per_fold] if metric == "top1" else [s.top2_hit_rate for s in self.per_fold]
        values = [v for v in values if v is not None]
        if len(values) < 3:
            return {"lower": float("nan"), "upper": float("nan"), "n_dates": len(values)}
        arr = np.array(values, dtype=float)
        rng = np.random.default_rng(seed)
        boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
        return {
            "lower": float(np.quantile(boots, 0.025)),
            "upper": float(np.quantile(boots, 0.975)),
            "one_sided_95_lower": float(np.quantile(boots, 0.05)),
            "n_dates": len(values),
        }


def evaluate_score_chronologically(
    frame: pd.DataFrame, score_col: str, folds: list[Fold], score_name: str | None = None
) -> ChronologicalReport:
    report = ChronologicalReport(score_name=score_name or score_col)
    for fold in folds:
        _, val = split(frame, fold)
        if val.empty:
            continue
        report.per_fold.append(score_one_day(val, score_col))
    return report


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    if n == 0:
        return float("nan")
    return float(beta.ppf(alpha, k, n - k + 1)) if k > 0 else 0.0
