from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from evaluate_cutoff_meta_append import _research_feasible_mask_by_day


def test_research_feasible_mask_by_day_is_index_aligned_series() -> None:
    pool = pd.DataFrame(
        [
            {"run_date": "2026-05-24", "agreement_count": 1, "corr_score": 0.10},
            {"run_date": "2026-05-24", "agreement_count": 1, "corr_score": 0.30},
            {"run_date": "2026-05-25", "agreement_count": 0, "corr_score": 0.05},
            {"run_date": "2026-05-25", "agreement_count": 1, "corr_score": 0.20},
        ],
        index=[10, 11, 20, 21],
    )

    mask = _research_feasible_mask_by_day(
        pool,
        research_pool_min_agreement=1.0,
        research_corr_mode="percentile",
        research_corr_threshold=0.5,
    )

    assert isinstance(mask, pd.Series)
    assert mask.index.tolist() == [10, 11, 20, 21]
    assert mask.tolist() == [True, False, False, False]
