from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from report_precision_pool_83_proof import _summarize, wilson_lower_bound


class Args:
    target_hit_rate = 0.83
    confidence = 0.95
    min_resolved_plays = 30
    min_resolved_days = 5


def test_wilson_lower_bound_requires_more_than_raw_hit_rate() -> None:
    assert 10 / 12 >= 0.83
    assert wilson_lower_bound(10, 12, confidence=0.95) < 0.83


def test_precision_pool_proof_marks_small_sample_as_not_proven() -> None:
    rows = pd.DataFrame(
        {
            "__run_date__": pd.to_datetime(["2026-04-24"] * 12),
            "__result__": ["win"] * 10 + ["loss"] * 2,
            "__prob__": [0.84] * 12,
        }
    )

    summary = _summarize(rows, Args(), "overall")

    assert summary["hit_rate"] >= 0.83
    assert summary["status"] == "insufficient_sample"
    assert summary["wilson_lower_bound"] < 0.83


def test_precision_pool_proof_can_pass_with_large_clean_sample() -> None:
    rows = pd.DataFrame(
        {
            "__run_date__": pd.to_datetime([f"2026-04-{day:02d}" for day in range(1, 11) for _ in range(10)]),
            "__result__": ["win"] * 92 + ["loss"] * 8,
            "__prob__": [0.88] * 100,
        }
    )

    summary = _summarize(rows, Args(), "overall")

    assert summary["status"] == "proven_at_confidence"
    assert summary["wilson_lower_bound"] >= 0.83
