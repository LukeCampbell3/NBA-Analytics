from __future__ import annotations

"""Raw pair-observation strata counts (mission section 11) -- accumulated
from day one, exposed for future research, and explicitly NOT used to
derive any action gate. joint_support stays OBSERVE_ONLY / UNESTABLISHED
regardless of how large these counts get -- see calibration/support.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PairStrata:
    n_pair_total: int
    n_settled: int
    n_cross_game: int
    n_same_game: int
    n_by_market_pair_type: dict
    n_by_line_pair_type: dict
    n_by_state_pair: dict
    n_by_price_bucket: dict

    sum_predicted_joint_probability: float
    sum_predicted_independence_probability: float
    sum_actual_both_win: int  # count of both_win=True among settled rows

    def as_dict(self) -> dict:
        return {
            "n_pair_total": self.n_pair_total,
            "n_settled": self.n_settled,
            "n_cross_game": self.n_cross_game,
            "n_same_game": self.n_same_game,
            "n_by_market_pair_type": dict(self.n_by_market_pair_type),
            "n_by_line_pair_type_sample": dict(list(self.n_by_line_pair_type.items())[:50]),  # line pair types are high-cardinality; cap for readability
            "n_by_state_pair": dict(self.n_by_state_pair),
            "n_by_price_bucket": dict(self.n_by_price_bucket),
            "sum_predicted_joint_probability": self.sum_predicted_joint_probability,
            "sum_predicted_independence_probability": self.sum_predicted_independence_probability,
            "sum_actual_both_win": self.sum_actual_both_win,
            # Diagnostics needed later for joint Brier/log-loss/calibration-slope
            # research (mission section 11) -- not computed here, just the raw
            # sums/counts that make them computable once enough data exists.
            "note": "raw counts/sums only -- no action gate is derived from this; joint_support remains OBSERVE_ONLY/UNESTABLISHED regardless of these values",
        }


def compute_pair_strata(rows: list[dict]) -> PairStrata:
    def bucket_counts(key: str) -> dict:
        counts: dict = {}
        for r in rows:
            counts[str(r.get(key))] = counts.get(str(r.get(key)), 0) + 1
        return counts

    settled = [r for r in rows if r.get("settlement_status") == "settled"]
    return PairStrata(
        n_pair_total=len(rows),
        n_settled=len(settled),
        n_cross_game=sum(1 for r in rows if not r.get("same_game")),
        n_same_game=sum(1 for r in rows if r.get("same_game")),
        n_by_market_pair_type=bucket_counts("market_pair_type"),
        n_by_line_pair_type=bucket_counts("line_pair_type"),
        n_by_state_pair=bucket_counts("state_bucket_pair"),
        n_by_price_bucket=bucket_counts("price_bucket"),
        sum_predicted_joint_probability=float(sum(float(r["predicted_joint_probability"]) for r in rows)),
        sum_predicted_independence_probability=float(sum(float(r["predicted_independence_probability"]) for r in rows)),
        sum_actual_both_win=sum(1 for r in settled if r.get("both_win")),
    )
