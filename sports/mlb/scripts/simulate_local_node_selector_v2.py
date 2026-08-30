#!/usr/bin/env python3
"""Simulate operating characteristics for the shadow local-node v2 test.

This is a design simulator, not an MLB outcome optimizer. It never reads the
August archive. Worlds are defined by a true calibration residual, a shared
slate shock, independent slates, and multiple propositions within each slate.
The output measures boundary false recovery, power, confidence-bound coverage,
and the approximate slate count needed for a desired effect.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass

import numpy as np

from local_node_selector_v2 import _student_t_critical


@dataclass(frozen=True)
class SimulationResult:
    slates: int
    rows_per_slate: int
    true_residual: float
    slate_shock_sd: float
    confidence: float
    practical_residual_floor: float
    trials: int
    acceptance_rate: float
    lcb_coverage: float
    mean_estimated_residual: float
    mean_lcb: float


def simulate_world(
    *,
    rng: np.random.Generator,
    trials: int,
    slates: int,
    rows_per_slate: int,
    true_residual: float,
    slate_shock_sd: float,
    confidence: float,
    practical_residual_floor: float = 0.02,
) -> SimulationResult:
    balanced = np.clip(rng.normal(0.56, 0.04, size=(trials, slates)), 0.42, 0.72)
    slate_shock = rng.normal(0.0, slate_shock_sd, size=(trials, slates))
    true_probability = np.clip(balanced + true_residual + slate_shock, 0.01, 0.99)
    wins = rng.binomial(rows_per_slate, true_probability)
    slate_residual = wins / rows_per_slate - balanced
    estimated = slate_residual.mean(axis=1)
    standard_error = slate_residual.std(axis=1, ddof=1) / math.sqrt(slates)
    lcb = estimated - _student_t_critical(confidence, slates - 1) * standard_error
    return SimulationResult(
        slates=slates,
        rows_per_slate=rows_per_slate,
        true_residual=true_residual,
        slate_shock_sd=slate_shock_sd,
        confidence=confidence,
        practical_residual_floor=practical_residual_floor,
        trials=trials,
        acceptance_rate=float(np.mean(lcb > practical_residual_floor)),
        lcb_coverage=float(np.mean(lcb <= true_residual)),
        mean_estimated_residual=float(estimated.mean()),
        mean_lcb=float(lcb.mean()),
    )


def approximate_required_slates(
    *,
    true_residual: float,
    practical_residual_floor: float,
    rows_per_slate: int,
    slate_shock_sd: float,
    confidence: float,
    desired_power: float = 0.80,
) -> int | None:
    gap = true_residual - practical_residual_floor
    if gap <= 0:
        return None
    # Conservative Bernoulli variance uses p(1-p) <= .25. Slate shocks remain
    # after within-slate averaging and therefore cannot be divided by row count.
    slate_sd = math.sqrt(0.25 / rows_per_slate + slate_shock_sd**2)
    z_confidence = statistics.NormalDist().inv_cdf(confidence)
    z_power = statistics.NormalDist().inv_cdf(desired_power)
    return math.ceil(((z_confidence + z_power) * slate_sd / gap) ** 2)


def run_design(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    worlds = []
    for slates in args.slates:
        for rows in args.rows_per_slate:
            for residual in args.true_residuals:
                result = simulate_world(
                    rng=rng,
                    trials=args.trials,
                    slates=slates,
                    rows_per_slate=rows,
                    true_residual=residual,
                    slate_shock_sd=args.slate_shock_sd,
                    confidence=args.confidence,
                    practical_residual_floor=args.practical_residual_floor,
                )
                worlds.append(asdict(result))
    sample_sizes = [
        {
            "true_residual": residual,
            "rows_per_slate": rows,
            "unadjusted_required_slates": approximate_required_slates(
                true_residual=residual,
                practical_residual_floor=args.practical_residual_floor,
                rows_per_slate=rows,
                slate_shock_sd=args.slate_shock_sd,
                confidence=args.confidence,
            ),
            "twenty_hypothesis_bonferroni_required_slates": approximate_required_slates(
                true_residual=residual,
                practical_residual_floor=args.practical_residual_floor,
                rows_per_slate=rows,
                slate_shock_sd=args.slate_shock_sd,
                confidence=1.0 - 0.05 / 20,
            ),
        }
        for residual in args.true_residuals
        if residual > args.practical_residual_floor
        for rows in args.rows_per_slate
    ]
    return {
        "design": {
            "archive_outcomes_used": False,
            "selector_version": "local_node_selector_v2_shadow_simulation",
            "seed": args.seed,
            "trials_per_world": args.trials,
            "slate_shock_sd": args.slate_shock_sd,
            "confidence": args.confidence,
            "practical_residual_floor": args.practical_residual_floor,
        },
        "worlds": worlds,
        "sample_size_theory": sample_sizes,
        "multiplicity_theory": {
            "uncorrected_family_false_positive_if_independent": {
                str(count): 1.0 - (1.0 - 0.05) ** count for count in (1, 5, 20, 100)
            },
            "bonferroni_familywise_upper_bound": 0.05,
            "note": "Candidate regions overlap, but dependence does not make uncorrected scanning valid; simultaneous coverage is retained.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--slates", type=int, nargs="+", default=[8, 15, 20, 30, 50])
    parser.add_argument("--rows-per-slate", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--true-residuals", type=float, nargs="+", default=[0.02, 0.06, 0.10])
    parser.add_argument("--slate-shock-sd", type=float, default=0.08)
    parser.add_argument("--confidence", type=float, default=0.975)
    parser.add_argument("--practical-residual-floor", type=float, default=0.02)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.dumps(run_design(args), indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

