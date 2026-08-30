#!/usr/bin/env python3
"""Rolling-origin functional backtest for the V4 shadow selector.

Every evaluation slate is scored by a V4 fit using strictly earlier settled
slates. Candidate rows are never treated as independent evidence; headline
results expose both pooled descriptive metrics and equal-slate summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import balanced_market_ensemble_v4 as v4  # noqa: E402
import balanced_ranking_v3 as v3  # noqa: E402


SCORE_FIELDS = (
    "balanced_probability",
    "market_probability",
    "base_ev",
    "v19_order_score",
    "v4_ensemble_probability",
    "v4_safe_ev",
)


def implied_price(probability: float) -> float:
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("market probability must be in (0, 1)")
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def exact_price_from_row(row: dict[str, Any]) -> float:
    if row.get("price") is not None:
        return float(row["price"])
    # Archived V3 rows retain exact-price EV even when their market
    # probability field is a no-vig/consensus diagnostic. Recover the exact
    # offered decimal price algebraically: EV = P * D - 1.
    if row.get("base_ev") is not None and float(row.get("balanced_probability") or 0) > 0:
        decimal = (float(row["base_ev"]) + 1.0) / float(row["balanced_probability"])
        if decimal > 1.0:
            return -100.0 / (decimal - 1.0) if decimal < 2.0 else 100.0 * (decimal - 1.0)
    return implied_price(float(row["market_probability"]))


def realized_profit(win: int, price: float) -> float:
    if not win:
        return -1.0
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def _log_loss(y: int, p: float) -> float:
    p = min(1 - 1e-12, max(1e-12, float(p)))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def score_walk_forward(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    materialized = v4._valid_rows(rows)
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_date[row["date"]].append(row)
    scored: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for slate_date in sorted(by_date):
        prior = [row for row in materialized if row["date"] < slate_date]
        if len({row["date"] for row in prior}) < v4.MIN_TRAINING_SLATES:
            continue
        fitted = v4.fit(prior, before_date=slate_date)
        fits.append({"date": slate_date, **v4.asdict(fitted)})
        for row in by_date[slate_date]:
            price = exact_price_from_row(row)
            result = v4.score({**row, "price": price}, fitted)
            scored.append(
                {
                    **row,
                    "price": price,
                    "realized_profit": realized_profit(int(row["win"]), price),
                    "v4_ensemble_probability": result.ensemble_probability,
                    "v4_safe_probability": result.safe_probability,
                    "v4_safe_ev": result.safe_ev,
                    "v4_probability_edge": result.probability_edge,
                    "v4_eligible": result.eligible,
                    "v4_reasons": list(result.reasons),
                }
            )
    return scored, fits


def _score_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)
    slates = []
    for slate_date, slate in sorted(by_date.items()):
        ordered = sorted(slate, key=lambda row: (-float(row[field]), str(row.get("candidate_id", ""))))
        auc, pairs = v3.pairwise_concordance([row[field] for row in slate], [row["win"] for row in slate])
        item: dict[str, Any] = {"date": slate_date, "rows": len(slate), "auc": auc, "pairs": pairs}
        for k in (1, 3, 5):
            top = ordered[: min(k, len(ordered))]
            item[f"top_{k}_hit_rate"] = statistics.fmean(row["win"] for row in top)
            item[f"top_{k}_roi"] = statistics.fmean(row["realized_profit"] for row in top)
        slates.append(item)
    defined_auc = [item["auc"] for item in slates if item["auc"] is not None]
    return {
        "field": field,
        "independent_slates": len(slates),
        "mean_slate_auc": statistics.fmean(defined_auc) if defined_auc else None,
        "pooled_auc_descriptive": v3.pairwise_concordance([row[field] for row in rows], [row["win"] for row in rows])[0] if rows else None,
        **{
            f"mean_slate_top_{k}_{metric}": statistics.fmean(item[f"top_{k}_{metric}"] for item in slates) if slates else None
            for k in (1, 3, 5)
            for metric in ("hit_rate", "roi")
        },
        "slates": slates,
    }


def build_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scored, fits = score_walk_forward(rows)
    selected = [row for row in scored if row["v4_eligible"]]
    probability_metrics = {}
    for field in ("balanced_probability", "market_probability", "v4_ensemble_probability", "v4_safe_probability"):
        if scored:
            probability_metrics[field] = {
                "brier": statistics.fmean((float(row[field]) - int(row["win"])) ** 2 for row in scored),
                "log_loss": statistics.fmean(_log_loss(int(row["win"]), float(row[field])) for row in scored),
                "mean_probability": statistics.fmean(float(row[field]) for row in scored),
                "hit_rate": statistics.fmean(int(row["win"]) for row in scored),
            }
    return {
        "version": v4.V4_VERSION,
        "spec_hash": v4.PREREGISTRATION_SPEC_HASH,
        "status": "FUNCTIONAL_BACKTEST_ONLY",
        "publication_authority": False,
        "evaluation_slates": sorted({row["date"] for row in scored}),
        "evaluation_rows": len(scored),
        "fits": fits,
        "ranking": {field: _score_summary(scored, field) for field in SCORE_FIELDS},
        "probability_metrics": probability_metrics,
        "selection": {
            "eligible_rows": len(selected),
            "coverage": len(selected) / len(scored) if scored else 0.0,
            "wins": sum(int(row["win"]) for row in selected),
            "hit_rate": statistics.fmean(int(row["win"]) for row in selected) if selected else None,
            "roi": statistics.fmean(float(row["realized_profit"]) for row in selected) if selected else None,
            "by_date": {
                date: {
                    "plays": len(day), "wins": sum(int(row["win"]) for row in day),
                    "roi": statistics.fmean(float(row["realized_profit"]) for row in day),
                }
                for date in sorted({row["date"] for row in selected})
                for day in [[row for row in selected if row["date"] == date]]
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, help="Optional list of settled V3-format rows; defaults to archived harvest.")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.input_json:
        rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    else:
        rows, errors, _ = v3.harvest_rows()
        if errors:
            print(f"warning: {len(errors)} archived slates failed to load", file=sys.stderr)
    report = build_report(rows)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
