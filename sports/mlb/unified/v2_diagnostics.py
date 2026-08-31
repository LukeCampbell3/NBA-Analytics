from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values)/len(values) if values else None


def _graded(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("outcome") in {0, 1} and row.get("usable_probability") is not None]


def uncertainty_diagnostics(rows: Iterable[dict[str, Any]], bins: int = 4) -> dict[str, Any]:
    graded = sorted(_graded(rows), key=lambda row: float(row.get("uncertainty", math.inf)))
    if len(graded) < bins*5 or any(row.get("uncertainty") is None for row in graded):
        return {"state": "INSUFFICIENT_PROSPECTIVE_EVIDENCE", "quantiles": [], "monotonic_degradation": None}
    parts = [graded[index::bins] for index in range(bins)]
    result = []
    for index, part in enumerate(parts, 1):
        result.append({
            "quantile": index, "count": len(part),
            "mean_uncertainty": _mean(float(row["uncertainty"]) for row in part),
            "absolute_probability_error": _mean(abs(float(row["outcome"])-float(row["usable_probability"])) for row in part),
            "brier": _mean((float(row["outcome"])-float(row["usable_probability"]))**2 for row in part),
            "mean_return": _mean(float(row["realized_return"]) for row in part if row.get("realized_return") is not None),
        })
    errors = [row["absolute_probability_error"] for row in result]
    return {"state": "DIAGNOSTIC", "quantiles": result,
            "monotonic_degradation": all(a <= b for a, b in zip(errors, errors[1:]))}


def rank_performance(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _graded(rows):
        rank = row.get("ranking_position")
        key = str(rank) if rank in {1, 2, 3} else "4+"
        groups[key].append(row)
    return {key: {"count": len(part), "hit_rate": _mean(float(row["outcome"]) for row in part),
                  "roi": _mean(float(row["realized_return"]) for row in part if row.get("realized_return") is not None)}
            for key, part in sorted(groups.items())}


def top_k_performance(rows: Iterable[dict[str, Any]], ks: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    graded = _graded(rows)
    slates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in graded:
        slates[str(row.get("slate_id"))].append(row)
    output = {}
    for k in ks:
        chosen = [row for slate in slates.values()
                  for row in sorted(slate, key=lambda item: int(item.get("ranking_position") or 10**9))[:k]]
        output[f"top_{k}"] = {"selections": len(chosen),
                              "hit_rate": _mean(float(row["outcome"]) for row in chosen),
                              "roi": _mean(float(row["realized_return"]) for row in chosen if row.get("realized_return") is not None)}
    output["all_admissible"] = {"selections": len(graded),
                                "hit_rate": _mean(float(row["outcome"]) for row in graded),
                                "roi": _mean(float(row["realized_return"]) for row in graded if row.get("realized_return") is not None)}
    return output


def boundary_diagnostic(rows: Iterable[dict[str, Any]], field: str, threshold: float, epsilon: float) -> dict[str, Any]:
    graded = _graded(rows)
    below = [row for row in graded if row.get(field) is not None and threshold-epsilon <= float(row[field]) < threshold]
    above = [row for row in graded if row.get(field) is not None and threshold <= float(row[field]) <= threshold+epsilon]
    summarize = lambda part: {"count": len(part), "hit_rate": _mean(float(row["outcome"]) for row in part),
                              "roi": _mean(float(row["realized_return"]) for row in part if row.get("realized_return") is not None)}
    return {"field": field, "threshold": threshold, "epsilon": epsilon,
            "just_below": summarize(below), "just_above": summarize(above),
            "state": "DIAGNOSTIC" if len(below) >= 20 and len(above) >= 20 else "INSUFFICIENT_PROSPECTIVE_EVIDENCE"}
