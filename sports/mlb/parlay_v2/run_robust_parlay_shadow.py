from __future__ import annotations

"""Standalone research CLI for robust MLB two-leg parlay valuation.

Input JSON schema (minimal):
{
  "policy": { ... RobustParlayPolicy overrides ... },
  "pairs": [
    {
      "pair_id": "A||B",
      "combined_decimal_price": 2.25,
      "leg_1": {
        "leg_id": "A", "game_id": "g1", "mean_probability": 0.82,
        "decimal_price": 1.40, "effective_support": 300,
        "probability_samples": [ ... ], "pca_edge_certified": true
      },
      "leg_2": { ... },
      "dependency": {"mode": "INDEPENDENT", "support_count": 5000}
    }
  ]
}

The CLI is shadow/research only. It never emits production authorization.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from .robust_parlay_search import (
    JointDependencyEvidence,
    LegProbabilityEvidence,
    LPAParlaySearch,
    RobustParlayPolicy,
    value_two_leg_parlay,
)


def _leg(payload: dict[str, Any]) -> LegProbabilityEvidence:
    return LegProbabilityEvidence(
        leg_id=str(payload["leg_id"]),
        game_id=str(payload["game_id"]),
        mean_probability=float(payload["mean_probability"]),
        decimal_price=(None if payload.get("decimal_price") is None else float(payload["decimal_price"])),
        effective_support=int(payload.get("effective_support", 0)),
        probability_samples=tuple(float(x) for x in payload.get("probability_samples", [])),
        pca_edge_certified=bool(payload.get("pca_edge_certified", False)),
        market=payload.get("market"),
        team=payload.get("team"),
        book=payload.get("book"),
        sampling_source=str(payload.get("sampling_source", "UNKNOWN")),
    )


def _dependency(payload: dict[str, Any]) -> JointDependencyEvidence:
    return JointDependencyEvidence(
        mode=str(payload.get("mode", "UNSUPPORTED")),
        joint_probability_samples=tuple(float(x) for x in payload.get("joint_probability_samples", [])),
        common_world_outcomes=tuple(tuple(int(y) for y in x) for x in payload.get("common_world_outcomes", [])),
        support_count=int(payload.get("support_count", 0)),
        note=str(payload.get("note", "")),
    )


def run(input_json: Path, output_json: Path) -> dict[str, Any]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    policy = RobustParlayPolicy(**payload.get("policy", {}))
    valuations = []
    for row in payload.get("pairs", []):
        valuations.append(
            value_two_leg_parlay(
                str(row["pair_id"]),
                _leg(row["leg_1"]),
                _leg(row["leg_2"]),
                _dependency(row.get("dependency", {})),
                combined_decimal_price=(None if row.get("combined_decimal_price") is None else float(row["combined_decimal_price"])),
                policy=policy,
            )
        )

    certified_search = LPAParlaySearch(valuations, policy=policy, mode="certified")
    shadow_search = LPAParlaySearch(valuations, policy=policy, mode="shadow")
    result = {
        "system": "MLB_ROBUST_PARLAY_LPA_SHADOW_V1",
        "production_authorized": False,
        "note": "Research/shadow output only. LPA* selects among already-valued pairs; it does not estimate outcome probabilities.",
        "counts": {},
        "valuations": [v.as_dict() for v in valuations],
        "certified_search": certified_search.result().as_dict(),
        "shadow_search": shadow_search.result().as_dict(),
    }
    for value in valuations:
        result["counts"][value.classification] = result["counts"].get(value.classification, 0) + 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_json, args.output_json)


if __name__ == "__main__":
    main()
