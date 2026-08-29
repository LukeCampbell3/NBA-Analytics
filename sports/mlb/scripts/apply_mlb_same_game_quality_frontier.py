#!/usr/bin/env python3
"""Apply the primary same-game probability/value frontier to a publication.

The raw same-game simulator still writes every real cross-market candidate so
nothing is lost for research.  This post-selection step moves candidates that
fail the selector's *existing* joint-probability/edge/value thresholds into a
research-only list before the normal frontend build.  The public primary SGP
card therefore abstains when the only available positions are low-hit,
high-synthetic-EV research candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PARLAY_ROOT = REPO_ROOT / "sports" / "mlb" / "parlay_v2"
if str(PARLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(PARLAY_ROOT))

import parlay_quality_frontier as quality  # noqa: E402

DEFAULT_PAYLOAD = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "same_game_predictions.json"


def apply_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality.apply_same_game_probability_frontier(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = apply_file(args.payload)
    frontier = payload.get("quality_frontier") or {}
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "policy": frontier.get("policy"),
                "original_candidate_count": frontier.get("original_candidate_count", 0),
                "primary_candidate_count": frontier.get("primary_candidate_count", 0),
                "research_only_candidate_count": frontier.get("research_only_candidate_count", 0),
                "decision": frontier.get("decision"),
                "written": str(args.payload),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
