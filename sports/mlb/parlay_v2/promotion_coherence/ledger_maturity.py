"""Ledger-maturity monitor for the real pair-observation ledger.

The strict_dominance_over_baseline flag in the pair-ledger backtest
becomes decision-quality when the ledger has enough real (non-
synthetic) slates behind it. Today the real ledger has 4 slates -- too
thin for a serious tuning call, as BACKTEST_ANALYSIS.md already says.

This module reports maturity in a machine-checkable, deterministic
form so a later pipeline step can gate on it (or a test can pin the
current status). It never mutates the ledger, never touches the live
promotion path, and never invents a "readiness" number that isn't
derivable from the ledger itself.

Two thresholds are named explicitly:

    * DEFAULT_MIN_SLATES_FOR_DECISION_QUALITY: 10 -- the threshold from
      the promotion-coherence next-steps document. Below this, the
      strict-dominance floor is a data point, not a decision.
    * DEFAULT_MIN_ADMITTED_PAIRS_AT_DOMINANT_FLOOR: 500 -- an anti-
      overfit floor on how many pairs the dominant admitted subset
      needs to contain before its dominance is trustworthy. The
      backtest's own strict-dominance helper already enforces >=20
      pairs at the row level; 500 pairs is the stronger bar for
      believing a real slice-level readout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEDGER = "sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl"

DEFAULT_MIN_SLATES_FOR_DECISION_QUALITY = 10
DEFAULT_MIN_ADMITTED_PAIRS_AT_DOMINANT_FLOOR = 500


@dataclass
class LedgerMaturity:
    ledger_path: str
    row_count: int
    settled_row_count: int
    slates_covered: list[str]
    first_slate: Optional[str]
    last_slate: Optional[str]
    same_game_row_count: int
    cross_game_row_count: int
    rows_with_leg_1_no_vig: int
    rows_with_both_leg_no_vig: int
    min_slates_for_decision_quality: int
    slates_short_of_decision_quality: int
    decision_quality_ready: bool
    generated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_path": self.ledger_path,
            "row_count": self.row_count,
            "settled_row_count": self.settled_row_count,
            "slates_covered": self.slates_covered,
            "first_slate": self.first_slate,
            "last_slate": self.last_slate,
            "same_game_row_count": self.same_game_row_count,
            "cross_game_row_count": self.cross_game_row_count,
            "rows_with_leg_1_no_vig": self.rows_with_leg_1_no_vig,
            "rows_with_both_leg_no_vig": self.rows_with_both_leg_no_vig,
            "min_slates_for_decision_quality": self.min_slates_for_decision_quality,
            "slates_short_of_decision_quality": self.slates_short_of_decision_quality,
            "decision_quality_ready": self.decision_quality_ready,
            "generated_at_utc": self.generated_at_utc,
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_maturity(
    ledger_path: Path,
    *,
    min_slates: int = DEFAULT_MIN_SLATES_FOR_DECISION_QUALITY,
) -> LedgerMaturity:
    rows = _load_jsonl(ledger_path)
    settled = [r for r in rows if r.get("settlement_status") == "settled"]
    slates = sorted({r.get("slate_id") for r in settled if r.get("slate_id")})
    same_game = sum(1 for r in settled if r.get("same_game"))
    with_leg_1_no_vig = sum(
        1 for r in settled if r.get("leg_1_no_vig_market_probability") is not None
    )
    with_both_no_vig = sum(
        1 for r in settled
        if r.get("leg_1_no_vig_market_probability") is not None
        and r.get("leg_2_no_vig_market_probability") is not None
    )
    slates_short = max(0, min_slates - len(slates))
    return LedgerMaturity(
        ledger_path=str(ledger_path),
        row_count=len(rows),
        settled_row_count=len(settled),
        slates_covered=slates,
        first_slate=(slates[0] if slates else None),
        last_slate=(slates[-1] if slates else None),
        same_game_row_count=same_game,
        cross_game_row_count=len(settled) - same_game,
        rows_with_leg_1_no_vig=with_leg_1_no_vig,
        rows_with_both_leg_no_vig=with_both_no_vig,
        min_slates_for_decision_quality=min_slates,
        slates_short_of_decision_quality=slates_short,
        decision_quality_ready=(slates_short == 0),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def maturity_message(m: LedgerMaturity) -> str:
    lines = []
    lines.append(f"Real pair ledger: {m.settled_row_count} settled rows across {len(m.slates_covered)} slates")
    if m.first_slate and m.last_slate:
        lines.append(f"  slate window: {m.first_slate} -> {m.last_slate}")
    lines.append(f"  same-game: {m.same_game_row_count}, cross-game: {m.cross_game_row_count}")
    lines.append(
        f"  rows with per-leg no-vig capture: {m.rows_with_both_leg_no_vig} "
        f"(pair-ingest v1.1+ capture rate)"
    )
    if m.decision_quality_ready:
        lines.append(f"  DECISION-QUALITY READY (>= {m.min_slates_for_decision_quality} slates)")
    else:
        lines.append(
            f"  NOT YET decision-quality: need {m.slates_short_of_decision_quality} "
            f"more slate{'s' if m.slates_short_of_decision_quality != 1 else ''} "
            f"(target >= {m.min_slates_for_decision_quality})"
        )
    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Report readiness of the real pair-observation ledger for a decision-quality backtest.")
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / DEFAULT_LEDGER)
    parser.add_argument("--min-slates", type=int, default=DEFAULT_MIN_SLATES_FOR_DECISION_QUALITY)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    maturity = compute_maturity(args.ledger, min_slates=args.min_slates)
    print(maturity_message(maturity))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(maturity.to_dict(), indent=2, sort_keys=True, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _cli()
