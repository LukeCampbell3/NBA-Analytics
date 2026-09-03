"""AUC computation and slice-conditioned discriminative-power diagnostics
for the pair-observation ledger.

The negative-slope finding on the fitted beta calibrator
(pair_ledger_calibration.py) says the raw joint model's confidence has
negative correlation with actual outcome inside its operating range.
That single number is a mix. Slice by market_pair_type / same_game /
slate / price_bucket, and the mix separates: some slices have real
predictive signal (AUC > 0.5), others are noise (~0.5), others are
strongly inverted (~0.3). This module produces those numbers so an
operator can see WHERE the model deserves trust and where it does not.

AUC here is the standard rank-based ROC-AUC: probability that a
randomly-selected positive is scored higher than a randomly-selected
negative. Ties get half-credit (mid-rank). AUC == 0.5 is random;
1.0 is perfect; 0.0 is perfectly inverted; below 0.5 says the raw
predictions are anti-informative.

Read-only. Writes only under this subpackage. Consumed by
`slice_conditioned_calibrator.py`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEDGER = "sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl"
DEFAULT_REPORT = "sports/mlb/parlay_v2/promotion_coherence/reports/discriminative_power.json"


def _finite(value: Any) -> Optional[float]:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n != n or n in (float("inf"), float("-inf")):
        return None
    return n


def auc(pairs: Iterable[tuple[float, int]]) -> Optional[float]:
    """Standard rank-based ROC-AUC with mid-rank ties.

    Returns None when the input is degenerate (empty, only positives,
    only negatives). O(n log n).
    """
    pairs = [(p, y) for p, y in pairs if p is not None and y in (0, 1)]
    n = len(pairs)
    if n < 2:
        return None
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Rank ascending; equal values share mid-rank.
    indexed = sorted(enumerate(pairs), key=lambda ip: ip[1][0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1][0] == indexed[i][1][0]:
            j += 1
        mid = (i + j) / 2 + 1  # 1-indexed mid-rank
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = mid
        i = j + 1
    sum_pos_ranks = sum(ranks[k] for k in range(n) if pairs[k][1] == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass
class SliceDiscriminativePower:
    slice_key: str            # e.g. "market_pair_type=R|R"
    n: int
    n_positive: int
    hit_rate: Optional[float]
    auc: Optional[float]
    # A conservative one-sided call at n>=100 and AUC>0.53 (approx
    # 1-sigma for the null hypothesis AUC=0.5 at typical n). Labelled
    # explicitly rather than pretending it's a hypothesis test.
    positive_signal_flag: bool
    inverted_signal_flag: bool


def _classify(n: int, a: Optional[float]) -> tuple[bool, bool]:
    if a is None or n < 100:
        return (False, False)
    return (a > 0.53, a < 0.47)


def slice_pool(
    rows: list[dict[str, Any]],
    *,
    key_fn,
    key_label: str,
) -> list[SliceDiscriminativePower]:
    groups: dict[Any, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        p = _finite(r.get("predicted_joint_probability"))
        y = 1 if r.get("both_win") else 0
        if p is None:
            continue
        k = key_fn(r)
        groups[k].append((p, y))
    out: list[SliceDiscriminativePower] = []
    for k in sorted(groups, key=lambda x: (str(x))):
        pairs = groups[k]
        n = len(pairs)
        n_pos = sum(1 for _, y in pairs if y == 1)
        a = auc(pairs)
        pos, inv = _classify(n, a)
        out.append(SliceDiscriminativePower(
            slice_key=f"{key_label}={k}",
            n=n,
            n_positive=n_pos,
            hit_rate=(n_pos / n) if n else None,
            auc=a,
            positive_signal_flag=pos,
            inverted_signal_flag=inv,
        ))
    return out


@dataclass
class DiscriminativePowerReport:
    generated_at_utc: str
    ledger_path: str
    row_count: int
    global_slice: SliceDiscriminativePower
    by_market_pair_type: list[SliceDiscriminativePower]
    by_same_game: list[SliceDiscriminativePower]
    by_slate: list[SliceDiscriminativePower]
    by_price_bucket: list[SliceDiscriminativePower]
    positive_signal_slices: list[str] = field(default_factory=list)
    inverted_signal_slices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "ledger_path": self.ledger_path,
            "row_count": self.row_count,
            "global": asdict(self.global_slice),
            "by_market_pair_type": [asdict(x) for x in self.by_market_pair_type],
            "by_same_game": [asdict(x) for x in self.by_same_game],
            "by_slate": [asdict(x) for x in self.by_slate],
            "by_price_bucket": [asdict(x) for x in self.by_price_bucket],
            "positive_signal_slices": list(self.positive_signal_slices),
            "inverted_signal_slices": list(self.inverted_signal_slices),
        }


def _load_settled(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    return [json.loads(l) for l in open(ledger) if l.strip() and json.loads(l).get("settlement_status") == "settled"]


def build_report(*, ledger_path: Path) -> DiscriminativePowerReport:
    rows = _load_settled(ledger_path)
    all_pairs = [(_finite(r.get("predicted_joint_probability")),
                  1 if r.get("both_win") else 0) for r in rows]
    all_pairs = [(p, y) for p, y in all_pairs if p is not None]
    global_auc = auc(all_pairs)
    global_hit = sum(y for _, y in all_pairs) / len(all_pairs) if all_pairs else None
    pos, inv = _classify(len(all_pairs), global_auc)
    global_slice = SliceDiscriminativePower(
        slice_key="global",
        n=len(all_pairs),
        n_positive=sum(y for _, y in all_pairs),
        hit_rate=global_hit,
        auc=global_auc,
        positive_signal_flag=pos,
        inverted_signal_flag=inv,
    )
    by_market = slice_pool(rows, key_fn=lambda r: r.get("market_pair_type") or "UNKNOWN", key_label="market_pair_type")
    by_sg = slice_pool(rows, key_fn=lambda r: bool(r.get("same_game")), key_label="same_game")
    by_slate = slice_pool(rows, key_fn=lambda r: r.get("slate_id") or "UNKNOWN", key_label="slate_id")
    by_price = slice_pool(rows, key_fn=lambda r: r.get("price_bucket") or "UNKNOWN", key_label="price_bucket")

    positive: list[str] = []
    inverted: list[str] = []
    for group in [by_market, by_sg, by_price]:
        for s in group:
            if s.positive_signal_flag:
                positive.append(s.slice_key)
            if s.inverted_signal_flag:
                inverted.append(s.slice_key)

    return DiscriminativePowerReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        ledger_path=str(ledger_path),
        row_count=len(rows),
        global_slice=global_slice,
        by_market_pair_type=by_market,
        by_same_game=by_sg,
        by_slate=by_slate,
        by_price_bucket=by_price,
        positive_signal_slices=positive,
        inverted_signal_slices=inverted,
    )


def _print_slice_table(name: str, slices: list[SliceDiscriminativePower]) -> None:
    print(f"\n--- {name} ---")
    print(f"{'slice':<35} {'n':>6} {'hit':>6} {'AUC':>7} {'flags':>15}")
    for s in slices:
        hit = f"{s.hit_rate:.3f}" if s.hit_rate is not None else "  -  "
        a = f"{s.auc:.4f}" if s.auc is not None else "  n/a "
        flag = ("SIGNAL" if s.positive_signal_flag else ("INVERTED" if s.inverted_signal_flag else "flat"))
        print(f"{s.slice_key:<35} {s.n:>6} {hit:>6} {a:>7} {flag:>15}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Discriminative-power (AUC) diagnostics for the pair-observation ledger, sliced.")
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_REPORT)
    args = parser.parse_args()

    report = build_report(ledger_path=args.ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out}")

    g = report.global_slice
    print(f"\nGLOBAL: n={g.n}  hit_rate={g.hit_rate:.3f}  AUC={g.auc:.4f}  "
          f"({'POSITIVE SIGNAL' if g.positive_signal_flag else ('INVERTED SIGNAL' if g.inverted_signal_flag else 'flat / no signal')})")
    _print_slice_table("BY MARKET PAIR TYPE", report.by_market_pair_type)
    _print_slice_table("BY SAME_GAME", report.by_same_game)
    _print_slice_table("BY PRICE BUCKET", report.by_price_bucket)
    _print_slice_table("BY SLATE", report.by_slate)

    print(f"\nSLICES WITH POSITIVE SIGNAL (AUC > 0.53, n >= 100): {report.positive_signal_slices or 'none'}")
    print(f"SLICES WITH INVERTED SIGNAL (AUC < 0.47, n >= 100): {report.inverted_signal_slices or 'none'}")


if __name__ == "__main__":
    _cli()
