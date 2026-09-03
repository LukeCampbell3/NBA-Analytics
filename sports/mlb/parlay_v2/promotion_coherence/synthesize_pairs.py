"""Build a synthetic cross-game pair-observation ledger from real
settled singles.

The real pair-observation ledger this repo carries today has 3,120 rows
across 4 slates -- too thin for `strict_dominance_over_baseline` in the
backtest to be decision-quality. The singles calibration ledger at
sports/mlb/parlay_v2/calibration/reports/calibration_ledger.jsonl has
9,051 settled rows across 25 slates. Each row already carries the exact
fields the promotion-margin rule uses: `quote_decimal`,
`predictive_probability_if_available`, `actual_outcome` (1.0 / 0.0),
`settlement_status` ("win" / "loss"), `slate_id`, `game_id`.

This module cross-joins settled singles WITHIN each slate but only
across DIFFERENT games (the same cross-game constraint the real pair
ledger imposes) to produce synthetic pair observations that can be fed
into the same backtest.

Guardrails (loud and named so nothing is silent):

    * Deterministic. Given the same singles ledger, the same synthetic
      output. No randomness anywhere; sampling uses stable sorts on
      observation_id.
    * Per-slate cap. Uncapped cross-joins would explode to ~2 million
      pairs and over-represent any hot single. The cap defaults to
      DEFAULT_PAIRS_PER_SLATE_CAP (800), matching the real pair
      ledger's per-slate scale; the actual value used is written into
      the ledger metadata AND every row's `synthetic_ledger_version`.
    * Every row is stamped `is_synthetic: true`, `source: "singles_
      calibration_ledger"`, `pair_pool: "SYNTHETIC_CROSS_GAME_FROM_
      SINGLES"`. A backtest reader that treats these rows as real
      pair-ledger evidence is reading them wrong; the flags exist so it
      can't be an accident.
    * Independence-assumed joint probability, same as the real pair
      ledger uses today (predicted_independence_probability ==
      predicted_joint_probability). Documented, not hidden.

The synthetic ledger is a separate file
(`reports/synthetic_cross_game_pair_ledger.jsonl`) and is never mixed
with the real pair-observation ledger on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SINGLES_LEDGER = "sports/mlb/parlay_v2/calibration/reports/calibration_ledger.jsonl"
DEFAULT_SYNTHETIC_OUT = "sports/mlb/parlay_v2/promotion_coherence/reports/synthetic_cross_game_pair_ledger.jsonl"

SYNTHETIC_LEDGER_VERSION = "SYNTHETIC_CROSS_GAME_PAIR_V1"
DEFAULT_PAIRS_PER_SLATE_CAP = 800
DEFAULT_MAX_SINGLES_PER_GAME = 6  # bounds per-game contribution before cross-joining


def _row_ok_for_synthesis(row: dict[str, Any]) -> bool:
    """Only fully-settled singles with a real quote and a real
    predictive probability qualify. Anything missing is silently
    dropped: a synthetic pair with a fabricated leg would be worse than
    fewer rows."""
    if row.get("settlement_status") not in ("win", "loss"):
        return False
    if row.get("actual_outcome") not in (0, 1, 0.0, 1.0):
        return False
    quote = row.get("quote_decimal")
    if not isinstance(quote, (int, float)) or not (quote > 1.0):
        return False
    p_hat = row.get("predictive_probability_if_available")
    if not isinstance(p_hat, (int, float)) or not (0.0 < p_hat < 1.0):
        return False
    if not row.get("game_id"):
        return False
    return True


def _load_singles(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_by_slate_and_game(
    rows: Iterable[dict[str, Any]],
    *,
    max_singles_per_game: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    slate_game: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if not _row_ok_for_synthesis(row):
            continue
        slate = row.get("slate_id")
        game = row.get("game_id")
        slate_game.setdefault(slate, {}).setdefault(game, []).append(row)

    # Deterministic per-game truncation. Sort by observation_id so the
    # per-game keep-order does not depend on file order.
    for slate, games in slate_game.items():
        for game, singles in games.items():
            singles.sort(key=lambda r: r.get("observation_id") or r.get("row_hash") or "")
            if len(singles) > max_singles_per_game:
                games[game] = singles[:max_singles_per_game]
    return slate_game


def _pair_id(leg_1: dict[str, Any], leg_2: dict[str, Any]) -> str:
    """Order-independent pair id -- (leg_a, leg_b) and (leg_b, leg_a)
    hash to the same string, so no synthetic pair is emitted twice."""
    a = leg_1.get("observation_id") or leg_1.get("row_hash") or ""
    b = leg_2.get("observation_id") or leg_2.get("row_hash") or ""
    lo, hi = sorted((a, b))
    return hashlib.sha256(f"{lo}|{hi}".encode()).hexdigest()


def _make_pair_row(leg_1: dict[str, Any], leg_2: dict[str, Any]) -> dict[str, Any]:
    p1 = float(leg_1["predictive_probability_if_available"])
    p2 = float(leg_2["predictive_probability_if_available"])
    price_1 = float(leg_1["quote_decimal"])
    price_2 = float(leg_2["quote_decimal"])
    win_1 = int(leg_1["actual_outcome"])
    win_2 = int(leg_2["actual_outcome"])

    predicted_joint = p1 * p2
    quoted_pair_price = price_1 * price_2
    both_win = bool(win_1 == 1 and win_2 == 1)
    actual_pair_return = (quoted_pair_price - 1.0) if both_win else -1.0

    slate = leg_1.get("slate_id")
    leg_1_event_id = (f"{leg_1.get('player_id','')}|{leg_1.get('game_id','')}|"
                      f"{leg_1.get('market_bucket','')}|{leg_1.get('side','')}|{leg_1.get('line','')}")
    leg_2_event_id = (f"{leg_2.get('player_id','')}|{leg_2.get('game_id','')}|"
                      f"{leg_2.get('market_bucket','')}|{leg_2.get('side','')}|{leg_2.get('line','')}")
    market_pair_type = f"{leg_1.get('market_bucket','')}|{leg_2.get('market_bucket','')}"
    line_pair_type = (f"{leg_1.get('market_bucket','')}|{leg_1.get('side','')}|{leg_1.get('line','')}"
                      f"__{leg_2.get('market_bucket','')}|{leg_2.get('side','')}|{leg_2.get('line','')}")

    return {
        # -- fields the real pair-observation schema also exposes,
        # populated with real values wherever a real value exists:
        "actual_pair_return": actual_pair_return,
        "both_win": both_win,
        "leg_1_event_id": leg_1_event_id,
        "leg_2_event_id": leg_2_event_id,
        "leg_1_result": win_1,
        "leg_2_result": win_2,
        "predicted_independence_probability": predicted_joint,
        "predicted_joint_probability": predicted_joint,
        "quoted_pair_price": quoted_pair_price,
        "market_pair_type": market_pair_type,
        "line_pair_type": line_pair_type,
        "same_game": False,   # cross-game by construction
        "same_team": False,
        "settlement_status": "settled",
        "slate_id": slate,
        "pair_id": _pair_id(leg_1, leg_2),
        # -- extra per-leg model probabilities. These are the exact
        # values a future v2 pair schema would carry natively (item 2 of
        # the promotion-coherence next steps). Populated here on every
        # synthetic row so the market-disagreement work can consume them
        # directly.
        "leg_1_model_probability": p1,
        "leg_2_model_probability": p2,
        "leg_1_price": price_1,
        "leg_2_price": price_2,
        # -- synthetic-provenance markers. Named so they can never be
        # accidentally treated as real pair-observation rows.
        "is_synthetic": True,
        "synthetic_ledger_version": SYNTHETIC_LEDGER_VERSION,
        "synthetic_source": "singles_calibration_ledger",
        "pair_pool": "SYNTHETIC_CROSS_GAME_FROM_SINGLES",
    }


def _emit_pairs_for_slate(
    slate: str,
    games: dict[str, list[dict[str, Any]]],
    *,
    pairs_cap: int,
) -> list[dict[str, Any]]:
    """Deterministically enumerate cross-game pairs, cap at `pairs_cap`.

    Enumeration order: game pairs sorted by (game_id_a, game_id_b), and
    within each game pair, singles sorted by observation_id -- the same
    stable order used to truncate per-game contributions.
    """
    game_ids = sorted(games.keys())
    out: list[dict[str, Any]] = []
    for i in range(len(game_ids)):
        if len(out) >= pairs_cap:
            break
        for j in range(i + 1, len(game_ids)):
            if len(out) >= pairs_cap:
                break
            singles_a = games[game_ids[i]]
            singles_b = games[game_ids[j]]
            for la in singles_a:
                if len(out) >= pairs_cap:
                    break
                for lb in singles_b:
                    if len(out) >= pairs_cap:
                        break
                    out.append(_make_pair_row(la, lb))
    return out


@dataclass
class SynthesisMetadata:
    generated_at_utc: str
    synthetic_ledger_version: str
    source_singles_ledger: str
    source_singles_row_count: int
    source_settled_singles_count: int
    slates_covered: list[str]
    max_singles_per_game: int
    pairs_per_slate_cap: int
    total_synthetic_pairs: int


def synthesize_pair_ledger(
    *,
    singles_path: Path,
    max_singles_per_game: int = DEFAULT_MAX_SINGLES_PER_GAME,
    pairs_per_slate_cap: int = DEFAULT_PAIRS_PER_SLATE_CAP,
) -> tuple[list[dict[str, Any]], SynthesisMetadata]:
    singles = _load_singles(singles_path)
    slate_game = _group_by_slate_and_game(singles, max_singles_per_game=max_singles_per_game)
    settled_count = sum(1 for r in singles if _row_ok_for_synthesis(r))

    pairs: list[dict[str, Any]] = []
    slates_covered = sorted(slate_game.keys())
    for slate in slates_covered:
        pairs.extend(_emit_pairs_for_slate(slate, slate_game[slate], pairs_cap=pairs_per_slate_cap))

    meta = SynthesisMetadata(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        synthetic_ledger_version=SYNTHETIC_LEDGER_VERSION,
        source_singles_ledger=str(singles_path),
        source_singles_row_count=len(singles),
        source_settled_singles_count=settled_count,
        slates_covered=slates_covered,
        max_singles_per_game=max_singles_per_game,
        pairs_per_slate_cap=pairs_per_slate_cap,
        total_synthetic_pairs=len(pairs),
    )
    return pairs, meta


def write_synthetic_ledger(pairs: Iterable[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in pairs:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Synthesize a cross-game pair-observation ledger from settled singles.")
    parser.add_argument("--singles", type=Path, default=REPO_ROOT / DEFAULT_SINGLES_LEDGER)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_SYNTHETIC_OUT)
    parser.add_argument("--max-singles-per-game", type=int, default=DEFAULT_MAX_SINGLES_PER_GAME)
    parser.add_argument("--pairs-per-slate-cap", type=int, default=DEFAULT_PAIRS_PER_SLATE_CAP)
    args = parser.parse_args()

    pairs, meta = synthesize_pair_ledger(
        singles_path=args.singles,
        max_singles_per_game=args.max_singles_per_game,
        pairs_per_slate_cap=args.pairs_per_slate_cap,
    )
    write_synthetic_ledger(pairs, args.out)

    meta_path = args.out.with_name(args.out.stem + "_metadata.json")
    meta_path.write_text(json.dumps(meta.__dict__, indent=2, sort_keys=True, default=str))

    print(f"wrote {args.out} (pairs: {meta.total_synthetic_pairs})")
    print(f"wrote {meta_path}")
    print(f"source: {meta.source_singles_ledger}  "
          f"({meta.source_settled_singles_count}/{meta.source_singles_row_count} settled)")
    print(f"slates covered: {len(meta.slates_covered)}  "
          f"cap: {meta.pairs_per_slate_cap} pairs/slate  "
          f"max singles/game: {meta.max_singles_per_game}")


if __name__ == "__main__":
    _cli()
