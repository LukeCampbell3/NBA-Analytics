"""Replay recorded MLB parlay slates through the coherent-promotion gate.

Reads every `daily_predictions.json` under `sports/mlb/web/data/history/
runs/*/*/` plus the currently-published `sports/mlb/web/data/daily_
predictions.json`, computes the coherent decision for each, joins with
`sports/mlb/data/predictions/unified/historical_settlements.json` where
graded legs exist, and emits a JSON report to
`sports/mlb/parlay_v2/promotion_coherence/reports/latest_shadow_report.json`.

Reads only. Writes only under this subpackage. Nothing here touches a
selector, a run script, the frontend payload, or the live publication
path. A pipeline that never invokes this script sees no change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .promotion_confidence import (
    CoherentPromotionDecision,
    PromotionPenalties,
    PromotionThresholds,
    decide_coherent_promotion,
    default_thresholds,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HISTORY_GLOB = "sports/mlb/web/data/history/runs/*/*/daily_predictions.json"
DEFAULT_LIVE_PAYLOAD = "sports/mlb/web/data/daily_predictions.json"
DEFAULT_SETTLEMENTS = "sports/mlb/data/predictions/unified/historical_settlements.json"
DEFAULT_REPORT_PATH = "sports/mlb/parlay_v2/promotion_coherence/reports/latest_shadow_report.json"


# --- settlement joining -------------------------------------------------

_MARKET_ALIASES = {
    "TB": "TB",
    "H": "H",
    "HR": "HR",
    "R": "R",
    "RBI": "RBI",
    "SO": "SO",
    "K": "SO",
}


def _normalize_player_id(name_or_id: str) -> str:
    if not name_or_id:
        return ""
    cleaned = name_or_id.strip().lower().replace(" ", "_").replace(".", "").replace("'", "")
    return cleaned


def _leg_settlement_key(leg: Mapping[str, Any]) -> tuple[str, str, str, float]:
    player_id = leg.get("player_id") or leg.get("player_name") or leg.get("player") or ""
    market = leg.get("target") or leg.get("market") or ""
    side = leg.get("side") or ""
    line = leg.get("line")
    try:
        line_f = float(line)
    except (TypeError, ValueError):
        line_f = float("nan")
    return (
        _normalize_player_id(str(player_id)),
        _MARKET_ALIASES.get(str(market).upper(), str(market).upper()),
        str(side).upper(),
        line_f,
    )


def _load_settlements_index(path: Path) -> dict[tuple[str, str, str, str, float], str]:
    """Return {(date, player_id, market, side, line): settlement}.

    Empty when the file is missing -- shadow report should still run and
    just leave outcomes as `unknown`.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    settlements = raw.get("settlements") if isinstance(raw, dict) else raw
    index: dict[tuple[str, str, str, str, float], str] = {}
    for record in settlements or []:
        try:
            line = float(record.get("line"))
        except (TypeError, ValueError):
            continue
        key = (
            str(record.get("event_date") or ""),
            _normalize_player_id(str(record.get("player_id") or "")),
            _MARKET_ALIASES.get(str(record.get("market") or "").upper(), str(record.get("market") or "").upper()),
            str(record.get("side") or "").upper(),
            line,
        )
        index[key] = str(record.get("settlement") or "unknown")
    return index


def _grade_selected(
    payload: Mapping[str, Any],
    settlements_index: Mapping[tuple[str, str, str, str, float], str],
) -> dict[str, Any]:
    """Grade the parlay's selected legs against the settlements index.

    Returns per-leg result, an aggregate `parlay_result` ('won', 'lost',
    'push', 'unknown'), and the realized unit return (only when every
    leg graded to won or lost).
    """
    parlays = payload.get("parlays") or {}
    selected = parlays.get("selected_parlay") or {}
    slate_date = str(payload.get("run_date") or payload.get("slate_date") or "")

    leg_results: list[dict[str, Any]] = []
    combined_price = None
    for leg_key in ("leg_1", "leg_2", "leg_3", "leg_4"):
        leg = selected.get(leg_key)
        if not leg:
            continue
        player_norm, market, side, line = _leg_settlement_key(leg)
        result = settlements_index.get((slate_date, player_norm, market, side, line), "unknown")
        leg_results.append({
            "leg": leg_key,
            "player": leg.get("player") or leg.get("player_name"),
            "market": market,
            "side": side,
            "line": line,
            "decimal_price": leg.get("decimal_price"),
            "result": result,
        })
        price = leg.get("decimal_price")
        if isinstance(price, (int, float)):
            combined_price = price if combined_price is None else combined_price * float(price)

    if not leg_results:
        return {
            "parlay_result": "unknown",
            "legs": leg_results,
            "realized_return_per_unit": None,
        }

    if any(r["result"] == "lost" for r in leg_results):
        parlay_result = "lost"
    elif all(r["result"] == "won" for r in leg_results):
        parlay_result = "won"
    elif any(r["result"] == "unknown" for r in leg_results):
        parlay_result = "unknown"
    else:
        parlay_result = "push"

    if parlay_result == "won" and combined_price is not None:
        realized = combined_price - 1.0
    elif parlay_result == "lost":
        realized = -1.0
    else:
        realized = None

    return {
        "parlay_result": parlay_result,
        "legs": leg_results,
        "realized_return_per_unit": realized,
    }


# --- report -------------------------------------------------------------

@dataclass
class SlateShadowRow:
    payload_path: str
    slate_date: Optional[str]
    live_action: str
    coherent_action: str
    live_action_agrees: bool
    blocking_reasons: list[str]
    joint_probability: Optional[float]
    break_even_probability: Optional[float]
    promotion_margin: Optional[float]
    leg_probabilities: list[float]
    combined_decimal_price: Optional[float]
    candidate_id: Optional[str]
    grading: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowReport:
    generated_at_utc: str
    thresholds: PromotionThresholds
    penalties: PromotionPenalties
    rows: list[SlateShadowRow]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "thresholds": self.thresholds.__dict__,
            "penalties": self.penalties.__dict__,
            "rows": [row.__dict__ for row in self.rows],
            "summary": self.summary,
        }


def build_row(
    payload_path: Path,
    payload: Mapping[str, Any],
    decision: CoherentPromotionDecision,
    grading: Mapping[str, Any],
) -> SlateShadowRow:
    comps = decision.components
    return SlateShadowRow(
        payload_path=str(payload_path),
        slate_date=decision.slate_date,
        live_action=decision.live_action,
        coherent_action=decision.action,
        live_action_agrees=decision.live_action_agrees,
        blocking_reasons=list(decision.blocking_reasons),
        joint_probability=comps.calibrated_joint_probability,
        break_even_probability=comps.break_even_probability,
        promotion_margin=comps.promotion_margin,
        leg_probabilities=list(comps.leg_probabilities),
        combined_decimal_price=comps.combined_decimal_price,
        candidate_id=decision.candidate_id,
        grading=dict(grading),
    )


def _aggregate(rows: Iterable[SlateShadowRow]) -> dict[str, Any]:
    live_returns: list[float] = []
    coherent_returns: list[float] = []
    counts = {
        "total_payloads": 0,
        "live_act": 0,
        "coherent_act": 0,
        "divergent_live_act_coherent_abstain": 0,
        "divergent_live_abstain_coherent_act": 0,
        "concurrent_act": 0,
        "concurrent_abstain": 0,
        "graded_parlays": 0,
        "graded_live_wins": 0,
        "graded_live_losses": 0,
        "graded_coherent_would_publish": 0,
        "graded_coherent_would_publish_wins": 0,
        "graded_coherent_would_publish_losses": 0,
    }
    for row in rows:
        counts["total_payloads"] += 1
        if row.live_action == "ACT":
            counts["live_act"] += 1
        if row.coherent_action == "ACT":
            counts["coherent_act"] += 1
        if row.live_action == "ACT" and row.coherent_action == "ABSTAIN":
            counts["divergent_live_act_coherent_abstain"] += 1
        if row.live_action == "ABSTAIN" and row.coherent_action == "ACT":
            counts["divergent_live_abstain_coherent_act"] += 1
        if row.live_action == "ACT" and row.coherent_action == "ACT":
            counts["concurrent_act"] += 1
        if row.live_action == "ABSTAIN" and row.coherent_action == "ABSTAIN":
            counts["concurrent_abstain"] += 1

        realized = row.grading.get("realized_return_per_unit") if row.grading else None
        parlay_result = row.grading.get("parlay_result") if row.grading else "unknown"
        if realized is None:
            continue
        counts["graded_parlays"] += 1
        if row.live_action == "ACT":
            live_returns.append(float(realized))
            if parlay_result == "won":
                counts["graded_live_wins"] += 1
            elif parlay_result == "lost":
                counts["graded_live_losses"] += 1
        if row.coherent_action == "ACT":
            counts["graded_coherent_would_publish"] += 1
            coherent_returns.append(float(realized))
            if parlay_result == "won":
                counts["graded_coherent_would_publish_wins"] += 1
            elif parlay_result == "lost":
                counts["graded_coherent_would_publish_losses"] += 1

    summary: dict[str, Any] = dict(counts)
    summary["live_realized_return_per_unit_sum"] = sum(live_returns) if live_returns else 0.0
    summary["coherent_realized_return_per_unit_sum"] = sum(coherent_returns) if coherent_returns else 0.0
    return summary


def run_shadow_report(
    *,
    repo_root: Path = REPO_ROOT,
    history_glob: str = DEFAULT_HISTORY_GLOB,
    live_payload_rel: str = DEFAULT_LIVE_PAYLOAD,
    settlements_rel: str = DEFAULT_SETTLEMENTS,
    thresholds: PromotionThresholds | None = None,
    penalties: PromotionPenalties | None = None,
) -> ShadowReport:
    thresholds = thresholds or default_thresholds()
    penalties = penalties or PromotionPenalties()

    settlements_index = _load_settlements_index(repo_root / settlements_rel)

    payload_paths: list[Path] = sorted(repo_root.glob(history_glob))
    live_path = repo_root / live_payload_rel
    if live_path.exists():
        payload_paths.append(live_path)

    rows: list[SlateShadowRow] = []
    for path in payload_paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        decision = decide_coherent_promotion(
            payload, thresholds=thresholds, penalties=penalties,
        )
        grading = _grade_selected(payload, settlements_index)
        rows.append(build_row(path, payload, decision, grading))

    return ShadowReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        thresholds=thresholds,
        penalties=penalties,
        rows=rows,
        summary=_aggregate(rows),
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Shadow replay of the coherent MLB parlay promotion gate.")
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help="Repository root (default: derived from this file's location).",
    )
    parser.add_argument(
        "--history-glob", type=str, default=DEFAULT_HISTORY_GLOB,
        help="Glob for historical run payloads, relative to repo root.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=f"Output JSON path (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--min-leg-probability", type=float, default=default_thresholds().min_leg_probability,
    )
    parser.add_argument(
        "--min-joint-probability", type=float, default=default_thresholds().min_joint_probability,
    )
    parser.add_argument(
        "--min-promotion-margin", type=float, default=default_thresholds().min_promotion_margin,
    )
    parser.add_argument(
        "--min-probability-edge", type=float, default=default_thresholds().min_probability_edge,
    )
    parser.add_argument(
        "--min-expected-value", type=float, default=default_thresholds().min_expected_value_per_unit,
    )
    args = parser.parse_args()

    thresholds = PromotionThresholds(
        min_leg_probability=args.min_leg_probability,
        min_joint_probability=args.min_joint_probability,
        min_probability_edge=args.min_probability_edge,
        min_expected_value_per_unit=args.min_expected_value,
        min_promotion_margin=args.min_promotion_margin,
    )
    report = run_shadow_report(
        repo_root=args.repo_root, history_glob=args.history_glob, thresholds=thresholds,
    )
    out_path = args.out or (args.repo_root / DEFAULT_REPORT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))

    summary = report.summary
    print(f"wrote {out_path}")
    print(f"payloads inspected: {summary['total_payloads']}")
    print(f"  live ACT: {summary['live_act']}  coherent ACT: {summary['coherent_act']}")
    print(f"  live ACT / coherent ABSTAIN (would have blocked): {summary['divergent_live_act_coherent_abstain']}")
    print(f"  live ABSTAIN / coherent ACT (would have added): {summary['divergent_live_abstain_coherent_act']}")
    print(f"  graded parlays: {summary['graded_parlays']}")
    if summary["graded_parlays"]:
        print(f"    live realized return / unit sum: {summary['live_realized_return_per_unit_sum']:.3f}")
        print(f"    coherent realized return / unit sum: {summary['coherent_realized_return_per_unit_sum']:.3f}")


if __name__ == "__main__":
    _cli()
