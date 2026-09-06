#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def metric_delta(item: dict[str, Any], metric: str) -> float | None:
    legacy = (item.get("legacy") or {}).get(metric)
    sequential = (item.get("sequential_raw") or {}).get(metric)
    if legacy is None or sequential is None:
        return None
    return float(sequential) - float(legacy)


def verdict(item: dict[str, Any]) -> str:
    brier = metric_delta(item, "brier")
    logloss = metric_delta(item, "log_loss")
    if brier is None or logloss is None:
        return "INSUFFICIENT_EVIDENCE"
    if brier < 0 and logloss < 0:
        return "IMPROVES_PREDICTIVE_PROBABILITY"
    if brier > 0 and logloss > 0:
        return "REGRESSES_PREDICTIVE_PROBABILITY"
    return "MIXED_OR_TIE"


def publication_parity() -> dict[str, Any]:
    source = REPO_ROOT / "sports/mlb/web/data/sequential_pa_hitter_predictions.json"
    public = REPO_ROOT / "dist/mlb/data/sequential_pa_hitter_predictions.json"
    protected = REPO_ROOT / "paywall/private-content/app/mlb/data/sequential_pa_hitter_predictions.json"
    exists = all(path.is_file() for path in (source, public, protected))
    identical = bool(exists and source.read_bytes() == public.read_bytes() == protected.read_bytes())
    return {
        "source": str(source.relative_to(REPO_ROOT)),
        "public": str(public.relative_to(REPO_ROOT)),
        "protected": str(protected.relative_to(REPO_ROOT)),
        "all_exist": exists,
        "byte_identical": identical,
    }


def build(daily: dict[str, Any], historical: dict[str, Any], web: dict[str, Any], board: dict[str, Any]) -> dict[str, Any]:
    source_authority = dict(daily.get("authority") or web.get("authority") or {})
    # This is an architecture/governance invariant, not a data-derived guess.
    # Missing older daily artifacts may omit the field, but v1 is never allowed
    # to gain upward authority merely because a report input predates the field.
    source_authority.setdefault(
        "promotion_status", "NEGATIVE_AUTHORITY_UNTIL_INDEPENDENT_ADVANCED_MODEL_CALIBRATION"
    )
    source_authority["can_raise_legacy_probability"] = False
    source_authority.setdefault("can_veto_overconfident_h_tb", True)

    manifest = daily.get("advanced_manifest") or {}
    source_status = web.get("source_status") or manifest.get("source_status") or {}
    history_summary = historical.get("summary") or {}
    targets: dict[str, Any] = {}
    for target in ("H", "TB"):
        item = history_summary.get(target) or {}
        targets[target] = {
            **item,
            "brier_delta_raw_minus_legacy": metric_delta(item, "brier"),
            "log_loss_delta_raw_minus_legacy": metric_delta(item, "log_loss"),
            "verdict": verdict(item),
        }

    rows = daily.get("rows") or []
    statuses = Counter(str(row.get("status") or "UNKNOWN") for row in rows if isinstance(row, dict))
    blocked_reasons: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for reason in row.get("missing") or []:
            blocked_reasons[str(reason)] += 1

    return {
        "schema_version": "mlb_sequential_pa_model_final_validation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": web.get("run_date") or daily.get("run_date") or board.get("run_date"),
        "model_version": web.get("model_version") or daily.get("model_version") or "sequential_pa_contact_model_v1",
        "authority": source_authority,
        "data": {
            "sources": manifest.get("sources") or list(source_status),
            "source_status": source_status,
            "effective_as_of_date": web.get("effective_as_of_date") or manifest.get("effective_as_of_date"),
            "fetched_at_utc": manifest.get("fetched_at_utc"),
            "lookback_start_date": manifest.get("lookback_start_date"),
            "candidate_identities": manifest.get("candidate_identities"),
            "batter_profiles": manifest.get("batter_profiles"),
            "pitcher_profiles": manifest.get("pitcher_profiles"),
            "direct_matchups": manifest.get("direct_matchups"),
            "cache": manifest.get("cache") or {},
            "failures": manifest.get("failures") or [],
            "freshness_policy": manifest.get("freshness_policy") or {},
        },
        "architecture": {
            "pa_event_tree": ["K", "BB", "HBP", "HR", "NON_HR_CONTACT", "OTHER"],
            "non_hr_contact_tree": ["OUT", "1B", "2B", "3B", "ROE_OTHER"],
            "pa_ab_separate": True,
            "hits_over_0_5_identity": "P(H>=1) = 1 - P(H=0)",
            "tb_over_1_5_identity": "P(TB>=2) = directly simulated fraction with TB >= 2",
            "defense": "zero-centered residual from average Statcast expected-result context; no fabricated OAA",
            "sprint_speed": "not re-applied as a full residual when Statcast expected metrics are used",
            "game_state": "batting-order/team-run opportunity plus sequential starter-to-bullpen transition",
        },
        "historical_validation": {
            "available": bool(historical),
            "evidence_class": historical.get("evidence_class"),
            "rows": historical.get("rows"),
            "targets": targets,
            "zero_hit": history_summary.get("zero_hit") or {},
            "economic_evidence": historical.get("economic_evidence") or {},
        },
        "daily_production_test": {
            "evaluated_h_tb_rows": daily.get("evaluated_h_tb_rows") or web.get("evaluated_h_tb_rows"),
            "modeled_rows": daily.get("modeled_rows") or web.get("modeled_rows"),
            "blocked_rows": daily.get("blocked_rows") or web.get("blocked_rows"),
            "status_counts": dict(statuses),
            "blocked_reason_counts": dict(blocked_reasons),
            "frontend_candidate_rows": len(web.get("candidates") or []),
            "public_play_count": len(board.get("plays") or []),
            "data_freshness_status": web.get("data_freshness_status") or daily.get("data_freshness_status"),
        },
        "publication": publication_parity(),
        "github_actions": {
            "parent_workflow_run_id": os.environ.get("PARENT_WORKFLOW_RUN_ID") or None,
            "parent_workflow_name": os.environ.get("PARENT_WORKFLOW_NAME") or None,
            "report_workflow_run_id": os.environ.get("GITHUB_RUN_ID") or None,
            "report_workflow": os.environ.get("GITHUB_WORKFLOW") or None,
        },
        "limitations": [
            "Specific fielder OAA/location assignment remains non-authoritative when reliable fielder/location data are unavailable.",
            "Direct BvP is strongly shrunk because samples are usually small.",
            "Rolling-origin replay lacks complete historical pitch-level xFIP/SIERA/OAA snapshots and is diagnostic rather than certification.",
            "No historical ROI claim is made without exact preserved decision-time prices.",
            "The sequential model remains negative-authority until independent calibration evidence earns promotion.",
        ],
    }


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def to_markdown(p: dict[str, Any]) -> str:
    d = p["data"]
    hist = p["historical_validation"]
    prod = p["daily_production_test"]
    pub = p["publication"]
    savant = (d.get("source_status") or {}).get("baseball_savant_statcast") or {}
    fg = (d.get("source_status") or {}).get("fangraphs") or {}
    lines = [
        "# MLB Sequential PA Model Validation", "",
        f"Run date: `{p.get('run_date')}`  ",
        f"Model: `{p.get('model_version')}`  ",
        f"Authority: `{p['authority']['promotion_status']}`", "",
        "## Data", "",
        f"- Baseball Savant / Statcast: **{savant.get('status', 'UNKNOWN')}**; coverage `{savant.get('covered_entities')}` / `{savant.get('required_entities')}` active-slate entities.",
        f"- FanGraphs-compatible pitching: **{fg.get('status', 'UNKNOWN')}**; fields `{', '.join(fg.get('available_fields') or []) or 'none reported'}`.",
        f"- Effective/as-of cutoff: `{d.get('effective_as_of_date')}`; fetched `{d.get('fetched_at_utc')}`.",
        f"- Profiles: `{d.get('batter_profiles')}` batters, `{d.get('pitcher_profiles')}` pitchers, `{d.get('direct_matchups')}` direct BvP pairs.",
        f"- Incremental cache: `{json.dumps(d.get('cache') or {}, sort_keys=True)}`.", "",
        "Raw season-scale pitch data are not committed. Pybaseball caches upstream responses; production keeps bounded dated feature partitions with source/fetch/effective timestamps and MLBAM IDs.", "",
        "## Architecture", "",
        "`EXPECTED PA -> batter × pitcher -> K | BB | HBP | HR | NON_HR_CONTACT | OTHER -> contact quality -> average-context expected result -> defense/park residual -> PA result -> state update -> next PA -> full-night H/TB distribution`", "",
        "PA and AB are separate. HR is exclusive and not double-counted. Hits O0.5 is exactly `1-P(H=0)` and TB O1.5 is directly simulated `P(TB>=2)`. Defense is a zero-centered residual against Statcast average-context expected outcomes; Sprint Speed is not double-counted.", "",
        "## Historical validation", "",
        f"Evidence: `{hist.get('evidence_class')}`; observations `{hist.get('rows')}`.", "",
        "| Target | Verdict | Legacy Brier | Seq raw Brier | Δ Brier | Legacy log loss | Seq raw log loss | Δ log loss | Seq usable Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ("H", "TB"):
        item = (hist.get("targets") or {}).get(target) or {}
        legacy, seq, usable = item.get("legacy") or {}, item.get("sequential_raw") or {}, item.get("sequential_usable") or {}
        lines.append(
            f"| {target} | {item.get('verdict')} | {fmt(legacy.get('brier'))} | {fmt(seq.get('brier'))} | {fmt(item.get('brier_delta_raw_minus_legacy'))} | "
            f"{fmt(legacy.get('log_loss'))} | {fmt(seq.get('log_loss'))} | {fmt(item.get('log_loss_delta_raw_minus_legacy'))} | {fmt(usable.get('brier'))} |"
        )
    zero = hist.get("zero_hit") or {}
    lines += ["", f"Zero-hit diagnostic: predicted `{fmt(zero.get('predicted_zero_hit_rate'))}`, observed `{fmt(zero.get('observed_zero_hit_rate'))}`, n=`{zero.get('rows')}`.", "",
              "No ROI claim is made without exact preserved decision-time sportsbook prices.", "",
              "## Daily production test", "",
              f"- Evaluated `{prod.get('evaluated_h_tb_rows')}`, modeled `{prod.get('modeled_rows')}`, blocked `{prod.get('blocked_rows')}`.",
              f"- Freshness: `{prod.get('data_freshness_status')}`; statuses `{json.dumps(prod.get('status_counts') or {}, sort_keys=True)}`.",
              f"- Blocked reasons: `{json.dumps(prod.get('blocked_reason_counts') or {}, sort_keys=True)}`.", "",
              "## Static publication", "",
              f"Source/dist/protected sequential artifacts all exist: **{pub.get('all_exist')}**; byte-identical: **{pub.get('byte_identical')}**.", "",
              "## GitHub Actions", "",
              f"Parent `{p['github_actions'].get('parent_workflow_name')}` run `{p['github_actions'].get('parent_workflow_run_id')}`; report run `{p['github_actions'].get('report_workflow_run_id')}`.", "",
              "## Limitations", ""]
    lines.extend(f"- {item}" for item in p.get("limitations") or [])
    lines += ["", "## Promotion decision", "",
              "The advanced model remains **negative-authority**. The historical proxy regressed versus legacy, so it cannot raise public H/TB confidence or force picks. It may only veto/down-rank where fresh advanced evidence and the existing integrity/lineup/quote gates all pass.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", type=Path, default=REPO_ROOT / "artifacts/mlb_sequential_pa_model_validation.json")
    parser.add_argument("--historical", type=Path, default=REPO_ROOT / "artifacts/mlb_sequential_pa_historical_validation.json")
    parser.add_argument("--web", type=Path, default=REPO_ROOT / "sports/mlb/web/data/sequential_pa_hitter_predictions.json")
    parser.add_argument("--board", type=Path, default=REPO_ROOT / "sports/mlb/web/data/daily_predictions.json")
    parser.add_argument("--out-json", type=Path, default=REPO_ROOT / "artifacts/mlb_sequential_pa_model_validation.json")
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "artifacts/mlb_sequential_pa_model_validation.md")
    args = parser.parse_args()
    payload = build(load(args.daily), load(args.historical), load(args.web), load(args.board))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(to_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "run_date": payload.get("run_date"),
        "historical_available": payload["historical_validation"]["available"],
        "H_verdict": payload["historical_validation"]["targets"].get("H", {}).get("verdict"),
        "TB_verdict": payload["historical_validation"]["targets"].get("TB", {}).get("verdict"),
        "static_byte_identical": payload["publication"]["byte_identical"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
