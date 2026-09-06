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
DEFAULT_DAILY = REPO_ROOT / "artifacts" / "mlb_sequential_pa_model_validation.json"
DEFAULT_HISTORICAL = REPO_ROOT / "artifacts" / "mlb_sequential_pa_historical_validation.json"
DEFAULT_WEB = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "sequential_pa_hitter_predictions.json"
DEFAULT_BOARD = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"
DEFAULT_OUT_JSON = DEFAULT_DAILY
DEFAULT_OUT_MD = REPO_ROOT / "artifacts" / "mlb_sequential_pa_model_validation.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def verdict(item: dict[str, Any]) -> str:
    legacy = item.get("legacy") or {}
    seq = item.get("sequential_raw") or {}
    lb, sb = legacy.get("brier"), seq.get("brier")
    ll, sl = legacy.get("log_loss"), seq.get("log_loss")
    if None in (lb, sb, ll, sl):
        return "INSUFFICIENT_EVIDENCE"
    brier_delta = float(sb) - float(lb)
    log_delta = float(sl) - float(ll)
    if brier_delta < 0 and log_delta < 0:
        return "IMPROVES_PREDICTIVE_PROBABILITY"
    if brier_delta > 0 and log_delta > 0:
        return "REGRESSES_PREDICTIVE_PROBABILITY"
    return "MIXED_OR_TIE"


def parity() -> dict[str, Any]:
    relative = Path("mlb/data/sequential_pa_hitter_predictions.json")
    source = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "sequential_pa_hitter_predictions.json"
    public = REPO_ROOT / "dist" / relative
    protected = REPO_ROOT / "paywall" / "private-content" / "app" / relative
    paths = [source, public, protected]
    exists = [path.is_file() for path in paths]
    same = bool(all(exists) and public.read_bytes() == source.read_bytes() and protected.read_bytes() == source.read_bytes())
    return {
        "source": str(source.relative_to(REPO_ROOT)),
        "public": str(public.relative_to(REPO_ROOT)),
        "protected": str(protected.relative_to(REPO_ROOT)),
        "all_exist": all(exists),
        "byte_identical": same,
    }


def build_payload(daily: dict[str, Any], historical: dict[str, Any], web: dict[str, Any], board: dict[str, Any]) -> dict[str, Any]:
    daily_rows = daily.get("rows") or []
    status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in daily_rows if isinstance(row, dict))
    missing = Counter()
    for row in daily_rows:
        if not isinstance(row, dict):
            continue
        for reason in row.get("missing") or []:
            missing[str(reason)] += 1
    hist_summary = historical.get("summary") or {}
    validations: dict[str, Any] = {}
    for target in ("H", "TB"):
        item = hist_summary.get(target) or {}
        legacy = item.get("legacy") or {}
        seq = item.get("sequential_raw") or {}
        usable = item.get("sequential_usable") or {}
        validations[target] = {
            "rows": item.get("rows"),
            "observed_hit_rate": item.get("observed_hit_rate"),
            "legacy": legacy,
            "sequential_raw": seq,
            "sequential_usable": usable,
            "brier_delta_raw_minus_legacy": (
                float(seq["brier"]) - float(legacy["brier"])
                if seq.get("brier") is not None and legacy.get("brier") is not None else None
            ),
            "log_loss_delta_raw_minus_legacy": (
                float(seq["log_loss"]) - float(legacy["log_loss"])
                if seq.get("log_loss") is not None and legacy.get("log_loss") is not None else None
            ),
            "verdict": verdict(item),
        }

    manifest = daily.get("advanced_manifest") or {}
    source_status = web.get("source_status") or manifest.get("source_status") or {}
    seq_candidates = web.get("candidates") or []
    public_plays = board.get("plays") or []
    model_versions = sorted({
        str(play.get("probability_model_version") or play.get("model_version") or "")
        for play in public_plays if isinstance(play, dict) and (play.get("probability_model_version") or play.get("model_version"))
    })
    return {
        "schema_version": "mlb_sequential_pa_model_final_validation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": web.get("run_date") or daily.get("run_date") or board.get("run_date"),
        "model_version": web.get("model_version") or daily.get("model_version"),
        "authority": daily.get("authority") or web.get("authority") or {},
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
            "hits_over_0_5_identity": "P(H>=1) = 1 - P(H=0)",
            "tb_over_1_5_identity": "P(TB>=2) = directly simulated fraction with TB >= 2",
            "pa_ab_separate": True,
            "defense": "zero-centered residual from average Statcast expected-result context; no fabricated OAA",
            "sprint_speed": "not re-applied as full residual when Statcast expected metrics are the baseline",
            "game_state": "batting-order/team-run opportunity plus sequential starter-to-bullpen transition",
        },
        "historical_validation": {
            "available": bool(historical),
            "evidence_class": historical.get("evidence_class"),
            "rows": historical.get("rows"),
            "targets": validations,
            "zero_hit": hist_summary.get("zero_hit") or {},
            "economic_evidence": historical.get("economic_evidence") or {},
        },
        "daily_production_test": {
            "evaluated_h_tb_rows": daily.get("evaluated_h_tb_rows") or web.get("evaluated_h_tb_rows"),
            "modeled_rows": daily.get("modeled_rows") or web.get("modeled_rows"),
            "blocked_rows": daily.get("blocked_rows") or web.get("blocked_rows"),
            "status_counts": dict(status_counts),
            "blocked_reason_counts": dict(missing),
            "frontend_candidate_rows": len(seq_candidates),
            "public_play_count": len(public_plays),
            "public_model_versions": model_versions,
            "data_freshness_status": web.get("data_freshness_status") or daily.get("data_freshness_status"),
        },
        "publication": parity(),
        "github_actions": {
            "canonical_parent_run_id": os.environ.get("PARENT_WORKFLOW_RUN_ID") or None,
            "consistency_run_id": os.environ.get("GITHUB_RUN_ID") or None,
            "consistency_workflow": os.environ.get("GITHUB_WORKFLOW") or None,
            "evidence_at_report_build": "report builder runs only after source generation and, in frontend consistency, after byte-parity validation",
        },
        "limitations": [
            "Specific fielder OAA/location assignment remains non-authoritative when reliable location/fielder data are unavailable.",
            "Direct BvP is strongly shrunk because samples are typically small.",
            "Historical rolling-origin replay lacks complete decision-time pitch-level xFIP/SIERA/OAA snapshots and is diagnostic, not certification.",
            "Historical ROI is not claimed without exact preserved decision-time prices.",
            "Named-reliever sequence modeling remains less specific than the starter portion when bullpen identities are not reliably known pregame.",
            "The sequential model remains negative-authority until independent calibration evidence earns promotion.",
        ],
    }


def markdown(payload: dict[str, Any]) -> str:
    data = payload["data"]
    hist = payload["historical_validation"]
    prod = payload["daily_production_test"]
    pub = payload["publication"]
    source_status = data.get("source_status") or {}
    savant = source_status.get("baseball_savant_statcast") or {}
    fangraphs = source_status.get("fangraphs") or {}
    lines = [
        "# MLB Sequential PA Model Validation",
        "",
        f"Run date: `{payload.get('run_date')}`  ",
        f"Model: `{payload.get('model_version')}`  ",
        f"Authority: `{(payload.get('authority') or {}).get('promotion_status', 'NEGATIVE_AUTHORITY')}`",
        "",
        "## Data",
        "",
        f"- Baseball Savant / Statcast: **{savant.get('status', 'UNKNOWN')}**; covered `{savant.get('covered_entities')}` / `{savant.get('required_entities')}` required active-slate entities.",
        f"- FanGraphs-compatible pitching: **{fangraphs.get('status', 'UNKNOWN')}**; available fields: `{', '.join(fangraphs.get('available_fields') or []) or 'none reported'}`.",
        f"- Leakage cutoff/effective as-of: `{data.get('effective_as_of_date')}`; fetched at `{data.get('fetched_at_utc')}`.",
        f"- Processed coverage: `{data.get('batter_profiles')}` batters, `{data.get('pitcher_profiles')}` pitchers, `{data.get('direct_matchups')}` direct BvP profiles.",
        f"- Incremental cache: `{json.dumps(data.get('cache') or {}, sort_keys=True)}`.",
        "",
        "Raw season-scale pitch data are not committed to Git. Pybaseball caches upstream responses; the production feature store contains bounded, dated processed partitions with source/fetch/effective timestamps and MLBAM identity.",
        "",
        "## Architecture",
        "",
        "`EXPECTED PA -> batter × pitcher -> K | BB | HBP | HR | NON_HR_CONTACT | OTHER -> contact-quality distribution -> average-context expected result -> defense/park residual -> final PA -> update starter/bullpen/opportunity state -> next PA -> full-night H/TB distribution`",
        "",
        "HR is an exclusive PA branch and is not counted again inside non-HR contact. PA and AB are separate. Hits O0.5 is exactly `1-P(H=0)`; TB O1.5 is the directly simulated fraction with `TB>=2`. Statcast expected outcomes are treated as average historical-context baselines, so defense is only a zero-centered residual. Sprint Speed is not double-counted when those expected metrics already contain it.",
        "",
        "## Historical validation",
        "",
        f"Evidence class: `{hist.get('evidence_class')}`; observations: `{hist.get('rows')}`.",
        "",
        "| Target | Verdict | Legacy Brier | Seq raw Brier | Δ Brier | Legacy log loss | Seq raw log loss | Δ log loss | Seq usable Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ("H", "TB"):
        item = (hist.get("targets") or {}).get(target) or {}
        legacy = item.get("legacy") or {}
        seq = item.get("sequential_raw") or {}
        usable = item.get("sequential_usable") or {}
        lines.append(
            f"| {target} | {item.get('verdict', 'INSUFFICIENT_EVIDENCE')} | {f(legacy.get('brier'))} | {f(seq.get('brier'))} | "
            f"{f(item.get('brier_delta_raw_minus_legacy'))} | {f(legacy.get('log_loss'))} | {f(seq.get('log_loss'))} | "
            f"{f(item.get('log_loss_delta_raw_minus_legacy'))} | {f(usable.get('brier'))} |"
        )
    zero = hist.get("zero_hit") or {}
    lines += [
        "",
        f"Zero-hit calibration diagnostic: predicted `{f(zero.get('predicted_zero_hit_rate'))}`, observed `{f(zero.get('observed_zero_hit_rate'))}` across `{zero.get('rows')}` H observations.",
        "",
        "No historical ROI claim is made without exact preserved prediction-time sportsbook prices. Reconstructed prices are not certification evidence.",
        "",
        "## Daily production test",
        "",
        f"- H/TB rows evaluated: `{prod.get('evaluated_h_tb_rows')}`; modeled: `{prod.get('modeled_rows')}`; blocked: `{prod.get('blocked_rows')}`.",
        f"- Data freshness: `{prod.get('data_freshness_status')}`.",
        f"- Sequential status counts: `{json.dumps(prod.get('status_counts') or {}, sort_keys=True)}`.",
        f"- Blocked-data reasons: `{json.dumps(prod.get('blocked_reason_counts') or {}, sort_keys=True)}`.",
        f"- Existing public board plays: `{prod.get('public_play_count')}`. No existing issued pick is retroactively changed by this report or the advanced-data refresh.",
        "",
        "## Static publication",
        "",
        f"Sequential source/public/protected artifacts all exist: **{pub.get('all_exist')}**. Byte-identical after static build: **{pub.get('byte_identical')}**.",
        "",
        "The frontend consistency workflow additionally rejects stale run dates, missing model versions, probabilities outside `[0,1]`, missing effective-as-of metadata, and an unverifiable Baseball Savant status.",
        "",
        "## GitHub Actions",
        "",
        f"Canonical parent run: `{payload['github_actions'].get('canonical_parent_run_id')}`; consistency run: `{payload['github_actions'].get('consistency_run_id')}`. The report-build step is downstream of source generation and static byte-parity validation in the consistency workflow.",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in payload.get("limitations") or []]
    lines += [
        "",
        "## Promotion decision",
        "",
        "The sequential model remains **negative-authority/shadow for upward probability changes**. It can veto/down-rank an overconfident legacy H/TB candidate when its support/freshness contract is satisfied, but it cannot manufacture more picks or raise legacy confidence until independent calibration evidence supports promotion.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--web", type=Path, default=DEFAULT_WEB)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    daily = read_json(args.daily)
    historical = read_json(args.historical)
    web = read_json(args.web)
    board = read_json(args.board)
    payload = build_payload(daily, historical, web, board)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown(payload), encoding="utf-8")
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
