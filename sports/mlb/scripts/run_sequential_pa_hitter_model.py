#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.advanced.data_layer import DEFAULT_ADVANCED_ROOT
from sports.mlb.advanced.integration import enrich_pool_with_sequential_pa

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = REPO_ROOT / "artifacts" / "mlb_sequential_pa_model_validation.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "mlb_sequential_pa_model_validation.md"
DEFAULT_WEB = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "sequential_pa_hitter_predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh advanced MLB data and enrich H/TB rows with sequential PA probabilities.")
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--advanced-root", type=Path, default=DEFAULT_ADVANCED_ROOT)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--web-json", type=Path, default=DEFAULT_WEB)
    parser.add_argument("--no-refresh", action="store_true")
    return parser.parse_args()


def markdown(report: dict) -> str:
    manifest = report.get("advanced_manifest") or {}
    rows = report.get("rows") or []
    ready = [row for row in rows if row.get("status") in {"READY", "WEAK_SUPPORT"}]
    blocked = [row for row in rows if row.get("status") == "BLOCKED_DATA"]
    lines = [
        "# MLB Sequential PA H/TB Model Validation",
        "",
        f"- Run date: `{report.get('run_date')}`",
        f"- Model: `{report.get('model_version')}`",
        f"- Evaluated H/TB rows: **{report.get('evaluated_h_tb_rows', 0)}**",
        f"- Modeled rows: **{report.get('modeled_rows', 0)}**",
        f"- Blocked rows: **{report.get('blocked_rows', 0)}**",
        f"- Data freshness: `{report.get('data_freshness_status')}`",
        "",
        "## Data",
        "",
        f"Sources: `{', '.join(manifest.get('sources') or [])}`",
        "",
        f"Profile coverage: {manifest.get('batter_profiles', 0)} batter profiles, {manifest.get('pitcher_profiles', 0)} pitcher profiles, {manifest.get('direct_matchups', 0)} direct BvP process profiles.",
        "",
        "Raw Statcast data are cached in the GitHub Actions cache rather than committed as a large repository artifact. Every profile partition is dated and carries fetch/effective timestamps.",
        "",
        "## Architecture",
        "",
        "`PA -> K | BB | HBP | HR | NON_HR_CONTACT | OTHER`",
        "",
        "`NON_HR_CONTACT -> OUT | 1B | 2B | 3B | ROE_OTHER`",
        "",
        "Expected PA is modeled separately from per-PA quality. Later PA transition toward a bullpen state rather than assuming identical independent PA against the starter. PA and AB are tracked separately.",
        "",
        "Statcast xBA/xSLG/xwOBA are treated as average-context expected-contact baselines. Specific defense is a zero-centered residual; v1 does not fabricate OAA when reliable fielder-level data are unavailable. Sprint Speed is not added as a second full adjustment on top of Statcast expected metrics.",
        "",
        "## Probability authority",
        "",
        "The raw H/TB probability comes directly from the simulated nightly distribution. `P(H over 0.5) = 1 - P(H=0)` and `P(TB over 1.5) = P(TB>=2)` by construction.",
        "",
        "The v1 advanced model has **negative authority only** until an independent advanced-model calibration holdout is accumulated. Its usable probability is a downward-only uncertainty-adjusted probability; it may veto an overconfident legacy H/TB candidate but may not inflate one above the already-authorized legacy probability.",
        "",
        "## Current modeled rows",
        "",
        "| Player | Target | Status | Raw P | Usable P | P(0H) | E[PA] | E[H] | E[TB] | Support |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ready[:50]:
        lines.append(
            f"| {row.get('player','')} | {row.get('target','')} | {row.get('status','')} | "
            f"{float(row.get('raw_probability') or 0):.3f} | {float(row.get('usable_probability') or 0):.3f} | "
            f"{float(row.get('p_h_0') or 0):.3f} | {float(row.get('expected_pa') or 0):.2f} | "
            f"{float(row.get('expected_h') or 0):.2f} | {float(row.get('expected_tb') or 0):.2f} | "
            f"{float(row.get('support') or 0):.2f} |"
        )
    lines.extend([
        "",
        "## Blocked/degraded data",
        "",
        f"{len(blocked)} rows were fail-closed because required MLBAM identity, Statcast profile, pitcher profile, or freshness evidence was unavailable.",
        "",
        "## Validation status",
        "",
        "Unit/invariant tests cover the event tree, HR non-double-counting, strikeout suppression, contact-quality directionality, average/elite/poor defense residuals, PA-order effects, PA-vs-AB accounting, H/TB distribution identities, and deterministic simulation. The existing legacy H/TB model remains the baseline and fallback. Formal economic promotion requires exact decision-time prices and independently calibrated prospective evidence; reconstructed prices are not used for certification.",
        "",
        "## Known limitations",
        "",
        "- Specific fielder OAA/location assignment is not given positive authority in v1 when unavailable; the model degrades to average-context defense with uncertainty.",
        "- Direct BvP is shrunk heavily because sample sizes are usually small.",
        "- FanGraphs xFIP/SIERA availability is source-dependent and missingness increases uncertainty.",
        "- Bullpen identity is represented as a transition toward league-average relief state until a named-reliever distribution is supported.",
        "- The advanced model is not permitted to raise public H/TB confidence until independent calibration evidence supports promotion.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = enrich_pool_with_sequential_pa(
        pool_csv=args.pool_csv,
        run_date=args.run_date,
        advanced_root=args.advanced_root,
        refresh_data=not args.no_refresh,
        trials=args.trials,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.web_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_md.write_text(markdown(report), encoding="utf-8")
    web = {
        "schema_version": "mlb_sequential_pa_frontend_v1",
        "run_date": args.run_date,
        "model_version": report.get("model_version"),
        "publication_authority": False,
        "authority": report.get("authority"),
        "data_freshness_status": report.get("data_freshness_status"),
        "evaluated_h_tb_rows": report.get("evaluated_h_tb_rows"),
        "modeled_rows": report.get("modeled_rows"),
        "blocked_rows": report.get("blocked_rows"),
        "candidates": [
            {key: row.get(key) for key in (
                "player", "target", "status", "raw_probability", "usable_probability", "lcb",
                "p_h_0", "expected_pa", "expected_h", "expected_tb", "support", "uncertainty"
            )}
            for row in (report.get("rows") or [])
        ],
    }
    args.web_json.write_text(json.dumps(web, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("run_date", "model_version", "evaluated_h_tb_rows", "modeled_rows", "blocked_rows", "data_freshness_status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
