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
from sports.mlb.advanced.production_refresh import refresh_advanced_profiles_incremental

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = REPO_ROOT / "artifacts" / "mlb_sequential_pa_model_validation.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "mlb_sequential_pa_model_validation.md"
DEFAULT_WEB = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "sequential_pa_hitter_predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh advanced MLB data and enrich H/TB rows with game-conditioned sequential PA probabilities.")
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
    source_status = manifest.get("source_status") or {}
    model_artifact = report.get("model_artifact") or {}
    rows = report.get("rows") or []
    ready = [row for row in rows if row.get("status") in {"READY", "WEAK_SUPPORT"}]
    blocked = [row for row in rows if row.get("status") == "BLOCKED_DATA"]
    lines = [
        "# MLB Game-Conditioned Sequential PA H/TB Validation",
        "",
        f"- Run date: `{report.get('run_date')}`",
        f"- Probability model: `{report.get('model_version')}`",
        f"- Structural simulator: `{report.get('structural_model_version')}`",
        f"- Residual fit: `{model_artifact.get('training_status')}`",
        f"- Evidence class: `{model_artifact.get('evidence_class')}`",
        f"- Evaluated H/TB rows: **{report.get('evaluated_h_tb_rows', 0)}**",
        f"- Modeled rows: **{report.get('modeled_rows', 0)}**",
        f"- Blocked rows: **{report.get('blocked_rows', 0)}**",
        f"- Data freshness: `{report.get('data_freshness_status')}`",
        "",
        "## Data",
        "",
        f"Sources: `{', '.join(manifest.get('sources') or [])}`",
        "",
        f"Baseball Savant / Statcast status: `{(source_status.get('baseball_savant_statcast') or {}).get('status', 'UNKNOWN')}`",
        "",
        f"FanGraphs status: `{(source_status.get('fangraphs') or {}).get('status', 'UNKNOWN')}`",
        "",
        f"Effective as-of date: `{manifest.get('effective_as_of_date')}`",
        "",
        f"Profile coverage: {manifest.get('batter_profiles', 0)} batter profiles, {manifest.get('pitcher_profiles', 0)} pitcher profiles, {manifest.get('direct_matchups', 0)} direct BvP process profiles.",
        "",
        "Raw Statcast data are cached by pybaseball and processed same-as-of partitions are cached rather than committed as large raw datasets. Every profile partition is dated and carries source, fetch, and effective timestamps.",
        "",
        "## Architecture",
        "",
        "`legacy/no-vig prior -> game state -> expert activations -> residual logit -> sequential PA distribution -> uncertainty/authority gate`",
        "",
        "Six experts are evaluated per game: strikeout/contact, contact quality, power/TB, defensive conversion, PA opportunity, and starter-removal/bullpen transition. Global residual coefficients are multiplied by game-specific activations, so a high-K matchup emphasizes contact survival while a low-K matchup can emphasize batted-ball quality. TB gives more relevance to the power-tail expert than H.",
        "",
        "The underlying event tree remains `PA -> K | BB | HBP | HR | NON_HR_CONTACT | OTHER`, followed by a non-HR contact outcome distribution. PA and AB are tracked separately and later PA transition away from the starter.",
        "",
        "## Probability authority",
        "",
        "The structural simulator still directly estimates `P(H=0)`, `P(H>=1)` and the TB tail. It no longer replaces the calibrated prior outright. The game-conditioned layer learns a bounded residual around the legacy/no-vig prior in logit space.",
        "",
        "Until exact point-in-time advanced-feature validation clears the target-specific Brier and log-loss gates, a positive residual remains shadow-only. A negative residual may lower/veto an overconfident H/TB candidate.",
        "",
        "## Current modeled rows",
        "",
        "| Player | Target | Prior P | Conditioned P | Production P | P(0H) | E[PA] | E[H] | E[TB] | Support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ready[:50]:
        lines.append(
            f"| {row.get('player','')} | {row.get('target','')} | {float(row.get('prior_probability') or 0):.3f} | "
            f"{float(row.get('game_conditioned_probability') or 0):.3f} | {float(row.get('usable_probability') or 0):.3f} | "
            f"{float(row.get('p_h_0') or 0):.3f} | {float(row.get('expected_pa') or 0):.2f} | "
            f"{float(row.get('expected_h') or 0):.2f} | {float(row.get('expected_tb') or 0):.2f} | {float(row.get('support') or 0):.2f} |"
        )
    lines.extend([
        "",
        "## Blocked/degraded data",
        "",
        f"{len(blocked)} rows were fail-closed because required MLBAM identity, Statcast profile, pitcher profile, or freshness evidence was unavailable.",
        "",
        "## Validation status",
        "",
        "The residual trainer uses a temporal split: coefficients are fit on earlier dates and scored on later dates. Historical proxy evidence may initialize shadow residuals but cannot unlock positive authority because the older processed corpus does not preserve every exact Savant/FanGraphs pregame state. Exact/prospective evidence is required for promotion.",
        "",
        "## Known limitations",
        "",
        "- Specific fielder OAA/location assignment remains zero-centered/uncertain when unavailable; no fielder data are fabricated.",
        "- Direct BvP remains heavily shrunk because sample sizes are usually small.",
        "- FanGraphs xFIP/SIERA availability is source-dependent and missingness lowers evidence strength.",
        "- Bullpen identity is still a transition toward neutral relief state until named-reliever distributions are supported.",
        "- Weather is consumed when present; missing weather is an uncertainty component rather than an invented value.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not args.no_refresh:
        refresh_advanced_profiles_incremental(pool_csv=args.pool_csv, run_date=args.run_date, advanced_root=args.advanced_root)
    report = enrich_pool_with_sequential_pa(pool_csv=args.pool_csv, run_date=args.run_date, advanced_root=args.advanced_root, refresh_data=False, trials=args.trials)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.web_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_md.write_text(markdown(report), encoding="utf-8")
    web = {
        "schema_version": "mlb_game_conditioned_hitter_frontend_v2",
        "run_date": args.run_date,
        "model_version": report.get("model_version"),
        "structural_model_version": report.get("structural_model_version"),
        "publication_authority": False,
        "authority": report.get("authority"),
        "model_artifact": report.get("model_artifact"),
        "data_freshness_status": report.get("data_freshness_status"),
        "source_status": (report.get("advanced_manifest") or {}).get("source_status") or {},
        "effective_as_of_date": (report.get("advanced_manifest") or {}).get("effective_as_of_date"),
        "evaluated_h_tb_rows": report.get("evaluated_h_tb_rows"),
        "modeled_rows": report.get("modeled_rows"),
        "blocked_rows": report.get("blocked_rows"),
        "candidates": [
            {key: row.get(key) for key in (
                "player", "target", "status", "raw_probability", "prior_probability", "game_conditioned_probability",
                "usable_probability", "lcb", "p_h_0", "expected_pa", "expected_h", "expected_tb", "support", "uncertainty",
                "authority", "expert_weights", "pitch_compatibility"
            )}
            for row in (report.get("rows") or [])
        ],
    }
    args.web_json.write_text(json.dumps(web, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        **{key: report.get(key) for key in ("run_date", "model_version", "structural_model_version", "evaluated_h_tb_rows", "modeled_rows", "blocked_rows", "data_freshness_status")},
        "source_status": web["source_status"],
        "effective_as_of_date": web["effective_as_of_date"],
        "model_artifact": web["model_artifact"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
