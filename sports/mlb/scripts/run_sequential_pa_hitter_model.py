#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.advanced.data_layer import DEFAULT_ADVANCED_ROOT
from sports.mlb.advanced.integration import enrich_pool_with_sequential_pa
from sports.mlb.advanced.production_refresh import refresh_advanced_profiles_incremental

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = REPO_ROOT / "artifacts" / "mlb_sequential_pa_model_validation.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "mlb_sequential_pa_model_validation.md"
DEFAULT_WEB = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "sequential_pa_hitter_predictions.json"
DEFAULT_PREGAME_ROOT = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "history" / "runs"

SNAPSHOT_MARKETS = {"H", "TB", "HR"}
SNAPSHOT_SCHEMA_VERSION = "mlb_game_conditioned_pregame_snapshot_v1"


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if number.is_integer() and not any(ch in text.lower() for ch in (".", "e")):
        return int(number)
    return number


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_pregame_feature_snapshot(
    *,
    pool_csv: Path,
    report: dict[str, Any],
    run_date: str,
    snapshot_root: Path = DEFAULT_PREGAME_ROOT,
    captured_at_utc: str | None = None,
) -> Path:
    """Persist an immutable, outcome-free copy of the exact live feature state.

    Only explicitly whitelisted prediction-time fields are copied. Outcome and
    settlement columns are never copied even if they happen to be present in
    the source CSV. The snapshot is content-addressed and never overwritten.
    """

    captured_at = captured_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows: list[dict[str, Any]] = []
    with pool_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            target = str(row.get("Target") or "").upper()
            player_type = str(row.get("Player_Type") or "").lower()
            if target not in SNAPSHOT_MARKETS or player_type != "hitter":
                continue

            diagnostics = _json_object(row.get("Sequential_PA_Diagnostics"))
            conditioned = _json_object(diagnostics.get("game_conditioned"))
            state = _json_object(conditioned.get("state"))
            pitch_detail = _json_object(conditioned.get("pitch_compatibility"))

            rows.append(
                {
                    "row_index": row_index,
                    "game_id": _clean_scalar(row.get("Game_ID")),
                    "player": _clean_scalar(row.get("Player")),
                    "player_mlbam_id": _clean_scalar(row.get("Player_MLBAM_ID")),
                    "team": _clean_scalar(row.get("Team")),
                    "opponent": _clean_scalar(row.get("Opponent")),
                    "opposing_pitcher": _clean_scalar(row.get("Opposing_Pitcher")),
                    "opposing_pitcher_id": _clean_scalar(row.get("Opposing_Pitcher_ID")),
                    "batting_order": _clean_scalar(row.get("Sequential_Batting_Order") or row.get("Batting_Order")),
                    "target": target,
                    "market_line": _clean_scalar(row.get("Market_Line")),
                    "market_over_price": _clean_scalar(row.get("Market_Over_Price")),
                    "market_under_price": _clean_scalar(row.get("Market_Under_Price")),
                    "market_source": _clean_scalar(row.get("Market_Source")),
                    "structural_probability": _clean_scalar(row.get("Sequential_PA_Raw_Probability")),
                    "prior_probability": _clean_scalar(row.get("Game_Conditioned_Prior_Probability")),
                    "candidate_probability": _clean_scalar(row.get("Game_Conditioned_Candidate_Probability")),
                    "production_probability": _clean_scalar(row.get("Game_Conditioned_Production_Probability")),
                    "probability_lcb": _clean_scalar(row.get("Game_Conditioned_Probability_LCB")),
                    "residual_logit": _clean_scalar(row.get("Game_Conditioned_Residual_Logit")),
                    "evidence_strength": _clean_scalar(row.get("Game_Conditioned_Evidence_Strength")),
                    "sequential_uncertainty": _clean_scalar(row.get("Sequential_PA_Uncertainty")),
                    "sequential_support": _clean_scalar(row.get("Sequential_PA_Support")),
                    "status": _clean_scalar(row.get("Sequential_PA_Status")),
                    "authority": _clean_scalar(row.get("Game_Conditioned_Authority")),
                    "expert_weights": _json_object(row.get("Game_Conditioned_Expert_Weights")),
                    "expert_signals": _json_object(row.get("Game_Conditioned_Expert_Signals")),
                    "expert_activations": _json_object(row.get("Game_Conditioned_Expert_Activations")),
                    "expert_contributions": _json_object(row.get("Game_Conditioned_Expert_Contributions")),
                    "game_state": state,
                    "pitch_compatibility": pitch_detail,
                }
            )

    manifest = report.get("advanced_manifest") or {}
    artifact = report.get("model_artifact") or {}
    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "run_date": run_date,
        "captured_at_utc": captured_at,
        "evidence_class": "EXACT_PREGAME_FEATURE_SNAPSHOT_UNSETTLED",
        "outcomes_included": False,
        "settlement_included": False,
        "feature_parity_purpose": "preserve_exact_live_state_for_future_strict_prior_date_training_and_certification",
        "model_version": report.get("model_version"),
        "structural_model_version": report.get("structural_model_version"),
        "data_freshness_status": report.get("data_freshness_status"),
        "effective_as_of_date": manifest.get("effective_as_of_date"),
        "advanced_sources": manifest.get("sources") or [],
        "model_training_status": artifact.get("training_status"),
        "model_evidence_class": artifact.get("evidence_class"),
        "row_count": len(rows),
        "rows": rows,
    }
    snapshot_hash = _canonical_hash(snapshot)
    snapshot["snapshot_sha256"] = snapshot_hash

    safe_time = captured_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    safe_time = safe_time.replace("+0000", "Z")
    target_dir = snapshot_root / run_date / "game_conditioned_pregame"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_time}_{snapshot_hash[:16]}.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("snapshot_sha256") != snapshot_hash:
            raise RuntimeError(f"immutable pregame snapshot collision: {target}")
        return target

    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh advanced MLB data and enrich H/TB/HR rows with game-conditioned sequential PA probabilities.")
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--advanced-root", type=Path, default=DEFAULT_ADVANCED_ROOT)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--web-json", type=Path, default=DEFAULT_WEB)
    parser.add_argument("--pregame-snapshot-root", type=Path, default=DEFAULT_PREGAME_ROOT)
    parser.add_argument("--no-refresh", action="store_true")
    return parser.parse_args()


def markdown(report: dict) -> str:
    manifest = report.get("advanced_manifest") or {}
    source_status = manifest.get("source_status") or {}
    model_artifact = report.get("model_artifact") or {}
    target_authority = model_artifact.get("target_authority") or {}
    rows = report.get("rows") or []
    ready = [row for row in rows if row.get("status") in {"READY", "WEAK_SUPPORT"}]
    blocked = [row for row in rows if row.get("status") == "BLOCKED_DATA"]
    lines = [
        "# MLB Game-Conditioned Sequential PA H/TB/HR Validation",
        "",
        f"- Run date: `{report.get('run_date')}`",
        f"- Probability model: `{report.get('model_version')}`",
        f"- Structural simulator: `{report.get('structural_model_version')}`",
        f"- Residual fit: `{model_artifact.get('training_status')}`",
        f"- Evidence class: `{model_artifact.get('evidence_class')}`",
        f"- Evaluated H/TB/HR rows: **{report.get('evaluated_h_tb_hr_rows', 0)}**",
        f"- Modeled rows: **{report.get('modeled_rows', 0)}**",
        f"- Blocked rows: **{report.get('blocked_rows', 0)}**",
        f"- Data freshness: `{report.get('data_freshness_status')}`",
        "",
        "## Target authority",
        "",
        "| Target | Diagnostic gate | Positive authority | Validation status |",
        "|---|---|---|---|",
    ]
    for target in ("H", "TB", "HR"):
        authority = target_authority.get(target) or {}
        lines.append(
            f"| {target} | {bool(authority.get('diagnostic_gate_passed'))} | "
            f"{bool(authority.get('positive_authority'))} | {authority.get('validation_status') or 'UNVALIDATED'} |"
        )
    lines += [
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
        "`legacy/no-vig prior -> game state -> expert activations -> residual logit -> sequential PA distribution -> target-specific uncertainty/authority gate`",
        "",
        "Six experts are evaluated per game: strikeout/contact, contact quality, power/TB/HR, defensive conversion, PA opportunity, and starter-removal/bullpen transition. Global residual coefficients are multiplied by game-specific activations, so a high-K matchup emphasizes contact survival while a low-K matchup can emphasize batted-ball quality. Power relevance increases from H to TB to HR.",
        "",
        "The event tree is `PA -> K | BB | HBP | HR | NON_HR_CONTACT | OTHER`, followed by a non-HR contact outcome distribution. PA and AB are tracked separately and later PA transition away from the starter. H, TB and HR probabilities are calculated from their own simulated outcome arrays rather than from point projections.",
        "",
        "## Probability authority",
        "",
        "The structural simulator directly estimates `P(H>=1)`, `P(TB>=2)` and `P(HR>=1)`. The game-conditioned layer learns a bounded residual around the legacy/no-vig prior in logit space.",
        "",
        "A target that fails expanding-window Brier/log-loss validation is shadow-only and cannot change production probability. A target that clears the diagnostic gate may apply conservative negative-only authority only after train/serve feature parity is independently proven. Positive residual authority additionally requires exact point-in-time advanced-feature evidence.",
        "",
        "Every live run now writes a content-addressed, outcome-free pregame feature snapshot under the publication history tree. These snapshots preserve the exact live expert state needed to eliminate reconstruction and train/serve skew in future validation.",
        "",
        "## Current modeled rows",
        "",
        "| Player | Target | Prior P | Conditioned P | Production P | P(0H) | P(HR>=1) | E[PA] | E[H] | E[TB] | Support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ready[:50]:
        lines.append(
            f"| {row.get('player','')} | {row.get('target','')} | {float(row.get('prior_probability') or 0):.3f} | "
            f"{float(row.get('game_conditioned_probability') or 0):.3f} | {float(row.get('usable_probability') or 0):.3f} | "
            f"{float(row.get('p_h_0') or 0):.3f} | {float(row.get('p_hr_ge_1') or 0):.3f} | "
            f"{float(row.get('expected_pa') or 0):.2f} | {float(row.get('expected_h') or 0):.2f} | "
            f"{float(row.get('expected_tb') or 0):.2f} | {float(row.get('support') or 0):.2f} |"
        )
    lines.extend([
        "",
        "## Blocked/degraded data",
        "",
        f"{len(blocked)} rows were fail-closed because required MLBAM identity, Statcast profile, pitcher profile, or freshness evidence was unavailable.",
        "",
        "## Validation status",
        "",
        "Residual validation uses expanding-window folds. Every held-out block is scored with coefficients fit only on strictly earlier dates. Aggregate Brier and log loss must both improve, at least three folds must exist, and at least 60% of folds must improve both metrics before negative-only authority is considered. Independent authority validation also requires minimum metric gains and train/serve feature parity.",
        "",
        "Historical proxy evidence may initialize shadow residuals but cannot unlock positive authority because the processed corpus does not preserve every exact Savant/FanGraphs pregame state. Exact point-in-time evidence is required for promotion.",
        "",
        "## Known limitations",
        "",
        "- Specific fielder OAA/location assignment remains zero-centered/uncertain when unavailable; no fielder data are fabricated.",
        "- Direct BvP remains heavily shrunk because sample sizes are usually small.",
        "- FanGraphs xFIP/SIERA availability is source-dependent and missingness lowers evidence strength.",
        "- Bullpen identity is still a transition toward neutral relief state until named-reliever distributions are supported.",
        "- Weather currently consumes temperature when present; wind/humidity are not invented when absent.",
        "- Handedness is preserved in game state, but no fixed platoon coefficient is fabricated without split evidence.",
        "- Live pitch-compatibility is preserved in snapshots but cannot authorize a residual until the same feature is represented in training evidence.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not args.no_refresh:
        refresh_advanced_profiles_incremental(pool_csv=args.pool_csv, run_date=args.run_date, advanced_root=args.advanced_root)
    report = enrich_pool_with_sequential_pa(pool_csv=args.pool_csv, run_date=args.run_date, advanced_root=args.advanced_root, refresh_data=False, trials=args.trials)
    snapshot_path = capture_pregame_feature_snapshot(
        pool_csv=args.pool_csv,
        report=report,
        run_date=args.run_date,
        snapshot_root=args.pregame_snapshot_root,
    )
    report["pregame_feature_snapshot"] = {
        "path": str(snapshot_path.relative_to(REPO_ROOT)) if snapshot_path.is_relative_to(REPO_ROOT) else str(snapshot_path),
        "captured": True,
        "outcomes_included": False,
    }
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
        "pregame_feature_snapshot": report.get("pregame_feature_snapshot"),
        "data_freshness_status": report.get("data_freshness_status"),
        "source_status": (report.get("advanced_manifest") or {}).get("source_status") or {},
        "effective_as_of_date": (report.get("advanced_manifest") or {}).get("effective_as_of_date"),
        "evaluated_h_tb_hr_rows": report.get("evaluated_h_tb_hr_rows"),
        "modeled_rows": report.get("modeled_rows"),
        "blocked_rows": report.get("blocked_rows"),
        "candidates": [
            {key: row.get(key) for key in (
                "player", "target", "status", "raw_probability", "prior_probability", "game_conditioned_probability",
                "usable_probability", "lcb", "p_h_0", "p_hr_ge_1", "expected_pa", "expected_h", "expected_tb", "support", "uncertainty",
                "authority", "expert_weights", "pitch_compatibility"
            )}
            for row in (report.get("rows") or [])
        ],
    }
    args.web_json.write_text(json.dumps(web, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        **{key: report.get(key) for key in ("run_date", "model_version", "structural_model_version", "evaluated_h_tb_hr_rows", "modeled_rows", "blocked_rows", "data_freshness_status")},
        "source_status": web["source_status"],
        "effective_as_of_date": web["effective_as_of_date"],
        "model_artifact": web["model_artifact"],
        "pregame_feature_snapshot": web["pregame_feature_snapshot"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
