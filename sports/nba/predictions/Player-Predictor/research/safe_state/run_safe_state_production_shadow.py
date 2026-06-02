from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.audit_price_provenance import run_price_provenance_audit
from research.market_quality.build_price_defense_shadow_boards import build_price_defense_shadow_boards
from research.market_quality.report_edge_defense import build_edge_defense_report
from research.safe_state.evaluate_safe_state_shadow_results import BOARD_VARIANTS, evaluate_safe_state_shadow_results
from research.safe_state.forecastability_blocker_resolution_report import build_forecastability_blocker_resolution_report
from research.safe_state.forecastability_root_cause_search import run_forecastability_root_cause_search
from research.safe_state.run_safe_state_evidence_lifecycle import run_safe_state_evidence_lifecycle
from research.safe_state.safe_state_evidence_gap_report import build_safe_state_evidence_gap_report
from research.safe_state.safe_state_promotion_gate import evaluate_safe_state_promotion_gate
from research.safe_state.safe_state_shadow_boards import (
    build_safe_state_shadow_boards,
    write_safe_state_shadow_boards_from_annotated,
)

DEFAULT_CONFIG = PLAYER_PREDICTOR_ROOT / "config" / "safe_state_production_test.yaml"
DEFAULT_OUTPUT_ROOT = PLAYER_PREDICTOR_ROOT.parents[1] / "validation" / "production_shadow" / "safe_state"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("safe_state_production_test", config)
    return value if isinstance(value, dict) else {}


def _copy_csv_or_empty(source: Path | None, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source and source.exists():
        shutil.copyfile(source, dest)
    else:
        pd.DataFrame(
            columns=[
                "candidate_id",
                "game_id",
                "market_date",
                "game_date",
                "player",
                "player_name",
                "target",
                "direction",
                "side",
                "line",
                "market_line",
                "market_type",
            ]
        ).to_csv(dest, index=False)
    return dest


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_from_csv(csv_path: Path, json_path: Path) -> None:
    frame = _read_csv(csv_path)
    _write_json(json_path, {"rows": frame.to_dict(orient="records")})


def _provider_health_status(report: dict[str, Any], *, require_event_start: bool, require_side_price: bool) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not bool(report.get("api_key_visible", False)):
        reasons.append("sportsgameodds_api_key_not_visible")
    if not bool(report.get("request_success", False)):
        reasons.append("provider_request_failed")
    if int(report.get("events_returned", 0) or 0) <= 0:
        reasons.append("no_events_returned")
    if require_event_start and int(report.get("startsAt_available_count", 0) or 0) <= 0:
        reasons.append("event_start_time_missing")
    if require_side_price and int(report.get("side_specific_price_count", 0) or 0) <= 0:
        reasons.append("side_specific_prices_missing")
    return ("PROVIDER_OK" if not reasons else "PROVIDER_BLOCKED"), reasons


def _sanitize_provider_report(report: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in report.items():
        key_text = str(key).lower()
        if key_text == "api_key_visible":
            sanitized[key] = bool(value)
            continue
        if "api_key" in key_text or "secret" in key_text or "token" in key_text:
            continue
        sanitized[key] = value
    return sanitized


def _write_provider_blocked(
    *,
    output_dir: Path,
    run_id: str,
    run_date: str,
    config_path: Path,
    provider_report: dict[str, Any],
    blocked_reasons: list[str],
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "run_date": run_date,
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "ring": "RING_1_PRODUCTION_SHADOW",
        "provider": "sportsgameodds",
        "api_key_visible": bool(provider_report.get("api_key_visible", False)),
        "provider_health_status": "PROVIDER_BLOCKED",
        "provider_healthcheck_path": str(output_dir / "provider_healthcheck.json"),
        "live_snapshot_path": "",
        "production_board_path": "",
        "candidate_pool_path": "",
        "all_output_paths": [str(output_dir / "provider_healthcheck.json")],
        "production_behavior_changed": False,
        "promotion_claim": False,
        "promotion_ready": False,
        "blocked_reasons": blocked_reasons,
    }
    _write_json(output_dir / "safe_state_production_shadow_manifest.json", manifest)
    report = {
        "status": "PROVIDER_BLOCKED",
        "manifest": manifest,
        "provider_healthcheck": provider_report,
        "production_behavior_changed": False,
        "promotion_claim": False,
        "promotion_ready": False,
    }
    _write_json(output_dir / "safe_state_production_shadow_report.json", report)
    _write_shadow_markdown(output_dir / "safe_state_production_shadow_report.md", report)
    return report


def _run_provider_healthcheck(
    *,
    output_dir: Path,
    override_json: Path | None,
    event_limit: int,
) -> dict[str, Any]:
    output_path = output_dir / "provider_healthcheck.json"
    if override_json and override_json.exists():
        report = _sanitize_provider_report(_read_json(override_json))
        _write_json(output_path, report)
        return report
    cmd = [
        sys.executable,
        str(PLAYER_PREDICTOR_ROOT / "scripts" / "check_nba_provider_health.py"),
        "--output-dir",
        str(output_dir),
        "--event-limit",
        str(int(event_limit)),
    ]
    subprocess.run(cmd, cwd=PLAYER_PREDICTOR_ROOT, check=False, capture_output=True, text=True)
    report = _sanitize_provider_report(_read_json(output_path))
    if not report:
        report = {
            "fetched_at_utc": _utc_now(),
            "api_key_visible": False,
            "request_success": False,
            "events_returned": 0,
            "odds_rows_returned": 0,
            "startsAt_available_count": 0,
            "side_specific_price_count": 0,
            "books_observed": [],
            "failure_reason": "provider healthcheck did not write output",
        }
        _write_json(output_path, report)
    return report


def _run_daily_pipeline(
    *,
    season: int,
    run_date: str,
    market_bookmakers: str | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PLAYER_PREDICTOR_ROOT / "scripts" / "run_daily_market_pipeline.py"),
        "--season",
        str(int(season)),
        "--run-date",
        str(run_date),
        "--snapshot-policy",
        "live_only",
        "--market-provider",
        "sportsgameodds",
        "--skip-cutoff-meta-monitor",
        "--skip-export-web",
        "--skip-build-site",
    ]
    if market_bookmakers:
        cmd.extend(["--market-bookmakers", str(market_bookmakers)])
    completed = subprocess.run(cmd, cwd=PLAYER_PREDICTOR_ROOT, check=True, capture_output=True, text=True)
    manifest = _locate_daily_pipeline_manifest(run_date)
    manifest["daily_pipeline_command"] = cmd
    manifest["daily_pipeline_returncode"] = int(completed.returncode)
    return manifest


def _locate_daily_pipeline_manifest(run_date: str) -> dict[str, Any]:
    stamp = run_date.replace("-", "")
    run_dir = PLAYER_PREDICTOR_ROOT / "model" / "analysis" / "daily_runs" / stamp
    matches = sorted(run_dir.glob(f"daily_market_pipeline_manifest_{stamp}.json"))
    return _read_json(matches[-1]) if matches else {}


def _resolve_paths(
    *,
    run_date: str,
    production_board_csv: Path | None,
    candidate_pool_csv: Path | None,
    live_snapshot_path: Path | None,
    slate_csv: Path | None,
    daily_manifest: dict[str, Any],
) -> dict[str, Path | None]:
    if not daily_manifest:
        daily_manifest = _locate_daily_pipeline_manifest(run_date)
    return {
        "production_board_csv": production_board_csv or (Path(str(daily_manifest.get("final_csv"))) if daily_manifest.get("final_csv") else None),
        "candidate_pool_csv": candidate_pool_csv or (Path(str(daily_manifest.get("selector_csv"))) if daily_manifest.get("selector_csv") else None),
        "live_snapshot_path": live_snapshot_path or (Path(str(daily_manifest.get("current_market_snapshot"))) if daily_manifest.get("current_market_snapshot") else None),
        "slate_csv": slate_csv or (Path(str(daily_manifest.get("slate_csv"))) if daily_manifest.get("slate_csv") else None),
        "daily_manifest_path": Path(str(daily_manifest.get("manifest_path"))) if daily_manifest.get("manifest_path") else None,
    }


def _ensure_evidence_inputs(
    *,
    output_dir: Path,
    candidate_pool_path: Path,
    production_board_path: Path,
    annotated_candidates_csv: Path | None,
    blocker_resolution_rows_csv: Path | None,
    root_cause_rows_csv: Path | None,
    data_proc_dir: Path | None,
    historical_csv: Path | None,
) -> tuple[Path, Path, Path, Path]:
    if annotated_candidates_csv and annotated_candidates_csv.exists():
        annotated = _read_csv(annotated_candidates_csv)
        production = _read_csv(production_board_path)
        write_safe_state_shadow_boards_from_annotated(
            output_dir=output_dir,
            annotated=annotated,
            production=production,
            input_paths={
                "annotated_candidates_csv": str(annotated_candidates_csv),
                "production_board_csv": str(production_board_path),
            },
        )
    else:
        build_safe_state_shadow_boards(
            output_dir=output_dir,
            candidate_pool_csv=candidate_pool_path,
            production_board_csv=production_board_path,
            historical_csv=historical_csv,
        )

    annotated_path = output_dir / "safe_state_annotated_candidates.csv"
    gap_report = build_safe_state_evidence_gap_report(
        output_dir=output_dir,
        candidate_pool_csv=candidate_pool_path,
        production_board_csv=production_board_path,
        safe_state_dir=output_dir,
        annotated_candidates_csv=annotated_path,
        historical_csv=historical_csv,
    )
    blockers_path = Path(gap_report["output_paths"]["candidate_blockers_csv"])

    resolution_path = output_dir / "forecastability_blocker_resolution_rows.csv"
    if blocker_resolution_rows_csv and blocker_resolution_rows_csv.exists():
        shutil.copyfile(blocker_resolution_rows_csv, resolution_path)
    else:
        build_forecastability_blocker_resolution_report(
            output_dir=output_dir,
            annotated_candidates_csv=annotated_path,
            candidate_blockers_csv=blockers_path,
            data_proc_dir=data_proc_dir,
        )

    root_path = output_dir / "forecastability_root_cause_rows.csv"
    if root_cause_rows_csv and root_cause_rows_csv.exists():
        shutil.copyfile(root_cause_rows_csv, root_path)
    else:
        run_forecastability_root_cause_search(
            output_dir=output_dir,
            annotated_candidates_csv=annotated_path,
            resolution_rows_csv=resolution_path,
            data_proc_dir=data_proc_dir,
            candidate_pool_csv=candidate_pool_path,
        )
    return annotated_path, blockers_path, resolution_path, root_path


def _write_membership(output_dir: Path) -> Path:
    production = _read_csv(output_dir / "production_board_as_is.csv")
    prod_ids = set(production.get("candidate_id", pd.Series(dtype=str)).fillna("").astype(str).tolist()) if not production.empty else set()
    records: list[dict[str, Any]] = []
    for variant in BOARD_VARIANTS:
        board = _read_csv(output_dir / f"{variant}.csv")
        if board.empty:
            continue
        for _, row in board.iterrows():
            candidate_id = str(row.get("candidate_id", ""))
            records.append(
                {
                    "board_variant": variant,
                    "candidate_id": candidate_id,
                    "player": row.get("player", row.get("player_name", "")),
                    "market_date": row.get("market_date", row.get("game_date", "")),
                    "target": row.get("target", ""),
                    "side": row.get("side", row.get("direction", "")),
                    "line": row.get("line", row.get("market_line", "")),
                    "in_production": bool(candidate_id and candidate_id in prod_ids),
                    "shadow_only": bool(candidate_id and candidate_id not in prod_ids and variant != "production_board_as_is"),
                }
            )
    path = output_dir / "safe_state_shadow_board_membership.csv"
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


def _summarize_shadow_comparison(output_dir: Path, filename: str) -> dict[str, Any]:
    summary = _read_csv(output_dir / "safe_state_shadow_variant_summary.csv")
    payload = {
        "variant_summaries": summary.to_dict(orient="records") if not summary.empty else [],
        "production_behavior_changed": False,
        "promotion_claim": False,
    }
    _write_json(output_dir / filename, payload)
    return payload


def _collect_output_paths(output_dir: Path) -> list[str]:
    return sorted(str(path) for path in output_dir.glob("*") if path.is_file())


def _empty_price_audit(output_dir: Path) -> dict[str, Any]:
    audit_path = output_dir / "price_provenance_audit.csv"
    untrusted_path = output_dir / "selected_rows_price_untrusted.csv"
    summary_path = output_dir / "price_provenance_audit_summary.json"
    columns = [
        "record_scope",
        "candidate_id",
        "market_type",
        "market_side_price",
        "market_side_break_even",
        "price_validity_status",
        "timestamp_safe_flag",
        "edge_defendability_tier",
        "price_gap_blocks_validation",
    ]
    pd.DataFrame(columns=columns).to_csv(audit_path, index=False)
    pd.DataFrame(columns=columns).to_csv(untrusted_path, index=False)
    summary = {
        "total_candidate_rows": 0,
        "total_selected_rows": 0,
        "percent_with_valid_timestamp_safe_market_side_price": 0.0,
        "percent_with_market_side_break_even": 0.0,
        "percent_with_odds_snapshot_time": 0.0,
        "percent_with_price_source": 0.0,
        "rows_where_edge_cannot_be_validated": 0,
        "rows_that_would_be_edge_defendable": 0,
        "rows_that_would_be_edge_price_dependent": 0,
        "rows_that_would_fail_price": 0,
        "output_paths": {
            "price_provenance_audit_csv": str(audit_path),
            "price_provenance_audit_summary_json": str(summary_path),
            "selected_rows_price_untrusted_csv": str(untrusted_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def run_safe_state_production_shadow(
    *,
    season: int,
    run_date: str,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path,
    provider_healthcheck_json: Path | None = None,
    production_board_csv: Path | None = None,
    candidate_pool_csv: Path | None = None,
    annotated_candidates_csv: Path | None = None,
    blocker_resolution_rows_csv: Path | None = None,
    root_cause_rows_csv: Path | None = None,
    live_snapshot_path: Path | None = None,
    slate_csv: Path | None = None,
    historical_csv: Path | None = None,
    data_proc_dir: Path | None = None,
    skip_production_pipeline: bool = False,
    market_bookmakers: str | None = None,
    provider_event_limit: int = 50,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _section(_load_config(config_path))
    provider_config = config.get("live_provider", {}) if isinstance(config.get("live_provider", {}), dict) else {}
    run_id = f"safe_state_prod_shadow_{run_date.replace('-', '')}_{datetime.now(timezone.utc).strftime('%H%M%SZ')}"

    provider_report = _run_provider_healthcheck(output_dir=output_dir, override_json=provider_healthcheck_json, event_limit=provider_event_limit)
    provider_status, provider_blockers = _provider_health_status(
        provider_report,
        require_event_start=bool(provider_config.get("require_event_start_time", True)),
        require_side_price=bool(provider_config.get("require_side_specific_price", True)),
    )
    if provider_status == "PROVIDER_BLOCKED":
        return _write_provider_blocked(
            output_dir=output_dir,
            run_id=run_id,
            run_date=run_date,
            config_path=config_path,
            provider_report=provider_report,
            blocked_reasons=provider_blockers,
        )

    daily_manifest: dict[str, Any] = {}
    if not skip_production_pipeline and (production_board_csv is None or candidate_pool_csv is None):
        daily_manifest = _run_daily_pipeline(season=season, run_date=run_date, market_bookmakers=market_bookmakers)

    paths = _resolve_paths(
        run_date=run_date,
        production_board_csv=production_board_csv,
        candidate_pool_csv=candidate_pool_csv,
        live_snapshot_path=live_snapshot_path,
        slate_csv=slate_csv,
        daily_manifest=daily_manifest,
    )
    production_path = _copy_csv_or_empty(paths["production_board_csv"], output_dir / "production_board_as_is.csv")
    candidate_path = _copy_csv_or_empty(paths["candidate_pool_csv"] or annotated_candidates_csv, output_dir / "candidate_pool.csv")
    _json_from_csv(production_path, output_dir / "production_board_as_is.json")
    production_hash = _hash_file(production_path)

    if _read_csv(production_path).empty and _read_csv(candidate_path).empty:
        price_audit = _empty_price_audit(output_dir)
    else:
        price_audit = run_price_provenance_audit(
            selected_board_paths=[production_path],
            candidate_pool_paths=[candidate_path],
            daily_runs_dirs=[],
            output_dir=output_dir,
        )
    edge_report = build_edge_defense_report(
        output_dir=output_dir,
        price_audit_csv=output_dir / "price_provenance_audit.csv",
        selected_board_csv=production_path,
        candidate_pool_csv=candidate_path,
    )
    price_shadow = build_price_defense_shadow_boards(
        output_dir=output_dir,
        candidate_pool_csv=candidate_path,
        production_board_csv=production_path,
    )
    annotated_path, blockers_path, resolution_path, root_path = _ensure_evidence_inputs(
        output_dir=output_dir,
        candidate_pool_path=candidate_path,
        production_board_path=production_path,
        annotated_candidates_csv=annotated_candidates_csv,
        blocker_resolution_rows_csv=blocker_resolution_rows_csv,
        root_cause_rows_csv=root_cause_rows_csv,
        data_proc_dir=data_proc_dir,
        historical_csv=historical_csv,
    )

    lifecycle = run_safe_state_evidence_lifecycle(
        output_dir=output_dir,
        annotated_candidates_csv=annotated_path,
        blocker_resolution_rows_csv=resolution_path,
        root_cause_rows_csv=root_path,
        candidate_blockers_csv=blockers_path,
        data_proc_dir=data_proc_dir,
        run_id=run_id,
        evaluate_settlement=False,
    )
    settlement = evaluate_safe_state_shadow_results(board_dir=output_dir, output_dir=output_dir)
    membership_path = _write_membership(output_dir)
    price_comparison = _summarize_shadow_comparison(output_dir, "price_defense_shadow_comparison.json")
    safe_comparison = _summarize_shadow_comparison(output_dir, "safe_state_shadow_comparison.json")
    gate = evaluate_safe_state_promotion_gate(aggregate_metrics_csv=output_dir / "safe_state_shadow_settlement_metrics.csv", config_path=config_path, output_dir=output_dir)

    manifest = {
        "run_id": run_id,
        "run_date": run_date,
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "ring": config.get("ring", "RING_1_PRODUCTION_SHADOW"),
        "provider": provider_config.get("provider", "sportsgameodds"),
        "api_key_visible": bool(provider_report.get("api_key_visible", False)),
        "provider_health_status": provider_status,
        "provider_healthcheck_path": str(output_dir / "provider_healthcheck.json"),
        "live_snapshot_path": str(paths["live_snapshot_path"] or ""),
        "production_board_path": str(production_path),
        "candidate_pool_path": str(candidate_path),
        "slate_csv": str(paths["slate_csv"] or ""),
        "production_board_hash": production_hash,
        "all_output_paths": _collect_output_paths(output_dir),
        "production_behavior_changed": False,
        "promotion_claim": False,
        "promotion_ready": False,
        "blocked_reasons": gate.get("blocked_reasons", []),
    }
    _write_json(output_dir / "safe_state_production_shadow_manifest.json", manifest)

    report = {
        "status": "WAITING_FOR_SETTLEMENT",
        "manifest": manifest,
        "provider_healthcheck": provider_report,
        "price_provenance_audit": price_audit,
        "edge_defense_report": edge_report,
        "price_defense_shadow": price_shadow,
        "safe_state_lifecycle": lifecycle,
        "settlement": settlement,
        "price_defense_shadow_comparison": price_comparison,
        "safe_state_shadow_comparison": safe_comparison,
        "membership_csv": str(membership_path),
        "promotion_gate": gate,
        "production_behavior_changed": False,
        "promotion_claim": False,
        "promotion_ready": False,
    }
    _write_json(output_dir / "safe_state_production_shadow_report.json", report)
    _write_shadow_markdown(output_dir / "safe_state_production_shadow_report.md", report)
    return report


def _write_shadow_markdown(path: Path, report: dict[str, Any]) -> None:
    manifest = report.get("manifest", {})
    lines = [
        "# Safe-State Production Shadow Report",
        "",
        f"- Status: {report.get('status', '')}",
        f"- Ring: {manifest.get('ring', 'RING_1_PRODUCTION_SHADOW')}",
        f"- Provider health: {manifest.get('provider_health_status', '')}",
        f"- Production board: {manifest.get('production_board_path', '')}",
        f"- Candidate pool: {manifest.get('candidate_pool_path', '')}",
        f"- Production board hash: {manifest.get('production_board_hash', '')}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "- Promotion ready: false",
        "",
        "## Blocked Reasons",
    ]
    blockers = manifest.get("blocked_reasons", []) or report.get("promotion_gate", {}).get("blocked_reasons", [])
    lines.extend([f"- {reason}" for reason in blockers] or ["- none"])
    lines.extend(
        [
            "",
            "## Guardrails",
            "- RING_1 is evidence collection only.",
            "- No staking, auto-betting, selector gate, or sidecar enforcement is enabled.",
            "- Pending outcomes remain PENDING until settlement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RING_1 production-shadow safe-state evidence collection.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--provider-healthcheck-json", type=Path)
    parser.add_argument("--production-board-csv", type=Path)
    parser.add_argument("--candidate-pool-csv", type=Path)
    parser.add_argument("--annotated-candidates-csv", type=Path)
    parser.add_argument("--blocker-resolution-rows-csv", type=Path)
    parser.add_argument("--root-cause-rows-csv", type=Path)
    parser.add_argument("--live-snapshot-path", type=Path)
    parser.add_argument("--slate-csv", type=Path)
    parser.add_argument("--historical-csv", type=Path)
    parser.add_argument("--data-proc-dir", type=Path)
    parser.add_argument("--skip-production-pipeline", action="store_true")
    parser.add_argument("--market-bookmakers")
    parser.add_argument("--provider-event-limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / str(args.run_date).replace("-", ""))
    report = run_safe_state_production_shadow(
        season=int(args.season),
        run_date=str(args.run_date),
        config_path=args.config,
        output_dir=output_dir,
        provider_healthcheck_json=args.provider_healthcheck_json,
        production_board_csv=args.production_board_csv,
        candidate_pool_csv=args.candidate_pool_csv,
        annotated_candidates_csv=args.annotated_candidates_csv,
        blocker_resolution_rows_csv=args.blocker_resolution_rows_csv,
        root_cause_rows_csv=args.root_cause_rows_csv,
        live_snapshot_path=args.live_snapshot_path,
        slate_csv=args.slate_csv,
        historical_csv=args.historical_csv,
        data_proc_dir=args.data_proc_dir,
        skip_production_pipeline=bool(args.skip_production_pipeline),
        market_bookmakers=args.market_bookmakers,
        provider_event_limit=int(args.provider_event_limit),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
