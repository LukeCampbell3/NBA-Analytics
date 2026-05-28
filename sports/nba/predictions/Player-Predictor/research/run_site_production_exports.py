from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[1]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.player_simulation.simulate_next_season_player_states import simulate_next_season_player_states
from research.safe_state.run_safe_state_production_shadow import run_safe_state_production_shadow
from research.site_export.export_safe_state_site_cards import export_safe_state_site_cards


NBA_ROOT = PLAYER_PREDICTOR_ROOT.parents[1]
DEFAULT_SITE_EXPORT_DIR = NBA_ROOT / "validation" / "production_shadow" / "site_exports"
DEFAULT_SAFE_STATE_BASE = NBA_ROOT / "validation" / "production_shadow" / "safe_state"
DEFAULT_SIM_DIR = NBA_ROOT / "validation" / "production_shadow" / "player_simulation"
DEFAULT_WEB_DATA_DIR = NBA_ROOT / "web" / "data"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_if_exists(source: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, dest)
        return str(dest)
    return ""


def _validate_site_exports(*, output_dir: Path, safe_report: dict[str, Any], sim_manifest: dict[str, Any] | None) -> dict[str, Any]:
    required = [
        "safe_state_latest.json",
        "safe_state_latest.csv",
        "safe_state_cards.json",
        "site_manifest.json",
    ]
    if sim_manifest:
        required.extend(["player_simulation_cards.json", "player_simulation_summary.csv"])
    missing = [name for name in required if not (output_dir / name).exists()]
    safe_validation = safe_report.get("validation", {})
    sim_validation_path = output_dir / "player_simulation_validation_report.json"
    sim_validation = {}
    if sim_validation_path.exists():
        sim_validation = json.loads(sim_validation_path.read_text(encoding="utf-8"))
    return {
        "missing_required_files": missing,
        "safe_state_cards_include_shadow_status": bool(safe_validation.get("safe_state_cards_include_shadow_status", True)),
        "player_cards_include_uncertainty": bool(sim_validation.get("all_cards_include_uncertainty", True)) if sim_manifest else True,
        "production_behavior_changed": False,
        "promotion_claim": False,
        "staking_field_enabled": False,
        "validation_passed": not missing
        and bool(safe_validation.get("promotion_ready", False)) is False
        and (bool(sim_validation.get("all_cards_include_uncertainty", True)) if sim_manifest else True),
    }


def run_site_production_exports(
    *,
    season: int,
    run_date: str,
    safe_state_run_dir: Path | None,
    site_output_dir: Path,
    simulate_next_season: bool = False,
    simulation_count: int = 10000,
    shadow_only: bool = True,
    skip_safe_state_run: bool = False,
    copy_to_web_data: Path | None = DEFAULT_WEB_DATA_DIR,
) -> dict[str, Any]:
    if not shadow_only:
        raise ValueError("Only --shadow-only production site exports are supported.")
    site_output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = run_date.replace("-", "")
    safe_state_run_dir = safe_state_run_dir or (DEFAULT_SAFE_STATE_BASE / run_stamp)

    safe_state_report: dict[str, Any] = {}
    if not skip_safe_state_run:
        safe_state_report = run_safe_state_production_shadow(
            season=season,
            run_date=run_date,
            output_dir=safe_state_run_dir,
        )

    safe_export = export_safe_state_site_cards(
        safe_state_run_dir=safe_state_run_dir,
        output_dir=site_output_dir,
        run_date=run_date,
    )

    sim_manifest: dict[str, Any] | None = None
    if simulate_next_season:
        sim_manifest = simulate_next_season_player_states(
            data_proc_dir=PLAYER_PREDICTOR_ROOT / "Data-Proc",
            output_dir=DEFAULT_SIM_DIR,
            cutoff_date=run_date,
            simulation_count=int(simulation_count),
        )
        _copy_if_exists(Path(sim_manifest["output_paths"]["cards_json"]), site_output_dir / "player_simulation_cards.json")
        _copy_if_exists(Path(sim_manifest["output_paths"]["csv"]), site_output_dir / "player_simulation_summary.csv")
        _copy_if_exists(Path(sim_manifest["output_paths"]["validation_report"]), site_output_dir / "player_simulation_validation_report.json")
    else:
        existing_cards = DEFAULT_SIM_DIR / "player_simulation_cards.json"
        existing_summary = DEFAULT_SIM_DIR / "next_season_player_simulations.csv"
        existing_validation = DEFAULT_SIM_DIR / "player_simulation_validation_report.json"
        _copy_if_exists(existing_cards, site_output_dir / "player_simulation_cards.json")
        _copy_if_exists(existing_summary, site_output_dir / "player_simulation_summary.csv")
        _copy_if_exists(existing_validation, site_output_dir / "player_simulation_validation_report.json")

    manifest = {
        "run_id": f"site_exports_{run_stamp}_{datetime.now(timezone.utc).strftime('%H%M%SZ')}",
        "run_date": run_date,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff_date": run_date,
        "provider": "sportsgameodds",
        "safe_state_run_dir": str(safe_state_run_dir),
        "site_output_dir": str(site_output_dir),
        "safe_state_report_status": safe_state_report.get("status", "existing_run_used"),
        "simulation_enabled": bool(simulate_next_season),
        "simulation_manifest": sim_manifest or {},
        "production_behavior_changed": False,
        "promotion_ready": False,
        "shadow_only": True,
        "staking_enabled": False,
        "auto_bet_enabled": False,
    }
    _write_json(site_output_dir / "site_manifest.json", manifest)

    validation = _validate_site_exports(output_dir=site_output_dir, safe_report=safe_export, sim_manifest=sim_manifest)
    _write_json(site_output_dir / "site_export_validation_report.json", validation)

    copied_to_web: list[str] = []
    if copy_to_web_data:
        for name in [
            "safe_state_latest.json",
            "safe_state_latest.csv",
            "safe_state_cards.json",
            "player_simulation_cards.json",
            "player_simulation_summary.csv",
            "player_simulation_validation_report.json",
            "safe_state_site_validation_report.json",
            "site_manifest.json",
        ]:
            dest = _copy_if_exists(site_output_dir / name, copy_to_web_data / name)
            if dest:
                copied_to_web.append(dest)

    report = {
        "site_manifest": manifest,
        "safe_state_export": safe_export,
        "simulation_manifest": sim_manifest or {},
        "validation": validation,
        "copied_to_web_data": copied_to_web,
        "production_behavior_changed": False,
        "promotion_ready": False,
        "shadow_only": True,
    }
    _write_json(site_output_dir / "site_export_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate production-site safe-state and player simulation exports.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--safe-state-run-dir", type=Path)
    parser.add_argument("--site-output-dir", type=Path, default=DEFAULT_SITE_EXPORT_DIR)
    parser.add_argument("--simulate-next-season", action="store_true")
    parser.add_argument("--simulation-count", type=int, default=10000)
    parser.add_argument("--shadow-only", action="store_true")
    parser.add_argument("--skip-safe-state-run", action="store_true")
    parser.add_argument("--no-copy-to-web-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_site_production_exports(
        season=int(args.season),
        run_date=str(args.run_date),
        safe_state_run_dir=args.safe_state_run_dir,
        site_output_dir=args.site_output_dir,
        simulate_next_season=bool(args.simulate_next_season),
        simulation_count=int(args.simulation_count),
        shadow_only=bool(args.shadow_only),
        skip_safe_state_run=bool(args.skip_safe_state_run),
        copy_to_web_data=None if args.no_copy_to_web_data else DEFAULT_WEB_DATA_DIR,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
