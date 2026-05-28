from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.aggregate_safe_state_shadow_evidence import DEFAULT_BASE_DIR


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    run_dirs = [path for path in base_dir.iterdir() if path.is_dir() and path.name != "aggregate"]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda path: path.name)[-1]


def build_safe_state_production_status(
    *,
    base_dir: Path = DEFAULT_BASE_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or (base_dir / "aggregate")
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_run_dir(base_dir)
    aggregate_gate = _read_json(base_dir / "aggregate" / "safe_state_promotion_gate_status.json")
    manifest = _read_json(latest / "safe_state_production_shadow_manifest.json") if latest else {}
    metrics = _read_csv(latest / "safe_state_shadow_settlement_metrics.csv") if latest else pd.DataFrame()

    if manifest.get("provider_health_status") == "PROVIDER_BLOCKED":
        status = "PROVIDER_BLOCKED"
    elif not metrics.empty and pd.to_numeric(metrics.get("resolved_rows", 0), errors="coerce").fillna(0).sum() > 0:
        status = "SETTLED_EVIDENCE_ACCUMULATING"
    elif latest:
        status = "WAITING_FOR_SETTLEMENT"
    else:
        status = "PROMOTION_BLOCKED"

    report = {
        "status": status,
        "latest_run_dir": str(latest) if latest else "",
        "latest_run_id": manifest.get("run_id", ""),
        "ring": manifest.get("ring", "RING_1_PRODUCTION_SHADOW"),
        "promotion_ready": False,
        "promotion_status": aggregate_gate.get("promotion_status", "NOT_PROMOTION_ELIGIBLE"),
        "blocked_reasons": aggregate_gate.get("blocked_reasons", ["no_aggregate_evidence"]),
        "production_behavior_changed": False,
        "promotion_claim": False,
        "next_required_steps": aggregate_gate.get(
            "next_required_steps",
            ["run_ring_1_production_shadow", "wait_for_settlement", "aggregate_multi_slate_evidence"],
        ),
    }
    (output_dir / "safe_state_production_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "safe_state_production_status.md", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Safe-State Production Status",
        "",
        f"- Status: {report['status']}",
        f"- Ring: {report['ring']}",
        "- Promotion ready: false",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "## Blockers",
    ]
    lines.extend([f"- {reason}" for reason in report.get("blocked_reasons", [])])
    lines.extend(["", "## Next Steps"])
    lines.extend([f"- {step}" for step in report.get("next_required_steps", [])])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write safe-state production-shadow status dashboard artifacts.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_safe_state_production_status(base_dir=args.base_dir, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
