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

from research.safe_state.safe_state_promotion_gate import evaluate_safe_state_promotion_gate

DEFAULT_BASE_DIR = PLAYER_PREDICTOR_ROOT.parents[1] / "validation" / "production_shadow" / "safe_state"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted([path for path in base_dir.iterdir() if path.is_dir() and path.name != "aggregate"])


def _load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics = _read_csv(run_dir / "safe_state_shadow_settlement_metrics.csv")
    if metrics.empty:
        return metrics
    manifest = _read_json(run_dir / "safe_state_production_shadow_manifest.json")
    metrics = metrics.copy()
    metrics["run_dir"] = str(run_dir)
    metrics["run_date"] = str(manifest.get("run_date") or run_dir.name)
    metrics["run_id"] = str(manifest.get("run_id") or run_dir.name)
    return metrics


def _weighted_mean(group: pd.DataFrame, column: str, weight_column: str = "resolved_rows") -> float | None:
    if column not in group.columns:
        return None
    values = pd.to_numeric(group[column], errors="coerce")
    weights = pd.to_numeric(group.get(weight_column, pd.Series(1, index=group.index)), errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0)
    if not mask.any():
        return None
    return float((values.loc[mask] * weights.loc[mask]).sum() / weights.loc[mask].sum())


def _aggregate_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "total_rows",
                "resolved_rows",
                "pending_rows",
                "wins",
                "losses",
                "pushes",
                "hit_rate",
                "roi",
                "profit_units",
                "brier",
                "ece",
                "calibration_gap",
            ]
        )
    variant_col = "variant" if "variant" in rows.columns else "board_variant"
    records: list[dict[str, Any]] = []
    for variant, group in rows.groupby(variant_col, dropna=False):
        wins = int(pd.to_numeric(group.get("wins", 0), errors="coerce").fillna(0).sum())
        losses = int(pd.to_numeric(group.get("losses", 0), errors="coerce").fillna(0).sum())
        resolved = int(pd.to_numeric(group.get("resolved_rows", 0), errors="coerce").fillna(0).sum())
        pending = int(pd.to_numeric(group.get("pending_rows", 0), errors="coerce").fillna(0).sum())
        profit = float(pd.to_numeric(group.get("profit_units", 0.0), errors="coerce").fillna(0.0).sum())
        denom = wins + losses
        records.append(
            {
                "variant": str(variant),
                "run_count": int(group["run_date"].nunique()) if "run_date" in group.columns else int(len(group)),
                "total_rows": int(pd.to_numeric(group.get("rows", 0), errors="coerce").fillna(0).sum()),
                "resolved_rows": resolved,
                "pending_rows": pending,
                "wins": wins,
                "losses": losses,
                "pushes": int(pd.to_numeric(group.get("pushes", 0), errors="coerce").fillna(0).sum()),
                "hit_rate": None if denom == 0 else float(wins / denom),
                "roi": None if denom == 0 else float(profit / denom),
                "profit_units": profit,
                "brier": _weighted_mean(group, "brier"),
                "ece": _weighted_mean(group, "ece"),
                "calibration_gap": _weighted_mean(group, "calibration_gap"),
                "production_rows_removed": int(pd.to_numeric(group.get("production_rows_removed", 0), errors="coerce").fillna(0).sum()),
                "production_wins_removed": int(pd.to_numeric(group.get("production_wins_removed", 0), errors="coerce").fillna(0).sum()),
                "production_losses_removed": int(pd.to_numeric(group.get("production_losses_removed", 0), errors="coerce").fillna(0).sum()),
                "shadow_rows_added": int(pd.to_numeric(group.get("shadow_rows_added", 0), errors="coerce").fillna(0).sum()),
                "shadow_added_wins": int(pd.to_numeric(group.get("shadow_added_wins", 0), errors="coerce").fillna(0).sum()),
                "shadow_added_losses": int(pd.to_numeric(group.get("shadow_added_losses", 0), errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_safe_state_shadow_evidence(
    *,
    base_dir: Path = DEFAULT_BASE_DIR,
    output_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or (base_dir / "aggregate")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = _run_dirs(base_dir)
    metrics_frames = [_load_metrics(path) for path in run_dirs]
    metrics_rows = pd.concat([frame for frame in metrics_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in metrics_frames) else pd.DataFrame()
    aggregate = _aggregate_metrics(metrics_rows)

    metrics_path = output_dir / "safe_state_shadow_aggregate_metrics.csv"
    report_json_path = output_dir / "safe_state_shadow_aggregate_report.json"
    report_md_path = output_dir / "safe_state_shadow_aggregate_report.md"
    aggregate.to_csv(metrics_path, index=False)

    gate = evaluate_safe_state_promotion_gate(aggregate_metrics_csv=metrics_path, config_path=config_path, output_dir=output_dir)
    report = {
        "base_dir": str(base_dir),
        "run_dirs": [str(path) for path in run_dirs],
        "run_count": int(len(run_dirs)),
        "runs_with_metrics": int(0 if metrics_rows.empty else metrics_rows["run_dir"].nunique()),
        "output_paths": {
            "metrics_csv": str(metrics_path),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
            "promotion_gate_status_json": str(output_dir / "safe_state_promotion_gate_status.json"),
        },
        "aggregate_metrics": aggregate.to_dict(orient="records"),
        "critical_questions": {
            "does_price_defense_alone_help": "requires_multi_slate_settlement",
            "does_forecastability_improve_price_defense": "requires_multi_slate_settlement",
            "does_structural_mispricing_improve_price_defense": "requires_multi_slate_settlement",
            "does_safe_state_core_outperform_production": "requires_multi_slate_settlement",
            "do_true_unstable_rejections_mostly_lose": "requires_multi_slate_settlement",
            "do_needs_more_sample_rows_mature": "requires_multi_slate_tracking",
            "does_system_avoid_coverage_collapse": "requires_promotion_gate_review",
        },
        "promotion_gate": gate,
        "production_behavior_changed": False,
        "promotion_claim": False,
    }
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report_md_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Safe-State Shadow Aggregate Report",
        "",
        f"- Run count: {report['run_count']}",
        f"- Runs with metrics: {report['runs_with_metrics']}",
        f"- Promotion ready: {str(report['promotion_gate'].get('promotion_ready', False)).lower()}",
        f"- Promotion status: {report['promotion_gate'].get('promotion_status', 'NOT_PROMOTION_ELIGIBLE')}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "## Variant Metrics",
    ]
    for row in report["aggregate_metrics"]:
        lines.append(
            f"- {row.get('variant')}: rows={row.get('total_rows')}, resolved={row.get('resolved_rows')}, "
            f"pending={row.get('pending_rows')}, wins={row.get('wins')}, losses={row.get('losses')}"
        )
    lines.extend(["", "Evidence is shadow-only until the promotion gate requirements are met."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate production-shadow safe-state evidence across slates.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = aggregate_safe_state_shadow_evidence(base_dir=args.base_dir, output_dir=args.output_dir, config_path=args.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
