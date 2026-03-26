#!/usr/bin/env python3
"""
Export the latest daily market prediction run to a static web JSON payload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parent.parent
SITE_ROOT = PLAYER_PREDICTOR_ROOT.parent
DAILY_RUNS_ROOT = PLAYER_PREDICTOR_ROOT / "model" / "analysis" / "daily_runs"
DEFAULT_WEB_JSON = SITE_ROOT / "web" / "data" / "daily_predictions.json"
DEFAULT_DIST_JSON = SITE_ROOT / "dist" / "data" / "daily_predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily market predictions to static web JSON.")
    parser.add_argument("--manifest", type=Path, default=None, help="Explicit daily pipeline manifest JSON.")
    parser.add_argument("--daily-runs-root", type=Path, default=DAILY_RUNS_ROOT, help="Root directory containing daily run folders.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_WEB_JSON, help="Static web JSON output path (web/data).")
    parser.add_argument("--out-dist", type=Path, default=DEFAULT_DIST_JSON, help="Static dist JSON output path (dist/data).")
    return parser.parse_args()


def find_latest_manifest(root: Path) -> Path:
    manifests = sorted(root.glob("**/daily_market_pipeline_manifest_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError(f"No daily pipeline manifest found under {root}")
    return manifests[0]


def resolve_artifact_path(raw_path: str | None, manifest_dir: Path) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    local_fallback = manifest_dir / candidate.name
    if local_fallback.exists():
        return local_fallback
    return candidate


def safe_float(value) -> float | None:
    try:
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def build_summary(plays: pd.DataFrame) -> dict:
    if plays.empty:
        return {
            "play_count": 0,
            "avg_expected_win_rate": None,
            "avg_ev": None,
            "avg_edge": None,
            "by_target": {},
            "by_recommendation": {},
        }

    return {
        "play_count": int(len(plays)),
        "avg_expected_win_rate": safe_float(pd.to_numeric(plays.get("expected_win_rate"), errors="coerce").mean()),
        "avg_ev": safe_float(pd.to_numeric(plays.get("ev"), errors="coerce").mean()),
        "avg_edge": safe_float(pd.to_numeric(plays.get("abs_edge"), errors="coerce").mean()),
        "by_target": plays.get("target", pd.Series(dtype=str)).value_counts().to_dict(),
        "by_recommendation": plays.get("recommendation", pd.Series(dtype=str)).value_counts().to_dict(),
    }


def normalize_play_rows(plays: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    ordered = plays.copy().reset_index(drop=True)
    ordered["rank"] = ordered.index + 1
    for _, row in ordered.iterrows():
        rows.append(
            {
                "rank": int(row["rank"]),
                "player": str(row.get("player", "")),
                "target": str(row.get("target", "")),
                "direction": str(row.get("direction", "")),
                "market_date": str(row.get("market_date", "")) if pd.notna(row.get("market_date")) else None,
                "last_history_date": str(row.get("last_history_date", "")) if pd.notna(row.get("last_history_date")) else None,
                "prediction": safe_float(row.get("prediction")),
                "market_line": safe_float(row.get("market_line")),
                "edge": safe_float(row.get("edge")),
                "abs_edge": safe_float(row.get("abs_edge")),
                "expected_win_rate": safe_float(row.get("expected_win_rate")),
                "raw_expected_win_rate": safe_float(row.get("raw_expected_win_rate")),
                "ev": safe_float(row.get("ev")),
                "final_confidence": safe_float(row.get("final_confidence")),
                "gap_percentile": safe_float(row.get("gap_percentile")),
                "recommendation": str(row.get("recommendation", "")),
                "history_rows": int(row.get("history_rows", 0)) if pd.notna(row.get("history_rows")) else None,
                "market_books": safe_float(row.get("market_books")),
                "uncertainty_sigma": safe_float(row.get("uncertainty_sigma")),
                "spike_probability": safe_float(row.get("spike_probability")),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve() if args.manifest else find_latest_manifest(args.daily_runs_root.resolve())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    final_csv = resolve_artifact_path(manifest.get("final_csv"), manifest_path.parent)
    final_json = resolve_artifact_path(manifest.get("final_json"), manifest_path.parent)
    plays = pd.read_csv(final_csv) if final_csv and final_csv.exists() else pd.DataFrame()
    final_payload = json.loads(final_json.read_text(encoding="utf-8")) if final_json and final_json.exists() else {}

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "run_date": manifest.get("run_date"),
        "season": manifest.get("season"),
        "through_date": manifest.get("through_date"),
        "current_market_rows": manifest.get("current_market_rows"),
        "current_market_snapshot_meta": manifest.get("current_market_snapshot_meta", {}),
        "used_latest_manifest": manifest.get("used_latest_manifest"),
        "market_snapshot": final_payload.get("market_snapshot") or manifest.get("current_market_snapshot"),
        "model_run_id": final_payload.get("run_id"),
        "policy_profile": final_payload.get("policy_profile"),
        "policy": final_payload.get("policy", {}),
        "input_validation": final_payload.get("input_validation", {}),
        "summary": build_summary(plays),
        "plays": normalize_play_rows(plays),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Also copy to dist/data for the predictor page
    args.out_dist.parent.mkdir(parents=True, exist_ok=True)
    args.out_dist.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("DAILY PREDICTIONS WEB EXPORT COMPLETE")
    print("=" * 90)
    print(f"Manifest: {manifest_path}")
    print(f"Rows:     {len(plays)}")
    print(f"Output:   {args.out_json}")
    print(f"Dist:     {args.out_dist}")


if __name__ == "__main__":
    main()
