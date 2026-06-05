#!/usr/bin/env python3
"""
Build v9.5 by replacing oracle lineup fields with pregame availability snapshots.

Availability is probability-weighted:
  expected lineup delta = sum(teammate_delta * out_probability * confidence)

This script is promotion-oriented: it refuses invalid snapshot schemas and
reports whether all snapshots are before game start. It does not use actual
same-game absences unless they are present in the supplied snapshot file.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
MARKET_CAPS = {"PTS": 4.0, "TRB": 2.0, "AST": 2.0}
REQUIRED_FIELDS = [
    "snapshot_time",
    "game_start_time",
    "date",
    "team",
    "player",
    "status",
    "out_probability",
    "availability_confidence",
    "source",
]
STATUS_DEFAULTS = {
    "out": 1.0,
    "confirmed_out": 1.0,
    "doubtful": 0.8,
    "questionable": 0.45,
    "probable": 0.15,
    "available": 0.0,
    "active": 0.0,
}


def _resolve(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / text).resolve()


def _read_table(path: Path) -> pd.DataFrame:
    path = _resolve(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path)


def _load_manifest(path: Path) -> dict:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _load_rows(manifest: dict) -> pd.DataFrame:
    output = _resolve(Path(manifest["output"]))
    rows = pd.read_csv(output / "data" / "prop_training_rows.csv")
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.date.astype(str)
    rows["player"] = rows["player"].astype(str).str.replace(" ", "_", regex=False)
    return rows


def _normalize_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(" ", "_", regex=False)


def _normal_sf(x: np.ndarray) -> np.ndarray:
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    return 1.0 - cdf


def _copy_tree_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def _normalize_availability(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    availability = raw.copy()
    rename = {
        "commence_time": "game_start_time",
        "commence_time_utc": "game_start_time",
        "start_time": "game_start_time",
        "snapshot_time_utc": "snapshot_time",
        "team_abbreviation": "team",
        "player_name": "player",
    }
    for old, new in rename.items():
        if old in availability.columns and new not in availability.columns:
            availability[new] = availability[old]

    if "out_probability" not in availability.columns and "status" in availability.columns:
        availability["out_probability"] = availability["status"].astype(str).str.lower().map(STATUS_DEFAULTS)
    if "availability_confidence" not in availability.columns:
        availability["availability_confidence"] = np.where(
            availability.get("status", "").astype(str).str.lower().isin(["out", "confirmed_out", "active", "available"]),
            1.0,
            0.75,
        )
    if "source" not in availability.columns:
        availability["source"] = "unknown"
    if "date" not in availability.columns and "game_start_time" in availability.columns:
        availability["date"] = pd.to_datetime(availability["game_start_time"], errors="coerce", utc=True).dt.date.astype(str)

    missing = [col for col in REQUIRED_FIELDS if col not in availability.columns]
    if missing:
        return availability, {"status": "fail", "missing_required_fields": missing}

    availability["date"] = pd.to_datetime(availability["date"], errors="coerce").dt.date.astype(str)
    availability["player"] = _normalize_name(availability["player"])
    availability["team"] = availability["team"].astype(str).str.upper().str.strip()
    availability["status"] = availability["status"].astype(str).str.lower().str.strip()
    availability["out_probability"] = pd.to_numeric(availability["out_probability"], errors="coerce").clip(0.0, 1.0)
    availability["availability_confidence"] = pd.to_numeric(availability["availability_confidence"], errors="coerce").clip(0.0, 1.0)
    availability["snapshot_ts"] = pd.to_datetime(availability["snapshot_time"], errors="coerce", utc=True)
    availability["game_start_ts"] = pd.to_datetime(availability["game_start_time"], errors="coerce", utc=True)
    invalid = {
        "snapshot_time": int(availability["snapshot_ts"].isna().sum()),
        "game_start_time": int(availability["game_start_ts"].isna().sum()),
        "date": int(availability["date"].eq("NaT").sum()),
        "player": int(availability["player"].eq("").sum()),
        "team": int(availability["team"].eq("").sum()),
        "out_probability": int(availability["out_probability"].isna().sum()),
        "availability_confidence": int(availability["availability_confidence"].isna().sum()),
    }
    before_lock = availability["snapshot_ts"] < availability["game_start_ts"]
    validation = {
        "status": "pass" if all(v == 0 for v in invalid.values()) and bool(before_lock.all()) else "fail",
        "missing_required_fields": [],
        "invalid_value_counts": invalid,
        "rows": int(len(availability)),
        "all_snapshots_before_game_start": bool(before_lock.all()),
        "late_snapshot_rows": int((~before_lock).sum()),
        "sources": sorted(str(v) for v in availability["source"].dropna().unique()),
    }
    return availability, validation


def _dedupe_availability(availability: pd.DataFrame) -> pd.DataFrame:
    availability = availability.sort_values(["date", "team", "player", "snapshot_ts"])
    return availability.drop_duplicates(["date", "team", "player"], keep="last")


def _apply_features(rows: pd.DataFrame, availability: pd.DataFrame, deltas: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    availability = _dedupe_availability(availability)
    availability_by_date_team = {
        key: frame for key, frame in availability.groupby(["date", "team"], sort=False)
    }
    deltas = deltas.copy()
    deltas["first_shared_date"] = pd.to_datetime(deltas["first_shared_date"], errors="coerce").dt.date.astype(str)
    deltas["last_shared_date"] = pd.to_datetime(deltas["last_shared_date"], errors="coerce").dt.date.astype(str)
    delta_groups = {key: frame for key, frame in deltas.groupby(["player", "market", "team"], sort=False)}

    feature_rows: list[dict] = []
    for _, row in rows.iterrows():
        player = str(row["player"])
        date = str(row["date"])
        market = str(row["market"])
        best = None
        best_abs = -1.0
        candidate_teams = [str(row.get("lineup_team"))] if pd.notna(row.get("lineup_team")) and row.get("lineup_team") else []
        candidate_teams += [team for d, team in availability_by_date_team if d == date and (player, market, team) in delta_groups]
        for team in dict.fromkeys(candidate_teams):
            avail = availability_by_date_team.get((date, team))
            delta_frame = delta_groups.get((player, market, team))
            if avail is None or delta_frame is None:
                continue
            eligible = delta_frame[
                (delta_frame["first_shared_date"] <= date)
                & (delta_frame["last_shared_date"] >= date)
            ].merge(
                avail[["player", "out_probability", "availability_confidence", "status", "source", "snapshot_time", "game_start_time"]],
                left_on="teammate",
                right_on="player",
                how="inner",
                suffixes=("", "_availability"),
            )
            if eligible.empty:
                continue
            eligible["weighted_delta"] = (
                eligible["shrunk_delta"]
                * eligible["out_probability"]
                * eligible["availability_confidence"]
            )
            weighted = float(eligible["weighted_delta"].sum())
            if abs(weighted) > best_abs:
                best_abs = abs(weighted)
                best = (team, eligible, weighted)

        if best is None:
            feature_rows.append(_empty_features())
            continue
        team, eligible, weighted = best
        cap = MARKET_CAPS.get(market, 2.0)
        adjustment = float(np.clip(weighted, -cap, cap))
        feature_rows.append({
            "pregame_lineup_team": team,
            "pregame_teammate_out_prob_sum": float(eligible["out_probability"].sum()),
            "pregame_teammate_out_expected_count": float((eligible["out_probability"] * eligible["availability_confidence"]).sum()),
            "pregame_lineup_delta_weighted": weighted,
            "pregame_lineup_adjustment": adjustment,
            "pregame_availability_confidence": float(np.average(eligible["availability_confidence"], weights=eligible["out_probability"].clip(lower=0.001))),
            "pregame_availability_rows": int(len(eligible)),
            "pregame_usage_removed_expected": float((eligible["out_probability"] * eligible["availability_confidence"] * eligible.get("baseline_rate", 0.0)).sum()),
            "pregame_ast_shift_expected": float(eligible.loc[eligible["market"].eq("AST"), "weighted_delta"].sum()) if "AST" in set(eligible["market"]) else 0.0,
            "pregame_reb_shift_expected": float(eligible.loc[eligible["market"].eq("TRB"), "weighted_delta"].sum()) if "TRB" in set(eligible["market"]) else 0.0,
            "pregame_teammates_considered": "|".join(eligible.sort_values("out_probability", ascending=False)["teammate"].head(8).tolist()),
        })
    features = pd.DataFrame(feature_rows, index=rows.index)
    return pd.concat([rows.reset_index(drop=True), features.reset_index(drop=True)], axis=1), {
        "joined_rows": int((features["pregame_availability_rows"] > 0).sum()),
        "join_rate": float((features["pregame_availability_rows"] > 0).mean()) if len(features) else 0.0,
    }


def _empty_features() -> dict:
    return {
        "pregame_lineup_team": "",
        "pregame_teammate_out_prob_sum": 0.0,
        "pregame_teammate_out_expected_count": 0.0,
        "pregame_lineup_delta_weighted": 0.0,
        "pregame_lineup_adjustment": 0.0,
        "pregame_availability_confidence": 0.0,
        "pregame_availability_rows": 0,
        "pregame_usage_removed_expected": 0.0,
        "pregame_ast_shift_expected": 0.0,
        "pregame_reb_shift_expected": 0.0,
        "pregame_teammates_considered": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply pregame availability snapshots to v9.4 and create v9.5")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "model" / "props" / "v9_4" / "manifest.json")
    parser.add_argument("--availability-snapshots", type=Path, required=True)
    parser.add_argument("--lineup-artifacts", type=Path, default=ROOT / "model" / "props" / "v9_4" / "lineup_delta_artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_5")
    parser.add_argument("--lineup-weight", type=float, default=1.0)
    parser.add_argument("--sigma-inflation-per-expected-out", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = _resolve(args.source_manifest)
    source_manifest = _load_manifest(source_manifest_path)
    rows = _load_rows(source_manifest)
    raw_availability = _read_table(args.availability_snapshots)
    availability, schema_validation = _normalize_availability(raw_availability)
    if schema_validation["status"] != "pass":
        raise ValueError(f"availability snapshot validation failed: {schema_validation}")
    delta_path = args.lineup_artifacts / "player_teammate_out_deltas.parquet"
    if not _resolve(delta_path).exists():
        delta_path = args.lineup_artifacts / "player_teammate_out_deltas.csv"
    deltas = _read_table(delta_path)

    adjusted, join_report = _apply_features(rows, availability, deltas)
    base_mean_col = "v92_model_mean" if "v92_model_mean" in adjusted.columns else "model_mean"
    base_sigma_col = "v92_sigma" if "v92_sigma" in adjusted.columns else "sigma"
    adjusted["v95_pregame_lineup_model_mean"] = (
        pd.to_numeric(adjusted[base_mean_col], errors="coerce").fillna(adjusted["model_mean"])
        + args.lineup_weight * adjusted["pregame_lineup_adjustment"]
    )
    base_sigma = pd.to_numeric(adjusted[base_sigma_col], errors="coerce").fillna(adjusted.get("sigma", 3.0)).clip(lower=0.25)
    adjusted["v95_pregame_lineup_sigma"] = (
        base_sigma * (1.0 + args.sigma_inflation_per_expected_out * adjusted["pregame_teammate_out_expected_count"].clip(upper=5))
    ).clip(lower=0.25)
    z = (pd.to_numeric(adjusted["line"], errors="coerce") - adjusted["v95_pregame_lineup_model_mean"]) / adjusted["v95_pregame_lineup_sigma"]
    adjusted["p_over_raw_v94_safe"] = adjusted["p_over_raw"]
    adjusted["p_over_raw"] = np.clip(_normal_sf(z.to_numpy(dtype=float)), 0.001, 0.999)
    adjusted["pregame_lineup_probability_delta"] = adjusted["p_over_raw"] - adjusted["p_over_raw_v94_safe"]

    output = _resolve(args.output)
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(data_dir / "prop_training_rows.csv", index=False)
    _copy_tree_if_exists(_resolve(Path(source_manifest["output"])) / "calibration", output / "calibration")

    oracle_columns = [col for col in adjusted.columns if "oracle" in col.lower()]
    lineup_field_safety = {
        "oracle_fields_present": bool(oracle_columns),
        "oracle_fields": oracle_columns,
        "availability_snapshot_joined": join_report["joined_rows"] > 0,
        "all_snapshots_before_game_start": schema_validation["all_snapshots_before_game_start"],
        "late_news_rows_flagged": schema_validation["late_snapshot_rows"] > 0,
        "availability_source_present": "source" in availability.columns and availability["source"].notna().any(),
        "availability_confidence_present": "availability_confidence" in availability.columns,
    }
    report = {
        "status": "built_pregame_availability_lineup_rows",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(adjusted)),
        "availability_rows": int(len(availability)),
        "feature_join": join_report,
        "avg_abs_probability_delta": float(adjusted["pregame_lineup_probability_delta"].abs().mean()),
        "avg_abs_lineup_adjustment": float(adjusted["pregame_lineup_adjustment"].abs().mean()),
        "lineup_weight": args.lineup_weight,
        "schema_validation": schema_validation,
        "lineup_field_safety": lineup_field_safety,
    }
    status = "pregame_lineup_shadow_candidate" if not lineup_field_safety["oracle_fields_present"] and lineup_field_safety["all_snapshots_before_game_start"] else "blocked_lineup_field_safety"
    manifest = {
        "model_version": "prop_engine_v9_5_pregame_lineup_distribution",
        "status": status,
        "trained_at": report["built_at"],
        "source_v9_4_manifest": str(source_manifest_path.relative_to(REPO_ROOT)) if source_manifest_path.is_relative_to(REPO_ROOT) else str(source_manifest_path),
        "availability_snapshots": str(_resolve(args.availability_snapshots)),
        "output": str(output.relative_to(REPO_ROOT)) if output.is_relative_to(REPO_ROOT) else str(output),
        "rows": int(len(adjusted)),
        "players": int(adjusted["player"].nunique()),
        "date_min": str(pd.to_datetime(adjusted["date"], errors="coerce").min().date()),
        "date_max": str(pd.to_datetime(adjusted["date"], errors="coerce").max().date()),
        "artifacts": {
            "data": "data/prop_training_rows.csv",
            "calibration": "calibration",
            "availability_snapshot_schema": "Player-Predictor/configs/availability_snapshot_schema_v1.json",
            "pregame_lineup_application_report": "pregame_lineup_application_report.json",
        },
        "pregame_lineup_application": report,
        "promotion_gates": {
            "require_no_oracle_lineup_fields": True,
            "require_snapshot_time_before_lock": True,
            "require_availability_source_present": True,
            "require_availability_confidence": True,
            "forbid_same_game_actual_absence": True,
            "must_beat_v9_4_safe_gated_brier": True,
            "must_beat_v9_4_safe_walk_forward_brier": True,
            "max_gated_ece": 0.025,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pregame_lineup_application_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": status, "report": report, "manifest": str(output / "manifest.json")}, indent=2, default=str))


if __name__ == "__main__":
    main()
