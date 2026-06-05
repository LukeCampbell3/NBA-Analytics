#!/usr/bin/env python3
"""
Apply lineup-delta artifacts to prop rows.

Default mode writes an explicit retrospective/oracle artifact: it uses actual
same-game absences from historical logs to estimate the ceiling impact of
teammate availability. This is useful for model representation research, but
must not be promoted unless the same availability labels come from a pregame
feed at prediction time.
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


def _resolve(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / text).resolve()


def _load_manifest(path: Path) -> dict:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _load_rows(manifest: dict, manifest_path: Path) -> pd.DataFrame:
    output = _resolve(Path(manifest["output"]))
    rows_path = output / "data" / "prop_training_rows.csv"
    rows = pd.read_csv(rows_path)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.date.astype(str)
    rows["player"] = rows["player"].astype(str).str.replace(" ", "_", regex=False)
    return rows


def _read_table(path: Path) -> pd.DataFrame:
    path = _resolve(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _coerce_logs(logs: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player",
        "TEAM_ABBREVIATION": "team",
        "GAME_ID": "game_id",
        "GAME_DATE": "date",
        "MIN": "MIN",
        "REB": "REB",
    }
    logs = logs.rename(columns={k: v for k, v in rename.items() if k in logs.columns}).copy()
    required = {"player_id", "player", "team", "game_id", "date", "MIN"}
    missing = sorted(required - set(logs.columns))
    if missing:
        raise ValueError(f"game logs missing required columns: {missing}")
    logs["date"] = pd.to_datetime(logs["date"], errors="coerce").dt.date.astype(str)
    logs["player"] = logs["player"].astype(str).str.strip().str.replace(" ", "_", regex=False)
    logs["player_id"] = logs["player_id"].astype(str)
    logs["team"] = logs["team"].astype(str)
    logs["MIN"] = pd.to_numeric(logs["MIN"], errors="coerce").fillna(0.0)
    if "AVAILABLE_FLAG" in logs.columns:
        available = pd.to_numeric(logs["AVAILABLE_FLAG"], errors="coerce").fillna(0) > 0
    else:
        available = pd.Series(True, index=logs.index)
    logs["played"] = available & (logs["MIN"] > 0)
    return logs.dropna(subset=["date"]).copy()


def _normal_sf(x: np.ndarray) -> np.ndarray:
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    return 1.0 - cdf


def _copy_tree_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def _build_lineup_features(rows: pd.DataFrame, logs: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    played = logs.loc[logs["played"]].copy()
    target_lookup = (
        played.sort_values(["date", "player", "MIN"], ascending=[True, True, False])
        .drop_duplicates(["date", "player"])
        .set_index(["date", "player"])[["team", "game_id"]]
        .to_dict("index")
    )
    active_by_date_team = played.groupby(["date", "team"])["player"].agg(lambda s: set(s.astype(str))).to_dict()

    deltas = deltas.copy()
    deltas["first_shared_date"] = pd.to_datetime(deltas["first_shared_date"], errors="coerce").dt.date.astype(str)
    deltas["last_shared_date"] = pd.to_datetime(deltas["last_shared_date"], errors="coerce").dt.date.astype(str)
    grouped = {key: frame for key, frame in deltas.groupby(["player", "market", "team"], sort=False)}

    feature_rows: list[dict] = []
    for _, row in rows.iterrows():
        player = str(row["player"])
        date = str(row["date"])
        market = str(row["market"])
        target = target_lookup.get((date, player))
        if not target:
            feature_rows.append(_empty_features())
            continue
        team = target["team"]
        active = active_by_date_team.get((date, team), set())
        candidates = grouped.get((player, market, team))
        if candidates is None or candidates.empty:
            feature_rows.append(_empty_features(team=team, game_id=target["game_id"]))
            continue
        eligible = candidates[
            (candidates["first_shared_date"] <= date)
            & (candidates["last_shared_date"] >= date)
            & ~candidates["teammate"].isin(active)
        ].copy()
        if eligible.empty:
            feature_rows.append(_empty_features(team=team, game_id=target["game_id"]))
            continue

        cap = MARKET_CAPS.get(market, 2.0)
        raw_sum = float(eligible["shrunk_delta"].sum())
        weighted_sum = float((eligible["shrunk_delta"] * eligible["confidence"].clip(0, 1)).sum())
        adjustment = float(np.clip(weighted_sum, -cap, cap))
        feature_rows.append({
            "lineup_team": team,
            "lineup_game_id": target["game_id"],
            "lineup_oracle_teammates_out_count": int(len(eligible)),
            "lineup_oracle_teammates_out": "|".join(eligible.sort_values("confidence", ascending=False)["teammate"].head(8).tolist()),
            "lineup_oracle_delta_raw_sum": raw_sum,
            "lineup_oracle_delta_weighted": weighted_sum,
            "lineup_oracle_adjustment": adjustment,
            "lineup_oracle_confidence_sum": float(eligible["confidence"].clip(0, 1).sum()),
            "lineup_oracle_max_abs_delta": float(eligible["shrunk_delta"].abs().max()),
        })
    return pd.DataFrame(feature_rows, index=rows.index)


def _empty_features(team: str | None = None, game_id: str | None = None) -> dict:
    return {
        "lineup_team": team,
        "lineup_game_id": game_id,
        "lineup_oracle_teammates_out_count": 0,
        "lineup_oracle_teammates_out": "",
        "lineup_oracle_delta_raw_sum": 0.0,
        "lineup_oracle_delta_weighted": 0.0,
        "lineup_oracle_adjustment": 0.0,
        "lineup_oracle_confidence_sum": 0.0,
        "lineup_oracle_max_abs_delta": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply v9.4 lineup delta features to prop rows")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "model" / "props" / "v9_4" / "manifest.json")
    parser.add_argument("--lineup-artifacts", type=Path, default=ROOT / "model" / "props" / "v9_4" / "lineup_delta_artifacts")
    parser.add_argument("--game-logs", type=Path, default=ROOT / "data copy" / "raw" / "nba_enrichment" / "season=2026" / "player_game_logs.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_4_oracle_lineup")
    parser.add_argument("--lineup-weight", type=float, default=1.0)
    parser.add_argument("--sigma-inflation-per-out", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = _resolve(args.source_manifest)
    source_manifest = _load_manifest(manifest_path)
    rows = _load_rows(source_manifest, manifest_path)
    logs = _coerce_logs(_read_table(args.game_logs))
    delta_path = args.lineup_artifacts / "player_teammate_out_deltas.parquet"
    if not _resolve(delta_path).exists():
        delta_path = args.lineup_artifacts / "player_teammate_out_deltas.csv"
    deltas = _read_table(delta_path)

    features = _build_lineup_features(rows, logs, deltas)
    adjusted = pd.concat([rows.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    base_mean_col = "v92_model_mean" if "v92_model_mean" in adjusted.columns else "model_mean"
    base_sigma_col = "v92_sigma" if "v92_sigma" in adjusted.columns else "sigma"
    adjusted["v94_lineup_model_mean"] = (
        pd.to_numeric(adjusted[base_mean_col], errors="coerce").fillna(adjusted["model_mean"])
        + args.lineup_weight * adjusted["lineup_oracle_adjustment"]
    )
    base_sigma = pd.to_numeric(adjusted[base_sigma_col], errors="coerce").fillna(adjusted.get("sigma", 3.0)).clip(lower=0.25)
    adjusted["v94_lineup_sigma"] = (
        base_sigma * (1.0 + args.sigma_inflation_per_out * adjusted["lineup_oracle_teammates_out_count"].clip(upper=5))
    ).clip(lower=0.25)
    z = (pd.to_numeric(adjusted["line"], errors="coerce") - adjusted["v94_lineup_model_mean"]) / adjusted["v94_lineup_sigma"]
    adjusted["p_over_raw_v93"] = adjusted["p_over_raw"]
    adjusted["p_over_raw"] = np.clip(_normal_sf(z.to_numpy(dtype=float)), 0.001, 0.999)
    adjusted["lineup_oracle_probability_delta"] = adjusted["p_over_raw"] - adjusted["p_over_raw_v93"]
    adjusted["lineup_oracle_feature_available"] = adjusted["lineup_oracle_teammates_out_count"] > 0

    output = _resolve(args.output)
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(data_dir / "prop_training_rows.csv", index=False)
    _copy_tree_if_exists(_resolve(Path(source_manifest["output"])) / "calibration", output / "calibration")

    report = {
        "status": "built_oracle_lineup_adjusted_rows",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(adjusted)),
        "feature_available_rows": int(adjusted["lineup_oracle_feature_available"].sum()),
        "feature_available_rate": float(adjusted["lineup_oracle_feature_available"].mean()),
        "avg_abs_probability_delta": float(adjusted["lineup_oracle_probability_delta"].abs().mean()),
        "avg_abs_lineup_adjustment": float(adjusted["lineup_oracle_adjustment"].abs().mean()),
        "lineup_weight": args.lineup_weight,
        "sigma_inflation_per_out": args.sigma_inflation_per_out,
        "leakage_status": "oracle_retro_only_not_live_safe",
        "production_rule": "Replace lineup_oracle_* fields with pregame availability joins before promotion.",
    }
    manifest = {
        "model_version": "prop_engine_v9_4_oracle_lineup_adjusted_distribution",
        "status": "research_only_oracle_lineup_not_promotable",
        "trained_at": report["built_at"],
        "source_v9_4_manifest": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.is_relative_to(REPO_ROOT) else str(manifest_path),
        "output": str(output.relative_to(REPO_ROOT)) if output.is_relative_to(REPO_ROOT) else str(output),
        "rows": int(len(adjusted)),
        "players": int(adjusted["player"].nunique()),
        "date_min": str(pd.to_datetime(adjusted["date"], errors="coerce").min().date()),
        "date_max": str(pd.to_datetime(adjusted["date"], errors="coerce").max().date()),
        "artifacts": {
            "data": "data/prop_training_rows.csv",
            "calibration": "calibration",
            "lineup_application_report": "lineup_application_report.json",
        },
        "lineup_application": report,
        "live_promotion_blockers": [
            "Uses actual same-game teammate absences from historical logs.",
            "Must be rebuilt from pregame injury/availability snapshots before live use.",
            "Oracle metrics can guide representation but cannot justify production promotion.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "lineup_application_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": report, "manifest": str(output / "manifest.json")}, indent=2, default=str))


if __name__ == "__main__":
    main()
