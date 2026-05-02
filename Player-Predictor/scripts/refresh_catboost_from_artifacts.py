#!/usr/bin/env python3
"""
Refresh the CatBoost production layer from recovered structured-stack artifacts.

This keeps the existing neural artifacts (LSTM ensemble, scaler(s), PTS branch)
and retrains the CatBoost delta models against the latest processed data using
the current training code paths and validation split heuristics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training"))
sys.path.insert(0, str(REPO_ROOT / "inference"))

from improved_lstm_v7 import (  # noqa: E402
    build_player_tail_split,
    build_recency_sample_weights,
    create_shared_trainer,
    evaluate_predictions,
    evaluate_rolling_windows,
    rolling_window_stats_1d,
    weighted_ensemble_latent_export,
)
from structured_stack_contract import build_schema_payload, normalize_catboost_model_info, validate_metadata_contract  # noqa: E402
from structured_stack_inference import StructuredStackInference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh CatBoost models using existing structured-stack artifacts.")
    parser.add_argument("--manifest", type=Path, default=MODEL_DIR / "production_structured_lstm_stack.json", help="Manifest to recover artifacts from.")
    parser.add_argument("--promote-if-better", action="store_true", help="Update production manifest if the refreshed validation MAE improves.")
    parser.add_argument("--val-fraction", type=float, default=0.20, help="Per-player tail holdout fraction.")
    return parser.parse_args()


def _copy_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_processed_frame(feature_trainer) -> pd.DataFrame:
    candidate_dirs = [
        REPO_ROOT / "Data",
        REPO_ROOT / "Data-Proc",
        REPO_ROOT / "Data-Proc-OG",
        REPO_ROOT / "Data-org",
    ]
    data_dir = next((path for path in candidate_dirs if path.exists() and path.is_dir()), None)
    if data_dir is None:
        searched = ", ".join(str(path) for path in candidate_dirs)
        raise FileNotFoundError(f"No data directory found. Searched: {searched}")

    all_dfs = []
    for player_dir in data_dir.iterdir():
        if not player_dir.is_dir():
            continue
        player_name = player_dir.name
        processed_files = list(player_dir.glob("*_processed.csv"))
        if not processed_files:
            processed_files = list(player_dir.glob("*_processed_processed.csv"))
        if not processed_files:
            processed_files = list(player_dir.glob("*.csv"))
        for file in processed_files:
            df = pd.read_csv(file)
            df["Player"] = player_name
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No processed player files were found.")

    df = pd.concat(all_dfs, ignore_index=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feature_trainer.create_hybrid_features(df)


def build_sequences_for_predictor(df: pd.DataFrame, predictor: StructuredStackInference):
    if "Date" in df.columns:
        df = df.sort_values(["Player", "Date"]).reset_index(drop=True)
    elif "Game_Index" in df.columns:
        df = df.sort_values(["Player", "Game_Index"]).reset_index(drop=True)
    else:
        df = df.sort_values(["Player"]).reset_index(drop=True)

    df, _repair_info = predictor._repair_required_columns(df)
    df["Player_ID"] = df["Player"].astype(str).map(predictor.player_mapping).fillna(0).astype(int)
    categorical_features = list(predictor.feature_columns[:3])
    numeric_features = list(predictor.feature_columns[3:])

    numeric_frame = df[numeric_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    baseline_frame = df[predictor.baseline_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target_frame = df[predictor.target_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df.loc[:, numeric_features] = predictor.scaler_x.transform(numeric_frame.to_numpy(dtype=np.float32))
    baseline_scaled = predictor.scaler_y.transform(baseline_frame.to_numpy(dtype=np.float32))
    target_scaled = predictor.scaler_y.transform(target_frame.to_numpy(dtype=np.float32))

    baseline_cols = [f"_baseline_scaled_{idx}" for idx in range(len(predictor.target_columns))]
    target_cols = [f"_target_scaled_{idx}" for idx in range(len(predictor.target_columns))]
    df.loc[:, baseline_cols] = baseline_scaled
    df.loc[:, target_cols] = target_scaled

    X_list = []
    baseline_list = []
    y_list = []
    sequence_meta_rows = []
    seq_len = int(predictor.seq_len)

    for player, player_df in df.groupby("Player", sort=False):
        player_df = player_df.reset_index(drop=True)
        for i in range(len(player_df) - seq_len):
            seq = player_df.iloc[i:i + seq_len]
            target_row = player_df.iloc[i + seq_len]
            X_seq = np.concatenate(
                [
                    np.column_stack(
                        [
                            np.clip(seq["Player_ID"].to_numpy(dtype=np.int32), 0, predictor.counts["players"] - 1),
                            np.clip(seq["Team_ID"].to_numpy(dtype=np.int32), 0, predictor.counts["teams"] - 1),
                            np.clip(seq["Opponent_ID"].to_numpy(dtype=np.int32), 0, predictor.counts["opponents"] - 1),
                        ]
                    ).astype(np.float32),
                    seq[numeric_features].to_numpy(dtype=np.float32),
                ],
                axis=1,
            )
            X_list.append(X_seq)
            baseline_list.append(seq.iloc[-1][baseline_cols].to_numpy(dtype=np.float32))
            y_list.append(target_row[target_cols].to_numpy(dtype=np.float32))
            sequence_meta_rows.append(
                {
                    "player": player,
                    "sequence_end_index": int(i + seq_len - 1),
                    "target_index": int(i + seq_len),
                    "target_date": target_row.get("Date"),
                    "game_index": float(target_row.get("Game_Index")) if "Game_Index" in target_row.index else np.nan,
                }
            )

    if not X_list:
        raise RuntimeError("No predictor-space sequences were created.")

    X = np.asarray(X_list, dtype=np.float32)
    baselines = np.asarray(baseline_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    sequence_meta = pd.DataFrame.from_records(sequence_meta_rows)
    return X, baselines, y, sequence_meta


def _artifact_path_strings(run_dir: Path, predictor: StructuredStackInference) -> dict[str, str | list[str]]:
    artifact_paths: dict[str, str | list[str]] = {}
    artifact_paths["lstm_weights"] = [
        str((run_dir / predictor._artifact_path("lstm_weights", index=idx).name).as_posix())
        for idx in range(len(predictor.models))
    ]
    artifact_paths["pts_branch_weights"] = str((run_dir / predictor._artifact_path("pts_branch_weights").name).as_posix())
    artifact_paths["catboost_models"] = str((run_dir / "lstm_v7_catboost_models.pkl").as_posix())
    artifact_paths["scaler_x"] = str((run_dir / predictor._artifact_path("scaler_x").name).as_posix())
    artifact_paths["scaler_y"] = str((run_dir / predictor._artifact_path("scaler_y").name).as_posix())
    artifact_paths["metadata"] = str((run_dir / "lstm_v7_metadata.json").as_posix())
    artifact_paths["schema"] = str((run_dir / "lstm_v7_feature_schema.json").as_posix())
    meta_models_path = predictor.artifact_paths.get("meta_models")
    if meta_models_path:
        artifact_paths["meta_models"] = str((run_dir / Path(str(meta_models_path)).name).as_posix())
    return artifact_paths


def candidate_score(mae: float, rolling_stats: dict) -> float:
    return float(mae + 0.025 * rolling_stats["std"] + 0.01 * max(0.0, rolling_stats["max"] - mae))


def fit_positive_weights(candidate_matrix: np.ndarray, target_delta: np.ndarray) -> np.ndarray | None:
    weights, *_ = np.linalg.lstsq(candidate_matrix, target_delta, rcond=None)
    weights = np.clip(weights.astype(np.float64), 0.0, None)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-8:
        return None
    weights /= weights.sum()
    return weights.astype(np.float32)


def refresh_candidate_specs() -> dict[str, list[tuple[str, str, dict, dict]]]:
    return {
        "PTS": [
            ("pts_deep_v3", "v3", dict(
                loss_function="MAE", iterations=1100, learning_rate=0.025,
                depth=8, l2_leaf_reg=3.0, min_data_in_leaf=28,
                subsample=0.75, colsample_bylevel=0.75, early_stopping_rounds=60,
                verbose=0, random_seed=242,
            ), {}),
            ("pts_latent_v3", "latent_v3", dict(
                loss_function="MAE", iterations=1100, learning_rate=0.022,
                depth=8, l2_leaf_reg=3.0, min_data_in_leaf=24,
                subsample=0.75, colsample_bylevel=0.72, early_stopping_rounds=65,
                verbose=0, random_seed=942,
            ), {}),
            ("stable_v3", "v3", dict(
                loss_function="MAE", iterations=800, learning_rate=0.03,
                depth=6, l2_leaf_reg=2.0, min_data_in_leaf=24,
                subsample=0.8, colsample_bylevel=0.8, early_stopping_rounds=45,
                verbose=0, random_seed=142,
            ), {}),
            ("lossguide_latent_v3", "latent_v3", dict(
                loss_function="MAE", iterations=900, learning_rate=0.028,
                depth=8, grow_policy="Lossguide", max_leaves=64,
                l2_leaf_reg=3.0, min_data_in_leaf=20,
                subsample=0.8, colsample_bylevel=0.72, early_stopping_rounds=60,
                verbose=0, random_seed=3742,
            ), {}),
        ],
        "TRB": [
            ("base_v2_s2", "v2", dict(
                loss_function="MAE", iterations=900, learning_rate=0.028,
                depth=7, l2_leaf_reg=2.5, min_data_in_leaf=18,
                subsample=0.82, colsample_bylevel=0.78, early_stopping_rounds=55,
                verbose=0, random_seed=1043,
            ), {}),
            ("stable_v3", "v3", dict(
                loss_function="MAE", iterations=800, learning_rate=0.03,
                depth=6, l2_leaf_reg=2.0, min_data_in_leaf=24,
                subsample=0.8, colsample_bylevel=0.8, early_stopping_rounds=45,
                verbose=0, random_seed=143,
            ), {}),
            ("base_v2", "v2", dict(
                loss_function="MAE", iterations=700, learning_rate=0.035,
                depth=6, l2_leaf_reg=1.5, min_data_in_leaf=20,
                subsample=0.8, colsample_bylevel=0.8, early_stopping_rounds=40,
                verbose=0, random_seed=43,
            ), {}),
            ("robust_latent_v2", "latent_v2", dict(
                loss_function="MAE", iterations=950, learning_rate=0.026,
                depth=7, l2_leaf_reg=3.5, min_data_in_leaf=24,
                bootstrap_type="Bayesian", bagging_temperature=0.8,
                random_strength=1.5, model_shrink_rate=0.02, model_shrink_mode="Constant",
                colsample_bylevel=0.72, early_stopping_rounds=60,
                verbose=0, random_seed=4743,
            ), {}),
        ],
        "AST": [
            ("latent_v3", "latent_v3", dict(
                loss_function="MAE", iterations=950, learning_rate=0.025,
                depth=7, l2_leaf_reg=2.5, min_data_in_leaf=24,
                subsample=0.78, colsample_bylevel=0.75, early_stopping_rounds=55,
                verbose=0, random_seed=744,
            ), {}),
            ("recency_latent_v2", "latent_v2", dict(
                loss_function="MAE", iterations=950, learning_rate=0.026,
                depth=7, l2_leaf_reg=3.0, min_data_in_leaf=22,
                subsample=0.80, colsample_bylevel=0.72, early_stopping_rounds=60,
                verbose=0, random_seed=6744,
            ), {"use_recency_weight": True}),
        ],
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    if predictor.artifact_free:
        raise RuntimeError(f"Recovered artifacts are unavailable: {predictor.artifact_free_reason}")

    feature_trainer = create_shared_trainer()
    raw_df = load_processed_frame(feature_trainer)
    X, baselines, y, sequence_meta = build_sequences_for_predictor(raw_df, predictor)
    train_idx, val_idx = build_player_tail_split(sequence_meta, val_fraction=float(args.val_fraction), min_train_rows=8, min_val_rows=1)
    split_strategy = "player_tail_holdout"
    if train_idx is None or val_idx is None:
        split_idx = int(len(X) * 0.8)
        train_idx = np.arange(split_idx, dtype=int)
        val_idx = np.arange(split_idx, len(X), dtype=int)
        split_strategy = "global_position_fallback"

    train_meta = None if sequence_meta is None else pd.DataFrame(sequence_meta).iloc[train_idx].reset_index(drop=True)
    val_meta = None if sequence_meta is None else pd.DataFrame(sequence_meta).iloc[val_idx].reset_index(drop=True)

    X_train, X_val = X[train_idx], X[val_idx]
    b_train, b_val = baselines[train_idx], baselines[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    delta_train = y_train - b_train
    delta_val = y_val - b_val
    y_val_orig = predictor.scaler_y.inverse_transform(y_val)
    b_val_orig = predictor.scaler_y.inverse_transform(b_val)

    structured_train_latents = weighted_ensemble_latent_export(predictor.models, predictor.val_losses, X_train, b_train)
    structured_val_latents = weighted_ensemble_latent_export(predictor.models, predictor.val_losses, X_val, b_val)
    _, train_sets, _ = predictor._feature_sets(X_train, b_train)
    _, val_sets, _ = predictor._feature_sets(X_val, b_val)

    feature_pair_map = {
        key: (train_sets[key], val_sets[key])
        for key in train_sets.keys()
        if key in val_sets
    }
    recency_weight_train = build_recency_sample_weights(train_meta)
    val_order_values = None
    if val_meta is not None and not val_meta.empty:
        parsed = pd.to_datetime(val_meta["target_date"], errors="coerce")
        if parsed.notna().any():
            val_order_values = parsed.map(lambda value: float(value.value) if pd.notna(value) else np.nan).to_numpy(dtype=np.float64)

    cb_models = []
    cb_delta = np.zeros_like(delta_val)
    cb_model_info = []
    refresh_specs = refresh_candidate_specs()
    print("\n" + "=" * 80)
    print("TARGETED CATBOOST REFRESH")
    print("=" * 80)
    for t_idx, target in enumerate(predictor.target_columns):
        records = []
        for candidate_name, feature_version, params, fit_options in refresh_specs[target]:
            X_gbm_train, X_gbm_val = feature_pair_map[feature_version]
            model = CatBoostRegressor(**params)
            fit_kwargs = {}
            if fit_options.get("use_recency_weight") and recency_weight_train is not None:
                fit_kwargs["sample_weight"] = recency_weight_train
            model.fit(X_gbm_train, delta_train[:, t_idx], eval_set=(X_gbm_val, delta_val[:, t_idx]), verbose=0, **fit_kwargs)
            candidate_delta = np.asarray(model.predict(X_gbm_val), dtype=np.float32)
            pred_orig = predictor.scaler_y.inverse_transform(
                np.column_stack(
                    [
                        b_val[:, 0] + (candidate_delta if t_idx == 0 else cb_delta[:, 0]),
                        b_val[:, 1] + (candidate_delta if t_idx == 1 else cb_delta[:, 1]),
                        b_val[:, 2] + (candidate_delta if t_idx == 2 else cb_delta[:, 2]),
                    ]
                )
            )[:, t_idx]
            mae = float(np.mean(np.abs(y_val_orig[:, t_idx] - pred_orig)))
            rolling_stats = rolling_window_stats_1d(pred_orig, y_val_orig[:, t_idx], n_windows=4, order_values=val_order_values)
            score = candidate_score(mae, rolling_stats)
            records.append(
                {
                    "name": candidate_name,
                    "feature_version": feature_version,
                    "model": model,
                    "delta": candidate_delta,
                    "mae": mae,
                    "score": score,
                    "best_iteration": int(model.best_iteration_),
                }
            )
            print(f"  {target} | {candidate_name:22s} mae={mae:.4f} score={score:.4f}")

        records.sort(key=lambda item: (item["score"], item["mae"]))
        candidate_matrix = np.column_stack([record["delta"] for record in records])
        blend_weights = fit_positive_weights(candidate_matrix, delta_val[:, t_idx])
        if blend_weights is None:
            blend_weights = np.full(len(records), 1.0 / len(records), dtype=np.float32)
        blended_delta = candidate_matrix @ blend_weights
        best_single = records[0]
        best_single_pred = predictor.scaler_y.inverse_transform(
            np.column_stack(
                [
                    b_val[:, 0] + (best_single["delta"] if t_idx == 0 else cb_delta[:, 0]),
                    b_val[:, 1] + (best_single["delta"] if t_idx == 1 else cb_delta[:, 1]),
                    b_val[:, 2] + (best_single["delta"] if t_idx == 2 else cb_delta[:, 2]),
                ]
            )
        )[:, t_idx]
        best_single_score = candidate_score(
            float(np.mean(np.abs(y_val_orig[:, t_idx] - best_single_pred))),
            rolling_window_stats_1d(best_single_pred, y_val_orig[:, t_idx], n_windows=4, order_values=val_order_values),
        )
        blended_pred = predictor.scaler_y.inverse_transform(
            np.column_stack(
                [
                    b_val[:, 0] + (blended_delta if t_idx == 0 else cb_delta[:, 0]),
                    b_val[:, 1] + (blended_delta if t_idx == 1 else cb_delta[:, 1]),
                    b_val[:, 2] + (blended_delta if t_idx == 2 else cb_delta[:, 2]),
                ]
            )
        )[:, t_idx]
        blended_mae = float(np.mean(np.abs(y_val_orig[:, t_idx] - blended_pred)))
        blended_score = candidate_score(
            blended_mae,
            rolling_window_stats_1d(blended_pred, y_val_orig[:, t_idx], n_windows=4, order_values=val_order_values),
        )

        if blended_score + 1e-8 < best_single_score:
            cb_delta[:, t_idx] = blended_delta
            cb_models.append(
                {
                    "members": [
                        {
                            "model": record["model"],
                            "feature_version": record["feature_version"],
                            "candidate": record["name"],
                            "best_iteration": record["best_iteration"],
                        }
                        for record in records
                    ],
                    "weights": [float(weight) for weight in blend_weights],
                    "ensemble_size": int(len(records)),
                }
            )
            cb_model_info.append(
                {
                    "target": target,
                    "feature_version": "+".join(record["feature_version"] for record in records),
                    "candidate": "targeted_refresh_ensemble",
                    "feature_versions": [record["feature_version"] for record in records],
                    "candidates": [record["name"] for record in records],
                    "weights": [float(weight) for weight in blend_weights],
                    "ensemble_size": int(len(records)),
                    "best_iteration": int(np.mean([record["best_iteration"] for record in records])),
                    "mae": blended_mae,
                    "selection_score": blended_score,
                }
            )
            print(f"  {target} -> ensemble selected | mae={blended_mae:.4f} score={blended_score:.4f}")
        else:
            cb_delta[:, t_idx] = best_single["delta"]
            cb_models.append(
                {
                    "members": [
                        {
                            "model": best_single["model"],
                            "feature_version": best_single["feature_version"],
                            "candidate": best_single["name"],
                            "best_iteration": best_single["best_iteration"],
                        }
                    ],
                    "weights": [1.0],
                    "ensemble_size": 1,
                }
            )
            cb_model_info.append(
                {
                    "target": target,
                    "feature_version": best_single["feature_version"],
                    "candidate": best_single["name"],
                    "feature_versions": [best_single["feature_version"]],
                    "candidates": [best_single["name"]],
                    "weights": [1.0],
                    "ensemble_size": 1,
                    "best_iteration": best_single["best_iteration"],
                    "mae": best_single["mae"],
                    "selection_score": best_single["score"],
                }
            )
            print(f"  {target} -> single selected   | mae={best_single['mae']:.4f} score={best_single['score']:.4f}")

    cb_pred_orig = predictor.scaler_y.inverse_transform(b_val + cb_delta)
    cb_avg_mae, cb_metrics = evaluate_predictions(
        cb_pred_orig,
        y_val_orig,
        b_val_orig,
        predictor.target_columns,
        "REFRESHED CATBOOST DELTA",
    )
    cb_rolling_summary = evaluate_rolling_windows(
        cb_pred_orig,
        y_val_orig,
        predictor.target_columns,
        "REFRESHED CATBOOST ROLLING-WINDOW ROBUSTNESS",
        n_windows=4,
        sequence_meta=val_meta,
    )

    base_run_id = str(predictor.metadata.get("run_id") or predictor.production.get("run_id") or "lstm_v7")
    run_id = f"{base_run_id}_catboost_refresh_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(predictor.models)):
        source = predictor._artifact_path("lstm_weights", index=idx)
        _copy_artifact(source, run_dir / source.name)
    for key in ("pts_branch_weights", "scaler_x", "scaler_y", "schema"):
        source = predictor._artifact_path(key)
        _copy_artifact(source, run_dir / source.name)
    if predictor.artifact_paths.get("meta_models"):
        source = predictor._artifact_path("meta_models")
        _copy_artifact(source, run_dir / source.name)

    joblib.dump(cb_models, run_dir / "lstm_v7_catboost_models.pkl")

    metadata = dict(predictor.metadata)
    metadata["run_id"] = run_id
    metadata["best_method"] = "catboost_delta"
    metadata["promoted_to_production"] = False
    metadata["avg_mae"] = float(cb_avg_mae)
    metadata["results"] = dict(metadata.get("results", {}))
    metadata["results"]["catboost_delta"] = cb_metrics
    metadata["rolling_window_validation"] = {"catboost_delta": cb_rolling_summary}
    metadata["pts_latent_ablation"] = predictor.metadata.get("pts_latent_ablation")
    metadata["validation_split"] = {
        "strategy": split_strategy,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "val_fraction": float(args.val_fraction),
        "val_date_min": str(pd.to_datetime(val_meta["target_date"], errors="coerce").min().date()) if val_meta is not None and not val_meta.empty and pd.to_datetime(val_meta["target_date"], errors="coerce").notna().any() else None,
        "val_date_max": str(pd.to_datetime(val_meta["target_date"], errors="coerce").max().date()) if val_meta is not None and not val_meta.empty and pd.to_datetime(val_meta["target_date"], errors="coerce").notna().any() else None,
    }
    metadata["refresh_source_run_id"] = base_run_id
    metadata["refresh_mode"] = "catboost_only"
    metadata["catboost_model_info"] = normalize_catboost_model_info(cb_model_info, predictor.target_columns, cb_models)

    artifact_paths = _artifact_path_strings(run_dir, predictor)
    metadata["artifact_paths"] = artifact_paths

    contract_errors = validate_metadata_contract(
        metadata,
        scaler_x=predictor.scaler_x,
        scaler_y=predictor.scaler_y,
        cb_models=cb_models,
    )
    if contract_errors:
        raise ValueError("Refreshed metadata contract is invalid:\n" + "\n".join(f"  - {line}" for line in contract_errors))

    (run_dir / "lstm_v7_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "lstm_v7_feature_schema.json").write_text(json.dumps(build_schema_payload(metadata), indent=2), encoding="utf-8")

    latest_manifest = {
        "run_id": run_id,
        "avg_mae": float(cb_avg_mae),
        "best_method": "catboost_delta",
        "artifact_paths": artifact_paths,
    }
    (MODEL_DIR / "latest_structured_lstm_stack.json").write_text(json.dumps(latest_manifest, indent=2), encoding="utf-8")

    current_production_mae = float(predictor.production.get("avg_mae", predictor.metadata.get("avg_mae", 999.0)))
    if args.promote_if_better and float(cb_avg_mae) < current_production_mae:
        production_manifest = dict(latest_manifest)
        (MODEL_DIR / "production_structured_lstm_stack.json").write_text(json.dumps(production_manifest, indent=2), encoding="utf-8")
        metadata["promoted_to_production"] = True
        (run_dir / "lstm_v7_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Promoted refreshed CatBoost run to production: {run_id}")

    print("\n" + "=" * 80)
    print("CATBOOST REFRESH COMPLETE")
    print("=" * 80)
    print(f"Recovered from manifest: {manifest_path}")
    print(f"Source run id:           {base_run_id}")
    print(f"New run id:              {run_id}")
    print(f"Validation avg MAE:      {cb_avg_mae:.4f}")
    print(f"Production avg MAE:      {current_production_mae:.4f}")
    print(f"Run directory:           {run_dir}")


if __name__ == "__main__":
    main()
