#!/usr/bin/env python3
"""
Inference for the structured LSTM + CatBoost stack.

This loader:
- rebuilds the structured LSTM ensemble
- loads the production CatBoost models
- computes production predictions
- emits a latent environment explanation alongside the prediction
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from improved_lstm_v7 import (
    PTSEmbeddingTrainingModel,
    StructuredLSTMTrainingModel,
    build_feature_spec,
    build_pts_embedding_model,
    build_pts_state_feature_matrix,
    build_structured_latent_feature_matrix,
    build_structured_lstm,
    create_shared_trainer,
    get_ensemble_variants,
    prepare_gbm_features_v3,
    weighted_ensemble_diagnostics,
)
from improved_stacking_trainer import prepare_gbm_features_v2
try:
    from structured_stack_contract import (
        apply_schema_sidecar,
        normalize_catboost_model_info,
        validate_metadata_contract,
    )
except Exception:  # pragma: no cover - fallback when contract helpers are unavailable
    def normalize_catboost_model_info(catboost_model_info, target_columns, cb_models):
        targets = [str(target) for target in (target_columns or [])]
        models = list(cb_models) if isinstance(cb_models, (list, tuple)) else []
        existing = {}
        if isinstance(catboost_model_info, list):
            for item in catboost_model_info:
                if isinstance(item, dict) and item.get("target") is not None:
                    existing[str(item["target"])] = dict(item)
        normalized = []
        for idx, target in enumerate(targets):
            model_bundle = models[idx] if idx < len(models) else None
            entry = existing.get(target, {})
            feature_versions = entry.get("feature_versions")
            if not feature_versions:
                if entry.get("feature_version"):
                    feature_versions = [entry.get("feature_version")]
                elif isinstance(model_bundle, dict):
                    members = model_bundle.get("members", [])
                    feature_versions = [member.get("feature_version") for member in members if isinstance(member, dict) and member.get("feature_version")]
                else:
                    feature_versions = ["v3"]
            feature_versions = [str(value) for value in feature_versions if value]
            if not feature_versions:
                feature_versions = ["v3"]
            normalized.append(
                {
                    "target": target,
                    "feature_version": str(entry.get("feature_version") or feature_versions[0]),
                    "feature_versions": feature_versions,
                }
            )
        return normalized

    def validate_metadata_contract(metadata, scaler_x=None, scaler_y=None, cb_models=None):
        errors = []
        if not isinstance(metadata, dict):
            return ["metadata must be a dictionary"]
        for key in ["target_columns", "feature_columns", "baseline_features", "n_targets", "n_features", "catboost_model_info"]:
            if key not in metadata:
                errors.append(f"missing required metadata key: {key}")
        return errors

    def apply_schema_sidecar(metadata, schema_payload):
        repaired = dict(metadata or {})
        sidecar = dict(schema_payload or {})
        for key in [
            "target_columns",
            "feature_columns",
            "baseline_features",
            "feature_spec",
            "n_targets",
            "n_features",
            "seq_len",
            "catboost_model_info",
            "counts",
            "schema_signature",
            "artifact_contract_version",
        ]:
            if key not in repaired or repaired.get(key) in (None, [], {}, ""):
                if key in sidecar:
                    repaired[key] = sidecar[key]
        return repaired


TEAM_ID_BY_ABBREV = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}

TARGET_FLOOR_RATIOS = {
    "PTS": 0.50,
    "TRB": 0.35,
    "AST": 0.35,
}

TARGET_DOWNSIDE_TRIGGER_RATIOS = {
    "PTS": 0.72,
    "TRB": 0.58,
    "AST": 0.58,
}

DETERMINISTIC_SEED = 42


def _configure_deterministic_runtime():
    np.random.seed(DETERMINISTIC_SEED)
    try:
        tf.keras.utils.set_random_seed(DETERMINISTIC_SEED)
    except Exception:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -12.0, 12.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


_configure_deterministic_runtime()


def infer_counts_from_weights(weight_path: Path) -> dict[str, int]:
    with h5py.File(weight_path, "r") as f:
        counts = {}
        candidates = [
            ("net\\layers\\embedding", "players"),
            ("net\\layers\\embedding_1", "teams"),
            ("net\\layers\\embedding_2", "opponents"),
            ("layers/player_embed", "players"),
            ("layers/team_embed", "teams"),
            ("layers/opp_embed", "opponents"),
        ]
        for group_name, key in candidates:
            if key in counts or group_name not in f:
                continue
            var_group = f[group_name]["vars"]
            first_key = sorted(var_group.keys())[0]
            counts[key] = int(var_group[first_key].shape[0])
    return counts


class StructuredStackInference:
    DEFAULT_TARGET_COLUMNS = ["PTS", "TRB", "AST"]

    @staticmethod
    def _build_forward_fn(net):
        @tf.function(reduce_retracing=True)
        def forward(x, baseline):
            return net([x, baseline], training=False)
        return forward

    def _init_artifact_free_mode(self, reason: str):
        self.artifact_free = True
        self.artifact_free_reason = str(reason)
        self.metadata = {
            "model_type": "structured_lstm_stack",
            "run_id": "artifact_free_heuristic",
            "target_columns": list(self.DEFAULT_TARGET_COLUMNS),
        }
        self.target_columns = list(self.DEFAULT_TARGET_COLUMNS)
        self.feature_columns = []
        self.baseline_features = [f"{target}_rolling_avg" for target in self.target_columns]
        self.feature_spec = {}
        self.seq_len = 10
        self.n_features = 0
        self.n_targets = len(self.target_columns)
        self.player_mapping = {}
        self.team_mapping = {}
        self.opponent_mapping = {}
        self.counts = {"players": 1, "teams": 1, "opponents": 1}
        self.member_configs = []
        self.val_losses = []
        self.catboost_model_info = {
            target: {
                "target": target,
                "feature_version": "heuristic",
                "feature_versions": ["heuristic"],
                "model_type": "heuristic",
                "member_count": 1,
            }
            for target in self.target_columns
        }
        self.required_feature_versions = {"heuristic"}
        self.feature_trainer = None
        self.models = []
        self.pts_branch = None
        self.pts_ablate_feature_key = None
        self.pts_ablate_blocks = []
        self.enable_pts_residual_split = False

    @staticmethod
    def _heuristic_prediction_payload(history_df: pd.DataFrame, reason: str | None = None):
        targets = ["PTS", "TRB", "AST"]
        active = history_df.copy()
        if "Did_Not_Play" in active.columns:
            active = active.loc[pd.to_numeric(active["Did_Not_Play"], errors="coerce").fillna(0.0) < 0.5].copy()
        if active.empty:
            active = history_df.copy()

        predicted: dict[str, float] = {}
        baseline: dict[str, float] = {}
        target_factors: dict[str, dict] = {}
        sigma_values: list[float] = []

        for target in targets:
            values = pd.to_numeric(active.get(target), errors="coerce").dropna()
            if values.empty:
                values = pd.to_numeric(history_df.get(target), errors="coerce").dropna()

            base_col = f"{target}_rolling_avg"
            baseline_series = pd.to_numeric(history_df.get(base_col), errors="coerce").dropna()
            baseline_value = float(baseline_series.iloc[-1]) if not baseline_series.empty else float(values.mean()) if not values.empty else 0.0

            if values.empty:
                pred_value = max(0.0, baseline_value)
                sigma = 0.0
                spike_prob = 0.10
            else:
                recent = values.tail(12)
                weights = np.linspace(1.0, 2.2, len(recent))
                recency_mean = float(np.average(recent.to_numpy(dtype=float), weights=weights))
                season_mean = float(values.mean())
                trend = float(recent.tail(min(3, len(recent))).mean() - recent.head(min(3, len(recent))).mean())

                pred_value = 0.55 * recency_mean + 0.30 * season_mean + 0.15 * (baseline_value + 0.35 * trend)
                pred_value = float(max(0.0, pred_value))
                sigma = float(np.std(recent.to_numpy(dtype=float), ddof=0)) if len(recent) > 1 else 0.0
                if len(recent) > 1:
                    recent_std = float(np.std(recent.to_numpy(dtype=float), ddof=0)) + 1e-6
                    z_score = float((recent.iloc[-1] - recent.mean()) / recent_std)
                    spike_prob = float(np.clip(0.50 + 0.20 * z_score, 0.05, 0.95))
                else:
                    spike_prob = 0.10

            predicted[target] = pred_value
            baseline[target] = float(max(0.0, baseline_value))
            sigma_values.append(sigma)
            target_factors[target] = {
                "uncertainty_sigma": sigma,
                "spike_probability": spike_prob,
            }

        avg_prediction = float(np.mean(list(predicted.values()))) if predicted else 0.0
        avg_sigma = float(np.mean(sigma_values)) if sigma_values else 0.0
        sigma_ratio = avg_sigma / max(1.0, avg_prediction)
        belief_uncertainty = float(np.clip(0.20 + 0.80 * sigma_ratio, 0.05, 0.95))
        mp_series = pd.to_numeric(active.get("MP"), errors="coerce").dropna()
        feasibility = 0.70 if mp_series.empty else float(np.clip(mp_series.tail(10).mean() / 34.0, 0.25, 0.98))
        fallback_reasons = ["artifact_free_heuristic"]
        if reason:
            fallback_reasons.append(str(reason))

        return {
            "baseline": baseline,
            "predicted": predicted,
            "predicted_raw_model": dict(predicted),
            "predicted_split_model": dict(predicted),
            "catboost_feature_versions": {target: ["heuristic"] for target in targets},
            "data_quality": {
                "schema_repaired": False,
                "used_default_ids": False,
                "repaired_columns": [],
                "nan_feature_repaired": False,
                "nan_feature_count": 0,
                "nan_feature_columns": [],
                "fallback_blend": 1.0,
                "fallback_reasons": fallback_reasons,
                "active_like": True,
                "floor_guard_applied": False,
                "pts_residual_split_applied": False,
                "pts_spike_gate": 0.0,
                "pts_spike_delta": 0.0,
                "pts_split_activation": 0.0,
            },
            "latent_environment": {
                "slow_state_strength": 0.0,
                "environment_strength": 0.0,
                "belief_uncertainty": belief_uncertainty,
                "feasibility": feasibility,
                "role_shift_risk": 0.35,
                "volatility_regime_risk": float(np.clip(sigma_ratio, 0.05, 0.95)),
                "context_pressure_risk": 0.30,
            },
            "target_factors": target_factors,
        }

    def _load_schema_sidecar(self):
        candidates = []
        try:
            candidates.append(self._artifact_path("schema", fallback="lstm_v7_feature_schema.json"))
        except Exception:
            pass
        candidates.append(self.model_dir / "lstm_v7_feature_schema.json")
        for path in candidates:
            try:
                if path.exists():
                    return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    def _maybe_persist_metadata(self):
        try:
            self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _validate_or_repair_metadata_contract(self):
        normalized = dict(self.metadata or {})
        normalized["catboost_model_info"] = normalize_catboost_model_info(
            normalized.get("catboost_model_info"),
            normalized.get("target_columns", []),
            self.cb_models,
        )
        errors = validate_metadata_contract(
            normalized,
            scaler_x=self.scaler_x,
            scaler_y=self.scaler_y,
            cb_models=self.cb_models,
        )
        if not errors:
            self.metadata = normalized
            return

        schema_payload = self._load_schema_sidecar()
        if schema_payload:
            repaired = apply_schema_sidecar(normalized, schema_payload)
            repaired["catboost_model_info"] = normalize_catboost_model_info(
                repaired.get("catboost_model_info"),
                repaired.get("target_columns", []),
                self.cb_models,
            )
            repaired_errors = validate_metadata_contract(
                repaired,
                scaler_x=self.scaler_x,
                scaler_y=self.scaler_y,
                cb_models=self.cb_models,
            )
            if not repaired_errors:
                self.metadata = repaired
                self._maybe_persist_metadata()
                return
            error_lines = "\n  - ".join(repaired_errors[:12])
            raise ValueError(
                "Structured stack metadata is invalid even after schema sidecar repair.\n"
                f"  - {error_lines}"
            )

        error_lines = "\n  - ".join(errors[:12])
        raise ValueError(
            "Structured stack metadata contract is invalid and no schema sidecar is available.\n"
            "  Run: python scripts/repair_structured_stack_contract.py\n"
            f"  - {error_lines}"
        )

    def __init__(self, model_dir="model", manifest_path=None, allow_schema_repair: bool = True):
        self.model_dir = Path(model_dir)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.allow_schema_repair = bool(allow_schema_repair)
        self.artifact_free = False
        self.artifact_free_reason = None
        self.production = self._load_manifest()
        self.artifact_paths = self.production.get("artifact_paths", {})
        try:
            self.metadata_path = self._artifact_path("metadata", fallback="lstm_v7_metadata.json")
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.scaler_x = joblib.load(self._artifact_path("scaler_x", fallback="lstm_v7_scaler_x.pkl"))
            self.scaler_y = joblib.load(self._artifact_path("scaler_y", fallback="lstm_v7_scaler_y.pkl"))
            self.cb_models = joblib.load(self._artifact_path("catboost_models", fallback="lstm_v7_catboost_models.pkl"))
        except FileNotFoundError as exc:
            self._init_artifact_free_mode(reason=f"{type(exc).__name__}: {exc}")
            return
        self._validate_or_repair_metadata_contract()
        self.target_columns = self.metadata["target_columns"]
        self.feature_columns = self.metadata["feature_columns"]
        self.baseline_features = self.metadata["baseline_features"]
        raw_feature_spec = self.metadata.get("feature_spec", {})
        if isinstance(raw_feature_spec, dict) and {"player_idx", "team_idx", "opp_idx"}.issubset(raw_feature_spec.keys()):
            self.feature_spec = raw_feature_spec
        else:
            # Backward compatibility: older metadata stored grouped feature names
            # instead of index-based spec required by the current model builders.
            self.feature_spec = build_feature_spec(self.feature_columns)
        self.seq_len = int(self.metadata["seq_len"])
        self.n_features = int(self.metadata["n_features"])
        self.n_targets = int(self.metadata["n_targets"])
        self.player_mapping = self.metadata.get("player_mapping", {})
        self.team_mapping = self.metadata.get("team_mapping", {})
        self.opponent_mapping = self.metadata.get("opponent_mapping", {})
        counts = self.metadata.get("counts")
        if not counts:
            counts = infer_counts_from_weights(self.model_dir / "lstm_v7_member_0.weights.h5")
        self.counts = counts
        self.member_configs = self.metadata.get("ensemble_member_configs") or get_ensemble_variants()[: self.metadata["n_models"]]
        self.val_losses = list(self.metadata.get("val_losses") or [1.0] * len(self.member_configs))
        self.catboost_model_info = {entry["target"]: entry for entry in self.metadata["catboost_model_info"]}
        self.enable_pts_residual_split = bool(self.metadata.get("enable_pts_residual_split", False))
        self.required_feature_versions = {
            feature_version
            for entry in self.metadata.get("catboost_model_info", [])
            for feature_version in entry.get("feature_versions", [entry.get("feature_version")])
            if feature_version
        }
        self.feature_trainer = create_shared_trainer()
        self.models = self._load_models()
        self.pts_branch = self._load_pts_branch() if self._needs_pts_branch() else None
        pts_ablation = self.metadata.get("pts_latent_ablation") or {}
        best_base_name = pts_ablation.get("best_base_name")
        best_blocks = list(pts_ablation.get("best_subset_blocks", []))
        self.pts_ablate_feature_key = None
        self.pts_ablate_blocks = []
        if best_base_name in {"v2", "v3"} and best_blocks:
            self.pts_ablate_feature_key = f"pts_ablate_{best_base_name}"
            self.pts_ablate_blocks = best_blocks

    @staticmethod
    def _mapping_key_kind(mapping: dict) -> str:
        keys = [str(k) for k in mapping.keys()]
        non_unk = [k for k in keys if k != "UNK"]
        if not non_unk:
            return "unk"
        if all(k.isdigit() or (k.startswith("-") and k[1:].isdigit()) for k in non_unk):
            return "numeric"
        if all(re.fullmatch(r"[A-Z]{2,4}", k or "") for k in non_unk):
            return "abbrev"
        return "string"

    @staticmethod
    def _extract_team_abbrev(matchup: str) -> str | None:
        if not isinstance(matchup, str):
            return None
        token = matchup.split()[0].strip().upper()
        return token if token in TEAM_ID_BY_ABBREV else None

    def _map_categorical_value(self, value, mapping: dict, default: int = 0) -> int:
        if value is None:
            return default
        kind = self._mapping_key_kind(mapping)
        if kind == "numeric":
            return int(mapping.get(str(int(value)), default))
        return int(mapping.get(str(value), default))

    def _repair_required_columns(self, df: pd.DataFrame):
        out = df.copy()
        repair_flags = {
            "repaired_columns": [],
            "used_default_ids": False,
            "schema_repaired": False,
            "nan_feature_repaired": False,
            "nan_feature_count": 0,
            "nan_feature_columns": [],
        }

        # This raw timestamp exists to complete the processed-data contract.
        # The current production artifacts were effectively trained with it as
        # a zeroed placeholder, so keep inference consistent and numeric-safe.
        if "Market_Fetched_At_UTC" in out.columns:
            out["Market_Fetched_At_UTC"] = 0.0

        if "Player_ID" not in out.columns:
            out["Player_ID"] = out["Player"].astype(str).map(self.player_mapping).fillna(0).astype(int)
            repair_flags["repaired_columns"].append("Player_ID")

        if "Team_ID" not in out.columns:
            team_kind = self._mapping_key_kind(self.team_mapping)
            if team_kind == "numeric":
                team_ids = out.get("MATCHUP", pd.Series([None] * len(out))).map(self._extract_team_abbrev).map(TEAM_ID_BY_ABBREV)
                if team_ids is not None:
                    out["Team_ID"] = team_ids.map(lambda v: self._map_categorical_value(v, self.team_mapping, 0)).fillna(0).astype(int)
                else:
                    out["Team_ID"] = 0
            elif team_kind == "abbrev":
                out["Team_ID"] = out.get("MATCHUP", pd.Series([None] * len(out))).map(self._extract_team_abbrev).map(lambda v: self._map_categorical_value(v, self.team_mapping, 0)).fillna(0).astype(int)
            else:
                out["Team_ID"] = 0
            repair_flags["repaired_columns"].append("Team_ID")

        if "Opponent_ID" not in out.columns:
            opp_kind = self._mapping_key_kind(self.opponent_mapping)
            if opp_kind == "numeric" and "Opponent" in out.columns:
                opp_values = out["Opponent"].map(TEAM_ID_BY_ABBREV)
                out["Opponent_ID"] = opp_values.map(lambda v: self._map_categorical_value(v, self.opponent_mapping, 0)).fillna(0).astype(int)
            elif opp_kind == "abbrev" and "Opponent" in out.columns:
                out["Opponent_ID"] = out["Opponent"].astype(str).map(lambda v: self._map_categorical_value(v, self.opponent_mapping, 0)).fillna(0).astype(int)
            else:
                out["Opponent_ID"] = 0
            repair_flags["repaired_columns"].append("Opponent_ID")
        elif self._mapping_key_kind(self.opponent_mapping) == "numeric":
            out["Opponent_ID"] = pd.Series(out["Opponent_ID"]).fillna(0).astype(int).map(lambda v: self._map_categorical_value(v, self.opponent_mapping, 0)).astype(int)

        if self._mapping_key_kind(self.team_mapping) == "numeric" and "Team_ID" in out.columns:
            out["Team_ID"] = pd.Series(out["Team_ID"]).fillna(0).astype(int)
            if out["Team_ID"].max() >= self.counts["teams"]:
                out["Team_ID"] = out["Team_ID"].map(lambda v: self._map_categorical_value(v, self.team_mapping, 0)).astype(int)

        if any(col in repair_flags["repaired_columns"] for col in ["Player_ID", "Team_ID", "Opponent_ID"]):
            repair_flags["schema_repaired"] = True
            repair_flags["used_default_ids"] = bool(
                (out[["Player_ID", "Team_ID", "Opponent_ID"]] == 0).any(axis=None)
            )

        for col in self.baseline_features:
            if col not in out.columns:
                target = col.replace("_rolling_avg", "")
                if target in out.columns:
                    out[col] = out[target].rolling(window=5, min_periods=1).mean()
                else:
                    out[col] = 0.0
                repair_flags["repaired_columns"].append(col)

        missing = [col for col in self.feature_columns if col not in out.columns]
        if missing:
            for col in missing:
                out[col] = 0.0
            repair_flags["repaired_columns"].extend(missing)
            repair_flags["schema_repaired"] = True

        return out, repair_flags

    def _load_manifest(self):
        if self.manifest_path is not None:
            if self.manifest_path.exists():
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return {}
        for name in ["production_structured_lstm_stack.json", "latest_structured_lstm_stack.json"]:
            path = self.model_dir / name
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _artifact_path(self, key, fallback=None, index=None):
        value = self.artifact_paths.get(key)
        if isinstance(value, list):
            if index is None:
                raise ValueError(f"Artifact '{key}' requires an index")
            if index < len(value):
                path = Path(value[index])
                return path if path.is_absolute() else self.model_dir.parent / path
            if fallback is None:
                raise IndexError(f"Artifact '{key}' index {index} is unavailable")
            return self.model_dir / fallback
        if isinstance(value, str):
            path = Path(value)
            return path if path.is_absolute() else self.model_dir.parent / path
        if fallback is None:
            raise FileNotFoundError(f"Artifact '{key}' is not defined in the manifest")
        return self.model_dir / fallback

    def _load_models(self):
        models = []
        for idx, cfg in enumerate(self.member_configs):
            net = build_structured_lstm(
                seq_len=self.seq_len,
                n_features=self.n_features,
                n_targets=self.n_targets,
                feature_spec=self.feature_spec,
                counts=self.counts,
                seed=self.metadata["seeds"][idx] if idx < len(self.metadata["seeds"]) else 42 + idx * 19,
                config=cfg,
            )
            wrapped = StructuredLSTMTrainingModel(net, feature_spec=self.feature_spec)
            # Keras 3 requires subclassed models to be built before loading weights.
            wrapped([np.zeros((1, self.seq_len, self.n_features), dtype=np.float32), np.zeros((1, self.n_targets), dtype=np.float32)], training=False)
            wrapped.load_weights(self._artifact_path("lstm_weights", fallback=f"lstm_v7_member_{idx}.weights.h5", index=idx), skip_mismatch=True)

            class Wrapper:
                def __init__(self, net):
                    self.net = net
                    self._inference_forward = StructuredStackInference._build_forward_fn(net)

            models.append(Wrapper(net))
        return models

    def _needs_pts_branch(self):
        if any(feature_version in {"pts_v2", "pts_v3"} for feature_version in self.required_feature_versions):
            return True
        if not self.enable_pts_residual_split:
            return False
        try:
            return self._artifact_path("pts_branch_weights", fallback="lstm_v7_pts_branch.weights.h5").exists()
        except Exception:
            return False

    def _load_pts_branch(self):
        cfg = self.metadata["config"]
        net = build_pts_embedding_model(
            self.seq_len,
            self.n_features,
            self.feature_spec,
            self.counts,
            seed=777,
            config=cfg,
        )
        weights_path = self._artifact_path("pts_branch_weights", fallback="lstm_v7_pts_branch.weights.h5")
        if weights_path.exists():
            net.load_weights(weights_path, skip_mismatch=True)
        net._inference_forward = self._build_forward_fn(net)
        return net

    def prepare_input(self, recent_games_df: pd.DataFrame, assume_prepared: bool = False):
        df = recent_games_df.copy()
        if "Player" not in df.columns:
            df["Player"] = "Inference_Player"
        if not assume_prepared:
            with contextlib.redirect_stdout(io.StringIO()):
                df = self.feature_trainer.create_hybrid_features(df)
        if not self.allow_schema_repair:
            missing = [col for col in self.feature_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing[:10]}")
        df, repair_info = self._repair_required_columns(df)
        if len(df) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} rows, got {len(df)}")
        df = df.tail(self.seq_len).copy()
        categorical_features = self.feature_columns[:3]
        numeric_features = self.feature_columns[3:]

        # Production guardrail: never feed NaN/inf numeric values into the scaler/model.
        numeric_frame = df[numeric_features].apply(pd.to_numeric, errors="coerce")
        invalid_mask = ~np.isfinite(numeric_frame.to_numpy(dtype=np.float32))
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            repair_info["nan_feature_repaired"] = True
            repair_info["nan_feature_count"] = invalid_count
            invalid_columns = [
                column
                for column, has_invalid in zip(numeric_features, invalid_mask.any(axis=0).tolist())
                if has_invalid
            ]
            repair_info["nan_feature_columns"] = invalid_columns
            numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)
            numeric_frame = numeric_frame.ffill().bfill().fillna(0.0)
            df.loc[:, numeric_features] = numeric_frame.astype(np.float32)

        cat_values = df[categorical_features].values.astype(np.float32)
        cat_values[:, 0] = np.clip(cat_values[:, 0], 0, self.counts["players"] - 1)
        cat_values[:, 1] = np.clip(cat_values[:, 1], 0, self.counts["teams"] - 1)
        cat_values[:, 2] = np.clip(cat_values[:, 2], 0, self.counts["opponents"] - 1)
        numeric_values = df[numeric_features].values.astype(np.float32)
        numeric_scaled = self.scaler_x.transform(numeric_values).astype(np.float32)
        X_scaled = np.concatenate([cat_values, numeric_scaled], axis=1).reshape(1, self.seq_len, self.n_features)
        baseline = df.iloc[-1][self.baseline_features].values.astype(np.float32)
        baseline_scaled = self.scaler_y.transform(baseline.reshape(1, -1)).astype(np.float32)
        latest_row = df.iloc[-1]
        recent_window = df.tail(5)
        recent_window_3 = df.tail(3)
        latest_context = {
            "did_not_play": float(latest_row["Did_Not_Play"]) if "Did_Not_Play" in latest_row.index else 0.0,
            "rest_days": float(latest_row["Rest_Days"]) if "Rest_Days" in latest_row.index else 2.0,
            "mp": float(latest_row["MP"]) if "MP" in latest_row.index else None,
            "usg": float(latest_row["USG%"]) if "USG%" in latest_row.index else None,
            "recent_pts_last": float(latest_row["PTS"]) if "PTS" in latest_row.index else 0.0,
            "recent_pts_max_5": float(recent_window["PTS"].max()) if "PTS" in recent_window.columns else 0.0,
            "recent_pts_avg_3": float(recent_window_3["PTS"].mean()) if "PTS" in recent_window_3.columns else 0.0,
            "recent_mp_avg_3": float(recent_window_3["MP"].mean()) if "MP" in recent_window_3.columns else 0.0,
            "recent_usg_avg_3": float(recent_window_3["USG%"].mean()) if "USG%" in recent_window_3.columns else 0.0,
            "recent_dnp_rate_5": float(recent_window["Did_Not_Play"].mean()) if "Did_Not_Play" in recent_window.columns else 0.0,
        }

        return X_scaled, baseline_scaled, baseline, repair_info, latest_context

    def _predict_catboost_target(self, model_bundle, feature_sets, feature_version=None):
        if isinstance(model_bundle, dict):
            members = model_bundle.get("members", [])
            weights = np.array(model_bundle.get("weights", []), dtype=np.float32)
            preds = [member["model"].predict(feature_sets[member["feature_version"]]) for member in members]
            if not preds:
                raise ValueError("CatBoost model bundle has no members")
            pred_matrix = np.column_stack(preds)
            if weights.size != pred_matrix.shape[1] or float(weights.sum()) <= 0.0:
                weights = np.full(pred_matrix.shape[1], 1.0 / pred_matrix.shape[1], dtype=np.float32)
            else:
                weights = weights / weights.sum()
            return pred_matrix @ weights
        return model_bundle.predict(feature_sets[feature_version])

    def _feature_sets(self, X, baseline):
        diagnostics = weighted_ensemble_diagnostics(self.models, self.val_losses, X, baseline)
        latent_blocks = {
            key: diagnostics[key]
            for key in [
                "bottleneck",
                "belief_mu",
                "belief_std",
                "slow_mode",
                "stable_env",
                "sigma",
                "spike_prob",
                "feasibility",
                "role_shift_prob",
                "volatility_regime_prob",
                "context_pressure_prob",
            ]
        }
        latent_matrix = build_structured_latent_feature_matrix(latent_blocks)
        sets = {
            "v2": prepare_gbm_features_v2(X, baseline),
            "v3": prepare_gbm_features_v3(X, baseline),
            "latent_v2": np.hstack([prepare_gbm_features_v2(X, baseline), latent_matrix]),
            "latent_v3": np.hstack([prepare_gbm_features_v3(X, baseline), latent_matrix]),
        }
        if self.pts_ablate_feature_key and self.pts_ablate_blocks:
            selected = [latent_blocks[name] for name in self.pts_ablate_blocks if name in latent_blocks]
            if selected:
                pts_ablate_matrix = np.hstack(selected)
                base_key = self.pts_ablate_feature_key.replace("pts_ablate_", "")
                sets[self.pts_ablate_feature_key] = np.hstack([sets[base_key], pts_ablate_matrix])
        pts_outputs = None
        if self.pts_branch is not None:
            pts_forward = getattr(self.pts_branch, "_inference_forward", None)
            if pts_forward is not None:
                pts_outputs = pts_forward(X, baseline[:, :1])
                pts_outputs = {
                    key: value.numpy() if hasattr(value, "numpy") else np.asarray(value)
                    for key, value in pts_outputs.items()
                }
            else:
                pts_outputs = self.pts_branch.predict([X, baseline[:, :1]], verbose=0, batch_size=1)
            pts_state = build_pts_state_feature_matrix(pts_outputs)
            sets["pts_v2"] = np.hstack([prepare_gbm_features_v2(X, baseline), pts_state])
            sets["pts_v3"] = np.hstack([prepare_gbm_features_v3(X, baseline), pts_state])
        return diagnostics, sets, pts_outputs

    def _apply_pts_residual_split(
        self,
        pred_raw_model: np.ndarray,
        baseline_raw: np.ndarray,
        diagnostics_1: dict,
        pts_outputs,
        latest_context: dict,
        repair_info: dict,
    ) -> tuple[np.ndarray, dict]:
        split_pred = pred_raw_model.copy()
        split_debug = {
            "applied": False,
            "pts_normal_delta": 0.0,
            "pts_spike_delta": 0.0,
            "pts_spike_gate": 0.0,
            "pts_split_activation": 0.0,
            "pts_latent_projected_mean": float(pred_raw_model[0]),
        }
        if pts_outputs is None or repair_info.get("schema_repaired") or repair_info.get("used_default_ids"):
            split_debug["pts_normal_delta"] = float(pred_raw_model[0] - baseline_raw[0])
            split_debug["pts_latent_projected_mean"] = float(
                baseline_raw[0]
                + self.scaler_y.scale_[0] * diagnostics_1["delta_normal"][0]
                + self.scaler_y.scale_[0] * diagnostics_1["delta_tail"][0] * diagnostics_1["spike_prob"][0]
            )
            return split_pred, split_debug

        active_like = (
            latest_context.get("did_not_play", 0.0) < 0.5
            and latest_context.get("rest_days", 2.0) <= 4.0
        )
        if not active_like:
            split_debug["pts_normal_delta"] = float(pred_raw_model[0] - baseline_raw[0])
            return split_pred, split_debug

        pts_idx = self.target_columns.index("PTS")
        baseline_pts = float(baseline_raw[pts_idx])
        raw_pts = float(pred_raw_model[pts_idx])
        raw_delta = raw_pts - baseline_pts
        continuation = float(pts_outputs["pts_continuation_prob"][0][0])
        opportunity = float(pts_outputs["pts_opportunity_jump_prob"][0][0])
        trend_trust = float(pts_outputs["pts_trend_trust"][0][0])
        baseline_trust = float(pts_outputs["pts_baseline_trust"][0][0])
        elasticity = float(pts_outputs["pts_elasticity"][0][0])
        downside_risk = float(pts_outputs["pts_downside_risk"][0][0])
        spike_prob = float(diagnostics_1["spike_prob"][pts_idx])
        feasibility = float(diagnostics_1["feasibility"][0])
        belief_uncertainty = float(np.mean(diagnostics_1["belief_std"]))
        role_shift_risk = float(diagnostics_1["role_shift_prob"][0])
        volatility_risk = float(diagnostics_1["volatility_regime_prob"][0])
        context_pressure_risk = float(diagnostics_1["context_pressure_prob"][0])
        sigma_pts = float(self.scaler_y.scale_[pts_idx] * diagnostics_1["sigma"][pts_idx])
        latent_projected_mean = float(
            baseline_pts
            + self.scaler_y.scale_[pts_idx] * diagnostics_1["delta_normal"][pts_idx]
            + self.scaler_y.scale_[pts_idx] * diagnostics_1["delta_tail"][pts_idx] * diagnostics_1["spike_prob"][pts_idx]
        )

        gate_input = (
            2.2 * opportunity
            + 1.6 * continuation
            + 1.3 * trend_trust
            + 1.0 * elasticity
            + 0.9 * spike_prob
            + 0.5 * feasibility
            - 1.6 * downside_risk
            - 0.7 * baseline_trust
            - 0.35 * belief_uncertainty
            - 2.35
        )
        model_gate = float(pts_outputs["pts_spike_gate"][0][0]) if "pts_spike_gate" in pts_outputs else None
        heuristic_gate = _sigmoid(gate_input)
        context_gate = _sigmoid(
            1.65 * role_shift_risk
            + 1.55 * volatility_risk
            + 1.20 * context_pressure_risk
            + 0.85 * opportunity
            + 0.60 * continuation
            + 0.55 * elasticity
            + 0.30 * max(0.0, belief_uncertainty - 0.70)
            - 2.20
        )
        if model_gate is None:
            spike_gate = float(np.clip(0.60 * heuristic_gate + 0.40 * context_gate, 0.0, 0.95))
        else:
            spike_gate = float(np.clip(0.45 * model_gate + 0.30 * heuristic_gate + 0.25 * context_gate, 0.0, 0.95))

        downside_ratio = float(np.clip((0.92 * baseline_pts - raw_pts) / max(baseline_pts, 1.0), 0.0, 1.0))
        upside_room = max(0.0, latent_projected_mean - raw_pts)
        baseline_recovery = max(0.0, baseline_pts - raw_pts)
        sigma_gap = max(0.0, 0.32 * sigma_pts)
        activation = float(
            np.clip(
                0.55 * spike_gate
                + 0.30 * downside_ratio
                + 0.15 * max(0.0, context_gate - 0.25),
                0.0,
                0.92,
            )
        )
        desired_uplift = max(
            upside_room,
            0.55 * baseline_recovery,
            sigma_gap,
        )
        if activation >= 0.10 and desired_uplift > 0.20:
            upside_cap = max(
                2.5,
                0.45 * baseline_pts + 0.22 * sigma_pts,
                0.75 * sigma_pts,
            )
            spike_delta = min(upside_cap, desired_uplift) * activation
            split_pred[pts_idx] = raw_pts + spike_delta
            split_debug["applied"] = spike_delta > 0.05
            split_debug["pts_spike_delta"] = float(spike_delta)

        split_debug["pts_normal_delta"] = float(raw_delta)
        split_debug["pts_spike_gate"] = float(spike_gate)
        split_debug["pts_split_activation"] = float(activation)
        split_debug["pts_latent_projected_mean"] = float(latent_projected_mean)
        return split_pred, split_debug

    def _selective_guardrail(
        self,
        pred_raw_pre_fallback: np.ndarray,
        baseline_raw: np.ndarray,
        latest_context: dict,
        repair_info: dict,
        diagnostics_1: dict,
    ) -> tuple[np.ndarray, float, list[str], bool]:
        feasibility = float(diagnostics_1["feasibility"][0])
        sigma_mean = float(np.mean(diagnostics_1["sigma"]))
        belief_uncertainty = float(np.mean(diagnostics_1["belief_std"]))
        active_like = (
            latest_context.get("did_not_play", 0.0) < 0.5
            and (latest_context.get("rest_days", 2.0) <= 4.0)
        )

        fallback_blend = 0.0
        fallback_reasons = []
        if repair_info["schema_repaired"]:
            fallback_blend = max(fallback_blend, 0.45)
            fallback_reasons.append("schema_repaired")
        if repair_info["used_default_ids"]:
            fallback_blend = max(fallback_blend, 0.60)
            fallback_reasons.append("default_ids")

        low_feas = feasibility < 0.50
        medium_feas = feasibility < 0.62
        high_uncertainty = belief_uncertainty > 0.90
        elevated_uncertainty = belief_uncertainty > 0.78
        high_sigma = sigma_mean > 8.0

        downside_flags = []
        trigger_strength = 0.0
        for idx, target in enumerate(self.target_columns):
            baseline_value = float(baseline_raw[idx])
            if baseline_value <= 0.0:
                continue
            pred_value = float(pred_raw_pre_fallback[idx])
            ratio = pred_value / baseline_value
            if ratio < TARGET_DOWNSIDE_TRIGGER_RATIOS[target]:
                downside_flags.append(f"{target.lower()}_downside")
                trigger_strength = max(trigger_strength, TARGET_DOWNSIDE_TRIGGER_RATIOS[target] - ratio)

        if downside_flags and active_like and (medium_feas or elevated_uncertainty):
            selective_blend = 0.22 + 0.75 * trigger_strength
            if low_feas:
                selective_blend += 0.18
            if high_uncertainty:
                selective_blend += 0.12
            if high_sigma:
                selective_blend += 0.08
            fallback_blend = max(fallback_blend, float(np.clip(selective_blend, 0.25, 0.82)))
            fallback_reasons.extend(downside_flags)
            if low_feas:
                fallback_reasons.append("low_feasibility")
            if high_uncertainty:
                fallback_reasons.append("high_uncertainty")
            elif elevated_uncertainty:
                fallback_reasons.append("elevated_uncertainty")
            if high_sigma:
                fallback_reasons.append("high_sigma")

        # Re-enable upside: if context is healthy and the model wants to move above baseline,
        # do not let the fallback dominate just because uncertainty is elevated.
        if active_like and feasibility >= 0.62:
            upside_allowed = False
            for idx, target in enumerate(self.target_columns):
                if pred_raw_pre_fallback[idx] > baseline_raw[idx]:
                    upside_allowed = True
                    break
            if upside_allowed and fallback_blend > 0.0 and not repair_info["schema_repaired"]:
                fallback_blend = min(fallback_blend, 0.18)
                fallback_reasons.append("upside_release")

        pred_raw = pred_raw_pre_fallback.copy()
        if fallback_blend > 0.0:
            pred_raw = (1.0 - fallback_blend) * pred_raw + fallback_blend * baseline_raw

        floor_guard_applied = False
        if active_like:
            floor_needed = (
                repair_info["schema_repaired"]
                or low_feas
                or high_uncertainty
                or bool(downside_flags)
            )
            if floor_needed:
                for idx, target in enumerate(self.target_columns):
                    baseline_value = float(baseline_raw[idx])
                    if baseline_value <= 0.0:
                        continue
                    floor_value = TARGET_FLOOR_RATIOS[target] * baseline_value
                    if pred_raw[idx] < floor_value:
                        pred_raw[idx] = floor_value
                        fallback_reasons.append(f"{target.lower()}_floor_guard")
                        floor_guard_applied = True

        pred_raw = np.maximum(pred_raw, 0.0)
        fallback_reasons = list(dict.fromkeys(fallback_reasons))
        return pred_raw, float(np.clip(fallback_blend, 0.0, 0.90)), fallback_reasons, bool(floor_guard_applied)

    def predict(self, recent_games_df: pd.DataFrame, assume_prepared: bool = False, return_debug: bool = False):
        if self.artifact_free:
            return self._heuristic_prediction_payload(recent_games_df, reason=self.artifact_free_reason)
        X, baseline_scaled, baseline_raw, repair_info, latest_context = self.prepare_input(recent_games_df, assume_prepared=assume_prepared)
        diagnostics, feature_sets, pts_outputs = self._feature_sets(X, baseline_scaled)

        delta_scaled = np.zeros((1, self.n_targets), dtype=np.float32)
        feature_versions = []
        for idx, target in enumerate(self.target_columns):
            info = self.catboost_model_info[target]
            model_bundle = self.cb_models[idx]
            if isinstance(model_bundle, dict):
                feature_versions.append(info.get("feature_versions", [info["feature_version"]]))
                delta_scaled[:, idx] = self._predict_catboost_target(model_bundle, feature_sets)
            else:
                feature_version = info["feature_version"]
                feature_versions.append([feature_version])
                delta_scaled[:, idx] = self._predict_catboost_target(model_bundle, feature_sets, feature_version=feature_version)

        pred_raw = self.scaler_y.inverse_transform(baseline_scaled + delta_scaled)[0]
        baseline_scaled_vec = baseline_scaled[0]
        diagnostics_1 = {key: value[0] for key, value in diagnostics.items()}
        feasibility = float(diagnostics_1["feasibility"][0])
        sigma_mean = float(np.mean(diagnostics_1["sigma"]))
        belief_uncertainty = float(np.mean(diagnostics_1["belief_std"]))
        active_like = (
            latest_context.get("did_not_play", 0.0) < 0.5
            and (latest_context.get("rest_days", 2.0) <= 4.0)
        )
        pred_raw_model = pred_raw.copy()
        if self.enable_pts_residual_split:
            pred_split_model, residual_split_debug = self._apply_pts_residual_split(
                pred_raw_model,
                baseline_raw,
                diagnostics_1,
                pts_outputs,
                latest_context,
                repair_info,
            )
        else:
            pred_split_model = pred_raw_model.copy()
            residual_split_debug = {
                "applied": False,
                "pts_normal_delta": float(pred_raw_model[0] - baseline_raw[0]),
                "pts_spike_delta": 0.0,
                "pts_spike_gate": 0.0,
                "pts_split_activation": 0.0,
                "pts_latent_projected_mean": float(pred_raw_model[0]),
            }
        pred_raw, fallback_blend, fallback_reasons, floor_guard_applied = self._selective_guardrail(
            pred_split_model,
            baseline_raw,
            latest_context,
            repair_info,
            diagnostics_1,
        )
        explanation = {
            "baseline": {t: float(v) for t, v in zip(self.target_columns, baseline_raw)},
            "predicted": {t: float(v) for t, v in zip(self.target_columns, pred_raw)},
            "predicted_raw_model": {t: float(v) for t, v in zip(self.target_columns, pred_raw_model)},
            "predicted_split_model": {t: float(v) for t, v in zip(self.target_columns, pred_split_model)},
            "catboost_feature_versions": {t: fv for t, fv in zip(self.target_columns, feature_versions)},
            "data_quality": {
                "schema_repaired": bool(repair_info["schema_repaired"]),
                "used_default_ids": bool(repair_info["used_default_ids"]),
                "repaired_columns": repair_info["repaired_columns"],
                "nan_feature_repaired": bool(repair_info.get("nan_feature_repaired", False)),
                "nan_feature_count": int(repair_info.get("nan_feature_count", 0)),
                "nan_feature_columns": repair_info.get("nan_feature_columns", []),
                "fallback_blend": fallback_blend,
                "fallback_reasons": fallback_reasons,
                "active_like": bool(active_like),
                "floor_guard_applied": bool(floor_guard_applied),
                "pts_residual_split_applied": bool(residual_split_debug["applied"]),
                "pts_spike_gate": float(residual_split_debug["pts_spike_gate"]),
                "pts_spike_delta": float(residual_split_debug["pts_spike_delta"]),
                "pts_split_activation": float(residual_split_debug["pts_split_activation"]),
            },
            "latent_environment": {
                "slow_state_strength": float(np.mean(np.abs(diagnostics_1["slow_mode"]))),
                "environment_strength": float(np.mean(np.abs(diagnostics_1["stable_env"]))),
                "belief_uncertainty": belief_uncertainty,
                "feasibility": feasibility,
                "role_shift_risk": float(diagnostics_1["role_shift_prob"][0]),
                "volatility_regime_risk": float(diagnostics_1["volatility_regime_prob"][0]),
                "context_pressure_risk": float(diagnostics_1["context_pressure_prob"][0]),
            },
            "target_factors": {},
        }
        if pts_outputs is not None:
            explanation["latent_environment"].update({
                "pts_trend_trust": float(pts_outputs["pts_trend_trust"][0][0]),
                "pts_baseline_trust": float(pts_outputs["pts_baseline_trust"][0][0]),
                "pts_elasticity": float(pts_outputs["pts_elasticity"][0][0]),
                "pts_opportunity_jump": float(pts_outputs["pts_opportunity_jump_prob"][0][0]),
                "pts_spike_gate": float(pts_outputs["pts_spike_gate"][0][0]) if "pts_spike_gate" in pts_outputs else None,
            })
        for idx, target in enumerate(self.target_columns):
            baseline_component = float(baseline_raw[idx])
            normal_component = float(self.scaler_y.scale_[idx] * diagnostics_1["delta_normal"][idx])
            tail_component = float(self.scaler_y.scale_[idx] * diagnostics_1["delta_tail"][idx] * diagnostics_1["spike_prob"][idx])
            sigma_component = float(self.scaler_y.scale_[idx] * diagnostics_1["sigma"][idx])
            explanation["target_factors"][target] = {
                "baseline_anchor": baseline_component,
                "normal_adjustment": normal_component,
                "tail_adjustment": tail_component,
                "spike_probability": float(diagnostics_1["spike_prob"][idx]),
                "uncertainty_sigma": sigma_component,
                "projected_mean_from_latent": baseline_component + normal_component + tail_component,
                "split_model_prediction": float(pred_split_model[idx]),
                "production_prediction": float(pred_raw[idx]),
            }
        if pts_outputs is not None:
            explanation["target_factors"]["PTS"].update({
                "normal_delta_component": float(pts_outputs["pts_normal_delta"][0][0]) if "pts_normal_delta" in pts_outputs else None,
                "spike_delta_component": float(pts_outputs["pts_spike_delta"][0][0]) if "pts_spike_delta" in pts_outputs else None,
                "spike_gate": float(pts_outputs["pts_spike_gate"][0][0]) if "pts_spike_gate" in pts_outputs else None,
                "continuation_probability": float(pts_outputs["pts_continuation_prob"][0][0]),
                "reversion_probability": float(pts_outputs["pts_reversion_prob"][0][0]),
                "opportunity_jump_probability": float(pts_outputs["pts_opportunity_jump_prob"][0][0]),
                "trend_trust": float(pts_outputs["pts_trend_trust"][0][0]),
                "baseline_trust": float(pts_outputs["pts_baseline_trust"][0][0]),
                "elasticity": float(pts_outputs["pts_elasticity"][0][0]),
                "downside_risk": float(pts_outputs["pts_downside_risk"][0][0]),
            })
        if return_debug:
            feature_debug = {}
            for key, value in feature_sets.items():
                arr = np.asarray(value, dtype=np.float32)
                feature_debug[key] = {
                    "shape": list(arr.shape),
                    "nan_count": int(np.isnan(arr).sum()),
                    "zero_fraction": float(np.mean(arr == 0.0)),
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                }
            explanation["debug"] = {
                "baseline_raw": {t: float(v) for t, v in zip(self.target_columns, baseline_raw)},
                "baseline_scaled": {t: float(v) for t, v in zip(self.target_columns, baseline_scaled[0])},
                "catboost_delta_scaled": {t: float(v) for t, v in zip(self.target_columns, delta_scaled[0])},
                "predicted_raw_model": {t: float(v) for t, v in zip(self.target_columns, pred_raw_model)},
                "predicted_post_split_pre_guard": {t: float(v) for t, v in zip(self.target_columns, pred_split_model)},
                "predicted_post_fallback": {t: float(v) for t, v in zip(self.target_columns, pred_raw)},
                "pts_residual_split": residual_split_debug,
                "feature_set_stats": feature_debug,
            }
        return explanation


def print_explanation(player_name: str, explanation: dict):
    print("\n" + "=" * 80)
    print(f"STRUCTURED STACK PREDICTION: {player_name}")
    print("=" * 80)
    print("\nBaseline anchor:")
    for target, value in explanation["baseline"].items():
        print(f"  {target}: {value:.2f}")
    print("\nPredicted:")
    for target, value in explanation["predicted"].items():
        print(f"  {target}: {value:.2f}")
    env = explanation["latent_environment"]
    print("\nLatent environment:")
    print(f"  slow_state_strength: {env['slow_state_strength']:.3f}")
    print(f"  environment_strength: {env['environment_strength']:.3f}")
    print(f"  belief_uncertainty: {env['belief_uncertainty']:.3f}")
    print(f"  feasibility: {env['feasibility']:.3f}")
    print(f"  role_shift_risk: {env['role_shift_risk']:.3f}")
    print(f"  volatility_regime_risk: {env['volatility_regime_risk']:.3f}")
    print(f"  context_pressure_risk: {env['context_pressure_risk']:.3f}")
    print("\nTarget factors:")
    for target, factors in explanation["target_factors"].items():
        print(f"  {target}:")
        print(f"    normal_adjustment: {factors['normal_adjustment']:.2f}")
        print(f"    tail_adjustment: {factors['tail_adjustment']:.2f}")
        print(f"    spike_probability: {factors['spike_probability']:.3f}")
        print(f"    uncertainty_sigma: {factors['uncertainty_sigma']:.2f}")
        print(f"    projected_mean_from_latent: {factors['projected_mean_from_latent']:.2f}")
        print(f"    production_prediction: {factors['production_prediction']:.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run structured stack inference with latent explanation.")
    parser.add_argument("--csv", required=True, help="Recent games csv with prepared feature columns")
    parser.add_argument("--player", default="Player", help="Display name for output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    inference = StructuredStackInference(model_dir="model")
    explanation = inference.predict(df)
    print_explanation(args.player, explanation)


if __name__ == "__main__":
    main()
