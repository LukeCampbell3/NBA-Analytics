from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import math
import re
import tempfile
import unicodedata

import numpy as np
import pandas as pd


PAYOUT_MINUS_110 = 100.0 / 110.0
PLAYER_NULL_TOKENS = {"", "nan", "none", "null", "nat"}


def normalize_player_name(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return "" if text in PLAYER_NULL_TOKENS else text


def coerce_event_datetime(values: Any) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    out = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    ymd_mask = numeric.between(19000101, 21001231, inclusive="both")
    if bool(ymd_mask.any()):
        ymd_text = numeric.round().astype("Int64").astype(str)
        parsed_ymd = pd.to_datetime(ymd_text.where(ymd_mask), format="%Y%m%d", errors="coerce")
        out = out.where(~ymd_mask, parsed_ymd)
    return out


def abbreviate_player_name(normalized: str) -> str:
    if not normalized:
        return ""
    parts = [part for part in normalized.split("_") if part]
    if len(parts) < 2:
        return normalized
    return f"{parts[0][0]}_{'_'.join(parts[1:])}"


def player_key_variants(*values: Any) -> set[str]:
    out: set[str] = set()
    for value in values:
        normalized = normalize_player_name(value)
        if not normalized:
            continue
        out.add(normalized)
        abbr = abbreviate_player_name(normalized)
        if abbr:
            out.add(abbr)
    return out


def build_pick_key(
    frame: pd.DataFrame,
    *,
    player_col: str = "market_player_raw",
    fallback_player_col: str = "player",
    market_date_col: str = "market_date",
    target_col: str = "target",
    direction_col: str = "direction",
    line_col: str = "market_line",
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    player_a = frame.get(player_col, pd.Series("", index=frame.index)).map(normalize_player_name)
    player_b = frame.get(fallback_player_col, pd.Series("", index=frame.index)).map(normalize_player_name)
    player = player_a.where(player_a.ne(""), player_b)
    market_date = pd.to_datetime(frame.get(market_date_col), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    target = frame.get(target_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    direction = frame.get(direction_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    line = pd.to_numeric(frame.get(line_col), errors="coerce").round(4)
    line_text = line.map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return (market_date + "|" + player + "|" + target + "|" + direction + "|" + line_text).astype("object")


def expected_utility_from_result(result: str, payout_per_win: float = PAYOUT_MINUS_110) -> float:
    tag = str(result).strip().lower()
    if tag == "win":
        return float(payout_per_win)
    if tag == "loss":
        return -1.0
    if tag == "push":
        return 0.0
    return float("nan")


def result_to_keep_label(result: str) -> float:
    tag = str(result).strip().lower()
    if tag == "win":
        return 1.0
    if tag == "loss":
        return 0.0
    return float("nan")


@dataclass
class LogisticGateConfig:
    learning_rate: float = 0.05
    l2_strength: float = 2.5
    max_iter: int = 3500
    tolerance: float = 1e-7
    class_weight_positive: float = 1.0
    class_weight_negative: float = 1.0


@dataclass
class CatBoostGateConfig:
    iterations: int = 320
    depth: int = 5
    learning_rate: float = 0.05
    l2_leaf_reg: float = 8.0
    random_strength: float = 1.0
    bagging_temperature: float = 0.0
    min_data_in_leaf: int = 20
    random_seed: int = 42
    class_weight_positive: float = 1.0
    class_weight_negative: float = 1.0
    segment_weight_power: float = 0.0
    segment_weight_cap: float = 3.0


@dataclass
class UpliftCatBoostGateConfig:
    iterations: int = 320
    depth: int = 5
    learning_rate: float = 0.05
    l2_leaf_reg: float = 8.0
    random_strength: float = 1.0
    bagging_temperature: float = 0.0
    min_data_in_leaf: int = 20
    random_seed: int = 42
    class_weight_positive: float = 1.0
    class_weight_negative: float = 1.0
    segment_weight_power: float = 0.0
    segment_weight_cap: float = 3.0
    keep_prob_temperature: float = 0.25
    harm_head_enabled: bool = True
    harm_positive_weight: float = 1.5
    harm_temperature: float = 1.0


@dataclass
class WalkForwardConfig:
    train_window_days: int = 120
    test_window_days: int = 14
    step_days: int = 7
    min_train_rows: int = 250
    min_test_rows: int = 20


@dataclass
class GatePolicyConfig:
    max_fire_rate: float = 0.10
    min_coverage_rate: float = 0.85
    max_removed_per_day: int = 1
    max_removed_per_player_per_day: int = 1
    max_removed_per_segment_per_day: int = 2
    max_removed_per_target_per_day: int = 2
    max_global_removed_per_player: int = 0
    max_global_removed_share_per_player: float = 1.0
    global_player_cap_min_veto_rows: int = 8
    relax_player_global_cap_for_budget: bool = True
    player_penalty_alpha: float = 0.0
    player_penalty_power: float = 1.0
    optimizer_enabled: bool = True
    optimizer_beam_width: int = 96
    optimizer_max_candidates_per_day: int = 64
    optimizer_keep_prob_weight: float = 1.0
    optimizer_gap_weight: float = 0.60
    optimizer_harm_weight: float = 0.80
    optimizer_rank_weight: float = 0.0
    segment_threshold_overrides: dict[str, float] = field(default_factory=dict)
    segment_budget_overrides: dict[str, int] = field(default_factory=dict)
    tail_slots_only: int = 2
    min_veto_gap: float = 0.02
    require_keep_prob_filter: bool = True
    allowed_segments: tuple[str, ...] = ()
    allowed_meta_cohorts: tuple[str, ...] = ()


def _bucketize_numeric(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bucket = pd.cut(numeric, bins=bins, labels=labels, include_lowest=True)
    out = bucket.astype("object").astype(str)
    out = out.where(out.str.lower().ne("nan"), "")
    out = out.fillna("")
    return out.astype("object")


def build_meta_cohort_columns(
    frame: pd.DataFrame,
    *,
    target_col: str = "target",
    direction_col: str = "direction",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    target = frame.get(target_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    direction = frame.get(direction_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    segment = (target + "|" + direction).where(target.str.len().gt(0) & direction.str.len().gt(0), "")

    cohorts = pd.DataFrame(index=frame.index)
    cohorts["meta_seg"] = ("SEG|" + segment).where(segment.str.len().gt(0), "")

    if "p_calibrated" in frame.columns:
        p_band = _bucketize_numeric(
            frame.get("p_calibrated", pd.Series(np.nan, index=frame.index)),
            bins=[-np.inf, 0.55, 0.60, 0.65, 0.70, 0.75, np.inf],
            labels=["LE_055", "B055_060", "B060_065", "B065_070", "B070_075", "GT_075"],
        )
        cohorts["meta_seg_p"] = ("SEGP|" + segment + "|" + p_band).where(segment.str.len().gt(0) & p_band.str.len().gt(0), "")

    if "expected_win_rate" in frame.columns:
        ew_band = _bucketize_numeric(
            frame.get("expected_win_rate", pd.Series(np.nan, index=frame.index)),
            bins=[-np.inf, 0.55, 0.60, 0.65, 0.70, 0.75, np.inf],
            labels=["LE_055", "B055_060", "B060_065", "B065_070", "B070_075", "GT_075"],
        )
        cohorts["meta_seg_ew"] = ("SEGEW|" + segment + "|" + ew_band).where(segment.str.len().gt(0) & ew_band.str.len().gt(0), "")

    if "ev" in frame.columns:
        ev_band = _bucketize_numeric(
            frame.get("ev", pd.Series(np.nan, index=frame.index)),
            bins=[-np.inf, 0.00, 0.02, 0.04, 0.06, 0.08, np.inf],
            labels=["LE_000", "B000_002", "B002_004", "B004_006", "B006_008", "GT_008"],
        )
        cohorts["meta_seg_ev"] = ("SEGEV|" + segment + "|" + ev_band).where(segment.str.len().gt(0) & ev_band.str.len().gt(0), "")

    if "market_line" in frame.columns:
        line_band = _bucketize_numeric(
            frame.get("market_line", pd.Series(np.nan, index=frame.index)),
            bins=[-np.inf, 0.5, 1.5, 3.5, 5.5, 8.5, np.inf],
            labels=["L00_05", "L05_15", "L15_35", "L35_55", "L55_85", "L85P"],
        )
        cohorts["meta_seg_line"] = ("SEGLINE|" + segment + "|" + line_band).where(segment.str.len().gt(0) & line_band.str.len().gt(0), "")

    if "board_instability_score" in frame.columns:
        inst_band = _bucketize_numeric(
            frame.get("board_instability_score", pd.Series(np.nan, index=frame.index)),
            bins=[-np.inf, 0.05, 0.10, 0.20, 0.30, np.inf],
            labels=["I00_05", "I05_10", "I10_20", "I20_30", "I30P"],
        )
        cohorts["meta_seg_instability"] = ("SEGINST|" + segment + "|" + inst_band).where(segment.str.len().gt(0) & inst_band.str.len().gt(0), "")

    if "board_segment_recent_weakness" in frame.columns:
        weakness_band = _bucketize_numeric(
            frame.get("board_segment_recent_weakness", pd.Series(np.nan, index=frame.index)),
            bins=[-np.inf, -0.10, 0.0, 0.10, 0.20, np.inf],
            labels=["WLT_M010", "WM010_000", "W000_010", "W010_020", "W020P"],
        )
        cohorts["meta_seg_weakness"] = ("SEGWEAK|" + segment + "|" + weakness_band).where(
            segment.str.len().gt(0) & weakness_band.str.len().gt(0),
            "",
        )

    for col in cohorts.columns:
        cohorts[col] = cohorts[col].fillna("").astype(str).str.upper().str.strip()
    return cohorts


@dataclass
class PromotionCriteria:
    min_broad_profit_delta_units: float = 0.0
    min_recent_profit_delta_units: float = 0.0
    min_recent_hit_rate_delta_pp: float = 0.0
    min_broad_hit_rate_delta_pp: float = -0.05
    min_coverage_retention: float = 0.96
    min_affected_share: float = 0.01
    max_top_removed_player_share: float = 0.30
    max_top_removed_segment_share: float = 0.35
    max_top_removed_target_share: float = 0.55
    concentration_min_veto_rows: int = 25
    conditional_target_cap_enabled: bool = False
    conditional_target_relaxed_cap: float = 0.90
    conditional_target_player_share_max: float = 0.15
    conditional_target_require_recent_non_negative: bool = True
    conditional_target_recent_hit_floor_pp: float = 0.0
    conditional_target_recent_profit_floor: float = 0.0
    min_rolling_pass_rate: float = 0.55
    max_observed_fire_rate: float = 0.20
    rolling_profit_delta_floor: float = 0.0
    rolling_hit_rate_delta_floor_pp: float = -0.10


@dataclass
class RegularizedLogisticGate:
    config: LogisticGateConfig = field(default_factory=LogisticGateConfig)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: dict[str, list[str]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    weights: np.ndarray | None = None
    bias: float = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _fit_preprocessor(self, frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> None:
        self.numeric_features = [str(col) for col in numeric_features]
        self.categorical_features = {}
        self.means = {}
        self.stds = {}
        for col in self.numeric_features:
            values = pd.to_numeric(frame.get(col), errors="coerce")
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std(ddof=0)) if values.notna().any() else 1.0
            if not np.isfinite(std) or std <= 1e-9:
                std = 1.0
            self.means[col] = mean
            self.stds[col] = std
        for col in categorical_features:
            series = frame.get(col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
            levels = sorted(level for level in series.unique().tolist() if level)
            self.categorical_features[str(col)] = levels

    def _matrix_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        n_rows = len(frame)
        blocks: list[np.ndarray] = []
        for col in self.numeric_features:
            raw = frame.get(col, pd.Series(np.nan, index=frame.index))
            values = pd.to_numeric(raw, errors="coerce").fillna(float(self.means.get(col, 0.0)))
            standardized = (values.to_numpy(dtype="float64") - float(self.means.get(col, 0.0))) / float(self.stds.get(col, 1.0))
            blocks.append(standardized.reshape(n_rows, 1))
        for col, levels in self.categorical_features.items():
            if not levels:
                continue
            series = frame.get(col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
            matrix = np.zeros((n_rows, len(levels)), dtype="float64")
            lookup = {level: idx for idx, level in enumerate(levels)}
            for idx, value in enumerate(series.tolist()):
                col_idx = lookup.get(value)
                if col_idx is not None:
                    matrix[idx, col_idx] = 1.0
            blocks.append(matrix)
        if not blocks:
            return np.zeros((n_rows, 0), dtype="float64")
        return np.concatenate(blocks, axis=1)

    def feature_names(self) -> list[str]:
        out: list[str] = []
        out.extend(self.numeric_features)
        for col, levels in self.categorical_features.items():
            for level in levels:
                out.append(f"{col}={level}")
        return out

    def fit_dataframe(
        self,
        frame: pd.DataFrame,
        label_col: str,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> None:
        train = frame.copy()
        labels = pd.to_numeric(train.get(label_col), errors="coerce")
        keep = labels.notna()
        train = train.loc[keep].copy()
        labels = labels.loc[keep].astype("float64")
        if train.empty:
            raise RuntimeError("Cannot fit logistic gate: no labeled rows.")

        self._fit_preprocessor(train, numeric_features=numeric_features, categorical_features=categorical_features)
        x = self._matrix_from_frame(train)
        n_rows, n_cols = x.shape
        if n_cols <= 0:
            raise RuntimeError("Cannot fit logistic gate: no feature columns.")
        y = labels.to_numpy(dtype="float64")

        w = np.zeros(n_cols, dtype="float64")
        b = 0.0
        cfg = self.config
        pos_w = float(max(cfg.class_weight_positive, 1e-9))
        neg_w = float(max(cfg.class_weight_negative, 1e-9))
        sample_w = np.where(y > 0.5, pos_w, neg_w).astype("float64")
        sample_w /= max(float(sample_w.mean()), 1e-9)
        lr = float(max(cfg.learning_rate, 1e-6))
        l2 = float(max(cfg.l2_strength, 0.0))
        prev_loss = math.inf

        for _ in range(int(max(cfg.max_iter, 10))):
            logits = np.dot(x, w) + b
            probs = self._sigmoid(logits)
            probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
            residual = probs - y
            weighted_residual = residual * sample_w
            grad_w = np.dot(x.T, weighted_residual) / float(n_rows) + l2 * w
            grad_b = float(np.mean(weighted_residual))
            w -= lr * grad_w
            b -= lr * grad_b

            if cfg.tolerance > 0.0:
                weighted_nll = -np.mean(sample_w * (y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
                loss = float(weighted_nll + 0.5 * l2 * np.sum(w * w))
                if abs(prev_loss - loss) <= float(cfg.tolerance):
                    prev_loss = loss
                    break
                prev_loss = loss

        self.weights = w.astype("float64")
        self.bias = float(b)

    def predict_proba_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("RegularizedLogisticGate is not fitted.")
        x = self._matrix_from_frame(frame)
        if x.shape[1] != self.weights.shape[0]:
            raise RuntimeError(
                f"Feature shape mismatch: matrix has {x.shape[1]} columns, model expects {self.weights.shape[0]}."
            )
        logits = np.dot(x, self.weights) + float(self.bias)
        return self._sigmoid(logits).astype("float64")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "regularized_logistic_gate_v1",
            "config": asdict(self.config),
            "numeric_features": list(self.numeric_features),
            "categorical_features": {k: list(v) for k, v in self.categorical_features.items()},
            "means": {k: float(v) for k, v in self.means.items()},
            "stds": {k: float(v) for k, v in self.stds.items()},
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": float(self.bias),
            "feature_names": self.feature_names(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RegularizedLogisticGate":
        cfg_payload = payload.get("config", {}) if isinstance(payload, dict) else {}
        gate = cls(config=LogisticGateConfig(**cfg_payload))
        gate.numeric_features = [str(col) for col in payload.get("numeric_features", [])]
        cat_payload = payload.get("categorical_features", {})
        gate.categorical_features = {str(k): [str(v) for v in values] for k, values in dict(cat_payload).items()}
        gate.means = {str(k): float(v) for k, v in dict(payload.get("means", {})).items()}
        gate.stds = {str(k): float(v) for k, v in dict(payload.get("stds", {})).items()}
        weights = payload.get("weights", [])
        gate.weights = np.asarray(weights, dtype="float64") if len(weights) else None
        gate.bias = float(payload.get("bias", 0.0))
        return gate


@dataclass
class CatBoostKeepDropGate:
    config: CatBoostGateConfig = field(default_factory=CatBoostGateConfig)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    _model: Any | None = field(default=None, repr=False)

    @staticmethod
    def _require_catboost() -> Any:
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise RuntimeError(
                "CatBoost is required for catboost gate inference/training but is not available."
            ) from exc
        return CatBoostClassifier

    def _matrix_from_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for col in self.numeric_features:
            values = pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")
            data[col] = values.astype("float64")
        for col in self.categorical_features:
            series = frame.get(col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
            data[col] = series.astype("object")
        if not data:
            return pd.DataFrame(index=frame.index)
        return pd.DataFrame(data, index=frame.index)

    def feature_names(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    @staticmethod
    def _serialize_model_blob(model: Any) -> str:
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as handle:
            model_path = Path(handle.name)
        try:
            model.save_model(str(model_path), format="cbm")
            blob = model_path.read_bytes()
        finally:
            try:
                model_path.unlink()
            except OSError:
                pass
        return base64.b64encode(blob).decode("ascii")

    @staticmethod
    def _deserialize_model_blob(blob_b64: str) -> Any:
        CatBoostClassifier = CatBoostKeepDropGate._require_catboost()
        raw = base64.b64decode(str(blob_b64).encode("ascii"))
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as handle:
            model_path = Path(handle.name)
        try:
            model_path.write_bytes(raw)
            model = CatBoostClassifier()
            model.load_model(str(model_path), format="cbm")
        finally:
            try:
                model_path.unlink()
            except OSError:
                pass
        return model

    def fit_dataframe(
        self,
        frame: pd.DataFrame,
        label_col: str,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> None:
        train = frame.copy()
        labels = pd.to_numeric(train.get(label_col), errors="coerce")
        keep = labels.notna()
        train = train.loc[keep].copy()
        labels = labels.loc[keep].astype("float64")
        if train.empty:
            raise RuntimeError("Cannot fit catboost gate: no labeled rows.")

        self.numeric_features = [str(col) for col in numeric_features if str(col) in train.columns]
        self.categorical_features = [str(col) for col in categorical_features if str(col) in train.columns]
        if not self.numeric_features and not self.categorical_features:
            raise RuntimeError("Cannot fit catboost gate: no feature columns.")

        x = self._matrix_from_frame(train)
        y = labels.to_numpy(dtype="int64")
        if len(np.unique(y)) < 2:
            raise RuntimeError("Cannot fit catboost gate: labels contain only one class.")

        cfg = self.config
        pos_w = float(max(cfg.class_weight_positive, 1e-9))
        neg_w = float(max(cfg.class_weight_negative, 1e-9))
        sample_w = np.where(y > 0, pos_w, neg_w).astype("float64")

        seg_power = float(max(0.0, getattr(cfg, "segment_weight_power", 0.0)))
        seg_cap = float(max(1.0, getattr(cfg, "segment_weight_cap", 3.0)))
        if seg_power > 1e-9:
            seg_target = train.get("target", pd.Series("", index=train.index)).fillna("").astype(str).str.upper().str.strip()
            seg_direction = train.get("direction", pd.Series("", index=train.index)).fillna("").astype(str).str.upper().str.strip()
            segment = (seg_target + "|" + seg_direction).where(
                seg_target.str.len().gt(0) & seg_direction.str.len().gt(0),
                "UNKNOWN|UNKNOWN",
            )
            segment_counts = segment.value_counts(dropna=False)
            median_count = float(segment_counts.median()) if not segment_counts.empty else 1.0
            median_count = max(1.0, median_count)
            per_row_count = segment.map(segment_counts).fillna(median_count).astype("float64")
            segment_multiplier = np.power(median_count / np.maximum(per_row_count.to_numpy(dtype="float64"), 1.0), seg_power)
            segment_multiplier = np.clip(segment_multiplier, 1.0 / seg_cap, seg_cap)
            sample_w *= segment_multiplier

        sample_w /= max(float(sample_w.mean()), 1e-9)

        CatBoostClassifier = self._require_catboost()
        model = CatBoostClassifier(
            iterations=int(max(20, cfg.iterations)),
            depth=int(max(2, cfg.depth)),
            learning_rate=float(max(1e-5, cfg.learning_rate)),
            l2_leaf_reg=float(max(0.0, cfg.l2_leaf_reg)),
            random_strength=float(max(0.0, cfg.random_strength)),
            bagging_temperature=float(max(0.0, cfg.bagging_temperature)),
            min_data_in_leaf=int(max(1, cfg.min_data_in_leaf)),
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=int(cfg.random_seed),
            verbose=False,
            allow_writing_files=False,
        )
        cat_features = [
            idx for idx, col in enumerate(x.columns.tolist()) if col in set(self.categorical_features)
        ]
        model.fit(
            x,
            y,
            cat_features=cat_features if cat_features else None,
            sample_weight=sample_w,
        )
        self._model = model

    def predict_proba_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CatBoostKeepDropGate is not fitted.")
        x = self._matrix_from_frame(frame)
        expected_cols = len(self.numeric_features) + len(self.categorical_features)
        if x.shape[1] != expected_cols:
            raise RuntimeError(
                f"Feature shape mismatch: matrix has {x.shape[1]} columns, model expects {expected_cols}."
            )
        probs = self._model.predict_proba(x)
        probs_arr = np.asarray(probs, dtype="float64")
        if probs_arr.ndim == 1:
            out = probs_arr
        elif probs_arr.shape[1] == 1:
            out = probs_arr[:, 0]
        else:
            out = probs_arr[:, 1]
        return np.clip(out.astype("float64"), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Cannot serialize CatBoost gate: model is not fitted.")
        return {
            "model_type": "catboost_gate_v1",
            "config": asdict(self.config),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "model_blob_b64": self._serialize_model_blob(self._model),
            "feature_names": self.feature_names(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CatBoostKeepDropGate":
        cfg_payload = payload.get("config", {}) if isinstance(payload, dict) else {}
        gate = cls(config=CatBoostGateConfig(**cfg_payload))
        gate.numeric_features = [str(col) for col in payload.get("numeric_features", [])]
        cat_payload = payload.get("categorical_features", [])
        if isinstance(cat_payload, dict):
            gate.categorical_features = [str(k) for k in cat_payload.keys()]
        else:
            gate.categorical_features = [str(col) for col in cat_payload]
        blob_b64 = str(payload.get("model_blob_b64", "")).strip()
        if blob_b64:
            gate._model = cls._deserialize_model_blob(blob_b64)
        return gate


@dataclass
class UpliftCatBoostGate:
    config: UpliftCatBoostGateConfig = field(default_factory=UpliftCatBoostGateConfig)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    _model: Any | None = field(default=None, repr=False)
    _harm_model: Any | None = field(default=None, repr=False)

    @staticmethod
    def _require_catboost() -> Any:
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            raise RuntimeError(
                "CatBoost is required for uplift catboost gate inference/training but is not available."
            ) from exc
        return CatBoostRegressor

    @staticmethod
    def _require_catboost_classifier() -> Any:
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise RuntimeError(
                "CatBoost is required for uplift harm-head inference/training but is not available."
            ) from exc
        return CatBoostClassifier

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _matrix_from_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for col in self.numeric_features:
            values = pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")
            data[col] = values.astype("float64")
        for col in self.categorical_features:
            series = frame.get(col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
            data[col] = series.astype("object")
        if not data:
            return pd.DataFrame(index=frame.index)
        return pd.DataFrame(data, index=frame.index)

    def feature_names(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    def fit_dataframe(
        self,
        frame: pd.DataFrame,
        label_col: str,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> None:
        train = frame.copy()
        labels = pd.to_numeric(train.get(label_col), errors="coerce")
        keep = labels.notna()
        train = train.loc[keep].copy()
        if train.empty:
            raise RuntimeError("Cannot fit uplift catboost gate: no labeled rows.")

        self.numeric_features = [str(col) for col in numeric_features if str(col) in train.columns]
        self.categorical_features = [str(col) for col in categorical_features if str(col) in train.columns]
        if not self.numeric_features and not self.categorical_features:
            raise RuntimeError("Cannot fit uplift catboost gate: no feature columns.")

        utility = pd.to_numeric(train.get("utility_units"), errors="coerce")
        if utility.isna().all():
            labels_binary = pd.to_numeric(train.get(label_col), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
            utility_arr = np.where(labels_binary > 0.5, float(PAYOUT_MINUS_110), -1.0).astype("float64")
        else:
            utility_arr = utility.fillna(0.0).to_numpy(dtype="float64")

        x = self._matrix_from_frame(train)
        if len(np.unique(np.round(utility_arr, 6))) < 2:
            raise RuntimeError("Cannot fit uplift catboost gate: utility target has no variation.")

        cfg = self.config
        pos_w = float(max(cfg.class_weight_positive, 1e-9))
        neg_w = float(max(cfg.class_weight_negative, 1e-9))
        sample_w = np.where(utility_arr >= 0.0, pos_w, neg_w).astype("float64")

        seg_power = float(max(0.0, getattr(cfg, "segment_weight_power", 0.0)))
        seg_cap = float(max(1.0, getattr(cfg, "segment_weight_cap", 3.0)))
        if seg_power > 1e-9:
            seg_target = train.get("target", pd.Series("", index=train.index)).fillna("").astype(str).str.upper().str.strip()
            seg_direction = train.get("direction", pd.Series("", index=train.index)).fillna("").astype(str).str.upper().str.strip()
            segment = (seg_target + "|" + seg_direction).where(
                seg_target.str.len().gt(0) & seg_direction.str.len().gt(0),
                "UNKNOWN|UNKNOWN",
            )
            segment_counts = segment.value_counts(dropna=False)
            median_count = float(segment_counts.median()) if not segment_counts.empty else 1.0
            median_count = max(1.0, median_count)
            per_row_count = segment.map(segment_counts).fillna(median_count).astype("float64")
            segment_multiplier = np.power(median_count / np.maximum(per_row_count.to_numpy(dtype="float64"), 1.0), seg_power)
            segment_multiplier = np.clip(segment_multiplier, 1.0 / seg_cap, seg_cap)
            sample_w *= segment_multiplier

        sample_w /= max(float(sample_w.mean()), 1e-9)

        CatBoostRegressor = self._require_catboost()
        model = CatBoostRegressor(
            iterations=int(max(20, cfg.iterations)),
            depth=int(max(2, cfg.depth)),
            learning_rate=float(max(1e-5, cfg.learning_rate)),
            l2_leaf_reg=float(max(0.0, cfg.l2_leaf_reg)),
            random_strength=float(max(0.0, cfg.random_strength)),
            bagging_temperature=float(max(0.0, cfg.bagging_temperature)),
            min_data_in_leaf=int(max(1, cfg.min_data_in_leaf)),
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=int(cfg.random_seed),
            verbose=False,
            allow_writing_files=False,
        )
        cat_features = [
            idx for idx, col in enumerate(x.columns.tolist()) if col in set(self.categorical_features)
        ]
        model.fit(
            x,
            utility_arr.astype("float64"),
            cat_features=cat_features if cat_features else None,
            sample_weight=sample_w,
        )
        self._model = model
        self._harm_model = None

        if bool(getattr(cfg, "harm_head_enabled", True)):
            harm_target = (utility_arr < 0.0).astype("int64")
            if len(np.unique(harm_target)) >= 2:
                pos_weight = float(max(1e-9, getattr(cfg, "harm_positive_weight", 1.5)))
                harm_w = np.where(harm_target > 0, pos_weight, 1.0).astype("float64")
                harm_w *= sample_w
                harm_w /= max(float(harm_w.mean()), 1e-9)
                CatBoostClassifier = self._require_catboost_classifier()
                harm_model = CatBoostClassifier(
                    iterations=int(max(20, cfg.iterations)),
                    depth=int(max(2, cfg.depth)),
                    learning_rate=float(max(1e-5, cfg.learning_rate)),
                    l2_leaf_reg=float(max(0.0, cfg.l2_leaf_reg)),
                    random_strength=float(max(0.0, cfg.random_strength)),
                    bagging_temperature=float(max(0.0, cfg.bagging_temperature)),
                    min_data_in_leaf=int(max(1, cfg.min_data_in_leaf)),
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=int(cfg.random_seed),
                    verbose=False,
                    allow_writing_files=False,
                )
                harm_model.fit(
                    x,
                    harm_target,
                    cat_features=cat_features if cat_features else None,
                    sample_weight=harm_w,
                )
                self._harm_model = harm_model

    def predict_proba_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("UpliftCatBoostGate is not fitted.")
        x = self._matrix_from_frame(frame)
        expected_cols = len(self.numeric_features) + len(self.categorical_features)
        if x.shape[1] != expected_cols:
            raise RuntimeError(
                f"Feature shape mismatch: matrix has {x.shape[1]} columns, model expects {expected_cols}."
            )
        keep_utility = np.asarray(self._model.predict(x), dtype="float64")
        temp = float(max(1e-6, self.config.keep_prob_temperature))
        keep_prob = self._sigmoid(keep_utility / temp)
        return np.clip(keep_prob.astype("float64"), 0.0, 1.0)

    def predict_harm_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        keep_prob = self.predict_proba_dataframe(frame)
        if self._harm_model is None:
            return np.clip((1.0 - keep_prob).astype("float64"), 0.0, 1.0)
        x = self._matrix_from_frame(frame)
        probs = self._harm_model.predict_proba(x)
        probs_arr = np.asarray(probs, dtype="float64")
        if probs_arr.ndim == 1:
            harm_prob = probs_arr
        elif probs_arr.shape[1] == 1:
            harm_prob = probs_arr[:, 0]
        else:
            harm_prob = probs_arr[:, 1]
        temp = float(max(1e-6, getattr(self.config, "harm_temperature", 1.0)))
        logits = np.log(np.clip(harm_prob, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - harm_prob, 1e-8, 1.0))
        adjusted = self._sigmoid(logits / temp)
        return np.clip(adjusted.astype("float64"), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Cannot serialize uplift catboost gate: model is not fitted.")
        return {
            "model_type": "catboost_uplift_gate_v1",
            "config": asdict(self.config),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "model_blob_b64": CatBoostKeepDropGate._serialize_model_blob(self._model),
            "harm_model_blob_b64": CatBoostKeepDropGate._serialize_model_blob(self._harm_model) if self._harm_model is not None else "",
            "feature_names": self.feature_names(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UpliftCatBoostGate":
        cfg_payload = payload.get("config", {}) if isinstance(payload, dict) else {}
        gate = cls(config=UpliftCatBoostGateConfig(**cfg_payload))
        gate.numeric_features = [str(col) for col in payload.get("numeric_features", [])]
        cat_payload = payload.get("categorical_features", [])
        if isinstance(cat_payload, dict):
            gate.categorical_features = [str(k) for k in cat_payload.keys()]
        else:
            gate.categorical_features = [str(col) for col in cat_payload]
        blob_b64 = str(payload.get("model_blob_b64", "")).strip()
        if blob_b64:
            raw = base64.b64decode(blob_b64.encode("ascii"))
            with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as handle:
                model_path = Path(handle.name)
            try:
                model_path.write_bytes(raw)
                model_cls = cls._require_catboost()
                model = model_cls()
                model.load_model(str(model_path), format="cbm")
                gate._model = model
            finally:
                try:
                    model_path.unlink()
                except OSError:
                    pass
        harm_blob_b64 = str(payload.get("harm_model_blob_b64", "")).strip()
        if harm_blob_b64:
            gate._harm_model = CatBoostKeepDropGate._deserialize_model_blob(harm_blob_b64)
        return gate


def load_gate_model_from_payload(payload: dict[str, Any] | None) -> Any:
    model_payload = payload if isinstance(payload, dict) else {}
    model_type = str(model_payload.get("model_type", "")).strip().lower()
    model_family = str(model_payload.get("model_family", "")).strip().lower()
    if model_type == "catboost_uplift_gate_v1" or model_family == "uplift_catboost":
        return UpliftCatBoostGate.from_dict(model_payload)
    if model_type == "catboost_gate_v1" or model_family == "catboost":
        return CatBoostKeepDropGate.from_dict(model_payload)
    return RegularizedLogisticGate.from_dict(model_payload)


def predict_keep_harm_scores(gate_model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    keep_prob = np.asarray(gate_model.predict_proba_dataframe(frame), dtype="float64")
    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    harm_prob: np.ndarray | None = None
    if hasattr(gate_model, "predict_harm_dataframe"):
        try:
            raw = getattr(gate_model, "predict_harm_dataframe")(frame)
            harm_prob = np.asarray(raw, dtype="float64")
        except Exception:
            harm_prob = None
    if harm_prob is None or harm_prob.shape[0] != keep_prob.shape[0]:
        harm_prob = 1.0 - keep_prob
    harm_prob = np.clip(np.asarray(harm_prob, dtype="float64"), 0.0, 1.0)
    return keep_prob, harm_prob


def iter_walk_forward_windows(
    frame: pd.DataFrame,
    *,
    date_col: str,
    config: WalkForwardConfig,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    dates = coerce_event_datetime(frame.get(date_col))
    if dates.isna().all():
        return []
    min_date = dates.min()
    max_date = dates.max()
    if pd.isna(min_date) or pd.isna(max_date):
        return []

    folds: list[dict[str, Any]] = []
    test_start = min_date + pd.Timedelta(days=int(max(config.train_window_days, 1)))
    while test_start <= max_date:
        train_start = test_start - pd.Timedelta(days=int(max(config.train_window_days, 1)))
        test_end = test_start + pd.Timedelta(days=int(max(config.test_window_days, 1)))
        train_mask = (dates >= train_start) & (dates < test_start)
        test_mask = (dates >= test_start) & (dates < test_end)
        train_rows = int(train_mask.sum())
        test_rows = int(test_mask.sum())
        if train_rows >= int(config.min_train_rows) and test_rows >= int(config.min_test_rows):
            folds.append(
                {
                    "train_mask": train_mask,
                    "test_mask": test_mask,
                    "train_start": str(train_start.date()),
                    "train_end": str((test_start - pd.Timedelta(days=1)).date()),
                    "test_start": str(test_start.date()),
                    "test_end": str((test_end - pd.Timedelta(days=1)).date()),
                    "train_rows": train_rows,
                    "test_rows": test_rows,
                }
            )
        test_start = test_start + pd.Timedelta(days=int(max(config.step_days, 1)))
    return folds


def _violates_global_player_cap(
    *,
    player: str,
    next_total_vetoes: int,
    next_player_vetoes: int,
    max_rows_per_player: int,
    max_share_per_player: float,
    min_rows_before_share_cap: int,
) -> bool:
    if not player:
        return False
    if int(max_rows_per_player) > 0 and int(next_player_vetoes) > int(max_rows_per_player):
        return True
    share_cap = float(np.clip(max_share_per_player, 0.0, 1.0))
    if share_cap >= 0.999999:
        return False
    if int(next_total_vetoes) < int(max(1, min_rows_before_share_cap)):
        return False
    return float(next_player_vetoes) / float(max(1, next_total_vetoes)) > share_cap + 1e-12


def _row_veto_score(
    row: pd.Series,
    *,
    require_keep_prob_filter: bool,
    keep_prob_weight: float,
    gap_weight: float,
    harm_weight: float,
    rank_weight: float,
) -> float:
    keep_prob = float(pd.to_numeric(pd.Series([row.get("gate_keep_prob", 0.5)]), errors="coerce").fillna(0.5).iloc[0])
    keep_prob = float(np.clip(keep_prob, 0.0, 1.0))
    gap = float(pd.to_numeric(pd.Series([row.get("gate_keep_prob_gap", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    gap = float(max(0.0, gap if bool(require_keep_prob_filter) else (1.0 - keep_prob)))
    harm = float(pd.to_numeric(pd.Series([row.get("gate_harm_score", np.nan)]), errors="coerce").fillna(1.0 - keep_prob).iloc[0])
    harm = float(np.clip(harm, 0.0, 1.0))
    rank_raw = pd.to_numeric(pd.Series([row.get("selected_rank", np.nan)]), errors="coerce").iloc[0]
    if rank_raw == rank_raw and float(rank_raw) > 0.0:
        rank_signal = 1.0 / (1.0 + float(rank_raw))
    else:
        rank_signal = 0.0
    keep_signal = 1.0 - keep_prob
    return (
        float(max(0.0, keep_prob_weight)) * keep_signal
        + float(max(0.0, gap_weight)) * gap
        + float(max(0.0, harm_weight)) * harm
        + float(max(0.0, rank_weight)) * rank_signal
    )


def _optimize_day_veto_indices(
    candidates: pd.DataFrame,
    *,
    max_veto: int,
    policy: GatePolicyConfig,
    global_player_counts: dict[str, int],
    global_total_vetoes: int,
    segment_budget_overrides: dict[str, int],
    enforce_global_player_cap: bool,
    preselected_idx: set[Any] | None = None,
    day_player_counts_seed: dict[str, int] | None = None,
    day_segment_counts_seed: dict[str, int] | None = None,
    day_target_counts_seed: dict[str, int] | None = None,
) -> list[Any]:
    if candidates.empty or int(max_veto) <= 0:
        return []

    preselected = set(preselected_idx or ())
    day_player_seed = dict(day_player_counts_seed or {})
    day_segment_seed = dict(day_segment_counts_seed or {})
    day_target_seed = dict(day_target_counts_seed or {})

    work = candidates.loc[~candidates.index.isin(preselected)].copy()
    if work.empty:
        return []
    work["_gate_veto_score"] = [
        _row_veto_score(
            row,
            require_keep_prob_filter=bool(policy.require_keep_prob_filter),
            keep_prob_weight=float(policy.optimizer_keep_prob_weight),
            gap_weight=float(policy.optimizer_gap_weight),
            harm_weight=float(policy.optimizer_harm_weight),
            rank_weight=float(policy.optimizer_rank_weight),
        )
        for _, row in work.iterrows()
    ]
    work = work.sort_values(["_gate_veto_score", "gate_keep_prob"], ascending=[False, True])
    max_candidates = int(max(8, policy.optimizer_max_candidates_per_day))
    if len(work) > max_candidates:
        work = work.head(max_candidates).copy()
    records = list(work.iterrows())
    if not records:
        return []

    beam_width = int(max(8, policy.optimizer_beam_width))
    # state: (score, selected_idx_tuple, day_player_counts, day_segment_counts, day_target_counts)
    states: list[tuple[float, tuple[Any, ...], dict[str, int], dict[str, int], dict[str, int]]] = [
        (0.0, tuple(), dict(day_player_seed), dict(day_segment_seed), dict(day_target_seed))
    ]

    for idx, row in records:
        next_states: list[tuple[float, tuple[Any, ...], dict[str, int], dict[str, int], dict[str, int]]] = []
        for score, selected, day_player_counts, day_segment_counts, day_target_counts in states:
            # Skip option
            next_states.append((score, selected, day_player_counts, day_segment_counts, day_target_counts))

            if len(selected) >= int(max_veto):
                continue
            player = str(row.get("_gate_player", ""))
            target = str(row.get("_gate_target", ""))
            segment = str(row.get("_gate_segment", ""))
            if player and day_player_counts.get(player, 0) >= int(max(0, policy.max_removed_per_player_per_day)):
                continue
            seg_cap = int(max(0, segment_budget_overrides.get(segment, int(policy.max_removed_per_segment_per_day))))
            if segment and day_segment_counts.get(segment, 0) >= seg_cap:
                continue
            if target and day_target_counts.get(target, 0) >= int(max(0, policy.max_removed_per_target_per_day)):
                continue
            if bool(enforce_global_player_cap) and player:
                next_total = int(global_total_vetoes + len(selected) + 1)
                base_player_total = int(global_player_counts.get(player, 0))
                next_player = int(base_player_total + day_player_counts.get(player, 0) + 1)
                if _violates_global_player_cap(
                    player=player,
                    next_total_vetoes=next_total,
                    next_player_vetoes=next_player,
                    max_rows_per_player=int(policy.max_global_removed_per_player),
                    max_share_per_player=float(policy.max_global_removed_share_per_player),
                    min_rows_before_share_cap=int(policy.global_player_cap_min_veto_rows),
                ):
                    continue

            new_player_counts = dict(day_player_counts)
            new_segment_counts = dict(day_segment_counts)
            new_target_counts = dict(day_target_counts)
            if player:
                new_player_counts[player] = new_player_counts.get(player, 0) + 1
            if segment:
                new_segment_counts[segment] = new_segment_counts.get(segment, 0) + 1
            if target:
                new_target_counts[target] = new_target_counts.get(target, 0) + 1
            add_score = float(pd.to_numeric(pd.Series([row.get("_gate_veto_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            next_states.append(
                (
                    float(score + add_score),
                    tuple([*selected, idx]),
                    new_player_counts,
                    new_segment_counts,
                    new_target_counts,
                )
            )

        # Deduplicate by count-signature and keep best score state.
        dedup: dict[tuple[Any, ...], tuple[float, tuple[Any, ...], dict[str, int], dict[str, int], dict[str, int]]] = {}
        for state in next_states:
            score, selected, day_player_counts, day_segment_counts, day_target_counts = state
            key = (
                len(selected),
                tuple(sorted(day_player_counts.items())),
                tuple(sorted(day_segment_counts.items())),
                tuple(sorted(day_target_counts.items())),
            )
            prev = dedup.get(key)
            if prev is None or float(score) > float(prev[0]) + 1e-12:
                dedup[key] = state
        states = sorted(
            dedup.values(),
            key=lambda item: (float(item[0]), len(item[1])),
            reverse=True,
        )[:beam_width]

    if not states:
        return []
    best = max(states, key=lambda item: (float(item[0]), len(item[1])))
    return list(best[1])


def apply_shadow_gate_policy(
    frame: pd.DataFrame,
    *,
    keep_prob_col: str,
    date_col: str,
    player_col: str,
    target_col: str,
    direction_col: str,
    threshold: float,
    policy: GatePolicyConfig,
) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        out["gate_keep_prob"] = np.nan
        out["gate_threshold"] = float(threshold)
        out["gate_veto"] = False
        out["gate_veto_reason"] = ""
        return out

    out = frame.copy()
    keep_prob_raw = out.get(keep_prob_col, pd.Series(np.nan, index=out.index))
    if not isinstance(keep_prob_raw, pd.Series):
        keep_prob_raw = pd.Series(keep_prob_raw, index=out.index)
    out["gate_keep_prob"] = pd.to_numeric(keep_prob_raw, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)

    harm_raw = out.get("gate_harm_score", pd.Series(np.nan, index=out.index))
    if not isinstance(harm_raw, pd.Series):
        harm_raw = pd.Series(harm_raw, index=out.index)
    out["gate_harm_score"] = (
        pd.to_numeric(harm_raw, errors="coerce")
        .fillna(1.0 - out["gate_keep_prob"])
        .clip(lower=0.0, upper=1.0)
    )
    out["gate_threshold"] = float(threshold)
    out["_gate_threshold_effective"] = float(threshold)
    out["gate_veto"] = False
    out["gate_veto_reason"] = ""
    out["_gate_player"] = out.get(player_col, pd.Series("", index=out.index)).map(normalize_player_name)
    out["_gate_target"] = out.get(target_col, pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    out["_gate_direction"] = out.get(direction_col, pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    out["_gate_segment"] = (out["_gate_target"] + "|" + out["_gate_direction"]).astype("object")
    out["_gate_date"] = coerce_event_datetime(out.get(date_col)).dt.strftime("%Y-%m-%d").fillna("")
    segment_threshold_overrides = {
        str(seg).upper().strip(): float(val)
        for seg, val in dict(policy.segment_threshold_overrides or {}).items()
        if str(seg).strip()
    }
    segment_budget_overrides = {
        str(seg).upper().strip(): int(max(0, int(val)))
        for seg, val in dict(policy.segment_budget_overrides or {}).items()
        if str(seg).strip()
    }
    if segment_threshold_overrides:
        for seg, seg_threshold in segment_threshold_overrides.items():
            if not np.isfinite(seg_threshold):
                continue
            seg_mask = out["_gate_segment"].eq(seg)
            if not bool(seg_mask.any()):
                continue
            out.loc[seg_mask, "_gate_threshold_effective"] = np.maximum(
                pd.to_numeric(out.loc[seg_mask, "_gate_threshold_effective"], errors="coerce").fillna(float(threshold)),
                float(seg_threshold),
            )
    out["gate_threshold_effective"] = pd.to_numeric(out["_gate_threshold_effective"], errors="coerce").fillna(float(threshold))
    out["gate_keep_prob_gap"] = out["gate_threshold_effective"] - out["gate_keep_prob"]
    allowed_segments = {
        str(seg).upper().strip()
        for seg in (policy.allowed_segments or ())
        if str(seg).strip()
    }
    allowed_meta_cohorts = {
        str(tag).upper().strip()
        for tag in (policy.allowed_meta_cohorts or ())
        if str(tag).strip()
    }
    meta_cohorts = build_meta_cohort_columns(out, target_col=target_col, direction_col=direction_col)
    global_player_counts: dict[str, int] = {}
    global_total_vetoes = 0
    global_cap_rows = int(max(0, policy.max_global_removed_per_player))
    global_cap_share = float(np.clip(policy.max_global_removed_share_per_player, 0.0, 1.0))
    global_cap_min_rows = int(max(1, policy.global_player_cap_min_veto_rows))
    global_cap_enabled = bool(global_cap_rows > 0 or global_cap_share < 0.999999)

    for gate_date, part in out.groupby("_gate_date", sort=True):
        if not gate_date:
            continue
        n_rows = int(len(part))
        if n_rows <= 0:
            continue
        max_veto = int(math.floor(float(np.clip(policy.max_fire_rate, 0.0, 1.0)) * n_rows))
        min_keep = int(math.ceil(float(np.clip(policy.min_coverage_rate, 0.0, 1.0)) * n_rows))
        max_veto = max(0, min(max_veto, n_rows - min_keep))
        day_cap = int(max(0, policy.max_removed_per_day))
        max_veto = min(max_veto, day_cap)
        if max_veto <= 0:
            continue

        if bool(policy.require_keep_prob_filter):
            candidates = part.loc[part["gate_keep_prob"] < part["gate_threshold_effective"]].copy()
        else:
            candidates = part.copy()
            # Safety: never allow unrestricted global rewrites when no cohort/segment scope exists.
            if not allowed_segments and not allowed_meta_cohorts:
                candidates = part.loc[part["gate_keep_prob"] < part["gate_threshold_effective"]].copy()
        if allowed_segments and not candidates.empty:
            candidates = candidates.loc[candidates["_gate_segment"].isin(allowed_segments)].copy()
        if allowed_meta_cohorts and not candidates.empty:
            cohort_part = meta_cohorts.loc[candidates.index] if not meta_cohorts.empty else pd.DataFrame(index=candidates.index)
            cohort_mask = pd.Series(False, index=candidates.index, dtype=bool)
            for cohort_col in cohort_part.columns:
                cohort_mask = cohort_mask | cohort_part[cohort_col].isin(allowed_meta_cohorts)
            candidates = candidates.loc[cohort_mask].copy()
        min_gap = float(max(0.0, policy.min_veto_gap)) if bool(policy.require_keep_prob_filter) else 0.0
        if min_gap > 0.0 and not candidates.empty:
            candidates = candidates.loc[candidates["gate_keep_prob_gap"] >= min_gap].copy()
        tail_slots = int(max(0, policy.tail_slots_only))
        if tail_slots > 0 and "selected_rank" in candidates.columns and not candidates.empty:
            rank = pd.to_numeric(candidates.get("selected_rank"), errors="coerce")
            tail_start = max(1.0, float(n_rows - tail_slots + 1))
            candidates = candidates.loc[rank >= tail_start].copy()
        if candidates.empty:
            continue

        player_counts: dict[str, int] = {}
        segment_counts: dict[str, int] = {}
        target_counts: dict[str, int] = {}
        selected_idx: list[Any] = []
        if bool(policy.optimizer_enabled):
            selected_idx = _optimize_day_veto_indices(
                candidates,
                max_veto=int(max_veto),
                policy=policy,
                global_player_counts=global_player_counts,
                global_total_vetoes=int(global_total_vetoes),
                segment_budget_overrides=segment_budget_overrides,
                enforce_global_player_cap=bool(global_cap_enabled),
            )
            for idx in selected_idx:
                row = candidates.loc[idx]
                player = str(row.get("_gate_player", ""))
                segment = str(row.get("_gate_segment", ""))
                target = str(row.get("_gate_target", ""))
                if player:
                    player_counts[player] = player_counts.get(player, 0) + 1
                if segment:
                    segment_counts[segment] = segment_counts.get(segment, 0) + 1
                if target:
                    target_counts[target] = target_counts.get(target, 0) + 1
            if (
                bool(global_cap_enabled)
                and bool(policy.relax_player_global_cap_for_budget)
                and len(selected_idx) < int(max_veto)
            ):
                relaxed_extra = _optimize_day_veto_indices(
                    candidates,
                    max_veto=int(max_veto - len(selected_idx)),
                    policy=policy,
                    global_player_counts=global_player_counts,
                    global_total_vetoes=int(global_total_vetoes + len(selected_idx)),
                    segment_budget_overrides=segment_budget_overrides,
                    enforce_global_player_cap=False,
                    preselected_idx=set(selected_idx),
                    day_player_counts_seed=player_counts,
                    day_segment_counts_seed=segment_counts,
                    day_target_counts_seed=target_counts,
                )
                selected_idx.extend(relaxed_extra)
        else:
            # Backward-compatible greedy fallback.
            candidates = candidates.sort_values(["gate_keep_prob", "gate_harm_score"], ascending=[True, False])
            for idx, row in candidates.iterrows():
                if len(selected_idx) >= int(max_veto):
                    break
                player = str(row.get("_gate_player", ""))
                segment = str(row.get("_gate_segment", ""))
                target = str(row.get("_gate_target", ""))
                if player and player_counts.get(player, 0) >= int(max(0, policy.max_removed_per_player_per_day)):
                    continue
                seg_cap = int(max(0, segment_budget_overrides.get(segment, int(policy.max_removed_per_segment_per_day))))
                if segment and segment_counts.get(segment, 0) >= seg_cap:
                    continue
                if target and target_counts.get(target, 0) >= int(max(0, policy.max_removed_per_target_per_day)):
                    continue
                selected_idx.append(idx)
                if player:
                    player_counts[player] = player_counts.get(player, 0) + 1
                if segment:
                    segment_counts[segment] = segment_counts.get(segment, 0) + 1
                if target:
                    target_counts[target] = target_counts.get(target, 0) + 1

        for idx in selected_idx:
            row = candidates.loc[idx]
            player = str(row.get("_gate_player", ""))
            if bool(policy.require_keep_prob_filter):
                reason = "optimizer_low_keep_prob"
            else:
                reason = "optimizer_meta_cohort_rule"
            if bool(global_cap_enabled) and player and _violates_global_player_cap(
                player=player,
                next_total_vetoes=int(global_total_vetoes + 1),
                next_player_vetoes=int(global_player_counts.get(player, 0) + 1),
                max_rows_per_player=global_cap_rows,
                max_share_per_player=global_cap_share,
                min_rows_before_share_cap=global_cap_min_rows,
            ):
                reason = reason + "_player_cap_relaxed"
            out.loc[idx, "gate_veto"] = True
            out.loc[idx, "gate_veto_reason"] = reason
            global_total_vetoes += 1
            if player:
                global_player_counts[player] = global_player_counts.get(player, 0) + 1

    out = out.drop(
        columns=[
            "_gate_player",
            "_gate_target",
            "_gate_direction",
            "_gate_segment",
            "_gate_date",
            "_gate_threshold_effective",
            "gate_keep_prob_gap",
        ],
        errors="ignore",
    )
    return out


def summarize_paired_outcomes(
    frame: pd.DataFrame,
    *,
    veto_col: str = "gate_veto",
    result_col: str = "result",
    payout_per_win: float = PAYOUT_MINUS_110,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "baseline": {},
            "gated": {},
            "delta": {},
        }

    out = frame.copy()
    result = out.get(result_col, pd.Series("", index=out.index)).astype(str).str.lower().str.strip()
    out["is_win"] = (result == "win").astype(int)
    out["is_loss"] = (result == "loss").astype(int)
    resolved = out.loc[(out["is_win"] + out["is_loss"]) > 0].copy()

    def _metrics(part: pd.DataFrame) -> dict[str, Any]:
        wins = int(part["is_win"].sum()) if not part.empty else 0
        losses = int(part["is_loss"].sum()) if not part.empty else 0
        resolved_rows = int(wins + losses)
        hit_rate = float(wins / resolved_rows) if resolved_rows > 0 else float("nan")
        profit = float(wins * payout_per_win - losses)
        ev_per_resolved = float(profit / resolved_rows) if resolved_rows > 0 else float("nan")
        return {
            "rows": int(len(part)),
            "resolved": resolved_rows,
            "wins": wins,
            "losses": losses,
            "hit_rate": hit_rate,
            "profit_units": profit,
            "ev_per_resolved": ev_per_resolved,
        }

    baseline = _metrics(resolved)
    kept = resolved.loc[~pd.to_numeric(resolved.get(veto_col), errors="coerce").fillna(0).astype(bool)].copy()
    gated = _metrics(kept)
    delta = {
        "rows": int(gated["rows"] - baseline["rows"]),
        "resolved": int(gated["resolved"] - baseline["resolved"]),
        "wins": int(gated["wins"] - baseline["wins"]),
        "losses": int(gated["losses"] - baseline["losses"]),
        "hit_rate_pp": float((gated["hit_rate"] - baseline["hit_rate"]) * 100.0)
        if baseline["hit_rate"] == baseline["hit_rate"] and gated["hit_rate"] == gated["hit_rate"]
        else float("nan"),
        "profit_units": float(gated["profit_units"] - baseline["profit_units"]),
        "ev_per_resolved": float(gated["ev_per_resolved"] - baseline["ev_per_resolved"])
        if baseline["ev_per_resolved"] == baseline["ev_per_resolved"] and gated["ev_per_resolved"] == gated["ev_per_resolved"]
        else float("nan"),
    }
    return {
        "rows": int(len(out)),
        "baseline": baseline,
        "gated": gated,
        "delta": delta,
    }


def rolling_window_paired_deltas(
    frame: pd.DataFrame,
    *,
    date_col: str,
    veto_col: str = "gate_veto",
    result_col: str = "result",
    window_days: int = 21,
    step_days: int = 7,
    payout_per_win: float = PAYOUT_MINUS_110,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    dates = coerce_event_datetime(frame.get(date_col))
    if dates.isna().all():
        return pd.DataFrame()
    start = dates.min()
    end = dates.max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        window_end = cursor + pd.Timedelta(days=int(max(window_days, 1)) - 1)
        mask = (dates >= cursor) & (dates <= window_end)
        part = frame.loc[mask].copy()
        if not part.empty:
            summary = summarize_paired_outcomes(part, veto_col=veto_col, result_col=result_col, payout_per_win=payout_per_win)
            delta = summary.get("delta", {})
            rows.append(
                {
                    "start": str(cursor.date()),
                    "end": str(window_end.date()),
                    "rows": int(len(part)),
                    "delta_profit_units": float(delta.get("profit_units", float("nan"))),
                    "delta_hit_rate_pp": float(delta.get("hit_rate_pp", float("nan"))),
                    "delta_ev_per_resolved": float(delta.get("ev_per_resolved", float("nan"))),
                    "delta_resolved": int(delta.get("resolved", 0)),
                }
            )
        cursor = cursor + pd.Timedelta(days=int(max(step_days, 1)))
    return pd.DataFrame.from_records(rows)


def concentration_stats(
    frame: pd.DataFrame,
    *,
    veto_col: str,
    player_col: str,
    target_col: str,
    direction_col: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "veto_rows": 0,
            "top_player_share": 0.0,
            "top_segment_share": 0.0,
            "top_target_share": 0.0,
            "player_hhi": 0.0,
            "segment_hhi": 0.0,
            "target_hhi": 0.0,
            "player_diversity": 0.0,
            "unique_players": 0,
            "unique_segments": 0,
            "unique_targets": 0,
        }
    vetoed = frame.loc[pd.to_numeric(frame.get(veto_col), errors="coerce").fillna(0).astype(bool)].copy()
    if vetoed.empty:
        return {
            "veto_rows": 0,
            "top_player_share": 0.0,
            "top_segment_share": 0.0,
            "top_target_share": 0.0,
            "player_hhi": 0.0,
            "segment_hhi": 0.0,
            "target_hhi": 0.0,
            "player_diversity": 0.0,
            "unique_players": 0,
            "unique_segments": 0,
            "unique_targets": 0,
        }
    players = vetoed.get(player_col, pd.Series("", index=vetoed.index)).map(normalize_player_name)
    targets = vetoed.get(target_col, pd.Series("", index=vetoed.index)).fillna("").astype(str).str.upper().str.strip()
    directions = vetoed.get(direction_col, pd.Series("", index=vetoed.index)).fillna("").astype(str).str.upper().str.strip()
    segments = (targets + "|" + directions).astype("object")
    n_rows = int(len(vetoed))
    player_freq = players.value_counts(normalize=True) if n_rows > 0 and players.nunique() > 0 else pd.Series(dtype="float64")
    segment_freq = segments.value_counts(normalize=True) if n_rows > 0 and segments.nunique() > 0 else pd.Series(dtype="float64")
    target_freq = targets.value_counts(normalize=True) if n_rows > 0 and targets.nunique() > 0 else pd.Series(dtype="float64")
    top_player = float(player_freq.iloc[0]) if not player_freq.empty else 0.0
    top_segment = float(segment_freq.iloc[0]) if not segment_freq.empty else 0.0
    top_target = float(target_freq.iloc[0]) if not target_freq.empty else 0.0
    player_hhi = float((player_freq.astype("float64") ** 2).sum()) if not player_freq.empty else 0.0
    segment_hhi = float((segment_freq.astype("float64") ** 2).sum()) if not segment_freq.empty else 0.0
    target_hhi = float((target_freq.astype("float64") ** 2).sum()) if not target_freq.empty else 0.0
    unique_players = int(players.nunique())
    return {
        "veto_rows": n_rows,
        "top_player_share": top_player,
        "top_segment_share": top_segment,
        "top_target_share": top_target,
        "player_hhi": player_hhi,
        "segment_hhi": segment_hhi,
        "target_hhi": target_hhi,
        "player_diversity": float(unique_players) / float(max(1, n_rows)),
        "unique_players": unique_players,
        "unique_segments": int(segments.nunique()),
        "unique_targets": int(targets.nunique()),
    }


def build_promotion_recommendation(
    *,
    broad_delta: dict[str, Any],
    recent_delta: dict[str, Any],
    affected_share: float,
    concentration: dict[str, Any],
    rolling: pd.DataFrame,
    observed_fire_rate: float,
    criteria: PromotionCriteria,
) -> dict[str, Any]:
    failures: list[str] = []
    broad_profit = float(broad_delta.get("profit_units", 0.0))
    broad_hit = float(broad_delta.get("hit_rate_pp", float("nan")))
    recent_profit = float(recent_delta.get("profit_units", 0.0))
    recent_hit = float(recent_delta.get("hit_rate_pp", float("nan")))

    if broad_profit < float(criteria.min_broad_profit_delta_units):
        failures.append("broad_profit_delta_below_floor")
    if recent_profit < float(criteria.min_recent_profit_delta_units):
        failures.append("recent_profit_delta_below_floor")
    if recent_hit == recent_hit and recent_hit < float(criteria.min_recent_hit_rate_delta_pp):
        failures.append("recent_hit_rate_delta_below_floor")
    if broad_hit == broad_hit and broad_hit < float(criteria.min_broad_hit_rate_delta_pp):
        failures.append("broad_hit_rate_delta_below_floor")
    coverage_retention = float(max(0.0, 1.0 - float(max(0.0, affected_share))))
    if coverage_retention < float(criteria.min_coverage_retention):
        failures.append("coverage_retention_below_floor")
    if float(affected_share) < float(criteria.min_affected_share):
        failures.append("affected_share_below_floor")

    top_player = float(concentration.get("top_player_share", 0.0))
    top_segment = float(concentration.get("top_segment_share", 0.0))
    top_target = float(concentration.get("top_target_share", 0.0))
    veto_rows = int(max(0, int(concentration.get("veto_rows", 0))))
    concentration_enforced = bool(veto_rows >= int(max(1, getattr(criteria, "concentration_min_veto_rows", 12))))
    effective_target_cap = float(criteria.max_top_removed_target_share)
    conditional_target_cap_active = False
    if bool(getattr(criteria, "conditional_target_cap_enabled", False)):
        player_ok = top_player <= float(getattr(criteria, "conditional_target_player_share_max", 0.15))
        recent_ok = True
        if bool(getattr(criteria, "conditional_target_require_recent_non_negative", True)):
            recent_ok = bool(
                recent_profit >= float(getattr(criteria, "conditional_target_recent_profit_floor", 0.0))
                and (
                    recent_hit != recent_hit
                    or recent_hit >= float(getattr(criteria, "conditional_target_recent_hit_floor_pp", 0.0))
                )
            )
        if player_ok and recent_ok:
            conditional_target_cap_active = True
            effective_target_cap = float(
                max(
                    float(criteria.max_top_removed_target_share),
                    float(getattr(criteria, "conditional_target_relaxed_cap", 0.90)),
                )
            )
    effective_player_cap = float(criteria.max_top_removed_player_share)
    effective_segment_cap = float(criteria.max_top_removed_segment_share)
    if not concentration_enforced:
        effective_player_cap = float(max(effective_player_cap, 1.0))
        effective_segment_cap = float(max(effective_segment_cap, 1.0))
        effective_target_cap = float(max(effective_target_cap, 1.0))
    if top_player > float(effective_player_cap):
        failures.append("removed_player_concentration_above_cap")
    if top_segment > float(effective_segment_cap):
        failures.append("removed_segment_concentration_above_cap")
    if top_target > float(effective_target_cap):
        failures.append("removed_target_concentration_above_cap")
    if float(observed_fire_rate) > float(criteria.max_observed_fire_rate):
        failures.append("observed_fire_rate_above_cap")

    rolling_pass_rate = float("nan")
    if rolling is not None and not rolling.empty:
        win_pass = pd.to_numeric(rolling.get("delta_hit_rate_pp"), errors="coerce").fillna(-np.inf) >= float(criteria.rolling_hit_rate_delta_floor_pp)
        profit_pass = pd.to_numeric(rolling.get("delta_profit_units"), errors="coerce").fillna(-np.inf) >= float(criteria.rolling_profit_delta_floor)
        pass_mask = win_pass & profit_pass
        rolling_pass_rate = float(pass_mask.mean()) if len(pass_mask) else float("nan")
        if rolling_pass_rate == rolling_pass_rate and rolling_pass_rate < float(criteria.min_rolling_pass_rate):
            failures.append("rolling_window_pass_rate_below_floor")
    else:
        failures.append("insufficient_rolling_windows")

    return {
        "pass": bool(len(failures) == 0),
        "failures": failures,
        "checks": {
            "broad_profit_delta_units": broad_profit,
            "broad_hit_rate_delta_pp": broad_hit,
            "recent_profit_delta_units": recent_profit,
            "recent_hit_rate_delta_pp": recent_hit,
            "coverage_retention": coverage_retention,
            "affected_share": float(affected_share),
            "observed_fire_rate": float(observed_fire_rate),
            "top_removed_player_share": top_player,
            "top_removed_segment_share": top_segment,
            "top_removed_target_share": top_target,
            "veto_rows": int(veto_rows),
            "concentration_enforced": bool(concentration_enforced),
            "effective_player_cap": float(effective_player_cap),
            "effective_segment_cap": float(effective_segment_cap),
            "effective_target_cap": float(effective_target_cap),
            "conditional_target_cap_active": bool(conditional_target_cap_active),
            "rolling_pass_rate": rolling_pass_rate,
        },
        "criteria": asdict(criteria),
    }


def _month_token(value: Any) -> str:
    ts = coerce_event_datetime(pd.Series([value])).iloc[0]
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m")


def _resolve_month_payload(payload: dict[str, Any], month_hint: str) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return "", None
    months = payload.get("months")
    if not isinstance(months, dict) or not months:
        return "", payload
    if month_hint and month_hint in months and isinstance(months[month_hint], dict):
        return month_hint, months[month_hint]
    prior = sorted(str(key) for key in months.keys() if str(key) <= str(month_hint))
    if prior:
        chosen = prior[-1]
        chosen_payload = months.get(chosen)
        return chosen, chosen_payload if isinstance(chosen_payload, dict) else None
    return "", None


def apply_accepted_pick_gate(
    frame: pd.DataFrame,
    payload: dict[str, Any] | None,
    *,
    run_date_hint: str | None = None,
    date_col: str = "market_date",
    player_col: str = "market_player_raw",
    target_col: str = "target",
    direction_col: str = "direction",
    live: bool = False,
    min_rows: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    live_requested = bool(live)
    if out.empty:
        return out, {
            "enabled": bool(payload),
            "enforced": False,
            "live": False,
            "live_requested": bool(live_requested),
            "reason": "empty_frame",
            "threshold": float("nan"),
            "month": "",
        }

    out["accepted_pick_gate_keep_prob"] = np.nan
    out["accepted_pick_gate_harm_score"] = np.nan
    out["accepted_pick_gate_threshold"] = np.nan
    out["accepted_pick_gate_veto"] = False
    out["accepted_pick_gate_veto_reason"] = ""
    out["accepted_pick_gate_enabled"] = bool(payload is not None)
    out["accepted_pick_gate_enforced"] = False
    out["accepted_pick_gate_live"] = False
    out["accepted_pick_gate_live_requested"] = bool(live_requested)
    out["accepted_pick_gate_month"] = ""
    out["accepted_pick_gate_drop_applied"] = False
    out["accepted_pick_gate_drop_count"] = 0
    out["accepted_pick_gate_policy"] = ""

    if payload is None:
        return out, {
            "enabled": False,
            "enforced": False,
            "live": False,
            "live_requested": bool(live_requested),
            "reason": "no_payload",
            "threshold": float("nan"),
            "month": "",
        }
    if int(len(out)) < int(max(0, min_rows)):
        return out, {
            "enabled": True,
            "enforced": False,
            "live": False,
            "live_requested": bool(live_requested),
            "reason": "insufficient_rows",
            "threshold": float("nan"),
            "month": "",
            "rows": int(len(out)),
            "min_rows": int(max(0, min_rows)),
        }

    month_hint = _month_token(run_date_hint)
    if not month_hint and date_col in out.columns:
        month_hint = _month_token(coerce_event_datetime(out[date_col]).max())
    if not month_hint:
        month_hint = _month_token(pd.Timestamp.utcnow())
    resolved_month, month_payload = _resolve_month_payload(payload, month_hint)
    if month_payload is None:
        return out, {
            "enabled": True,
            "enforced": False,
            "live": False,
            "live_requested": bool(live_requested),
            "reason": "no_month_payload",
            "threshold": float("nan"),
            "month": str(resolved_month or month_hint),
        }

    model_payload = month_payload.get("model", month_payload) if isinstance(month_payload, dict) else {}
    threshold = float(pd.to_numeric(pd.Series([month_payload.get("threshold", 0.5)]), errors="coerce").fillna(0.5).iloc[0])
    payload_shadow_only = bool(month_payload.get("shadow_only", payload.get("shadow_only", False)))
    payload_live_ready = bool(month_payload.get("live_ready", not payload_shadow_only))
    live_effective = bool(live_requested)
    live_guard_reason = ""
    if live_effective and payload_shadow_only:
        live_effective = False
        live_guard_reason = "payload_shadow_only"
    elif live_effective and not payload_live_ready:
        live_effective = False
        live_guard_reason = "payload_live_not_ready"
    policy_payload = month_payload.get("gate_policy", {})
    policy = GatePolicyConfig()
    if isinstance(policy_payload, dict):
        segment_threshold_payload = policy_payload.get("segment_threshold_overrides", {})
        if not isinstance(segment_threshold_payload, dict):
            segment_threshold_payload = {}
        segment_budget_payload = policy_payload.get("segment_budget_overrides", {})
        if not isinstance(segment_budget_payload, dict):
            segment_budget_payload = {}
        policy = GatePolicyConfig(
            max_fire_rate=float(policy_payload.get("max_fire_rate", policy.max_fire_rate)),
            min_coverage_rate=float(policy_payload.get("min_coverage_rate", policy.min_coverage_rate)),
            max_removed_per_day=int(policy_payload.get("max_removed_per_day", policy.max_removed_per_day)),
            max_removed_per_player_per_day=int(policy_payload.get("max_removed_per_player_per_day", policy.max_removed_per_player_per_day)),
            max_removed_per_segment_per_day=int(policy_payload.get("max_removed_per_segment_per_day", policy.max_removed_per_segment_per_day)),
            max_removed_per_target_per_day=int(policy_payload.get("max_removed_per_target_per_day", policy.max_removed_per_target_per_day)),
            max_global_removed_per_player=int(policy_payload.get("max_global_removed_per_player", policy.max_global_removed_per_player)),
            max_global_removed_share_per_player=float(
                policy_payload.get("max_global_removed_share_per_player", policy.max_global_removed_share_per_player)
            ),
            global_player_cap_min_veto_rows=int(
                policy_payload.get("global_player_cap_min_veto_rows", policy.global_player_cap_min_veto_rows)
            ),
            relax_player_global_cap_for_budget=bool(
                policy_payload.get("relax_player_global_cap_for_budget", policy.relax_player_global_cap_for_budget)
            ),
            player_penalty_alpha=float(policy_payload.get("player_penalty_alpha", policy.player_penalty_alpha)),
            player_penalty_power=float(policy_payload.get("player_penalty_power", policy.player_penalty_power)),
            optimizer_enabled=bool(policy_payload.get("optimizer_enabled", policy.optimizer_enabled)),
            optimizer_beam_width=int(policy_payload.get("optimizer_beam_width", policy.optimizer_beam_width)),
            optimizer_max_candidates_per_day=int(
                policy_payload.get("optimizer_max_candidates_per_day", policy.optimizer_max_candidates_per_day)
            ),
            optimizer_keep_prob_weight=float(
                policy_payload.get("optimizer_keep_prob_weight", policy.optimizer_keep_prob_weight)
            ),
            optimizer_gap_weight=float(policy_payload.get("optimizer_gap_weight", policy.optimizer_gap_weight)),
            optimizer_harm_weight=float(policy_payload.get("optimizer_harm_weight", policy.optimizer_harm_weight)),
            optimizer_rank_weight=float(policy_payload.get("optimizer_rank_weight", policy.optimizer_rank_weight)),
            segment_threshold_overrides={
                str(seg).upper().strip(): float(val)
                for seg, val in segment_threshold_payload.items()
                if str(seg).strip()
            },
            segment_budget_overrides={
                str(seg).upper().strip(): int(max(0, int(val)))
                for seg, val in segment_budget_payload.items()
                if str(seg).strip()
            },
            tail_slots_only=int(policy_payload.get("tail_slots_only", policy.tail_slots_only)),
            min_veto_gap=float(policy_payload.get("min_veto_gap", policy.min_veto_gap)),
            require_keep_prob_filter=bool(policy_payload.get("require_keep_prob_filter", policy.require_keep_prob_filter)),
            allowed_segments=tuple(
                str(seg).upper().strip()
                for seg in policy_payload.get("allowed_segments", [])
                if str(seg).strip()
            ),
            allowed_meta_cohorts=tuple(
                str(seg).upper().strip()
                for seg in policy_payload.get("allowed_meta_cohorts", [])
                if str(seg).strip()
            ),
        )

    model_type = str((model_payload or {}).get("model_type", "regularized_logistic_gate_v1")).strip().lower()
    try:
        gate_model = load_gate_model_from_payload(model_payload if isinstance(model_payload, dict) else {})
        keep_prob, harm_score = predict_keep_harm_scores(gate_model, out)
    except Exception as exc:
        return out, {
            "enabled": True,
            "enforced": False,
            "live": False,
            "live_requested": bool(live_requested),
            "live_guard_reason": str(live_guard_reason),
            "reason": "model_error",
            "error": f"{type(exc).__name__}: {exc}",
            "threshold": float(threshold),
            "month": str(resolved_month or month_hint),
            "model_type": model_type,
        }

    scored = out.copy()
    scored["accepted_pick_gate_keep_prob"] = pd.Series(keep_prob, index=scored.index, dtype="float64")
    scored["accepted_pick_gate_harm_score"] = pd.Series(harm_score, index=scored.index, dtype="float64")
    policy_scored = apply_shadow_gate_policy(
        scored,
        keep_prob_col="accepted_pick_gate_keep_prob",
        date_col=date_col,
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
        threshold=float(threshold),
        policy=policy,
    )
    veto_mask = pd.to_numeric(policy_scored.get("gate_veto"), errors="coerce").fillna(0).astype(bool)
    scored["accepted_pick_gate_threshold"] = float(threshold)
    scored["accepted_pick_gate_veto"] = veto_mask
    scored["accepted_pick_gate_harm_score"] = pd.to_numeric(
        policy_scored.get("gate_harm_score", scored.get("accepted_pick_gate_harm_score")),
        errors="coerce",
    ).fillna(0.0)
    scored["accepted_pick_gate_veto_reason"] = policy_scored.get("gate_veto_reason", pd.Series("", index=policy_scored.index)).fillna("").astype(str)
    scored["accepted_pick_gate_enabled"] = True
    scored["accepted_pick_gate_enforced"] = True
    scored["accepted_pick_gate_live"] = bool(live_effective)
    scored["accepted_pick_gate_live_requested"] = bool(live_requested)
    scored["accepted_pick_gate_month"] = str(resolved_month or month_hint)
    scored["accepted_pick_gate_policy"] = (
        f"fire<={float(policy.max_fire_rate):.3f};coverage>={float(policy.min_coverage_rate):.3f};"
        f"day<={int(policy.max_removed_per_day)};"
        f"player<={int(policy.max_removed_per_player_per_day)};"
        f"segment<={int(policy.max_removed_per_segment_per_day)};"
        f"target<={int(policy.max_removed_per_target_per_day)};"
        f"global_player_rows<={int(policy.max_global_removed_per_player)};"
        f"global_player_share<={float(policy.max_global_removed_share_per_player):.3f};"
        f"global_cap_min_rows={int(policy.global_player_cap_min_veto_rows)};"
        f"global_cap_relax={1 if bool(policy.relax_player_global_cap_for_budget) else 0};"
        f"player_penalty_alpha={float(policy.player_penalty_alpha):.3f};"
        f"player_penalty_power={float(policy.player_penalty_power):.3f};"
        f"optimizer={1 if bool(policy.optimizer_enabled) else 0};"
        f"beam={int(policy.optimizer_beam_width)};"
        f"opt_keep_w={float(policy.optimizer_keep_prob_weight):.3f};"
        f"opt_gap_w={float(policy.optimizer_gap_weight):.3f};"
        f"opt_harm_w={float(policy.optimizer_harm_weight):.3f};"
        f"seg_thr_overrides={int(len(policy.segment_threshold_overrides or {}))};"
        f"seg_budget_overrides={int(len(policy.segment_budget_overrides or {}))};"
        f"tail={int(policy.tail_slots_only)};gap>={float(policy.min_veto_gap):.3f};"
        f"keep_prob_filter={1 if bool(policy.require_keep_prob_filter) else 0};"
        f"allowed_segments={int(len(policy.allowed_segments or ()))};"
        f"allowed_meta_cohorts={int(len(policy.allowed_meta_cohorts or ()))}"
    )
    drop_count = int(veto_mask.sum()) if bool(live_effective) else 0
    scored["accepted_pick_gate_drop_applied"] = bool(live_effective and drop_count > 0)
    scored["accepted_pick_gate_drop_count"] = int(drop_count)

    if bool(live_effective):
        kept = scored.loc[~veto_mask].copy()
        if kept.empty:
            # Keep at least one row (lowest veto risk) to avoid a dead board.
            sort_cols = ["accepted_pick_gate_keep_prob"]
            sort_asc = [False]
            if "selected_rank" in scored.columns:
                sort_cols.append("selected_rank")
                sort_asc.append(True)
            rescue = scored.sort_values(sort_cols, ascending=sort_asc).head(1).copy()
            rescue["accepted_pick_gate_veto"] = False
            rescue["accepted_pick_gate_veto_reason"] = "minimum_board_rescue"
            rescue["accepted_pick_gate_drop_applied"] = True
            rescue["accepted_pick_gate_drop_count"] = int(max(0, int(len(scored) - 1)))
            kept = rescue
        scored = kept

    details = {
        "enabled": True,
        "enforced": True,
        "live": bool(live_effective),
        "live_requested": bool(live_requested),
        "live_guard_reason": str(live_guard_reason),
        "payload_shadow_only": bool(payload_shadow_only),
        "payload_live_ready": bool(payload_live_ready),
        "threshold": float(threshold),
        "month": str(resolved_month or month_hint),
        "model_type": model_type,
        "rows_in": int(len(frame)),
        "rows_out": int(len(scored)),
        "veto_rows": int(veto_mask.sum()),
        "drop_rows": int(drop_count),
        "veto_share": float(veto_mask.mean()) if len(veto_mask) else 0.0,
        "policy": asdict(policy),
    }
    return scored, details
