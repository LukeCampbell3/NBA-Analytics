from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math
import re
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
    min_coverage_retention: float = 0.98
    min_affected_share: float = 0.02
    max_top_removed_player_share: float = 0.30
    max_top_removed_segment_share: float = 0.35
    max_top_removed_target_share: float = 0.55
    min_rolling_pass_rate: float = 0.55
    max_observed_fire_rate: float = 0.12
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
    out["gate_keep_prob"] = pd.to_numeric(out.get(keep_prob_col), errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    out["gate_threshold"] = float(threshold)
    out["gate_keep_prob_gap"] = float(threshold) - out["gate_keep_prob"]
    out["gate_veto"] = False
    out["gate_veto_reason"] = ""
    out["_gate_player"] = out.get(player_col, pd.Series("", index=out.index)).map(normalize_player_name)
    out["_gate_target"] = out.get(target_col, pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    out["_gate_direction"] = out.get(direction_col, pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    out["_gate_segment"] = (out["_gate_target"] + "|" + out["_gate_direction"]).astype("object")
    out["_gate_date"] = coerce_event_datetime(out.get(date_col)).dt.strftime("%Y-%m-%d").fillna("")
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

    for gate_date, part in out.groupby("_gate_date", sort=True):
        if not gate_date:
            continue
        day_idx = part.index
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
            candidates = part.loc[part["gate_keep_prob"] < float(threshold)].copy()
        else:
            candidates = part.copy()
            # Safety: never allow unrestricted global rewrites when no cohort/segment scope exists.
            if not allowed_segments and not allowed_meta_cohorts:
                candidates = part.loc[part["gate_keep_prob"] < float(threshold)].copy()
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
        if "selected_rank" in candidates.columns:
            candidates = candidates.sort_values(["gate_keep_prob", "selected_rank"], ascending=[True, True])
        else:
            candidates = candidates.sort_values(["gate_keep_prob"], ascending=[True])

        veto_count = 0
        player_counts: dict[str, int] = {}
        segment_counts: dict[str, int] = {}
        target_counts: dict[str, int] = {}
        for idx, row in candidates.iterrows():
            if veto_count >= max_veto:
                break
            player = str(row.get("_gate_player", ""))
            target = str(row.get("_gate_target", ""))
            segment = str(row.get("_gate_segment", ""))
            if player and player_counts.get(player, 0) >= int(max(0, policy.max_removed_per_player_per_day)):
                continue
            if segment and segment_counts.get(segment, 0) >= int(max(0, policy.max_removed_per_segment_per_day)):
                continue
            if target and target_counts.get(target, 0) >= int(max(0, policy.max_removed_per_target_per_day)):
                continue
            out.loc[idx, "gate_veto"] = True
            out.loc[idx, "gate_veto_reason"] = "low_keep_prob" if bool(policy.require_keep_prob_filter) else "meta_cohort_rule"
            veto_count += 1
            if player:
                player_counts[player] = player_counts.get(player, 0) + 1
            if segment:
                segment_counts[segment] = segment_counts.get(segment, 0) + 1
            if target:
                target_counts[target] = target_counts.get(target, 0) + 1

    out = out.drop(columns=["_gate_player", "_gate_target", "_gate_direction", "_gate_segment", "_gate_date", "gate_keep_prob_gap"], errors="ignore")
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
            "unique_players": 0,
            "unique_segments": 0,
            "unique_targets": 0,
        }
    players = vetoed.get(player_col, pd.Series("", index=vetoed.index)).map(normalize_player_name)
    targets = vetoed.get(target_col, pd.Series("", index=vetoed.index)).fillna("").astype(str).str.upper().str.strip()
    directions = vetoed.get(direction_col, pd.Series("", index=vetoed.index)).fillna("").astype(str).str.upper().str.strip()
    segments = (targets + "|" + directions).astype("object")
    n_rows = int(len(vetoed))
    top_player = float(players.value_counts(normalize=True).iloc[0]) if n_rows > 0 and players.nunique() > 0 else 0.0
    top_segment = float(segments.value_counts(normalize=True).iloc[0]) if n_rows > 0 and segments.nunique() > 0 else 0.0
    top_target = float(targets.value_counts(normalize=True).iloc[0]) if n_rows > 0 and targets.nunique() > 0 else 0.0
    return {
        "veto_rows": n_rows,
        "top_player_share": top_player,
        "top_segment_share": top_segment,
        "top_target_share": top_target,
        "unique_players": int(players.nunique()),
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
    if top_player > float(criteria.max_top_removed_player_share):
        failures.append("removed_player_concentration_above_cap")
    if top_segment > float(criteria.max_top_removed_segment_share):
        failures.append("removed_segment_concentration_above_cap")
    if top_target > float(criteria.max_top_removed_target_share):
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
        policy = GatePolicyConfig(
            max_fire_rate=float(policy_payload.get("max_fire_rate", policy.max_fire_rate)),
            min_coverage_rate=float(policy_payload.get("min_coverage_rate", policy.min_coverage_rate)),
            max_removed_per_day=int(policy_payload.get("max_removed_per_day", policy.max_removed_per_day)),
            max_removed_per_player_per_day=int(policy_payload.get("max_removed_per_player_per_day", policy.max_removed_per_player_per_day)),
            max_removed_per_segment_per_day=int(policy_payload.get("max_removed_per_segment_per_day", policy.max_removed_per_segment_per_day)),
            max_removed_per_target_per_day=int(policy_payload.get("max_removed_per_target_per_day", policy.max_removed_per_target_per_day)),
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

    try:
        gate_model = RegularizedLogisticGate.from_dict(model_payload if isinstance(model_payload, dict) else {})
        keep_prob = gate_model.predict_proba_dataframe(out)
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
        }

    scored = out.copy()
    scored["accepted_pick_gate_keep_prob"] = pd.Series(keep_prob, index=scored.index, dtype="float64")
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
        "rows_in": int(len(frame)),
        "rows_out": int(len(scored)),
        "veto_rows": int(veto_mask.sum()),
        "drop_rows": int(drop_count),
        "veto_share": float(veto_mask.mean()) if len(veto_mask) else 0.0,
        "policy": asdict(policy),
    }
    return scored, details
