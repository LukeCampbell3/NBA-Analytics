from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decision_engine.accepted_pick_gate import build_pick_key


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> Path:
    path.resolve().parent.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value: Any, default: float = np.nan) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def safe_int(value: Any, default: int = 0) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return int(default)
    return int(round(float(numeric)))


def series_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def series_text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def build_candidate_id(
    frame: pd.DataFrame,
    *,
    player_col: str = "market_player_raw",
    fallback_player_col: str = "player",
    market_date_col: str = "market_date",
    target_col: str = "target",
    direction_col: str = "direction",
    line_col: str = "market_line",
) -> pd.Series:
    working = frame.copy()
    if market_date_col not in working.columns and "run_date" in working.columns:
        working[market_date_col] = working["run_date"]
    candidate_key = build_pick_key(
        working,
        player_col=player_col,
        fallback_player_col=fallback_player_col,
        market_date_col=market_date_col,
        target_col=target_col,
        direction_col=direction_col,
        line_col=line_col,
    ).astype(str)
    return "candidate::" + candidate_key


def coerce_market_type(frame: pd.DataFrame) -> pd.Series:
    if "market_id" in frame.columns:
        return series_text(frame, "market_id").str.upper().str.strip()
    return series_text(frame, "target").str.upper().str.strip() + "_" + series_text(frame, "direction").str.upper().str.strip()


def coerce_market_family(frame: pd.DataFrame) -> pd.Series:
    market_type = coerce_market_type(frame)
    if frame.empty:
        return pd.Series(dtype="object")
    return market_type.str.split("_").str[0].fillna("")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.resolve().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_string_list(value: Any, *, split_delimiters: tuple[str, ...] = ("|", ";", ",")) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out
    text = str(value).strip()
    if not text:
        return []
    for delimiter in split_delimiters:
        if delimiter in text:
            return [token.strip() for token in text.split(delimiter) if token.strip()]
    return [text]


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return bool(default)
    return bool(int(round(float(numeric))))


def clip_probability(values: pd.Series | float, *, lower: float = 1e-6, upper: float = 1.0 - 1e-6) -> pd.Series | float:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").fillna(0.5).clip(lower=lower, upper=upper)
    numeric = safe_float(values, default=0.5)
    return float(np.clip(numeric, lower, upper))


def result_to_label(value: Any) -> float:
    text = str(value).strip().lower()
    if text == "win":
        return 1.0
    if text == "loss":
        return 0.0
    return np.nan


def expected_result_utility(value: Any, payout_per_win: float = 100.0 / 110.0) -> float:
    text = str(value).strip().lower()
    if text == "win":
        return float(payout_per_win)
    if text == "loss":
        return -1.0
    if text == "push":
        return 0.0
    return np.nan


def brier_score(probabilities: pd.Series, labels: pd.Series) -> float:
    prob = clip_probability(probabilities)
    label = pd.to_numeric(labels, errors="coerce")
    mask = prob.notna() & label.notna()
    if not bool(mask.any()):
        return np.nan
    residual = prob.loc[mask].to_numpy(dtype="float64") - label.loc[mask].to_numpy(dtype="float64")
    return float(np.mean(np.square(residual)))


def expected_calibration_error(probabilities: pd.Series, labels: pd.Series, *, bins: int = 10) -> float:
    prob = clip_probability(probabilities)
    label = pd.to_numeric(labels, errors="coerce")
    mask = prob.notna() & label.notna()
    if not bool(mask.any()):
        return np.nan
    prob_values = prob.loc[mask].to_numpy(dtype="float64")
    label_values = label.loc[mask].to_numpy(dtype="float64")
    if prob_values.size == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, int(max(2, bins)) + 1)
    ece = 0.0
    total = max(len(prob_values), 1)
    for left, right in zip(edges[:-1], edges[1:]):
        bucket_mask = (prob_values >= left) & (prob_values < right if right < 1.0 else prob_values <= right)
        if not bucket_mask.any():
            continue
        accuracy = float(label_values[bucket_mask].mean())
        confidence = float(prob_values[bucket_mask].mean())
        ece += (float(bucket_mask.sum()) / float(total)) * abs(accuracy - confidence)
    return float(ece)


def calibration_gap(probabilities: pd.Series, labels: pd.Series) -> float:
    prob = clip_probability(probabilities)
    label = pd.to_numeric(labels, errors="coerce")
    mask = prob.notna() & label.notna()
    if not bool(mask.any()):
        return np.nan
    return float(abs(prob.loc[mask].mean() - label.loc[mask].mean()))


def bootstrap_mean_delta(
    baseline_values: pd.Series,
    variant_values: pd.Series,
    *,
    samples: int = 500,
    seed: int = 17,
) -> dict[str, float]:
    baseline = pd.to_numeric(baseline_values, errors="coerce")
    variant = pd.to_numeric(variant_values, errors="coerce")
    mask = baseline.notna() & variant.notna()
    if int(mask.sum()) <= 1:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    base_arr = baseline.loc[mask].to_numpy(dtype="float64")
    var_arr = variant.loc[mask].to_numpy(dtype="float64")
    observed = float(var_arr.mean() - base_arr.mean())
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    idx = np.arange(base_arr.shape[0], dtype=int)
    for _ in range(int(max(1, samples))):
        sample_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        draws.append(float(var_arr[sample_idx].mean() - base_arr[sample_idx].mean()))
    ci_low, ci_high = np.quantile(np.asarray(draws, dtype="float64"), [0.025, 0.975])
    return {"delta": observed, "ci_low": float(ci_low), "ci_high": float(ci_high)}
