from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGETS = {"PTS", "TRB", "AST"}
PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PLAYER_PREDICTOR_ROOT / "model" / "analysis"
SHARED_VALIDATION_ROOT = Path(__file__).resolve().parents[4] / "validation"
HISTORY_SOURCE_PATTERNS: tuple[tuple[str, int], ...] = (
    ("validation_recent_pool_selector", 500),
    ("selector_replay_rows_rebuilt", 420),
    ("selector_replay_rows", 360),
    ("validation_current_prod_hitrate_rows", 300),
    ("board_size_history_rows", 260),
    ("latest_market_comparison_strict_rows", 180),
)


def _numeric_series(frame: pd.DataFrame, columns: tuple[str, ...], default: float = np.nan) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _string_series(frame: pd.DataFrame, columns: tuple[str, ...], default: str = "") -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _coerce_event_datetime(values: Any) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    out = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    ymd_mask = numeric.between(19000101, 21001231, inclusive="both")
    if bool(ymd_mask.any()):
        ymd_text = numeric.round().astype("Int64").astype(str)
        parsed_ymd = pd.to_datetime(ymd_text.where(ymd_mask), format="%Y%m%d", errors="coerce")
        out = out.where(~ymd_mask, parsed_ymd)
    return out


def _recommendation_rank(values: pd.Series) -> pd.Series:
    order = {"elite": 3, "strong": 2, "consider": 1, "pass": 0}
    return values.fillna("").astype(str).str.strip().str.lower().map(order).fillna(0).astype("float64")


def _resolve_cutoff_date(selector_df: pd.DataFrame) -> pd.Timestamp | pd.NaT:
    for column in ("market_date", "run_date", "event_date"):
        if column in selector_df.columns:
            dates = _coerce_event_datetime(selector_df[column]).dropna()
            if not dates.empty:
                return pd.Timestamp(dates.min()).normalize()
    return pd.NaT


def _prepare_history(frame: pd.DataFrame, *, cutoff_date: pd.Timestamp | pd.NaT) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    date_col = "market_date" if "market_date" in out.columns else "event_date" if "event_date" in out.columns else ""
    if not date_col:
        return pd.DataFrame()
    out["event_date"] = _coerce_event_datetime(out[date_col]).dt.normalize()
    out = out.loc[out["event_date"].notna()].copy()
    if pd.notna(cutoff_date):
        out = out.loc[out["event_date"] < pd.Timestamp(cutoff_date).normalize()].copy()
    if out.empty:
        return out

    out["target"] = _string_series(out, ("target",), "").str.upper().str.strip()
    out["direction"] = _string_series(out, ("direction",), "").str.upper().str.strip()
    out = out.loc[out["target"].isin(TARGETS) & out["direction"].isin({"OVER", "UNDER"})].copy()
    if out.empty:
        return out

    result = _string_series(out, ("result", "outcome"), "").str.lower().str.strip()
    out["label"] = result.map({"win": 1.0, "loss": 0.0})
    out = out.loc[out["label"].notna()].copy()
    if out.empty:
        return out

    out["hist_prob"] = _numeric_series(out, ("estimated_win_rate", "expected_win_rate", "p_calibrated"), 0.5).fillna(0.5)
    out["hist_ev"] = _numeric_series(out, ("estimated_ev", "selection_ev", "ev"), 0.0).fillna(0.0)
    out["hist_confidence"] = _numeric_series(out, ("selection_confidence", "final_confidence"), 0.0).fillna(0.0)
    out["hist_abs_edge"] = _numeric_series(out, ("abs_edge",), 0.0).fillna(0.0)
    out["hist_uncertainty_sigma"] = _numeric_series(out, ("uncertainty_sigma",), np.nan)
    out["hist_history_rows"] = _numeric_series(out, ("history_rows",), 0.0).fillna(0.0)
    out["hist_spike_probability"] = _numeric_series(out, ("spike_probability",), 0.0).fillna(0.0)
    out["hist_market_line"] = _numeric_series(out, ("market_line",), np.nan)
    out["segment_key"] = out["target"] + "|" + out["direction"]
    return out


def _history_csv_candidates() -> list[tuple[int, Path]]:
    roots = [ANALYSIS_ROOT, SHARED_VALIDATION_ROOT]
    seen: set[Path] = set()
    candidates: list[tuple[int, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern, priority in HISTORY_SOURCE_PATTERNS:
            for path in root.glob(f"**/{pattern}*.csv"):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append((int(priority), path))
    return candidates


def _choose_history_frame(
    history_df: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp | pd.NaT,
    min_train_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = _prepare_history(history_df, cutoff_date=cutoff_date)
    best = prepared
    best_meta: dict[str, Any] = {
        "source": "provided_history",
        "path": "",
        "rows": int(len(prepared)),
        "priority": 0,
    }

    for priority, path in _history_csv_candidates():
        try:
            candidate_raw = pd.read_csv(path)
        except Exception:
            continue
        candidate = _prepare_history(candidate_raw, cutoff_date=cutoff_date)
        if candidate.empty:
            continue
        better_size = len(candidate) > len(best)
        enough_when_current_not = len(best) < int(max(1, min_train_rows)) <= len(candidate)
        if enough_when_current_not or better_size:
            best = candidate
            best_meta = {
                "source": "discovered_history_csv",
                "path": str(path),
                "rows": int(len(candidate)),
                "priority": int(priority),
            }

    return best, best_meta


def _resolve_holdout_split(
    prepared_history: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp | pd.NaT,
    min_train_rows: int,
    holdout_days: int,
    min_holdout_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = prepared_history.copy()
    holdout = prepared_history.iloc[0:0].copy()
    meta = {"mode": "disabled"}
    if prepared_history.empty:
        return train, holdout, meta

    if int(max(0, holdout_days)) > 0 and pd.notna(cutoff_date):
        holdout_cutoff = pd.Timestamp(cutoff_date) - pd.Timedelta(days=int(max(0, holdout_days)))
        maybe_holdout = prepared_history.loc[prepared_history["event_date"] >= holdout_cutoff].copy()
        maybe_train = prepared_history.loc[prepared_history["event_date"] < holdout_cutoff].copy()
        if len(maybe_train) >= int(max(1, min_train_rows)) and len(maybe_holdout) >= int(max(1, min_holdout_rows)):
            return maybe_train, maybe_holdout, {
                "mode": "calendar_days",
                "holdout_days": int(max(0, holdout_days)),
            }

    return train, holdout, meta


def _segment_rates(train: pd.DataFrame) -> tuple[dict[str, dict[str, float]], float]:
    global_rate = float(train["label"].mean()) if not train.empty else 0.5
    global_rate = float(np.clip(global_rate, 0.05, 0.95))
    segment_stats: dict[str, dict[str, float]] = {}
    for segment_key, part in train.groupby("segment_key", dropna=False):
        rows = int(len(part))
        wins = float(part["label"].sum())
        # Moderate shrinkage keeps tiny segments from dominating the board.
        rate = (wins + (global_rate * 8.0)) / (rows + 8.0)
        segment_stats[str(segment_key)] = {
            "rows": float(rows),
            "wins": float(wins),
            "rate": float(np.clip(rate, 0.05, 0.95)),
        }
    return segment_stats, global_rate


def _selector_probabilities(
    selector_df: pd.DataFrame,
    *,
    segment_stats: dict[str, dict[str, float]],
    global_rate: float,
    probability_shrink_factor: float,
    min_candidate_expected_win_rate: float,
    min_candidate_final_confidence: float,
    min_candidate_recommendation: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    out = selector_df.copy()
    target = _string_series(out, ("target",), "").str.upper().str.strip()
    direction = _string_series(out, ("direction",), "").str.upper().str.strip()
    segment_key = target + "|" + direction

    meta_prob = _numeric_series(out, ("expected_win_rate", "estimated_win_rate", "p_calibrated"), 0.5).fillna(0.5).clip(0.05, 0.95)
    final_confidence = _numeric_series(out, ("final_confidence", "selection_confidence"), 0.0).fillna(0.0)
    rec_rank = _recommendation_rank(_string_series(out, ("recommendation",), "pass"))
    min_rec = _recommendation_rank(pd.Series([min_candidate_recommendation])).iloc[0]

    segment_rate = pd.Series(
        [float(segment_stats.get(str(key), {}).get("rate", global_rate)) for key in segment_key],
        index=out.index,
        dtype="float64",
    )
    segment_rows = pd.Series(
        [float(segment_stats.get(str(key), {}).get("rows", 0.0)) for key in segment_key],
        index=out.index,
        dtype="float64",
    )
    support_strength = (segment_rows / 24.0).clip(0.0, 1.0)
    candidate_eligible = (
        meta_prob.ge(float(min_candidate_expected_win_rate))
        & final_confidence.ge(float(min_candidate_final_confidence))
        & rec_rank.ge(float(min_rec))
    )

    shrink = float(np.clip(probability_shrink_factor, 0.0, 1.0))
    historical_signal = (support_strength * segment_rate) + ((1.0 - support_strength) * global_rate)
    blend_raw = (shrink * historical_signal) + ((1.0 - shrink) * meta_prob)
    reranker_prob = blend_raw.where(candidate_eligible, (0.75 * blend_raw) + (0.25 * meta_prob))
    return reranker_prob.clip(0.01, 0.99), blend_raw.clip(0.01, 0.99), candidate_eligible.astype(bool)


def score_selector_with_robust_reranker(
    selector_df: pd.DataFrame,
    history_df: pd.DataFrame,
    *,
    probability_shrink_factor: float = 0.75,
    elite_pct: float = 0.95,
    min_train_rows: int = 4000,
    holdout_days: int = 45,
    min_holdout_rows: int = 250,
    num_pair_per_sample: int = 12,
    min_candidate_expected_win_rate: float = 0.55,
    min_candidate_final_confidence: float = 0.03,
    min_candidate_recommendation: str = "consider",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    del elite_pct
    del num_pair_per_sample

    out = selector_df.copy()
    out["robust_reranker_prob"] = np.nan
    out["robust_reranker_blend_raw"] = np.nan
    out["robust_reranker_enabled"] = False
    out["robust_reranker_candidate_eligible"] = False
    out["robust_reranker_train_rows"] = 0
    out["robust_reranker_holdout_rows"] = 0
    out["robust_reranker_source"] = "disabled"

    if out.empty:
        return out, {"enabled": False, "reason": "empty_selector"}

    cutoff_date = _resolve_cutoff_date(out)
    prepared_history, history_meta = _choose_history_frame(
        history_df,
        cutoff_date=cutoff_date,
        min_train_rows=int(max(1, min_train_rows)),
    )
    if len(prepared_history) < int(max(1, min_train_rows)):
        return out, {
            "enabled": False,
            "reason": "insufficient_train_rows",
            "train_rows": int(len(prepared_history)),
            "min_train_rows": int(max(1, min_train_rows)),
            "cutoff_date": str(cutoff_date.date()) if pd.notna(cutoff_date) else "",
            "history_source": history_meta,
        }

    train, holdout, holdout_meta = _resolve_holdout_split(
        prepared_history,
        cutoff_date=cutoff_date,
        min_train_rows=int(max(1, min_train_rows)),
        holdout_days=int(max(0, holdout_days)),
        min_holdout_rows=int(max(0, min_holdout_rows)),
    )
    if len(train) < int(max(1, min_train_rows)):
        train = prepared_history
        holdout = prepared_history.iloc[0:0].copy()
        holdout_meta = {"mode": "disabled_train_too_small"}

    segment_stats, global_rate = _segment_rates(train)
    reranker_prob, blend_raw, candidate_eligible = _selector_probabilities(
        out,
        segment_stats=segment_stats,
        global_rate=global_rate,
        probability_shrink_factor=float(probability_shrink_factor),
        min_candidate_expected_win_rate=float(min_candidate_expected_win_rate),
        min_candidate_final_confidence=float(min_candidate_final_confidence),
        min_candidate_recommendation=str(min_candidate_recommendation),
    )

    out["robust_reranker_prob"] = reranker_prob.astype("float64")
    out["robust_reranker_blend_raw"] = blend_raw.astype("float64")
    out["robust_reranker_enabled"] = True
    out["robust_reranker_candidate_eligible"] = candidate_eligible.astype(bool)
    out["robust_reranker_train_rows"] = int(len(train))
    out["robust_reranker_holdout_rows"] = int(len(holdout))
    out["robust_reranker_source"] = str(history_meta.get("source") or "provided_history")

    holdout_brier = None
    if not holdout.empty:
        holdout_prob, _, _ = _selector_probabilities(
            holdout.rename(
                columns={
                    "hist_prob": "expected_win_rate",
                    "hist_confidence": "final_confidence",
                }
            ),
            segment_stats=segment_stats,
            global_rate=global_rate,
            probability_shrink_factor=float(probability_shrink_factor),
            min_candidate_expected_win_rate=0.0,
            min_candidate_final_confidence=0.0,
            min_candidate_recommendation="pass",
        )
        labels = holdout["label"].to_numpy(dtype=float)
        holdout_brier = float(np.mean((holdout_prob.to_numpy(dtype=float) - labels) ** 2)) if labels.size else None

    summary = {
        "enabled": True,
        "cutoff_date": str(cutoff_date.date()) if pd.notna(cutoff_date) else "",
        "history_source": history_meta,
        "pre_cutoff_rows": int(len(prepared_history)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "holdout": holdout_meta,
        "holdout_brier": holdout_brier,
        "global_hit_rate": float(global_rate),
        "segment_count": int(len(segment_stats)),
        "segment_rates": segment_stats,
        "candidate_eligible_count": int(candidate_eligible.sum()),
        "probability_shrink_factor": float(probability_shrink_factor),
    }
    return out, summary
