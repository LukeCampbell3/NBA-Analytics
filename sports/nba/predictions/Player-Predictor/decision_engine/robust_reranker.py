from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path

from .accepted_pick_gate import LogisticGateConfig, RegularizedLogisticGate, build_meta_cohort_columns


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
NUMERIC_FEATURES = [
    "meta_prob",
    "meta_ev",
    "meta_confidence",
    "meta_abs_edge",
    "meta_edge_to_sigma",
    "meta_history_rows",
    "meta_uncertainty_sigma",
    "meta_spike_probability",
    "meta_market_line",
]
CATEGORICAL_FEATURES = [
    "target",
    "direction",
    "meta_seg",
    "meta_seg_ew",
    "meta_seg_ev",
    "meta_seg_line",
]


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


def _compute_edge_to_sigma(abs_edge: pd.Series, uncertainty_sigma: pd.Series) -> pd.Series:
    sigma = pd.to_numeric(uncertainty_sigma, errors="coerce").abs()
    edge = pd.to_numeric(abs_edge, errors="coerce").abs()
    return edge / sigma.where(sigma > 1e-6, np.nan)


def _safe_brier(prob: np.ndarray, label: np.ndarray) -> float | None:
    if prob.size == 0 or label.size == 0:
        return None
    return float(np.mean((prob - label) ** 2))


def _safe_log_loss(prob: np.ndarray, label: np.ndarray) -> float | None:
    if prob.size == 0 or label.size == 0:
        return None
    clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(label * np.log(clipped) + (1.0 - label) * np.log(1.0 - clipped)))


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

    unique_days = sorted(pd.to_datetime(prepared_history["event_date"], errors="coerce").dt.normalize().dropna().unique().tolist())
    if len(unique_days) >= 6:
        holdout_day_count = max(2, int(np.ceil(float(len(unique_days)) * 0.20)))
        holdout_days_set = set(unique_days[-holdout_day_count:])
        mask = pd.to_datetime(prepared_history["event_date"], errors="coerce").dt.normalize().isin(holdout_days_set)
        maybe_holdout = prepared_history.loc[mask].copy()
        maybe_train = prepared_history.loc[~mask].copy()
        relaxed_train_floor = max(1000, int(max(1, min_train_rows) * 0.60))
        if len(maybe_train) >= int(relaxed_train_floor) and len(maybe_holdout) >= int(max(100, min_holdout_rows)):
            return maybe_train, maybe_holdout, {
                "mode": "tail_unique_days",
                "holdout_day_count": int(holdout_day_count),
                "relaxed_train_floor": int(relaxed_train_floor),
            }

    return train, holdout, meta


def _resolve_effective_shrink(
    requested_shrink: float,
    *,
    holdout_prob: np.ndarray | None,
    base_prob: np.ndarray | None,
    label: np.ndarray | None,
) -> tuple[float, dict[str, Any]]:
    requested = float(np.clip(requested_shrink, 0.0, 1.0))
    if holdout_prob is None or base_prob is None or label is None or len(holdout_prob) == 0:
        return max(0.25, requested * 0.60), {"mode": "no_holdout", "requested": requested}

    model_brier = _safe_brier(holdout_prob, label)
    base_brier = _safe_brier(base_prob, label)
    model_log_loss = _safe_log_loss(holdout_prob, label)
    base_log_loss = _safe_log_loss(base_prob, label)
    brier_gain = float(base_brier - model_brier) if model_brier is not None and base_brier is not None else 0.0
    log_loss_gain = float(base_log_loss - model_log_loss) if model_log_loss is not None and base_log_loss is not None else 0.0

    if brier_gain > 0.002 and log_loss_gain > 0.002:
        effective = requested
        mode = "full"
    elif brier_gain > 0.0 or log_loss_gain > 0.0:
        effective = max(0.35, requested * 0.75)
        mode = "partial"
    else:
        effective = min(0.25, requested * 0.35)
        mode = "degraded"

    return float(np.clip(effective, 0.0, 1.0)), {
        "mode": mode,
        "requested": requested,
        "effective": float(np.clip(effective, 0.0, 1.0)),
        "brier_gain": float(brier_gain),
        "log_loss_gain": float(log_loss_gain),
        "model_brier": model_brier,
        "base_brier": base_brier,
        "model_log_loss": model_log_loss,
        "base_log_loss": base_log_loss,
    }


def _resolve_cutoff_date(selector_df: pd.DataFrame) -> pd.Timestamp | pd.NaT:
    for column in ("market_date", "run_date", "target_date"):
        if column not in selector_df.columns:
            continue
        parsed = _coerce_event_datetime(selector_df[column]).dropna()
        if not parsed.empty:
            return pd.Timestamp(parsed.min()).normalize()
    return pd.NaT


def _path_pattern_score(path: Path) -> int:
    name = path.name.lower()
    for pattern, score in HISTORY_SOURCE_PATTERNS:
        if pattern in name:
            return int(score)
    return 100


def _history_summary(prepared: pd.DataFrame) -> dict[str, Any]:
    event_dates = pd.to_datetime(prepared.get("event_date"), errors="coerce").dropna() if "event_date" in prepared.columns else pd.Series(dtype="datetime64[ns]")
    unique_days = int(event_dates.dt.normalize().nunique()) if not event_dates.empty else 0
    latest_date = str(event_dates.max().date()) if not event_dates.empty else ""
    return {
        "rows": int(len(prepared)),
        "unique_days": int(unique_days),
        "latest_date": latest_date,
    }


def _discover_history_csv_paths() -> list[Path]:
    candidates: list[Path] = []
    for root in (ANALYSIS_ROOT, SHARED_VALIDATION_ROOT):
        if not root.exists():
            continue
        for pattern, _ in HISTORY_SOURCE_PATTERNS:
            candidates.extend(root.rglob(f"{pattern}*.csv"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except Exception:
            resolved = str(candidate)
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _choose_history_frame(
    history_df: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp | pd.NaT,
    min_train_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared_current = _prepare_history_rows(history_df, cutoff_date)
    current_summary = _history_summary(prepared_current)
    current_meta = {
        "source": "in_memory_history_df",
        "path": "",
        **current_summary,
    }

    best_frame = prepared_current
    best_meta = current_meta
    best_score = (0, current_summary["unique_days"], current_summary["rows"], current_summary["latest_date"])

    discovered_rows: list[dict[str, Any]] = []
    for path in _discover_history_csv_paths():
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        prepared = _prepare_history_rows(raw, cutoff_date)
        if prepared.empty:
            continue
        summary = _history_summary(prepared)
        name_score = int(_path_pattern_score(path))
        score = (name_score, summary["unique_days"], summary["rows"], summary["latest_date"])
        discovered_rows.append(
            {
                "path": str(path.resolve()),
                "rows": int(summary["rows"]),
                "unique_days": int(summary["unique_days"]),
                "latest_date": str(summary["latest_date"]),
                "name_score": int(name_score),
            }
        )
        if score > best_score:
            best_score = score
            best_frame = prepared
            best_meta = {
                "source": path.name,
                "path": str(path.resolve()),
                **summary,
            }

    current_is_rich_enough = (
        int(current_summary["rows"]) >= int(max(1, min_train_rows))
        and int(current_summary["rows"]) >= int(best_meta.get("rows", 0))
        and int(current_summary["unique_days"]) >= int(best_meta.get("unique_days", 0))
    )
    if current_is_rich_enough:
        best_frame = prepared_current
        best_meta = current_meta

    best_meta["discovered_candidates"] = discovered_rows
    return best_frame, best_meta


def _prepare_model_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if source == "history":
        prob = _numeric_series(frame, ("estimated_win_rate", "expected_win_rate", "p_calibrated", "board_play_win_prob"), default=0.5)
        ev = _numeric_series(frame, ("estimated_ev", "ev", "ev_adjusted"), default=0.0)
        confidence = _numeric_series(frame, ("selection_confidence", "final_confidence", "confidence_score"), default=0.0)
        market_line = _numeric_series(frame, ("market_line",), default=np.nan)
    else:
        prob = _numeric_series(frame, ("expected_win_rate", "p_calibrated", "board_play_win_prob"), default=0.5)
        ev = _numeric_series(frame, ("ev", "ev_adjusted", "estimated_ev"), default=0.0)
        confidence = _numeric_series(frame, ("final_confidence", "confidence_score", "selection_confidence"), default=0.0)
        market_line = _numeric_series(frame, ("market_line",), default=np.nan)

    out = pd.DataFrame(index=frame.index)
    out["target"] = _string_series(frame, ("target",)).str.upper().str.strip()
    out["direction"] = _string_series(frame, ("direction",)).str.upper().str.strip()
    out["meta_prob"] = prob.fillna(0.5).clip(lower=0.0, upper=1.0)
    out["meta_ev"] = ev.fillna(0.0)
    out["meta_confidence"] = confidence.fillna(0.0).clip(lower=0.0, upper=1.0)
    out["meta_abs_edge"] = _numeric_series(frame, ("abs_edge",), default=0.0).fillna(0.0).abs()
    out["meta_history_rows"] = _numeric_series(frame, ("history_rows",), default=0.0).fillna(0.0).clip(lower=0.0)
    out["meta_uncertainty_sigma"] = _numeric_series(frame, ("uncertainty_sigma",), default=np.nan).fillna(0.0).clip(lower=0.0)
    out["meta_spike_probability"] = _numeric_series(frame, ("spike_probability",), default=0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["meta_market_line"] = market_line.fillna(0.0)
    edge_to_sigma = _numeric_series(frame, ("edge_to_sigma", "sigma_ratio"), default=np.nan)
    edge_to_sigma = edge_to_sigma.where(edge_to_sigma.notna(), _compute_edge_to_sigma(out["meta_abs_edge"], out["meta_uncertainty_sigma"]))
    out["meta_edge_to_sigma"] = edge_to_sigma.fillna(0.0).clip(lower=0.0)
    out["meta_recommendation_rank"] = _recommendation_rank(_string_series(frame, ("recommendation",), default=""))

    cohort_frame = pd.DataFrame(
        {
            "target": out["target"],
            "direction": out["direction"],
            "expected_win_rate": out["meta_prob"],
            "ev": out["meta_ev"],
            "market_line": out["meta_market_line"],
        },
        index=out.index,
    )
    cohorts = build_meta_cohort_columns(cohort_frame, target_col="target", direction_col="direction")
    for column in ("meta_seg", "meta_seg_ew", "meta_seg_ev", "meta_seg_line"):
        out[column] = cohorts.get(column, pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    return out


def _prepare_history_rows(history_df: pd.DataFrame, cutoff_date: pd.Timestamp | pd.NaT) -> pd.DataFrame:
    if history_df.empty:
        return history_df.iloc[0:0].copy()
    working = history_df.copy()
    working["target"] = _string_series(working, ("target",)).str.upper().str.strip()
    working["direction"] = _string_series(working, ("direction",)).str.upper().str.strip()
    working = working.loc[working["target"].isin(TARGETS) & working["direction"].isin({"OVER", "UNDER"})].copy()
    result = _string_series(working, ("result",)).str.strip().str.lower()
    working = working.loc[result.isin({"win", "loss"})].copy()
    if working.empty:
        return working
    working["label"] = np.where(result.loc[working.index].eq("win"), 1.0, 0.0)
    working["event_date"] = _coerce_event_datetime(_string_series(working, ("market_date", "run_date", "target_date")))
    working = working.loc[working["event_date"].notna()].copy()
    if pd.notna(cutoff_date):
        working = working.loc[working["event_date"] < pd.Timestamp(cutoff_date)].copy()
    if working.empty:
        return working
    prepared = _prepare_model_frame(working, source="history")
    prepared["label"] = working["label"].astype("float64")
    prepared["event_date"] = working["event_date"]
    prepared["base_prob"] = prepared["meta_prob"]
    return prepared


def _normalize_blend_component(values: pd.Series, center: float, scale: float) -> pd.Series:
    return (0.5 + 0.5 * np.tanh((pd.to_numeric(values, errors="coerce").fillna(center) - center) / max(scale, 1e-6))).clip(0.0, 1.0)


def _soft_floor_strength(values: pd.Series, floor: float, span: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(float(floor))
    return ((numeric - float(floor)) / max(float(span), 1e-6) + 0.5).clip(0.0, 1.0)


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

    prepared_selector = _prepare_model_frame(out, source="selector")
    if prepared_selector.empty:
        return out, {
            "enabled": False,
            "reason": "empty_selector",
            "train_rows": int(len(prepared_history)),
        }

    train, holdout, holdout_meta = _resolve_holdout_split(
        prepared_history,
        cutoff_date=cutoff_date,
        min_train_rows=int(max(1, min_train_rows)),
        holdout_days=int(max(0, holdout_days)),
        min_holdout_rows=int(max(1, min_holdout_rows)),
    )

    gate = RegularizedLogisticGate(
        config=LogisticGateConfig(
            learning_rate=0.05,
            l2_strength=2.5,
            max_iter=3000,
            tolerance=1e-7,
            class_weight_positive=1.0,
            class_weight_negative=1.0,
        )
    )
    try:
        gate.fit_dataframe(
            train,
            label_col="label",
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
    except Exception as exc:
        return out, {
            "enabled": False,
            "reason": "fit_error",
            "error": f"{type(exc).__name__}: {exc}",
            "train_rows": int(len(train)),
            "cutoff_date": str(cutoff_date.date()) if pd.notna(cutoff_date) else "",
            "history_source": history_meta,
        }

    holdout_prob = None
    base_holdout_prob = None
    holdout_label = None
    if not holdout.empty:
        holdout_prob = gate.predict_proba_dataframe(holdout)
        base_holdout_prob = holdout["base_prob"].to_numpy(dtype="float64", copy=False)
        holdout_label = holdout["label"].to_numpy(dtype="float64", copy=False)

    effective_shrink, shrink_meta = _resolve_effective_shrink(
        float(np.clip(probability_shrink_factor, 0.0, 1.0)),
        holdout_prob=holdout_prob,
        base_prob=base_holdout_prob,
        label=holdout_label,
    )

    model_prob = gate.predict_proba_dataframe(prepared_selector)
    base_prob = prepared_selector["meta_prob"].to_numpy(dtype="float64", copy=False)
    confidence = prepared_selector["meta_confidence"].to_numpy(dtype="float64", copy=False)
    edge_component = _normalize_blend_component(prepared_selector["meta_edge_to_sigma"], center=0.20, scale=0.35).to_numpy(
        dtype="float64",
        copy=False,
    )
    history_component = _normalize_blend_component(prepared_selector["meta_history_rows"], center=90.0, scale=45.0).to_numpy(
        dtype="float64",
        copy=False,
    )
    fallback_blend = 0.55 * base_prob + 0.20 * confidence + 0.15 * edge_component + 0.10 * history_component
    shrink = float(np.clip(effective_shrink, 0.0, 1.0))
    reranker_prob = np.clip(shrink * model_prob + (1.0 - shrink) * fallback_blend, 0.0, 1.0)
    blend_raw = np.clip(0.70 * reranker_prob + 0.20 * fallback_blend + 0.10 * edge_component, 0.0, 1.0)

    min_reco_rank = {"elite": 3, "strong": 2, "consider": 1, "pass": 0}.get(str(min_candidate_recommendation).strip().lower(), 1)
    candidate_eligible = (
        prepared_selector["meta_prob"].ge(float(min_candidate_expected_win_rate))
        & prepared_selector["meta_confidence"].ge(float(min_candidate_final_confidence))
        & prepared_selector["meta_recommendation_rank"].ge(float(min_reco_rank))
    )
    prob_strength = _soft_floor_strength(prepared_selector["meta_prob"], float(min_candidate_expected_win_rate), 0.08)
    confidence_strength = _soft_floor_strength(prepared_selector["meta_confidence"], float(min_candidate_final_confidence), 0.10)
    reco_strength = ((prepared_selector["meta_recommendation_rank"] - float(min_reco_rank)) / 2.0 + 0.5).clip(0.0, 1.0)
    candidate_strength = (0.45 * prob_strength + 0.35 * confidence_strength + 0.20 * reco_strength).clip(0.0, 1.0)
    eligible_bonus = np.where(candidate_eligible.to_numpy(dtype=bool, copy=False), 0.06, -0.03)
    reranker_prob = np.clip(0.84 * reranker_prob + 0.16 * candidate_strength.to_numpy(dtype="float64", copy=False) + eligible_bonus, 0.0, 1.0)
    blend_raw = np.clip(0.78 * blend_raw + 0.22 * candidate_strength.to_numpy(dtype="float64", copy=False) + eligible_bonus, 0.0, 1.0)

    out["robust_reranker_prob"] = pd.Series(reranker_prob, index=out.index, dtype="float64")
    out["robust_reranker_blend_raw"] = pd.Series(blend_raw, index=out.index, dtype="float64")
    out["robust_reranker_enabled"] = True
    out["robust_reranker_candidate_eligible"] = candidate_eligible.astype(bool)
    out["robust_reranker_candidate_strength"] = candidate_strength.astype("float64")
    out["robust_reranker_train_rows"] = int(len(prepared_history))
    out["robust_reranker_holdout_rows"] = int(len(holdout))
    out["robust_reranker_source"] = str(history_meta.get("source") or "walk_forward_logistic")

    summary: dict[str, Any] = {
        "enabled": True,
        "model_type": "walk_forward_logistic_meta_selector_v1",
        "cutoff_date": str(cutoff_date.date()) if pd.notna(cutoff_date) else "",
        "pre_cutoff_rows": int(len(prepared_history)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "selector_rows": int(len(out)),
        "candidate_eligible_rows": int(candidate_eligible.sum()),
        "candidate_strength_mean": float(candidate_strength.mean()) if len(candidate_strength) else 0.0,
        "feature_count": int(len(gate.feature_names())),
        "source": "walk_forward_logistic",
        "history_source": history_meta,
        "holdout_split": holdout_meta,
        "shrink": shrink_meta,
    }

    if not holdout.empty:
        summary["holdout"] = {
            "rows": int(len(holdout)),
            "mean_prob": float(np.mean(holdout_prob)),
            "mean_label": float(np.mean(holdout_label)),
            "brier": _safe_brier(holdout_prob, holdout_label),
            "log_loss": _safe_log_loss(holdout_prob, holdout_label),
            "baseline_brier": _safe_brier(base_holdout_prob, holdout_label),
            "baseline_log_loss": _safe_log_loss(base_holdout_prob, holdout_label),
        }

    return out, summary
