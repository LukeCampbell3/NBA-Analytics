from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODE_FULL = "A_full_conditional"
MODE_REDUCED = "B_reduced_conditional"
MODE_BASELINE_ONLY = "C_baseline_only"
MODE_SAFE_SHUTDOWN = "D_safe_shutdown"

TARGET_PAIR_BASE_LIFT: dict[tuple[str, str], float] = {
    ("PTS", "PTS"): 0.040,
    ("PTS", "AST"): 0.032,
    ("PTS", "TRB"): 0.014,
    ("AST", "PTS"): 0.030,
    ("AST", "AST"): 0.036,
    ("AST", "TRB"): 0.016,
    ("TRB", "PTS"): 0.012,
    ("TRB", "AST"): 0.018,
    ("TRB", "TRB"): 0.034,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(float(default))
    return pd.Series(float(default), index=df.index, dtype="float64")


def _datetime_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_datetime(df[column], errors="coerce")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value))))


def _logit(probability: float) -> float:
    p = _clip(probability, 1e-6, 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def _to_direction_sign(direction: Any) -> float:
    label = str(direction).upper()
    if label == "OVER":
        return 1.0
    if label == "UNDER":
        return -1.0
    return 0.0


def _category(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "neutral"


def _build_game_key(df: pd.DataFrame) -> pd.Series:
    event = df.get("market_event_id", pd.Series("", index=df.index)).astype(str).str.strip()
    home = df.get("market_home_team", pd.Series("", index=df.index)).astype(str).str.strip()
    away = df.get("market_away_team", pd.Series("", index=df.index)).astype(str).str.strip()
    teams = np.where(home <= away, home + "@" + away, away + "@" + home)
    teams = pd.Series(teams, index=df.index, dtype=str)
    market_date = df.get("market_date", pd.Series("", index=df.index)).astype(str).str.slice(0, 10)
    player = df.get("player", pd.Series("", index=df.index)).astype(str).str.strip()
    target = df.get("target", pd.Series("", index=df.index)).astype(str).str.strip()
    fallback = market_date + "|" + teams
    fallback_player = market_date + "|" + player + "|" + target
    use_player = home.eq("") & away.eq("")
    fallback = pd.Series(np.where(use_player, fallback_player, fallback), index=df.index, dtype=str)
    out = np.where(event.ne("") & event.ne("nan"), event, fallback)
    return pd.Series(out, index=df.index, dtype=str)


@dataclass
class ConditionalPromotionConfig:
    enabled: bool = True
    mode: str = "auto"
    anchor_min_probability: float = 0.57
    anchor_min_confidence: float = 0.05
    anchor_max_risk_penalty: float = 0.62
    min_anchor_count: int = 2
    recoverability_threshold: float = 0.52
    contradiction_threshold: float = 0.62
    noise_threshold: float = 0.68
    conditional_lambda: float = 0.45
    max_conditional_score: float = 0.35
    lift_shrinkage_k: float = 40.0
    min_pair_count: float = 25.0
    min_recent_pair_count: float = 10.0
    min_regime_pair_count: float = 8.0
    min_conditional_support: float = 0.10
    promotion_min_probability: float = 0.56
    promotion_min_ev: float = 0.02
    max_promotions_per_slate: int = 3
    max_promotions_per_game: int = 1
    max_promotions_per_player: int = 1
    max_promotions_per_script_cluster: int = 2
    max_promoted_share_of_recoverable: float = 0.35
    contrastive_weight: float = 0.20
    contrastive_clip: float = 0.08
    market_modifier_strength: float = 0.04
    market_modifier_clip: float = 0.03
    max_failure_memory_penalty: float = 0.06
    recency_half_life_days: float = 35.0
    stale_history_days: int = 21
    min_script_anchors_per_game: int = 1
    kill_switch_failure_rate: float = 0.60
    kill_switch_min_failures: int = 25
    promoted_min_recommendation: str = "consider"
    baseline_min_recommendation: str = "consider"
    failure_memory_path: str = "model/analysis/conditional_failure_memory.json"

    @classmethod
    def from_policy(cls, payload: dict[str, Any] | None) -> "ConditionalPromotionConfig":
        if not payload:
            return cls()
        values = cls().__dict__.copy()
        alias: dict[str, str] = {
            "conditional_framework_enabled": "enabled",
            "conditional_framework_mode": "mode",
            "conditional_anchor_min_probability": "anchor_min_probability",
            "conditional_anchor_min_confidence": "anchor_min_confidence",
            "conditional_anchor_max_risk_penalty": "anchor_max_risk_penalty",
            "conditional_min_anchor_count": "min_anchor_count",
            "conditional_recoverability_threshold": "recoverability_threshold",
            "conditional_contradiction_threshold": "contradiction_threshold",
            "conditional_noise_threshold": "noise_threshold",
            "conditional_lambda": "conditional_lambda",
            "conditional_max_score": "max_conditional_score",
            "conditional_lift_shrinkage_k": "lift_shrinkage_k",
            "conditional_min_pair_count": "min_pair_count",
            "conditional_min_recent_pair_count": "min_recent_pair_count",
            "conditional_min_regime_pair_count": "min_regime_pair_count",
            "conditional_min_support": "min_conditional_support",
            "conditional_promotion_min_probability": "promotion_min_probability",
            "conditional_promotion_min_ev": "promotion_min_ev",
            "conditional_max_promotions_per_slate": "max_promotions_per_slate",
            "conditional_max_promotions_per_game": "max_promotions_per_game",
            "conditional_max_promotions_per_player": "max_promotions_per_player",
            "conditional_max_promotions_per_script_cluster": "max_promotions_per_script_cluster",
            "conditional_max_promoted_share_of_recoverable": "max_promoted_share_of_recoverable",
            "conditional_contrastive_weight": "contrastive_weight",
            "conditional_contrastive_clip": "contrastive_clip",
            "conditional_market_modifier_strength": "market_modifier_strength",
            "conditional_market_modifier_clip": "market_modifier_clip",
            "conditional_max_failure_memory_penalty": "max_failure_memory_penalty",
            "conditional_recency_half_life_days": "recency_half_life_days",
            "conditional_stale_history_days": "stale_history_days",
            "conditional_min_script_anchors_per_game": "min_script_anchors_per_game",
            "conditional_kill_switch_failure_rate": "kill_switch_failure_rate",
            "conditional_kill_switch_min_failures": "kill_switch_min_failures",
            "conditional_promoted_min_recommendation": "promoted_min_recommendation",
            "conditional_baseline_min_recommendation": "baseline_min_recommendation",
            "conditional_failure_memory_path": "failure_memory_path",
        }
        for source_key, target_key in alias.items():
            if source_key in payload and payload[source_key] is not None:
                values[target_key] = payload[source_key]
        return cls(**values)

def _load_failure_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "patterns": {}, "meta": {"attempts": 0, "failures": 0}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "patterns": {}, "meta": {"attempts": 0, "failures": 0}}
    if not isinstance(payload, dict):
        return {"version": 1, "patterns": {}, "meta": {"attempts": 0, "failures": 0}}
    payload.setdefault("version", 1)
    payload.setdefault("patterns", {})
    payload.setdefault("meta", {"attempts": 0, "failures": 0})
    return payload


def _save_failure_memory(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _failure_pattern_key(row: pd.Series, regime: dict[str, Any], script_state: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("target", "")),
            str(row.get("direction", "")),
            str(regime.get("volatility_level", "neutral")),
            str(script_state.get("usage_concentration", "neutral")),
            str(script_state.get("pace_state", "neutral")),
        ]
    )


def _base_prob_calibration(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expected_push_rate"] = _numeric_series(out, "expected_push_rate", 0.0).clip(lower=0.0, upper=1.0)
    non_push = np.clip(1.0 - out["expected_push_rate"], 0.0, 1.0)
    out["base_prob_raw"] = _numeric_series(out, "expected_win_rate", 0.5).clip(lower=0.0, upper=non_push)
    out["posterior_variance"] = _numeric_series(out, "posterior_variance", 0.25).clip(lower=0.0, upper=1.0)
    out["belief_confidence_factor"] = _numeric_series(out, "belief_confidence_factor", 0.5).clip(lower=0.0, upper=1.0)
    out["feasibility"] = _numeric_series(out, "feasibility", 0.5).clip(lower=0.0, upper=1.0)
    out["calibration_weight"] = _numeric_series(out, "calibration_weight", 0.0).clip(lower=0.0, upper=1.0)
    if "final_confidence" in out.columns:
        out["final_confidence"] = _numeric_series(out, "final_confidence", 0.0)
    else:
        out["final_confidence"] = _numeric_series(out, "confidence_score", 0.0)

    variance_term = np.sqrt(out["posterior_variance"]).clip(lower=0.0, upper=1.0)
    strength = (
        0.25
        + 0.35 * out["belief_confidence_factor"]
        + 0.20 * out["feasibility"]
        + 0.20 * out["calibration_weight"]
        - 0.25 * variance_term
    ).clip(lower=0.10, upper=0.95)
    out["base_calibration_strength"] = strength
    out["p_base"] = (
        out["base_calibration_strength"] * out["base_prob_raw"]
        + (1.0 - out["base_calibration_strength"]) * 0.5 * non_push
    ).clip(lower=0.0, upper=non_push)
    out["s_base"] = out["p_base"].map(_logit)
    return out


def _detect_regime_state(df: pd.DataFrame, cfg: ConditionalPromotionConfig) -> dict[str, Any]:
    market_dates = _datetime_series(df, "market_date")
    anchor_date = market_dates.max() if market_dates.notna().any() else pd.Timestamp.now(tz=timezone.utc)
    history_dates = _datetime_series(df, "last_history_date")
    if history_dates.notna().any():
        history_age_days = (anchor_date - history_dates).dt.days.astype("float64")
        stale_share = float((history_age_days > float(cfg.stale_history_days)).fillna(True).mean())
    else:
        stale_share = 1.0
    feasibility = _numeric_series(df, "feasibility", 0.5).clip(lower=0.0, upper=1.0)
    volatility = _numeric_series(df, "volatility_score", 0.5).clip(lower=0.0, upper=1.0)
    injury_pressure = float(np.clip(1.0 - feasibility.mean(), 0.0, 1.0))
    minutes_stability = float(np.clip(feasibility.mean(), 0.0, 1.0))
    volatility_level = float(np.clip(volatility.mean(), 0.0, 1.0))
    spread_proxy = float(np.clip(_numeric_series(df, "abs_edge", 0.0).std(ddof=0) / 3.0, 0.0, 1.0))
    return {
        "injury_pressure": injury_pressure,
        "minutes_stability_score": minutes_stability,
        "volatility_score": volatility_level,
        "spread_proxy_score": spread_proxy,
        "stale_history_share": stale_share,
        "injury_level": _category(injury_pressure, 0.35, 0.60),
        "minutes_stability_level": _category(minutes_stability, 0.45, 0.70),
        "volatility_level": _category(volatility_level, 0.35, 0.62),
        "spread_profile": _category(spread_proxy, 0.30, 0.58),
    }


def _infer_script_states(df: pd.DataFrame, anchor_mask: pd.Series, cfg: ConditionalPromotionConfig) -> dict[str, dict[str, Any]]:
    scripts: dict[str, dict[str, Any]] = {}
    anchors = df.loc[anchor_mask].copy()
    if anchors.empty:
        return scripts
    for game_key, game in anchors.groupby("game_key"):
        if len(game) < int(cfg.min_script_anchors_per_game):
            continue
        player_counts = game["player"].astype(str).value_counts()
        top_share = float(player_counts.iloc[0] / max(1, len(game))) if not player_counts.empty else 0.0
        usage = _category(top_share, 0.35, 0.60)

        pace_slice = game.loc[game["target"].isin(["PTS", "AST"])].copy()
        pace_bias = float(pace_slice["direction_sign"].mean()) if not pace_slice.empty else 0.0
        if pace_bias >= 0.20:
            pace = "high"
        elif pace_bias <= -0.20:
            pace = "low"
        else:
            pace = "neutral"

        minutes_mean = float(_numeric_series(game, "feasibility", 0.5).mean())
        minutes = "stable" if minutes_mean >= 0.65 else "unstable" if minutes_mean <= 0.45 else "neutral"

        trb_slice = game.loc[game["target"] == "TRB"].copy()
        trb_counts = trb_slice["player"].astype(str).value_counts()
        trb_share = float(trb_counts.iloc[0] / max(1, len(trb_slice))) if not trb_counts.empty else 0.0
        if trb_share >= 0.60:
            rebound = "concentrated"
        elif trb_share <= 0.34 and len(trb_slice) >= 2:
            rebound = "distributed"
        else:
            rebound = "neutral"

        cluster = "|".join(
            [
                f"usage={usage}",
                f"pace={pace}",
                f"minutes={minutes}",
                f"rebound={rebound}",
            ]
        )
        scripts[str(game_key)] = {
            "usage_concentration": usage,
            "pace_state": pace,
            "minutes_state": minutes,
            "rebound_distribution": rebound,
            "cluster_id": cluster,
            "anchor_count": int(len(game)),
            "top_player_share": top_share,
        }
    return scripts


def _script_compatibility(row: pd.Series, script_state: dict[str, Any]) -> float:
    if not script_state:
        return 0.50
    score = 0.50
    target = str(row.get("target", "")).upper()
    direction = str(row.get("direction", "")).upper()
    feasibility = _clip(_safe_float(row.get("feasibility", 0.5)), 0.0, 1.0)
    if script_state.get("pace_state") == "high" and target in {"PTS", "AST"}:
        score += 0.10 if direction == "OVER" else -0.10
    elif script_state.get("pace_state") == "low" and target in {"PTS", "AST"}:
        score += 0.10 if direction == "UNDER" else -0.10
    if script_state.get("usage_concentration") == "high" and target == "PTS":
        score += 0.06 if direction == "OVER" else -0.06
    if script_state.get("rebound_distribution") == "concentrated" and target == "TRB":
        score += 0.07 if direction == "OVER" else -0.07
    if script_state.get("minutes_state") == "stable":
        score += 0.08 * feasibility
    if script_state.get("minutes_state") == "unstable":
        score -= 0.08 * (1.0 - feasibility)
    return _clip(score, 0.0, 1.0)


def _lift_base(candidate_target: str, anchor_target: str) -> float:
    return float(TARGET_PAIR_BASE_LIFT.get((str(candidate_target), str(anchor_target)), 0.01))


def _resolve_mode(
    df: pd.DataFrame,
    cfg: ConditionalPromotionConfig,
    anchor_count: int,
    regime_state: dict[str, Any],
    memory_payload: dict[str, Any],
) -> tuple[str, list[str], bool]:
    reasons: list[str] = []
    if df.empty:
        return MODE_SAFE_SHUTDOWN, ["empty_selector"], True
    required = {"expected_win_rate", "prediction", "market_line"}
    if not required.issubset(df.columns):
        return MODE_SAFE_SHUTDOWN, ["missing_required_columns"], True
    if not bool(cfg.enabled):
        return MODE_BASELINE_ONLY, ["framework_disabled"], False

    mode_raw = str(cfg.mode).strip().lower()
    if mode_raw in {"baseline", "baseline_only", "off"}:
        return MODE_BASELINE_ONLY, ["forced_baseline_only"], False
    if mode_raw in {"shutdown", "safe_shutdown"}:
        return MODE_SAFE_SHUTDOWN, ["forced_safe_shutdown"], True

    meta = memory_payload.get("meta", {}) if isinstance(memory_payload, dict) else {}
    failures = int(meta.get("failures", 0))
    attempts = int(meta.get("attempts", 0))
    failure_rate = float(failures / attempts) if attempts > 0 else 0.0
    if failures >= int(cfg.kill_switch_min_failures) and failure_rate >= float(cfg.kill_switch_failure_rate):
        reasons.append("kill_switch_failure_memory")
        return MODE_BASELINE_ONLY, reasons, True

    stale_share = float(regime_state.get("stale_history_share", 1.0))
    if anchor_count < int(cfg.min_anchor_count):
        reasons.append("anchor_shortage")
        return MODE_BASELINE_ONLY, reasons, False
    if stale_share >= 0.85:
        reasons.append("stale_history")
        return MODE_BASELINE_ONLY, reasons, False
    if stale_share >= 0.55:
        reasons.append("reduced_due_to_stale_coverage")
        return MODE_REDUCED, reasons, False
    if mode_raw == "reduced":
        reasons.append("forced_reduced")
        return MODE_REDUCED, reasons, False
    if mode_raw == "full":
        reasons.append("forced_full")
        return MODE_FULL, reasons, False
    return MODE_FULL, reasons, False

def apply_conditional_promotion(
    selector_df: pd.DataFrame,
    policy_payload: dict[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
    american_odds: int = -110,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = ConditionalPromotionConfig.from_policy(policy_payload)
    out = selector_df.copy()
    if out.empty:
        return out, {"fallback_mode": MODE_SAFE_SHUTDOWN, "reason": "empty_selector"}
    if "target" not in out.columns:
        out["target"] = "UNKNOWN"
    if "direction" not in out.columns:
        out["direction"] = "PUSH"
    if "player" not in out.columns:
        out["player"] = ""

    payout = float(american_odds / 100.0) if int(american_odds) > 0 else float(100.0 / max(1, abs(int(american_odds))))
    out["game_key"] = _build_game_key(out)
    out["direction_sign"] = out.get("direction", pd.Series("", index=out.index)).map(_to_direction_sign)
    out["history_rows"] = _numeric_series(out, "history_rows", 0.0).clip(lower=0.0)
    out["risk_penalty"] = _numeric_series(out, "risk_penalty", 0.5).clip(lower=0.0, upper=1.0)
    out["market_books"] = _numeric_series(out, "market_books", np.nan)
    books_max = float(out["market_books"].max()) if out["market_books"].notna().any() else 1.0
    if not np.isfinite(books_max) or books_max <= 0.0:
        books_max = 1.0
    out["market_books_norm"] = (out["market_books"].fillna(0.0) / books_max).clip(lower=0.0, upper=1.0)
    out["abs_edge"] = _numeric_series(out, "abs_edge", 0.0).clip(lower=0.0)
    out["abs_edge_pct"] = out.groupby("target")["abs_edge"].rank(method="average", pct=True).fillna(0.0)
    out = _base_prob_calibration(out)

    out["recommendation"] = out.get("recommendation", pd.Series("pass", index=out.index)).astype(str).str.lower()
    is_strong_label = out["recommendation"].isin({"elite", "strong"})
    anchor_mask = (
        is_strong_label
        & (out["p_base"] >= float(cfg.anchor_min_probability))
        & (_numeric_series(out, "final_confidence", 0.0) >= float(cfg.anchor_min_confidence))
        & (out["risk_penalty"] <= float(cfg.anchor_max_risk_penalty))
    )
    out["is_anchor"] = anchor_mask

    regime_state = _detect_regime_state(out, cfg)
    scripts = _infer_script_states(out, anchor_mask, cfg)
    out["script_cluster_id"] = out["game_key"].map(lambda key: scripts.get(str(key), {}).get("cluster_id", "script=unknown"))
    out["script_compatibility"] = out.apply(
        lambda row: _script_compatibility(row, scripts.get(str(row.get("game_key")), {})),
        axis=1,
    )

    failure_memory_path = Path(str(cfg.failure_memory_path))
    memory = _load_failure_memory(failure_memory_path)
    fallback_mode, fallback_reasons, kill_switch_triggered = _resolve_mode(
        out,
        cfg,
        int(anchor_mask.sum()),
        regime_state,
        memory,
    )

    out["recoverability_score"] = 0.0
    out["contradiction_score"] = 0.0
    out["noise_score"] = 1.0
    out["conditional_support"] = 0.0
    out["conditional_score"] = 0.0
    out["conditional_score_clipped"] = 0.0
    out["conditional_failure_penalty"] = 0.0
    out["p_final"] = out["p_base"]
    out["s_final"] = out["s_base"]
    out["ev_final"] = _numeric_series(out, "ev", 0.0)
    out["weak_bucket"] = "K"
    out["conditional_promoted"] = False
    out["conditional_eligible_for_board"] = True
    out["decision_tier"] = np.where(out["is_anchor"], "Tier A - Baseline", "Tier C - Baseline Pool")
    out["promotion_reason_codes"] = ""
    out["conditional_audit_summary"] = np.where(
        out["is_anchor"],
        "Kept as baseline anchor (strong calibrated base signal).",
        "Not evaluated for conditional promotion.",
    )

    if fallback_mode == MODE_SAFE_SHUTDOWN:
        out["conditional_eligible_for_board"] = False
        out["decision_tier"] = "Tier D - Safe Shutdown"
        out["conditional_audit_summary"] = "No action: required baseline integrity checks failed."
        summary = {
            "fallback_mode": fallback_mode,
            "fallback_reasons": fallback_reasons,
            "kill_switch_triggered": bool(kill_switch_triggered),
            "regime_state": regime_state,
            "script_state": scripts,
            "anchor_count": int(anchor_mask.sum()),
            "bucket_counts": {},
            "promotions_selected": 0,
            "promotions_candidates": 0,
            "diagnostics_rows": int(len(out)),
            "failure_memory_path": str(failure_memory_path),
        }
        return out, summary

    direction_bias = (
        out.loc[out["is_anchor"]]
        .groupby(["game_key", "target"])["direction_sign"]
        .mean()
        .to_dict()
        if bool(anchor_mask.any())
        else {}
    )
    recency_days = (_datetime_series(out, "market_date") - _datetime_series(out, "last_history_date")).dt.days.astype("float64")
    out["recency_factor"] = np.exp(-np.maximum(0.0, recency_days.fillna(365.0)) / max(1e-6, float(cfg.recency_half_life_days)))
    out["recency_factor"] = out["recency_factor"].clip(lower=0.0, upper=1.0)

    candidate_mask = ~out["is_anchor"]
    if candidate_mask.any():
        uncertainty = np.sqrt(_numeric_series(out, "posterior_variance", 0.25)).clip(lower=0.0, upper=1.0)
        recoverability = (
            0.28 * out["abs_edge_pct"]
            + 0.20 * out["script_compatibility"]
            + 0.20 * out["belief_confidence_factor"]
            + 0.18 * out["feasibility"]
            + 0.14 * (1.0 - uncertainty)
        ).clip(lower=0.0, upper=1.0)
        contradiction_parts: list[pd.Series] = []
        for idx, row in out.iterrows():
            bias = direction_bias.get((str(row.get("game_key")), str(row.get("target"))), 0.0)
            mismatch = 0.0
            sign = float(row.get("direction_sign", 0.0))
            if abs(bias) > 0.05 and sign != 0.0 and np.sign(bias) != np.sign(sign):
                mismatch = min(1.0, abs(bias))
            script_penalty = max(0.0, 0.55 - float(row.get("script_compatibility", 0.5))) / 0.55
            contradiction_value = _clip(
                0.40 * mismatch
                + 0.35 * script_penalty
                + 0.15 * float(row.get("risk_penalty", 0.5))
                + 0.10 * (1.0 - float(row.get("market_books_norm", 0.0))),
                0.0,
                1.0,
            )
            contradiction_parts.append(pd.Series(contradiction_value, index=[idx]))
        contradiction = pd.concat(contradiction_parts).sort_index() if contradiction_parts else pd.Series(0.0, index=out.index)
        noise = (
            0.36 * (1.0 - out["abs_edge_pct"])
            + 0.24 * out["risk_penalty"]
            + 0.22 * uncertainty
            + 0.18 * (1.0 - out["feasibility"])
        ).clip(lower=0.0, upper=1.0)

        out.loc[candidate_mask, "recoverability_score"] = recoverability.loc[candidate_mask]
        out.loc[candidate_mask, "contradiction_score"] = contradiction.loc[candidate_mask]
        out.loc[candidate_mask, "noise_score"] = noise.loc[candidate_mask]

        out.loc[candidate_mask, "weak_bucket"] = "U_n"
        uc_mask = candidate_mask & (out["contradiction_score"] >= float(cfg.contradiction_threshold))
        out.loc[uc_mask, "weak_bucket"] = "U_c"
        ur_mask = (
            candidate_mask
            & (out["recoverability_score"] >= float(cfg.recoverability_threshold))
            & (out["noise_score"] <= float(cfg.noise_threshold))
            & (~uc_mask)
        )
        out.loc[ur_mask, "weak_bucket"] = "U_r"

    promotions_candidates = out.loc[out["weak_bucket"] == "U_r"].copy()
    promoted_rows: list[int] = []
    rejected_due_budget = 0

    if fallback_mode in {MODE_FULL, MODE_REDUCED} and not promotions_candidates.empty:
        anchors = out.loc[out["is_anchor"]].copy()
        promotion_payload: list[dict[str, Any]] = []
        for idx, row in promotions_candidates.iterrows():
            same_game_anchors = anchors.loc[anchors["game_key"] == row["game_key"]].copy()
            if same_game_anchors.empty:
                out.at[idx, "promotion_reason_codes"] = "no_anchor_support"
                out.at[idx, "conditional_audit_summary"] = "Not promoted: recoverable but no same-game anchors were available."
                continue

            contributions: list[float] = []
            reliabilities: list[float] = []
            eligible_pairs = 0
            for _, anchor in same_game_anchors.iterrows():
                support_total = min(float(row.get("history_rows", 0.0)), float(anchor.get("history_rows", 0.0)))
                recent_support = support_total * min(float(row.get("recency_factor", 0.0)), float(anchor.get("recency_factor", 0.0)))
                regime_support = support_total * (1.0 - abs(float(row.get("volatility_score", 0.5)) - float(regime_state.get("volatility_score", 0.5))))
                if (
                    support_total < float(cfg.min_pair_count)
                    or recent_support < float(cfg.min_recent_pair_count)
                    or regime_support < float(cfg.min_regime_pair_count)
                ):
                    continue

                direction_multiplier = 1.0 if str(row.get("direction")) == str(anchor.get("direction")) else -1.0
                raw_lift = _lift_base(str(row.get("target")), str(anchor.get("target"))) * direction_multiplier
                if str(row.get("player")) == str(anchor.get("player")):
                    raw_lift += 0.012 if direction_multiplier > 0 else -0.018
                shrink = support_total / (support_total + float(cfg.lift_shrinkage_k))
                shrunk_lift = shrink * raw_lift

                support_factor = support_total / (support_total + 25.0)
                recency_factor = min(float(row.get("recency_factor", 0.0)), float(anchor.get("recency_factor", 0.0)))
                regime_factor = _clip(1.0 - abs(float(row.get("volatility_score", 0.5)) - float(regime_state.get("volatility_score", 0.5))), 0.0, 1.0)
                anchor_factor = _clip((float(anchor.get("p_base", 0.5)) - 0.5) * 2.0, 0.0, 1.0)
                market_factor = _clip(
                    0.40 + 0.60 * ((float(row.get("market_books_norm", 0.0)) + float(anchor.get("market_books_norm", 0.0))) * 0.5),
                    0.0,
                    1.0,
                )
                reliability = support_factor * recency_factor * regime_factor * anchor_factor * market_factor
                contributions.append(float(shrunk_lift * reliability))
                reliabilities.append(float(reliability))
                eligible_pairs += 1

            if eligible_pairs <= 0:
                out.at[idx, "promotion_reason_codes"] = "support_floor_failed"
                out.at[idx, "conditional_audit_summary"] = "Not promoted: anchor links failed support/recent/regime eligibility floors."
                continue

            conditional_core = float(np.sum(contributions))
            conditional_support = float(np.mean(reliabilities))
            contrastive = 0.0
            market_modifier = 0.0
            if fallback_mode == MODE_FULL:
                competitors = promotions_candidates.loc[
                    (promotions_candidates.index != idx)
                    & (promotions_candidates["game_key"] == row["game_key"])
                    & (promotions_candidates["player"] == row["player"])
                ]
                if not competitors.empty:
                    competitor_best = float(competitors["p_base"].max())
                    contrastive_raw = float(row.get("p_base", 0.5)) - competitor_best
                    contrastive = _clip(contrastive_raw, -float(cfg.contrastive_clip), float(cfg.contrastive_clip)) * float(cfg.contrastive_weight)
                market_modifier = _clip(
                    (float(row.get("market_books_norm", 0.0)) - 0.5) * float(cfg.market_modifier_strength)
                    - 0.02 * float(row.get("risk_penalty", 0.5)),
                    -float(cfg.market_modifier_clip),
                    float(cfg.market_modifier_clip),
                )

            pattern_key = _failure_pattern_key(row, regime_state, scripts.get(str(row.get("game_key")), {}))
            pattern_entry = memory.get("patterns", {}).get(pattern_key, {})
            pattern_rate = _clip(_safe_float(pattern_entry.get("ema_failure_rate"), 0.0), 0.0, 1.0)
            pattern_weight = _clip(_safe_float(pattern_entry.get("penalty_weight"), 0.08), 0.0, 0.40)
            failure_penalty = _clip(pattern_rate * pattern_weight, 0.0, float(cfg.max_failure_memory_penalty))

            s_cond_raw = conditional_core + contrastive + market_modifier - failure_penalty
            s_cond = _clip(s_cond_raw, -float(cfg.max_conditional_score), float(cfg.max_conditional_score))
            s_final = float(row.get("s_base", _logit(float(row.get("p_base", 0.5))))) + float(cfg.conditional_lambda) * s_cond
            non_push = _clip(1.0 - float(row.get("expected_push_rate", 0.0)), 0.0, 1.0)
            p_raw = _clip(_sigmoid(s_final), 0.0, non_push)
            final_blend = _clip(
                0.20 + 0.45 * conditional_support + 0.20 * float(row.get("belief_confidence_factor", 0.5)),
                0.15,
                0.95,
            )
            p_final = _clip(final_blend * p_raw + (1.0 - final_blend) * float(row.get("p_base", 0.5)), 0.0, non_push)
            loss = _clip(non_push - p_final, 0.0, 1.0)
            ev_final = float(p_final * payout - loss)

            out.at[idx, "conditional_support"] = conditional_support
            out.at[idx, "conditional_score"] = s_cond_raw
            out.at[idx, "conditional_score_clipped"] = s_cond
            out.at[idx, "conditional_failure_penalty"] = failure_penalty
            out.at[idx, "s_final"] = s_final
            out.at[idx, "p_final"] = p_final
            out.at[idx, "ev_final"] = ev_final

            eligible = (
                conditional_support >= float(cfg.min_conditional_support)
                and float(row.get("contradiction_score", 1.0)) < float(cfg.contradiction_threshold)
                and (
                    p_final >= float(cfg.promotion_min_probability)
                    or ev_final >= float(cfg.promotion_min_ev)
                )
            )
            reason_codes = []
            if conditional_support < float(cfg.min_conditional_support):
                reason_codes.append("support_below_threshold")
            if float(row.get("contradiction_score", 1.0)) >= float(cfg.contradiction_threshold):
                reason_codes.append("contradiction_too_high")
            if p_final < float(cfg.promotion_min_probability) and ev_final < float(cfg.promotion_min_ev):
                reason_codes.append("prob_ev_threshold_not_met")
            if eligible:
                reason_codes.append("eligible")

            promotion_payload.append(
                {
                    "index": int(idx),
                    "eligible": bool(eligible),
                    "game_key": str(row.get("game_key")),
                    "player": str(row.get("player")),
                    "script_cluster_id": str(out.at[idx, "script_cluster_id"]),
                    "ev_final": ev_final,
                    "p_final": p_final,
                    "recoverability_score": float(row.get("recoverability_score", 0.0)),
                    "conditional_support": conditional_support,
                    "reason_codes": reason_codes,
                }
            )

        promotion_payload = sorted(
            promotion_payload,
            key=lambda item: (
                not bool(item["eligible"]),
                -float(item["ev_final"]),
                -float(item["p_final"]),
                -float(item["recoverability_score"]),
                -float(item["conditional_support"]),
            ),
        )
        player_counts: dict[str, int] = {}
        game_counts: dict[str, int] = {}
        script_counts: dict[str, int] = {}
        recoverable_count = int((out["weak_bucket"] == "U_r").sum())
        share_cap = int(math.floor(max(0.0, float(cfg.max_promoted_share_of_recoverable)) * recoverable_count))
        if recoverable_count > 0 and share_cap <= 0:
            share_cap = 1
        slate_cap = int(cfg.max_promotions_per_slate)
        if slate_cap > 0:
            share_cap = min(share_cap, slate_cap) if share_cap > 0 else slate_cap

        for item in promotion_payload:
            idx = int(item["index"])
            if not bool(item["eligible"]):
                out.at[idx, "promotion_reason_codes"] = "|".join(item["reason_codes"])
                out.at[idx, "conditional_audit_summary"] = (
                    "Not promoted: recoverable candidate failed conditional support/probability gates."
                )
                continue
            if share_cap > 0 and len(promoted_rows) >= share_cap:
                rejected_due_budget += 1
                out.at[idx, "promotion_reason_codes"] = "budget_share_cap"
                out.at[idx, "conditional_audit_summary"] = "Not promoted: promotion budget cap reached."
                continue
            player = str(item["player"])
            game = str(item["game_key"])
            script_cluster = str(item["script_cluster_id"])
            if int(cfg.max_promotions_per_player) > 0 and player_counts.get(player, 0) >= int(cfg.max_promotions_per_player):
                rejected_due_budget += 1
                out.at[idx, "promotion_reason_codes"] = "budget_player_cap"
                out.at[idx, "conditional_audit_summary"] = "Not promoted: player-level promotion cap reached."
                continue
            if int(cfg.max_promotions_per_game) > 0 and game_counts.get(game, 0) >= int(cfg.max_promotions_per_game):
                rejected_due_budget += 1
                out.at[idx, "promotion_reason_codes"] = "budget_game_cap"
                out.at[idx, "conditional_audit_summary"] = "Not promoted: game-level promotion cap reached."
                continue
            if int(cfg.max_promotions_per_script_cluster) > 0 and script_counts.get(script_cluster, 0) >= int(cfg.max_promotions_per_script_cluster):
                rejected_due_budget += 1
                out.at[idx, "promotion_reason_codes"] = "budget_script_cluster_cap"
                out.at[idx, "conditional_audit_summary"] = "Not promoted: script-cluster promotion cap reached."
                continue

            promoted_rows.append(idx)
            player_counts[player] = player_counts.get(player, 0) + 1
            game_counts[game] = game_counts.get(game, 0) + 1
            script_counts[script_cluster] = script_counts.get(script_cluster, 0) + 1
            out.at[idx, "conditional_promoted"] = True
            out.at[idx, "decision_tier"] = "Tier B - Conditional Promotion"
            out.at[idx, "promotion_reason_codes"] = "promoted"
            out.at[idx, "conditional_audit_summary"] = (
                "Promoted: near-threshold base edge became actionable with coherent anchor/script support and low contradiction."
            )

    if fallback_mode in {MODE_FULL, MODE_REDUCED}:
        out["conditional_eligible_for_board"] = out["is_anchor"] | out["conditional_promoted"]
        demoted = (~out["conditional_eligible_for_board"]) & (~out["is_anchor"])
        out.loc[demoted & (out["weak_bucket"] == "U_r"), "decision_tier"] = "Tier C - Recoverable Rejected"
        out.loc[demoted & (out["weak_bucket"] == "U_n"), "decision_tier"] = "Tier D - Noisy Rejected"
        out.loc[demoted & (out["weak_bucket"] == "U_c"), "decision_tier"] = "Tier D - Contradicted Rejected"
        out.loc[demoted & out["conditional_audit_summary"].eq("Not evaluated for conditional promotion."), "conditional_audit_summary"] = (
            "Not promoted: weak candidate did not qualify for recoverability-first conditional review."
        )
    else:
        out["conditional_eligible_for_board"] = True
        out.loc[out["is_anchor"], "decision_tier"] = "Tier A - Baseline"
        out.loc[~out["is_anchor"], "decision_tier"] = "Tier A - Baseline"
        out.loc[~out["is_anchor"], "conditional_audit_summary"] = "Conditional layer bypassed; using baseline-only operation."

    promoted_min_label = str(cfg.promoted_min_recommendation).lower()
    if promoted_min_label in {"consider", "strong", "elite"} and out["conditional_promoted"].any():
        order = {"pass": 0, "consider": 1, "strong": 2, "elite": 3}
        minimum = int(order.get(promoted_min_label, 1))
        mask = out["conditional_promoted"]
        upgraded = []
        for label in out.loc[mask, "recommendation"].astype(str).str.lower():
            if order.get(label, 0) >= minimum:
                upgraded.append(label)
            else:
                upgraded.append(promoted_min_label)
        out.loc[mask, "recommendation"] = upgraded

    if "p_final" in out.columns:
        p_final = _numeric_series(out, "p_final", 0.5)
    else:
        p_final = _numeric_series(out, "p_base", 0.5)
    push = _numeric_series(out, "expected_push_rate", 0.0).clip(lower=0.0, upper=1.0)
    non_push = np.clip(1.0 - push, 0.0, 1.0)
    p_final = p_final.clip(lower=0.0, upper=non_push)
    out["p_final"] = p_final
    out["expected_win_rate_pre_conditional"] = _numeric_series(out, "expected_win_rate", 0.5)
    out["expected_win_rate"] = p_final
    out["expected_loss_rate"] = np.clip(non_push - out["expected_win_rate"], 0.0, 1.0)
    out["ev"] = out["expected_win_rate"] * payout - out["expected_loss_rate"]
    if "ev_final" in out.columns:
        out["ev_final"] = _numeric_series(out, "ev_final", 0.0)
    else:
        out["ev_final"] = _numeric_series(out, "ev", 0.0)

    now_iso = datetime.now(timezone.utc).isoformat()
    patterns = memory.setdefault("patterns", {})
    meta = memory.setdefault("meta", {"attempts": 0, "failures": 0})
    for idx in promoted_rows:
        row = out.loc[idx]
        key = _failure_pattern_key(row, regime_state, scripts.get(str(row.get("game_key")), {}))
        entry = patterns.setdefault(
            key,
            {
                "attempts": 0,
                "failures": 0,
                "ema_failure_rate": 0.0,
                "penalty_weight": 0.08,
                "last_seen_utc": None,
            },
        )
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["last_seen_utc"] = now_iso
        # Realized outcomes are not available here; keep failure rates slow and bounded.
        ema = _clip(_safe_float(entry.get("ema_failure_rate"), 0.0), 0.0, 1.0)
        entry["ema_failure_rate"] = _clip(0.995 * ema, 0.0, 1.0)
        meta["attempts"] = int(meta.get("attempts", 0)) + 1
    memory["last_updated_utc"] = now_iso
    _save_failure_memory(failure_memory_path, memory)

    out = out.sort_values(
        ["conditional_eligible_for_board", "is_anchor", "conditional_promoted", "ev_final", "p_final", "recoverability_score"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    summary = {
        "fallback_mode": fallback_mode,
        "fallback_reasons": fallback_reasons,
        "kill_switch_triggered": bool(kill_switch_triggered),
        "regime_state": regime_state,
        "script_state": scripts,
        "anchor_count": int(anchor_mask.sum()),
        "bucket_counts": out["weak_bucket"].value_counts().to_dict(),
        "promotions_candidates": int((out["weak_bucket"] == "U_r").sum()),
        "promotions_selected": int(out["conditional_promoted"].sum()),
        "promotions_rejected_budget": int(rejected_due_budget),
        "eligible_for_board_rows": int(out["conditional_eligible_for_board"].sum()),
        "failure_memory_path": str(failure_memory_path),
        "history_rows": int(len(history_df)) if isinstance(history_df, pd.DataFrame) else 0,
        "tier_counts": out["decision_tier"].value_counts().to_dict(),
    }
    return out, summary
