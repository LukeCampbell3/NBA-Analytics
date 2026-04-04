#!/usr/bin/env python3
"""
Post-process ranked market plays into a final actionable board.

This layer is intentionally separate from selection so we can:
- compute EV from expected win rate
- de-correlate by player
- filter to positive-EV / minimum-quality plays
- produce a tighter, final board for execution
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decision_engine.sizing import apply_tiered_bet_sizing
try:
    from decision_engine.selected_board_calibration import apply_selected_board_calibration as apply_selected_board_calibration_fn
except Exception:  # pragma: no cover - fallback when calibrator module is unavailable
    def apply_selected_board_calibration_fn(
        frame: pd.DataFrame,
        payload: dict | None,
        run_date_hint: str | None = None,
        prob_col: str = "board_play_win_prob",
        target_col: str = "target",
        direction_col: str = "direction",
    ) -> tuple[pd.Series, pd.Series, str]:
        probs = pd.to_numeric(frame.get(prob_col), errors="coerce").fillna(0.5).clip(lower=0.01, upper=0.99)
        return probs.astype("float64"), pd.Series("identity_module_missing", index=frame.index, dtype="object"), ""

try:
    from decision_engine.learned_pool_gate import apply_learned_pool_gate as apply_learned_pool_gate_fn
except Exception:  # pragma: no cover - fallback when gate module is unavailable
    def apply_learned_pool_gate_fn(
        frame: pd.DataFrame,
        payload: dict | None,
        run_date_hint: str | None = None,
        prob_col: str = "expected_win_rate",
        target_col: str = "target",
        direction_col: str = "direction",
    ) -> tuple[pd.Series, pd.Series, pd.Series, str, dict]:
        index = frame.index
        pass_mask = pd.Series(True, index=index, dtype=bool)
        thresholds = pd.Series(float("-inf"), index=index, dtype="float64")
        source = pd.Series("identity_module_missing", index=index, dtype="object")
        return pass_mask, thresholds, source, "", {"enabled": False, "reason": "module_missing"}

try:
    from decision_engine.staking_bucket_model_v2 import apply_staking_bucket_model_v2 as apply_staking_bucket_model_v2_fn
except Exception:  # pragma: no cover - fallback when staking model module is unavailable
    def apply_staking_bucket_model_v2_fn(
        frame: pd.DataFrame,
        payload: dict | None,
        run_date_hint: str | None = None,
        prob_col: str = "p_calibrated",
        payout_per_unit: float = (100.0 / 110.0),
        min_train_rows: int = 0,
    ) -> tuple[pd.Series, pd.Series, str, dict]:
        probs = pd.to_numeric(frame.get(prob_col), errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
        source = pd.Series("identity_module_missing", index=frame.index, dtype="object")
        return probs.astype("float64"), source, "", {"enabled": False, "reason": "module_missing"}

try:
    from decision_engine.uncertainty import (
        BELIEF_UNCERTAINTY_LOWER,
        BELIEF_UNCERTAINTY_UPPER,
        belief_confidence_factor,
        normalize_belief_uncertainty,
    )
except Exception:  # pragma: no cover - fallback for standalone execution
    BELIEF_UNCERTAINTY_LOWER = 0.75
    BELIEF_UNCERTAINTY_UPPER = 1.15

    def normalize_belief_uncertainty(value, default: float = 1.0, lower: float = BELIEF_UNCERTAINTY_LOWER, upper: float = BELIEF_UNCERTAINTY_UPPER):
        span = max(float(upper) - float(lower), 1e-9)
        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce").fillna(float(default))
            return ((numeric - float(lower)) / span).clip(lower=0.0, upper=1.0)
        try:
            numeric = float(value)
            if np.isnan(numeric):
                numeric = float(default)
        except Exception:
            numeric = float(default)
        return float(np.clip((numeric - float(lower)) / span, 0.0, 1.0))

    def belief_confidence_factor(value, default: float = 1.0, lower: float = BELIEF_UNCERTAINTY_LOWER, upper: float = BELIEF_UNCERTAINTY_UPPER):
        normalized = normalize_belief_uncertainty(value, default=default, lower=lower, upper=upper)
        if isinstance(normalized, pd.Series):
            return (1.0 - normalized).clip(lower=0.0, upper=1.0)
        return float(np.clip(1.0 - float(normalized), 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process selected market plays into a final board.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("model/analysis/upcoming_market_play_selector.csv"),
        help="Selector CSV from select_market_plays.py",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("model/analysis/final_market_plays.csv"),
        help="Output CSV path for the final board.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("model/analysis/final_market_plays.json"),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--american-odds",
        type=int,
        default=-110,
        help="Assumed book odds for EV calculation when actual odds are unavailable.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.0,
        help="Minimum expected value required to keep a play.",
    )
    parser.add_argument(
        "--min-final-confidence",
        type=float,
        default=0.03,
        help="Minimum final confidence required to keep a play.",
    )
    parser.add_argument(
        "--min-recommendation",
        type=str,
        default="consider",
        choices=["pass", "consider", "strong", "elite"],
        help="Lowest selector recommendation allowed into the final board.",
    )
    parser.add_argument(
        "--max-plays-per-player",
        type=int,
        default=1,
        help="Maximum number of plays to keep per player after ranking.",
    )
    parser.add_argument(
        "--max-plays-per-target",
        type=int,
        default=0,
        help="Maximum number of final plays to keep per target when target-specific caps are not supplied.",
    )
    parser.add_argument(
        "--max-pts-plays",
        type=int,
        default=6,
        help="Maximum final PTS plays to keep.",
    )
    parser.add_argument(
        "--max-trb-plays",
        type=int,
        default=4,
        help="Maximum final TRB plays to keep.",
    )
    parser.add_argument(
        "--max-ast-plays",
        type=int,
        default=2,
        help="Maximum final AST plays to keep.",
    )
    parser.add_argument(
        "--max-total-plays",
        type=int,
        default=12,
        help="Maximum number of final plays to keep overall.",
    )
    parser.add_argument(
        "--min-board-plays",
        type=int,
        default=0,
        help="Minimum final-board rows to target by dynamically relaxing EV cutoff when possible.",
    )
    parser.add_argument(
        "--non-pts-min-gap-percentile",
        type=float,
        default=0.90,
        help="Minimum disagreement percentile required for TRB/AST plays.",
    )
    parser.add_argument(
        "--edge-adjust-k",
        type=float,
        default=0.30,
        help="Weight for edge-adjusted EV ranking.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="thompson_ev",
        choices=[
            "ev_adjusted",
            "edge",
            "abs_edge",
            "xgb_ltr",
            "robust_reranker",
            "thompson_ev",
            "set_theory",
            "edge_append_shadow",
            "board_objective",
        ],
        help="Final ranking mode used before portfolio constraints are applied.",
    )
    parser.add_argument(
        "--max-plays-per-game",
        type=int,
        default=2,
        help="Maximum selected plays from the same game/event to limit correlation.",
    )
    parser.add_argument(
        "--max-plays-per-script-cluster",
        type=int,
        default=2,
        help="Maximum selected plays from the same inferred script cluster.",
    )
    parser.add_argument(
        "--max-history-staleness-days",
        type=int,
        default=0,
        help="Optional max days between market_date and last_history_date (0 disables staleness filtering).",
    )
    parser.add_argument(
        "--min-recency-factor",
        type=float,
        default=0.0,
        help="Optional minimum recency_factor required to keep a play (0 disables).",
    )
    parser.add_argument(
        "--thompson-temperature",
        type=float,
        default=1.0,
        help="Temperature on Thompson beta posterior sampling (>1 explores more).",
    )
    parser.add_argument(
        "--thompson-seed",
        type=int,
        default=17,
        help="Seed salt for deterministic Thompson sampling.",
    )
    parser.add_argument("--min-bet-win-rate", type=float, default=0.57, help="Minimum expected win rate required to place any bet.")
    parser.add_argument("--medium-bet-win-rate", type=float, default=0.60, help="Expected win rate for a medium-sized bet.")
    parser.add_argument("--full-bet-win-rate", type=float, default=0.65, help="Expected win rate for a full-sized bet.")
    parser.add_argument("--medium-tier-percentile", type=float, default=0.80, help="Minimum percentile for a medium-tier candidate.")
    parser.add_argument("--strong-tier-percentile", type=float, default=0.90, help="Minimum percentile for a strong-tier candidate.")
    parser.add_argument("--elite-tier-percentile", type=float, default=0.95, help="Minimum percentile for an elite-tier candidate.")
    parser.add_argument("--small-bet-fraction", type=float, default=0.005, help="Bankroll fraction for a small bet.")
    parser.add_argument("--medium-bet-fraction", type=float, default=0.010, help="Bankroll fraction for a medium bet.")
    parser.add_argument("--full-bet-fraction", type=float, default=0.015, help="Bankroll fraction for a full bet.")
    parser.add_argument("--max-bet-fraction", type=float, default=0.02, help="Maximum bankroll fraction per play.")
    parser.add_argument("--max-total-bet-fraction", type=float, default=0.05, help="Maximum total bankroll fraction across the board.")
    parser.add_argument(
        "--sizing-method",
        type=str,
        default="tiered_probability",
        choices=["tiered_probability", "flat_fraction", "coarse_bucket"],
        help="Stake allocation method: tiered probabilities, conservative flat fraction, or conservative coarse buckets.",
    )
    parser.add_argument(
        "--flat-bet-fraction",
        type=float,
        default=0.0,
        help="When sizing-method=flat_fraction, per-play bankroll fraction before total-cap scaling (<=0 falls back to small-bet-fraction).",
    )
    parser.add_argument("--coarse-low-bet-fraction", type=float, default=0.003, help="When sizing-method=coarse_bucket, low-tier bankroll fraction.")
    parser.add_argument("--coarse-mid-bet-fraction", type=float, default=0.005, help="When sizing-method=coarse_bucket, mid-tier bankroll fraction.")
    parser.add_argument("--coarse-high-bet-fraction", type=float, default=0.007, help="When sizing-method=coarse_bucket, high-tier bankroll fraction.")
    parser.add_argument(
        "--coarse-high-max-share",
        type=float,
        default=0.30,
        help="When sizing-method=coarse_bucket, hard cap on high-tier share of selected plays.",
    )
    parser.add_argument(
        "--coarse-mid-max-share",
        type=float,
        default=0.50,
        help="When sizing-method=coarse_bucket, hard cap on mid-tier share of selected plays.",
    )
    parser.add_argument(
        "--coarse-high-max-plays",
        type=int,
        default=0,
        help="When sizing-method=coarse_bucket, optional hard cap on high-tier play count (0 disables explicit count cap).",
    )
    parser.add_argument(
        "--coarse-mid-max-plays",
        type=int,
        default=0,
        help="When sizing-method=coarse_bucket, optional hard cap on mid-tier play count (0 disables explicit count cap).",
    )
    parser.add_argument(
        "--coarse-score-alpha-uncertainty",
        type=float,
        default=0.18,
        help="Weight on uncertainty penalty in coarse bucket stake score.",
    )
    parser.add_argument(
        "--coarse-score-beta-dependency",
        type=float,
        default=0.12,
        help="Weight on dependency burden penalty in coarse bucket stake score.",
    )
    parser.add_argument(
        "--coarse-score-gamma-support",
        type=float,
        default=0.08,
        help="Weight on support-strength lift in coarse bucket stake score.",
    )
    parser.add_argument(
        "--coarse-score-model",
        type=str,
        default="legacy",
        choices=["legacy", "stake_score_v1", "stake_model_v2"],
        help="Coarse bucket stake-score model variant.",
    )
    parser.add_argument(
        "--coarse-score-delta-prob-weight",
        type=float,
        default=0.0,
        help="Additional weight on calibrated non-push win-edge strength in coarse stake score.",
    )
    parser.add_argument(
        "--coarse-score-ev-weight",
        type=float,
        default=0.0,
        help="Additional weight on within-board EV strength in coarse stake score.",
    )
    parser.add_argument(
        "--coarse-score-risk-weight",
        type=float,
        default=0.0,
        help="Penalty weight on risk composite (risk/volatility/spike/tail imbalance) in coarse stake score.",
    )
    parser.add_argument(
        "--coarse-score-recency-weight",
        type=float,
        default=0.0,
        help="Additional weight on recency support in coarse stake score.",
    )
    parser.add_argument(
        "--staking-bucket-model-json",
        type=Path,
        default=Path("model/analysis/calibration/staking_bucket_model_v2.json"),
        help="Optional walk-forward staking bucket model payload JSON.",
    )
    parser.add_argument(
        "--disable-staking-bucket-model",
        action="store_true",
        help="Disable walk-forward staking bucket model usage even when coarse-score-model=stake_model_v2.",
    )
    parser.add_argument(
        "--staking-bucket-model-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month hint used for walk-forward staking bucket model lookup.",
    )
    parser.add_argument(
        "--staking-bucket-model-min-rows",
        type=int,
        default=0,
        help="Minimum monthly training rows required before applying walk-forward staking bucket model (0 disables this guard).",
    )
    parser.add_argument(
        "--belief-uncertainty-lower",
        type=float,
        default=BELIEF_UNCERTAINTY_LOWER,
        help="Lower anchor used when converting latent belief uncertainty into a confidence penalty.",
    )
    parser.add_argument(
        "--belief-uncertainty-upper",
        type=float,
        default=BELIEF_UNCERTAINTY_UPPER,
        help="Upper anchor used when converting latent belief uncertainty into a confidence penalty.",
    )
    parser.add_argument(
        "--append-agreement-min",
        type=int,
        default=3,
        help="Minimum E/T/V agreement count required for append-only shadow candidates.",
    )
    parser.add_argument(
        "--append-edge-percentile-min",
        type=float,
        default=0.90,
        help="Minimum abs-edge percentile required for append-only shadow candidates.",
    )
    parser.add_argument(
        "--append-max-extra-plays",
        type=int,
        default=3,
        help="Maximum number of append-only shadow plays added beyond the edge base board.",
    )
    parser.add_argument(
        "--board-objective-overfetch",
        type=float,
        default=4.0,
        help="Multiplier for board-objective candidate overfetch relative to board size.",
    )
    parser.add_argument(
        "--board-objective-candidate-limit",
        type=int,
        default=36,
        help="Hard cap on candidate universe size for exact board-objective solving (0 disables).",
    )
    parser.add_argument(
        "--board-objective-max-search-nodes",
        type=int,
        default=750000,
        help="Maximum branch-and-bound nodes explored by exact board-objective search before fallback.",
    )
    parser.add_argument(
        "--board-objective-lambda-corr",
        type=float,
        default=0.12,
        help="Penalty weight for pairwise correlation in board-objective mode.",
    )
    parser.add_argument(
        "--board-objective-lambda-conc",
        type=float,
        default=0.07,
        help="Penalty weight for concentration in board-objective mode.",
    )
    parser.add_argument(
        "--board-objective-lambda-unc",
        type=float,
        default=0.06,
        help="Penalty weight for uncertainty in board-objective mode.",
    )
    parser.add_argument(
        "--board-objective-corr-same-game",
        type=float,
        default=0.65,
        help="Pairwise dependency contribution when two candidates share the same game.",
    )
    parser.add_argument(
        "--board-objective-corr-same-player",
        type=float,
        default=1.0,
        help="Pairwise dependency contribution when two candidates share the same player.",
    )
    parser.add_argument(
        "--board-objective-corr-same-target",
        type=float,
        default=0.15,
        help="Pairwise dependency contribution when two candidates share the same target family.",
    )
    parser.add_argument(
        "--board-objective-corr-same-direction",
        type=float,
        default=0.05,
        help="Pairwise dependency contribution when two candidates share the same direction.",
    )
    parser.add_argument(
        "--board-objective-corr-same-script-cluster",
        type=float,
        default=0.30,
        help="Pairwise dependency contribution when two candidates share the same inferred script cluster.",
    )
    parser.add_argument(
        "--board-objective-swap-candidates",
        type=int,
        default=18,
        help="Maximum number of out-of-universe swap candidates evaluated after exact board solve.",
    )
    parser.add_argument(
        "--board-objective-swap-rounds",
        type=int,
        default=2,
        help="Maximum number of improving swap rounds after exact board solve.",
    )
    parser.add_argument(
        "--board-objective-instability-enabled",
        action="store_true",
        help="Enable shadow-instability penalty/veto handling for near-cutoff board-objective candidates.",
    )
    parser.add_argument(
        "--board-objective-lambda-shadow-disagreement",
        type=float,
        default=0.0,
        help="Penalty weight on shadow-model disagreement for near-cutoff board-objective candidates.",
    )
    parser.add_argument(
        "--board-objective-lambda-segment-weakness",
        type=float,
        default=0.0,
        help="Penalty weight on segment-level recent weakness for near-cutoff board-objective candidates.",
    )
    parser.add_argument(
        "--board-objective-instability-near-cutoff-window",
        type=int,
        default=3,
        help="Rank-distance window around the board cutoff where instability penalties apply.",
    )
    parser.add_argument(
        "--board-objective-instability-top-protected",
        type=int,
        default=3,
        help="Top ranked candidates protected from instability penalties/vetoes.",
    )
    parser.add_argument(
        "--board-objective-instability-veto-enabled",
        action="store_true",
        help="Enable near-cutoff veto for high-instability in-cutoff candidates.",
    )
    parser.add_argument(
        "--board-objective-instability-veto-quantile",
        type=float,
        default=0.85,
        help="Quantile threshold for triggering near-cutoff instability vetoes.",
    )
    parser.add_argument(
        "--board-objective-dynamic-size-enabled",
        action="store_true",
        help="Allow board-objective mode to shrink board size on unstable slates.",
    )
    parser.add_argument(
        "--board-objective-dynamic-size-max-shrink",
        type=int,
        default=0,
        help="Maximum board rows that dynamic-size mode can shrink below requested max-total-plays.",
    )
    parser.add_argument(
        "--board-objective-dynamic-size-trigger",
        type=float,
        default=0.62,
        help="Composite instability trigger above which dynamic-size shrink activates.",
    )
    parser.add_argument(
        "--learned-gate-json",
        type=Path,
        default=Path("model/analysis/calibration/learned_pool_gate.json"),
        help="Optional learned pool-gate payload JSON.",
    )
    parser.add_argument(
        "--enable-learned-gate",
        action="store_true",
        help="Enable learned pool-gate filtering before final board construction.",
    )
    parser.add_argument(
        "--learned-gate-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month override for learned gate selection.",
    )
    parser.add_argument(
        "--learned-gate-min-rows",
        type=int,
        default=0,
        help="Minimum rows that must pass learned gate before enforcement (0 uses payload/default policy).",
    )
    return parser.parse_args()


def american_profit_per_unit(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def recommendation_rank(label: str) -> int:
    order = {"elite": 0, "strong": 1, "consider": 2, "pass": 3}
    return order.get(str(label), 3)


def minimum_recommendation_rank(label: str) -> int:
    return {"elite": 0, "strong": 1, "consider": 2, "pass": 3}[label]


def _numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _stable_seed_from_row(row: pd.Series, base_seed: int) -> int:
    key = "|".join(
        [
            str(base_seed),
            str(row.get("player", "")),
            str(row.get("target", "")),
            str(row.get("market_date", "")),
            str(row.get("market_event_id", "")),
            str(row.get("direction", "")),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32 - 1)


def _clean_key_component(series: pd.Series) -> pd.Series:
    raw = series.fillna("").astype(str).str.strip()
    lowered = raw.str.lower()
    return raw.mask(lowered.isin({"", "nan", "none", "null", "nat"}), "")


def _build_game_key(df: pd.DataFrame) -> pd.Series:
    if "market_event_id" in df.columns:
        event_key = _clean_key_component(df["market_event_id"])
    else:
        event_key = pd.Series("", index=df.index, dtype=str)

    home = _clean_key_component(df.get("market_home_team", pd.Series("", index=df.index)))
    away = _clean_key_component(df.get("market_away_team", pd.Series("", index=df.index)))
    teams_sorted = np.where(home <= away, home + "@" + away, away + "@" + home)
    teams_sorted = pd.Series(teams_sorted, index=df.index, dtype=str)
    market_date = _clean_key_component(df.get("market_date", pd.Series("", index=df.index))).str.slice(0, 10)
    player = _clean_key_component(df.get("player", pd.Series("", index=df.index)))
    target = _clean_key_component(df.get("target", pd.Series("", index=df.index)))
    teams_missing = home.eq("") & away.eq("")
    fallback_team_key = market_date + "|" + teams_sorted
    fallback_player_key = market_date + "|" + player + "|" + target
    fallback_key = pd.Series(
        np.where(teams_missing, fallback_player_key, fallback_team_key),
        index=df.index,
        dtype=str,
    )
    return np.where(event_key.ne(""), event_key, fallback_key)


def _normalize_script_cluster(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    if text in {"", "nan", "none", "null", "unknown", "script=unknown", "uninferred", "script=uninferred"}:
        return ""
    return text


def _resolve_target_caps(
    ranked: pd.DataFrame,
    max_plays_per_target: int,
    max_target_plays: dict[str, int] | None,
) -> dict[str, int]:
    caps = {k: int(v) for k, v in (max_target_plays or {}).items()}
    if not caps and max_plays_per_target > 0:
        caps = {target: int(max_plays_per_target) for target in ranked.get("target", pd.Series(dtype=str)).astype(str).unique()}
    return caps


def _zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    std = float(numeric.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    mean = float(numeric.mean())
    return (numeric - mean) / std


def _safe_logit(prob: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(prob, dtype="float64"), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _safe_sigmoid(logit_values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logit_values, dtype="float64"), -40.0, 40.0)))


def _build_board_probability_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Build calibrated board-level play probability and uncertainty columns.

    Anchor on the canonical calibrated probability (`p_calibrated` when
    available, else `expected_win_rate`) and add modest feature-driven
    separation for board-level solve quality.
    """
    if candidates.empty:
        return candidates.copy()

    out = candidates.copy()
    base_source = out["p_calibrated"] if "p_calibrated" in out.columns else out.get("expected_win_rate")
    base = pd.to_numeric(base_source, errors="coerce").fillna(0.5).clip(lower=0.01, upper=0.99)
    abs_edge = _numeric_series(out, "abs_edge", 0.0).clip(lower=0.0)
    final_conf = _numeric_series(out, "final_confidence", 0.0).clip(lower=0.0)
    history_rows = _numeric_series(out, "history_rows", 0.0).clip(lower=0.0)
    posterior_var = _numeric_series(out, "posterior_variance", 0.25).clip(lower=0.0, upper=1.0)

    abs_edge_z = _zscore_series(np.log1p(abs_edge))
    conf_z = _zscore_series(final_conf)
    support_norm = np.log1p(history_rows)
    support_denom = float(np.nanmax(support_norm.to_numpy(dtype="float64"))) if len(support_norm) else 1.0
    support_norm = support_norm / max(support_denom, 1e-9)
    unc_norm = np.sqrt(posterior_var)
    unc_min = float(unc_norm.min()) if len(unc_norm) else 0.0
    unc_span = max(float(unc_norm.max()) - unc_min, 1e-9) if len(unc_norm) else 1.0
    unc_norm = (unc_norm - unc_min) / unc_span

    # Small, conservative dispersion over the anchored probability.
    logit = _safe_logit(base.to_numpy(dtype="float64"))
    logit += 0.18 * abs_edge_z.to_numpy(dtype="float64")
    logit += 0.10 * conf_z.to_numpy(dtype="float64")
    logit += 0.08 * (support_norm.to_numpy(dtype="float64") - 0.5)
    logit -= 0.15 * (unc_norm.to_numpy(dtype="float64") - 0.5)
    modeled = pd.Series(_safe_sigmoid(logit), index=out.index, dtype="float64")

    if {"target", "direction"}.issubset(out.columns):
        grouped = out[["target", "direction"]].fillna("").astype(str)
        group_mean = modeled.groupby([grouped["target"], grouped["direction"]]).transform("mean")
    else:
        group_mean = pd.Series(float(modeled.mean()) if len(modeled) else 0.5, index=out.index, dtype="float64")

    calibration_k = 25.0
    support_weight = history_rows / (history_rows + calibration_k)
    support_weight = support_weight.clip(lower=0.0, upper=1.0)
    calibrated = support_weight * modeled + (1.0 - support_weight) * group_mean
    board_prob = (0.75 * base + 0.25 * calibrated).clip(lower=0.01, upper=0.99)

    out["board_play_strength"] = calibrated.clip(lower=0.01, upper=0.99)
    out["board_play_win_prob"] = board_prob
    out["board_uncertainty_penalty"] = unc_norm.clip(lower=0.0, upper=1.0)
    out["board_prob_dispersion"] = board_prob - base
    return out


def _build_pairwise_dependency_matrix(
    candidates: pd.DataFrame,
    same_game_weight: float,
    same_player_weight: float,
    same_target_weight: float,
    same_direction_weight: float,
    same_script_cluster_weight: float,
) -> np.ndarray:
    if candidates.empty:
        return np.zeros((0, 0), dtype="float64")

    n_rows = int(len(candidates))
    matrix = np.zeros((n_rows, n_rows), dtype="float64")

    players = _clean_key_component(candidates.get("player", pd.Series("", index=candidates.index))).to_numpy(dtype=str)
    games = _clean_key_component(candidates.get("game_key", pd.Series("", index=candidates.index))).to_numpy(dtype=str)
    targets = _clean_key_component(candidates.get("target", pd.Series("", index=candidates.index))).to_numpy(dtype=str)
    directions = _clean_key_component(candidates.get("direction", pd.Series("", index=candidates.index))).to_numpy(dtype=str)
    scripts = candidates.get("script_cluster_id", pd.Series("", index=candidates.index)).map(_normalize_script_cluster).to_numpy(dtype=str)

    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            rho = 0.0
            if players[i] and players[i] == players[j]:
                rho += float(same_player_weight)
            if games[i] and games[i] == games[j]:
                rho += float(same_game_weight)
            if targets[i] and targets[i] == targets[j]:
                rho += float(same_target_weight)
            if directions[i] and directions[i] == directions[j]:
                rho += float(same_direction_weight)
            if scripts[i] and scripts[i] == scripts[j]:
                rho += float(same_script_cluster_weight)
            matrix[i, j] = rho
            matrix[j, i] = rho
    return matrix


def _normalize_0_1(series: pd.Series, constant_value: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype("float64")
    if values.empty:
        return values
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    if span <= 1e-9:
        return pd.Series(float(constant_value), index=values.index, dtype="float64")
    return ((values - lo) / span).clip(lower=0.0, upper=1.0).astype("float64")


def _segment_key_series(frame: pd.DataFrame) -> pd.Series:
    target = _clean_key_component(frame.get("target", pd.Series("", index=frame.index))).str.upper().replace("", "UNK")
    direction = _clean_key_component(frame.get("direction", pd.Series("", index=frame.index))).str.upper().replace("", "UNK")
    return (target + "|" + direction).astype("object")


def _tier_from_rank_pct(rank_pct: pd.Series) -> pd.Series:
    pct = pd.to_numeric(rank_pct, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    values = np.where(pct >= (2.0 / 3.0), 2, np.where(pct >= (1.0 / 3.0), 1, 0))
    return pd.Series(values.astype(int), index=pct.index, dtype="int64")


def _segment_recent_weakness_series(frame: pd.DataFrame, disagreement: pd.Series) -> tuple[pd.Series, str]:
    if "segment_recent_weakness" in frame.columns:
        weak = pd.to_numeric(frame.get("segment_recent_weakness"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        return weak.astype("float64"), "segment_recent_weakness"

    cal_gap = _numeric_series(frame, "segment_recent_calibration_gap_pp", np.nan)
    margin_under = _numeric_series(frame, "segment_recent_margin_underperformance", np.nan)
    loss_conc = _numeric_series(frame, "segment_recent_loss_concentration", np.nan)
    has_recent = cal_gap.notna().any() or margin_under.notna().any() or loss_conc.notna().any()
    if has_recent:
        cal_component = _normalize_0_1(cal_gap.fillna(0.0).clip(lower=0.0), constant_value=0.0)
        margin_component = _normalize_0_1(margin_under.fillna(0.0).clip(lower=0.0), constant_value=0.0)
        loss_component = _normalize_0_1(loss_conc.fillna(0.0).clip(lower=0.0), constant_value=0.0)
        weak = (0.40 * cal_component + 0.35 * margin_component + 0.25 * loss_component).clip(lower=0.0, upper=1.0)
        return weak.astype("float64"), "segment_recent_metrics"

    segment_key = _segment_key_series(frame)
    seg_mean = pd.to_numeric(disagreement, errors="coerce").fillna(0.0).groupby(segment_key).transform("mean")
    weak = _normalize_0_1(seg_mean, constant_value=0.5)
    return weak.astype("float64"), "shadow_disagreement_proxy"


def _build_board_instability_features(
    candidates: pd.DataFrame,
    payout_per_unit: float,
    staking_bucket_model_payload: dict | None,
    staking_bucket_model_month: str | None,
    staking_bucket_model_min_rows: int,
    same_game_weight: float,
    same_player_weight: float,
    same_target_weight: float,
    same_direction_weight: float,
    same_script_cluster_weight: float,
) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        return out

    prob = _numeric_series(out, "board_play_win_prob", np.nan)
    prob = prob.where(prob.notna(), _numeric_series(out, "p_calibrated", np.nan))
    prob = prob.where(prob.notna(), _numeric_series(out, "expected_win_rate", 0.5)).fillna(0.5).clip(lower=0.0, upper=1.0)

    expected_push = _numeric_series(out, "expected_push_rate", 0.0).clip(lower=0.0, upper=0.95)
    non_push = (1.0 - expected_push).clip(lower=0.05, upper=1.0)
    cond_prob = (prob / non_push).clip(lower=0.0, upper=1.0)
    break_even_cond = 1.0 / (1.0 + max(float(payout_per_unit), 1e-9))
    delta_prob = (cond_prob - float(break_even_cond)).clip(lower=-1.0, upper=1.0)
    delta_prob_strength = _normalize_0_1(delta_prob, constant_value=0.5)

    ev_source = _numeric_series(out, "ev_adjusted", np.nan)
    ev_source = ev_source.where(ev_source.notna(), _numeric_series(out, "ev", 0.0)).fillna(0.0)
    ev_strength = _normalize_0_1(ev_source, constant_value=0.5)

    unc = pd.to_numeric(out.get("board_uncertainty_penalty"), errors="coerce")
    if unc.isna().all():
        unc = pd.to_numeric(out.get("belief_uncertainty_normalized"), errors="coerce")
    if unc.isna().all():
        unc = np.sqrt(pd.to_numeric(out.get("posterior_variance"), errors="coerce").fillna(0.25).clip(lower=0.0, upper=1.0))
    unc = pd.to_numeric(unc, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)

    dep_matrix = _build_pairwise_dependency_matrix(
        out,
        same_game_weight=float(same_game_weight),
        same_player_weight=float(same_player_weight),
        same_target_weight=float(same_target_weight),
        same_direction_weight=float(same_direction_weight),
        same_script_cluster_weight=float(same_script_cluster_weight),
    )
    dep_burden = pd.Series(dep_matrix.mean(axis=1) if dep_matrix.size else np.zeros(len(out), dtype="float64"), index=out.index, dtype="float64")
    dep_strength = _normalize_0_1(dep_burden, constant_value=0.0)

    history_rows = _numeric_series(out, "history_rows", 0.0).clip(lower=0.0)
    calibration_rows = _numeric_series(out, "calibration_bucket_rows", 0.0).clip(lower=0.0)
    support_raw = 0.70 * np.log1p(history_rows) + 0.30 * np.log1p(calibration_rows)
    support_strength = _normalize_0_1(pd.Series(support_raw, index=out.index, dtype="float64"), constant_value=0.0)

    recency_strength = _numeric_series(out, "recency_factor", 1.0).clip(lower=0.0, upper=1.0)
    risk_penalty = _numeric_series(out, "risk_penalty", 0.0).clip(lower=0.0)
    volatility_score = _numeric_series(out, "volatility_score", 0.0).clip(lower=0.0)
    spike_probability = _numeric_series(out, "spike_probability", 0.0).clip(lower=0.0)
    tail_imbalance = _numeric_series(out, "tail_imbalance", 0.0).abs()
    risk_component = (0.45 * risk_penalty + 0.30 * volatility_score + 0.15 * spike_probability + 0.10 * tail_imbalance).clip(lower=0.0)

    legacy_score = (
        cond_prob
        - 0.18 * unc
        - 0.12 * dep_strength
        + 0.08 * support_strength
    )
    score_v1 = (
        legacy_score
        + 0.18 * delta_prob_strength
        + 0.14 * ev_strength
        + 0.06 * recency_strength
        - 0.16 * risk_component
    )
    score_v2_prob, score_v2_source, score_v2_month, score_v2_details = apply_staking_bucket_model_v2_fn(
        out,
        payload=staking_bucket_model_payload,
        run_date_hint=staking_bucket_model_month,
        prob_col="board_play_win_prob",
        payout_per_unit=float(payout_per_unit),
        min_train_rows=int(staking_bucket_model_min_rows),
    )
    score_v2 = pd.to_numeric(score_v2_prob, errors="coerce").fillna(prob).clip(lower=0.0, upper=1.0)

    rank_legacy = pd.to_numeric(legacy_score, errors="coerce").rank(method="average", pct=True).fillna(0.5)
    rank_v1 = pd.to_numeric(score_v1, errors="coerce").rank(method="average", pct=True).fillna(0.5)
    rank_v2 = pd.to_numeric(score_v2, errors="coerce").rank(method="average", pct=True).fillna(0.5)
    rank_matrix = np.column_stack(
        [
            rank_legacy.to_numpy(dtype="float64"),
            rank_v1.to_numpy(dtype="float64"),
            rank_v2.to_numpy(dtype="float64"),
        ]
    )
    rank_std = pd.Series(np.std(rank_matrix, axis=1), index=out.index, dtype="float64")
    rank_std_norm = _normalize_0_1(rank_std, constant_value=0.0)

    bucket_legacy = _tier_from_rank_pct(rank_legacy)
    bucket_v1 = _tier_from_rank_pct(rank_v1)
    bucket_v2 = _tier_from_rank_pct(rank_v2)
    bucket_matrix = np.column_stack(
        [
            bucket_legacy.to_numpy(dtype="int64"),
            bucket_v1.to_numpy(dtype="int64"),
            bucket_v2.to_numpy(dtype="int64"),
        ]
    )
    bucket_unique_count = np.apply_along_axis(lambda row: len(set(int(value) for value in row)), 1, bucket_matrix)
    bucket_disagreement = pd.Series(np.clip((bucket_unique_count - 1.0) / 2.0, 0.0, 1.0), index=out.index, dtype="float64")

    shadow_disagreement = (0.70 * rank_std_norm + 0.30 * bucket_disagreement).clip(lower=0.0, upper=1.0)
    segment_weakness, segment_weakness_source = _segment_recent_weakness_series(out, shadow_disagreement)
    instability_raw = 0.50 * shadow_disagreement + 0.20 * unc + 0.20 * dep_strength + 0.10 * segment_weakness
    instability_score = _normalize_0_1(pd.Series(instability_raw, index=out.index, dtype="float64"), constant_value=0.5)

    out["board_dependency_burden"] = dep_strength.astype("float64")
    out["board_shadow_disagreement"] = shadow_disagreement.astype("float64")
    out["board_shadow_rank_std"] = rank_std_norm.astype("float64")
    out["board_shadow_bucket_disagreement"] = bucket_disagreement.astype("float64")
    out["board_shadow_score_legacy"] = pd.to_numeric(legacy_score, errors="coerce").astype("float64")
    out["board_shadow_score_v1"] = pd.to_numeric(score_v1, errors="coerce").astype("float64")
    out["board_shadow_score_v2"] = pd.to_numeric(score_v2, errors="coerce").astype("float64")
    out["board_shadow_bucket_legacy"] = bucket_legacy.astype("int64")
    out["board_shadow_bucket_v1"] = bucket_v1.astype("int64")
    out["board_shadow_bucket_v2"] = bucket_v2.astype("int64")
    out["board_segment_key"] = _segment_key_series(out)
    out["board_segment_recent_weakness"] = segment_weakness.astype("float64")
    out["board_segment_recent_weakness_source"] = str(segment_weakness_source)
    out["board_instability_score"] = instability_score.astype("float64")
    out["board_shadow_score_v2_source"] = score_v2_source.astype("object")
    out["board_shadow_score_v2_month"] = str(score_v2_month)
    out["board_shadow_score_v2_details"] = json.dumps(score_v2_details, sort_keys=True)
    return out


def _apply_coarse_bucket_sizing(
    board: pd.DataFrame,
    max_bet_fraction: float,
    max_total_bet_fraction: float,
    low_bet_fraction: float,
    mid_bet_fraction: float,
    high_bet_fraction: float,
    high_max_share: float,
    mid_max_share: float,
    high_max_plays: int,
    mid_max_plays: int,
    score_alpha_uncertainty: float,
    score_beta_dependency: float,
    score_gamma_support: float,
    score_model: str,
    score_delta_prob_weight: float,
    score_ev_weight: float,
    score_risk_weight: float,
    score_recency_weight: float,
    payout_per_unit: float,
    staking_bucket_model_payload: dict | None,
    staking_bucket_model_month: str | None,
    staking_bucket_model_min_rows: int,
    same_game_weight: float,
    same_player_weight: float,
    same_target_weight: float,
    same_direction_weight: float,
    same_script_cluster_weight: float,
) -> pd.DataFrame:
    out = board.copy()
    n_rows = int(len(out))
    if n_rows <= 0:
        return out

    # Conservative scoring inputs.
    score_model_name = str(score_model or "legacy").strip().lower()
    if score_model_name not in {"legacy", "stake_score_v1", "stake_model_v2"}:
        score_model_name = "legacy"

    prob = _numeric_series(out, "p_calibrated", np.nan)
    prob = prob.where(prob.notna(), _numeric_series(out, "expected_win_rate", 0.5)).fillna(0.5).clip(lower=0.0, upper=1.0)
    expected_push = _numeric_series(out, "expected_push_rate", 0.0).clip(lower=0.0, upper=0.95)
    non_push = (1.0 - expected_push).clip(lower=0.05, upper=1.0)
    cond_prob = (prob / non_push).clip(lower=0.0, upper=1.0)
    safe_payout = max(float(payout_per_unit), 1e-9)
    break_even_cond = 1.0 / (1.0 + safe_payout)
    delta_prob = (cond_prob - float(break_even_cond)).clip(lower=-1.0, upper=1.0)
    delta_prob_strength = _normalize_0_1(delta_prob, constant_value=0.5)

    ev_source = _numeric_series(out, "ev_adjusted", np.nan)
    ev_source = ev_source.where(ev_source.notna(), _numeric_series(out, "ev", 0.0)).fillna(0.0)
    ev_strength = _normalize_0_1(ev_source, constant_value=0.5)

    unc = pd.to_numeric(out.get("board_uncertainty_penalty"), errors="coerce")
    if unc.isna().all():
        unc = pd.to_numeric(out.get("belief_uncertainty_normalized"), errors="coerce")
    if unc.isna().all():
        unc = np.sqrt(pd.to_numeric(out.get("posterior_variance"), errors="coerce").fillna(0.25).clip(lower=0.0, upper=1.0))
    unc = pd.to_numeric(unc, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)

    history_rows = _numeric_series(out, "history_rows", 0.0).clip(lower=0.0)
    calibration_rows = _numeric_series(out, "calibration_bucket_rows", 0.0).clip(lower=0.0)
    market_books = _numeric_series(out, "market_books", 0.0).clip(lower=0.0)
    recency_strength = _numeric_series(out, "recency_factor", 1.0).clip(lower=0.0, upper=1.0)
    recency_strength = _normalize_0_1(recency_strength, constant_value=0.5)
    support_strength = (
        0.60 * _normalize_0_1(np.log1p(history_rows), constant_value=0.0)
        + 0.20 * _normalize_0_1(np.log1p(calibration_rows), constant_value=0.0)
        + 0.20 * _normalize_0_1(
        np.log1p(market_books), constant_value=0.0
        )
    )

    risk_penalty = _numeric_series(out, "risk_penalty", 0.0).clip(lower=0.0)
    volatility = _numeric_series(out, "volatility_score", 0.0).clip(lower=0.0)
    spike_probability = _numeric_series(out, "spike_probability", 0.0).clip(lower=0.0)
    tail_imbalance = _numeric_series(out, "tail_imbalance", 0.0).abs()
    risk_component = (
        0.45 * _normalize_0_1(risk_penalty, constant_value=0.0)
        + 0.30 * _normalize_0_1(volatility, constant_value=0.0)
        + 0.15 * _normalize_0_1(spike_probability, constant_value=0.0)
        + 0.10 * _normalize_0_1(tail_imbalance, constant_value=0.0)
    )

    dep_matrix = _build_pairwise_dependency_matrix(
        out,
        same_game_weight=float(same_game_weight),
        same_player_weight=float(same_player_weight),
        same_target_weight=float(same_target_weight),
        same_direction_weight=float(same_direction_weight),
        same_script_cluster_weight=float(same_script_cluster_weight),
    )
    if n_rows > 1:
        dep_raw = pd.Series(dep_matrix.sum(axis=1) / float(n_rows - 1), index=out.index, dtype="float64")
    else:
        dep_raw = pd.Series(0.0, index=out.index, dtype="float64")
    dep_burden = _normalize_0_1(dep_raw, constant_value=0.0)

    score = (
        prob
        - float(score_alpha_uncertainty) * unc
        - float(score_beta_dependency) * dep_burden
        + float(score_gamma_support) * support_strength
    )
    if score_model_name == "stake_model_v2":
        run_date_hint = str(staking_bucket_model_month or "")
        if not run_date_hint:
            run_date_hint = str(pd.to_datetime(out.get("market_date"), errors="coerce").max()) if "market_date" in out.columns else ""
        v2_score, v2_source, v2_month, v2_details = apply_staking_bucket_model_v2_fn(
            out,
            payload=staking_bucket_model_payload,
            run_date_hint=run_date_hint,
            prob_col="p_calibrated",
            payout_per_unit=float(payout_per_unit),
            min_train_rows=int(max(0, staking_bucket_model_min_rows)),
        )
        score = (
            pd.to_numeric(v2_score, errors="coerce").fillna(prob)
            - float(score_alpha_uncertainty) * unc
            - float(score_beta_dependency) * dep_burden
            + float(score_gamma_support) * support_strength
        )
        out["coarse_score_v2_source"] = pd.Series(v2_source, index=out.index, dtype="object")
        out["coarse_score_v2_month"] = str(v2_month)
        out["coarse_score_v2_details"] = json.dumps(v2_details, sort_keys=True)
    if score_model_name == "stake_score_v1":
        score = (
            score
            + float(score_delta_prob_weight) * delta_prob_strength
            + float(score_ev_weight) * ev_strength
            - float(score_risk_weight) * risk_component
            + float(score_recency_weight) * recency_strength
        )

    out["coarse_score_model"] = score_model_name
    out["coarse_score_prob"] = prob
    out["coarse_score_cond_prob"] = cond_prob
    out["coarse_score_delta_prob"] = delta_prob
    out["coarse_score_delta_prob_strength"] = delta_prob_strength
    out["coarse_score_ev_strength"] = ev_strength
    out["coarse_score_uncertainty"] = unc
    out["coarse_score_dependency_burden"] = dep_burden
    out["coarse_score_support"] = support_strength
    out["coarse_score_recency"] = recency_strength
    out["coarse_score_risk"] = risk_component
    out["coarse_score_break_even_conditional"] = float(break_even_cond)
    out["coarse_score"] = pd.to_numeric(score, errors="coerce").fillna(0.0)

    high_share = float(np.clip(high_max_share, 0.0, 1.0))
    mid_share = float(np.clip(mid_max_share, 0.0, 1.0))
    high_cap = int(np.floor(high_share * float(n_rows)))
    mid_cap = int(np.floor(mid_share * float(n_rows)))
    if int(high_max_plays) > 0:
        high_cap = min(high_cap, int(high_max_plays))
    high_cap = max(0, min(high_cap, n_rows))
    remaining = max(0, n_rows - high_cap)
    if int(mid_max_plays) > 0:
        mid_cap = min(mid_cap, int(mid_max_plays))
    mid_cap = max(0, min(mid_cap, remaining))

    out = out.sort_values(["coarse_score", "p_calibrated", "ev_adjusted", "abs_edge"], ascending=[False, False, False, False]).copy()
    out["coarse_score_rank"] = np.arange(1, len(out) + 1)
    out["coarse_bucket"] = "low"
    if high_cap > 0:
        out.iloc[:high_cap, out.columns.get_loc("coarse_bucket")] = "high"
    if mid_cap > 0:
        out.iloc[high_cap : high_cap + mid_cap, out.columns.get_loc("coarse_bucket")] = "mid"
    out["coarse_bucket_high_cap"] = int(high_cap)
    out["coarse_bucket_mid_cap"] = int(mid_cap)
    out["coarse_bucket_total_rows"] = int(n_rows)

    bucket_to_fraction = {
        "low": float(low_bet_fraction),
        "mid": float(mid_bet_fraction),
        "high": float(high_bet_fraction),
    }
    out["allocation_tier"] = "coarse_" + out["coarse_bucket"].astype(str)
    out["allocation_action"] = out["allocation_tier"]
    out["allocation_action_from_tier"] = out["allocation_tier"]
    out["allocation_action_from_probability"] = out["allocation_tier"]
    out["allocation_action_level"] = out["coarse_bucket"].map({"low": 1, "mid": 2, "high": 3}).fillna(1).astype(int)
    out["bet_fraction_raw"] = pd.to_numeric(out["coarse_bucket"].map(bucket_to_fraction), errors="coerce").fillna(0.0)
    out["bet_fraction_raw"] = out["bet_fraction_raw"].clip(lower=0.0, upper=float(max_bet_fraction))

    raw_total = float(pd.to_numeric(out["bet_fraction_raw"], errors="coerce").fillna(0.0).sum())
    scale = 1.0
    if raw_total > 0.0 and float(max_total_bet_fraction) > 0.0:
        scale = min(1.0, float(max_total_bet_fraction) / raw_total)
    out["bet_fraction_scale"] = float(scale)
    out["bet_fraction"] = pd.to_numeric(out["bet_fraction_raw"], errors="coerce").fillna(0.0) * float(scale)
    return out


def _compute_concentration_penalty(
    selected_indices: list[int],
    targets: np.ndarray,
    directions: np.ndarray,
    games: np.ndarray,
) -> float:
    if not selected_indices:
        return 0.0
    k = float(len(selected_indices))
    inv_k_sq = 1.0 / max(k * k, 1e-9)

    def hhi(values: np.ndarray) -> float:
        counts: dict[str, int] = {}
        for idx in selected_indices:
            key = str(values[idx])
            counts[key] = counts.get(key, 0) + 1
        return float(sum((count * count) * inv_k_sq for count in counts.values()))

    target_hhi = hhi(targets)
    direction_hhi = hhi(directions)
    game_hhi = hhi(games)
    return 0.50 * target_hhi + 0.30 * direction_hhi + 0.20 * game_hhi


def _is_candidate_feasible(
    idx: int,
    players: np.ndarray,
    targets: np.ndarray,
    games: np.ndarray,
    scripts: np.ndarray,
    player_counts: dict[str, int],
    target_counts: dict[str, int],
    game_counts: dict[str, int],
    script_counts: dict[str, int],
    target_caps: dict[str, int],
    max_plays_per_player: int,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
) -> bool:
    player = str(players[idx])
    target = str(targets[idx])
    game = str(games[idx])
    script = str(scripts[idx])

    if max_plays_per_player > 0 and player_counts.get(player, 0) >= int(max_plays_per_player):
        return False
    target_cap = int(target_caps.get(target, 0))
    if target_cap > 0 and target_counts.get(target, 0) >= target_cap:
        return False
    if max_plays_per_game > 0 and game_counts.get(game, 0) >= int(max_plays_per_game):
        return False
    if max_plays_per_script_cluster > 0 and script and script_counts.get(script, 0) >= int(max_plays_per_script_cluster):
        return False
    return True


def _score_selected_indices(
    selected_indices: list[int],
    probs: np.ndarray,
    unc: np.ndarray,
    node_penalty: np.ndarray | None,
    dep_matrix: np.ndarray,
    targets: np.ndarray,
    directions: np.ndarray,
    games: np.ndarray,
    lambda_corr: float,
    lambda_conc: float,
    lambda_unc: float,
) -> float:
    if not selected_indices:
        return -np.inf

    node_score = float(np.sum(probs[selected_indices] - float(lambda_unc) * unc[selected_indices]))
    if node_penalty is not None and len(node_penalty) > 0:
        node_score -= float(np.sum(np.asarray(node_penalty, dtype="float64")[selected_indices]))
    pair_penalty = 0.0
    for pos_i, idx_i in enumerate(selected_indices):
        for idx_j in selected_indices[pos_i + 1 :]:
            pair_penalty += float(dep_matrix[idx_i, idx_j])
    concentration = _compute_concentration_penalty(selected_indices, targets=targets, directions=directions, games=games)
    return node_score - float(lambda_corr) * pair_penalty - float(lambda_conc) * concentration


def _select_board_objective_board(
    candidates: pd.DataFrame,
    max_plays_per_player: int,
    max_plays_per_target: int,
    max_total_plays: int,
    max_target_plays: dict[str, int] | None,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
    min_board_plays: int,
    board_objective_overfetch: float,
    board_objective_candidate_limit: int,
    board_objective_max_search_nodes: int,
    board_objective_lambda_corr: float,
    board_objective_lambda_conc: float,
    board_objective_lambda_unc: float,
    board_objective_corr_same_game: float,
    board_objective_corr_same_player: float,
    board_objective_corr_same_target: float,
    board_objective_corr_same_direction: float,
    board_objective_corr_same_script_cluster: float,
    board_objective_swap_candidates: int,
    board_objective_swap_rounds: int,
    payout_per_unit: float,
    staking_bucket_model_payload: dict | None,
    staking_bucket_model_month: str | None,
    staking_bucket_model_min_rows: int,
    board_objective_instability_enabled: bool,
    board_objective_lambda_shadow_disagreement: float,
    board_objective_lambda_segment_weakness: float,
    board_objective_instability_near_cutoff_window: int,
    board_objective_instability_top_protected: int,
    board_objective_instability_veto_enabled: bool,
    board_objective_instability_veto_quantile: float,
    board_objective_dynamic_size_enabled: bool,
    board_objective_dynamic_size_max_shrink: int,
    board_objective_dynamic_size_trigger: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    board_size = int(max_total_plays) if int(max_total_plays) > 0 else int(len(candidates))
    if board_size <= 0:
        return candidates.iloc[0:0].copy()
    board_size = min(board_size, int(len(candidates)))
    requested_min = max(0, int(min_board_plays))
    if requested_min > 0:
        board_size = max(board_size, min(requested_min, int(len(candidates))))

    base = _build_board_probability_features(candidates.copy())
    base["_source_index"] = base.index
    instability_active = bool(
        board_objective_instability_enabled
        and (
            float(board_objective_lambda_shadow_disagreement) > 0.0
            or float(board_objective_lambda_segment_weakness) > 0.0
            or bool(board_objective_instability_veto_enabled)
            or bool(board_objective_dynamic_size_enabled)
        )
    )
    if instability_active:
        base = _build_board_instability_features(
            base,
            payout_per_unit=float(payout_per_unit),
            staking_bucket_model_payload=staking_bucket_model_payload,
            staking_bucket_model_month=staking_bucket_model_month,
            staking_bucket_model_min_rows=int(staking_bucket_model_min_rows),
            same_game_weight=float(board_objective_corr_same_game),
            same_player_weight=float(board_objective_corr_same_player),
            same_target_weight=float(board_objective_corr_same_target),
            same_direction_weight=float(board_objective_corr_same_direction),
            same_script_cluster_weight=float(board_objective_corr_same_script_cluster),
        )
    else:
        base["board_dependency_burden"] = 0.0
        base["board_shadow_disagreement"] = 0.0
        base["board_segment_recent_weakness"] = 0.0
        base["board_segment_recent_weakness_source"] = "disabled"
        base["board_instability_score"] = 0.0
        base["board_instability_penalty"] = 0.0

    overfetch_count = int(np.clip(np.ceil(max(float(board_objective_overfetch), 1.0) * board_size), board_size, len(base)))
    idx_abs = set(base.sort_values(["abs_edge", "ev_adjusted", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).head(overfetch_count).index.tolist())
    idx_ev = set(base.sort_values(["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False]).head(overfetch_count).index.tolist())
    idx_prob = set(base.sort_values(["board_play_win_prob", "final_confidence", "abs_edge", "ev_adjusted"], ascending=[False, False, False, False]).head(overfetch_count).index.tolist())
    idx_th = set(base.sort_values(["thompson_ev", "board_play_win_prob", "ev_adjusted", "abs_edge"], ascending=[False, False, False, False]).head(overfetch_count).index.tolist()) if "thompson_ev" in base.columns else set()

    universe_idx = sorted(idx_abs | idx_ev | idx_prob | idx_th)
    universe = base.loc[universe_idx].copy() if universe_idx else base.copy()
    if universe.empty:
        return universe

    universe["board_universe_rank_base"] = (
        0.55 * _zscore_series(universe["board_play_win_prob"])
        + 0.25 * _zscore_series(universe["abs_edge"])
        + 0.20 * _zscore_series(universe["ev_adjusted"])
    )
    universe["board_universe_rank"] = pd.to_numeric(universe["board_universe_rank_base"], errors="coerce").fillna(0.0)
    candidate_cap = int(board_objective_candidate_limit)
    if candidate_cap > 0 and len(universe) > candidate_cap:
        universe = universe.sort_values(["board_universe_rank", "board_play_win_prob", "abs_edge"], ascending=[False, False, False]).head(candidate_cap).copy()

    universe = universe.sort_values(["board_universe_rank", "board_play_win_prob", "abs_edge"], ascending=[False, False, False]).reset_index(drop=True)
    n = int(len(universe))
    if n <= 0:
        return base.iloc[0:0].copy()
    k = min(board_size, n)
    if k <= 0:
        return base.iloc[0:0].copy()

    dynamic_day_score = np.nan
    dynamic_cutoff_separation = np.nan
    dynamic_shrink = 0
    dynamic_target_size = int(k)
    instability_veto_count = 0

    if instability_active and not universe.empty:
        rank_positions = np.arange(1, len(universe) + 1, dtype="float64")
        near_window = max(1, int(board_objective_instability_near_cutoff_window))
        top_protected = max(0, int(board_objective_instability_top_protected))
        rank_distance = np.abs(rank_positions - float(k))
        near_cutoff_mask = (rank_distance <= float(near_window)) & (rank_positions > float(top_protected))

        if bool(board_objective_instability_veto_enabled) and np.any(near_cutoff_mask):
            q = float(np.clip(board_objective_instability_veto_quantile, 0.50, 0.99))
            instability = pd.to_numeric(universe.get("board_instability_score"), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
            inclusion_mask = near_cutoff_mask & (rank_positions <= float(k))
            if np.any(inclusion_mask):
                threshold = float(np.nanquantile(instability[inclusion_mask], q))
                veto_idx = np.where(inclusion_mask & (instability >= threshold))[0]
                max_drop = max(0, int(len(universe)) - int(k))
                if len(veto_idx) > 0 and max_drop > 0:
                    drop_n = min(int(len(veto_idx)), int(max_drop))
                    veto_sorted = sorted(veto_idx.tolist(), key=lambda idx: float(instability[idx]), reverse=True)
                    drop_idx = veto_sorted[:drop_n]
                    if drop_idx:
                        universe = universe.drop(index=drop_idx).reset_index(drop=True)
                        instability_veto_count = int(len(drop_idx))
                        n = int(len(universe))
                        if n <= 0:
                            return base.iloc[0:0].copy()
                        k = min(k, n)
                        if k <= 0:
                            return base.iloc[0:0].copy()
                        rank_positions = np.arange(1, len(universe) + 1, dtype="float64")
                        rank_distance = np.abs(rank_positions - float(k))
                        near_cutoff_mask = (rank_distance <= float(near_window)) & (rank_positions > float(top_protected))

        lambda_shadow = max(0.0, float(board_objective_lambda_shadow_disagreement))
        lambda_segment = max(0.0, float(board_objective_lambda_segment_weakness))
        proximity = np.clip(1.0 - (rank_distance / max(float(near_window), 1.0)), 0.0, 1.0)
        shadow_disagreement = pd.to_numeric(universe.get("board_shadow_disagreement"), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        segment_weakness = pd.to_numeric(universe.get("board_segment_recent_weakness"), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        penalty = np.where(
            near_cutoff_mask,
            proximity * (lambda_shadow * shadow_disagreement + lambda_segment * segment_weakness),
            0.0,
        )
        universe["board_instability_penalty"] = pd.Series(penalty, index=universe.index, dtype="float64")
        universe["board_instability_near_cutoff"] = pd.Series(near_cutoff_mask, index=universe.index, dtype=bool)
        universe["board_universe_rank"] = (
            pd.to_numeric(universe.get("board_universe_rank_base"), errors="coerce").fillna(0.0)
            - pd.to_numeric(universe.get("board_instability_penalty"), errors="coerce").fillna(0.0)
        )
        universe = universe.sort_values(["board_universe_rank", "board_play_win_prob", "abs_edge"], ascending=[False, False, False]).reset_index(drop=True)

        if bool(board_objective_dynamic_size_enabled) and len(universe) > k and k > 0:
            ranks_dyn = np.arange(1, len(universe) + 1, dtype="float64")
            distance_dyn = np.abs(ranks_dyn - float(k))
            near_dyn = (distance_dyn <= float(near_window)) & (ranks_dyn > float(top_protected))
            if not np.any(near_dyn):
                near_dyn = ranks_dyn <= float(min(len(universe), max(k, 1)))

            instability_near = float(pd.to_numeric(universe.loc[near_dyn, "board_instability_score"], errors="coerce").fillna(0.0).mean())
            dependency_near = float(pd.to_numeric(universe.loc[near_dyn, "board_dependency_burden"], errors="coerce").fillna(0.0).mean())
            rank_scores = pd.to_numeric(universe.get("board_universe_rank"), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
            if len(rank_scores) > k:
                raw_gap = float(rank_scores[k - 1] - rank_scores[k])
                diff_scale = float(np.std(rank_scores, ddof=0))
                if diff_scale <= 1e-9:
                    diff_scale = float(np.max(np.abs(np.diff(rank_scores)))) if len(rank_scores) > 1 else 1.0
                dynamic_cutoff_separation = float(np.clip(raw_gap / max(diff_scale, 1e-9), 0.0, 1.0))
            else:
                dynamic_cutoff_separation = 1.0

            dynamic_day_score = float(
                0.50 * np.clip(instability_near, 0.0, 1.0)
                + 0.30 * np.clip(dependency_near, 0.0, 1.0)
                + 0.20 * (1.0 - np.clip(dynamic_cutoff_separation, 0.0, 1.0))
            )
            trigger = float(np.clip(board_objective_dynamic_size_trigger, 0.20, 0.95))
            max_shrink = max(0, int(board_objective_dynamic_size_max_shrink))
            lower_bound = max(1, min(int(max(requested_min, 1)), int(k)))
            allowed_shrink = max(0, min(max_shrink, int(k) - int(lower_bound)))
            if dynamic_day_score > trigger and allowed_shrink > 0:
                ratio = (dynamic_day_score - trigger) / max(1.0 - trigger, 1e-9)
                dynamic_shrink = int(np.ceil(np.clip(ratio, 0.0, 1.0) * float(allowed_shrink)))
                dynamic_shrink = max(0, min(dynamic_shrink, allowed_shrink))
                if dynamic_shrink > 0:
                    k = max(lower_bound, int(k) - int(dynamic_shrink))
                    dynamic_target_size = int(k)
    else:
        universe["board_instability_penalty"] = 0.0
        universe["board_instability_near_cutoff"] = False

    n = int(len(universe))
    if n <= 0:
        return base.iloc[0:0].copy()
    k = min(k, n)
    if k <= 0:
        return base.iloc[0:0].copy()

    target_caps = _resolve_target_caps(universe, max_plays_per_target=max_plays_per_target, max_target_plays=max_target_plays)
    players = _clean_key_component(universe.get("player", pd.Series("", index=universe.index))).to_numpy(dtype=str)
    targets = _clean_key_component(universe.get("target", pd.Series("", index=universe.index))).to_numpy(dtype=str)
    games = _clean_key_component(universe.get("game_key", pd.Series("", index=universe.index))).to_numpy(dtype=str)
    directions = _clean_key_component(universe.get("direction", pd.Series("", index=universe.index))).to_numpy(dtype=str)
    scripts = universe.get("script_cluster_id", pd.Series("", index=universe.index)).map(_normalize_script_cluster).to_numpy(dtype=str)

    probs = pd.to_numeric(universe["board_play_win_prob"], errors="coerce").fillna(0.5).to_numpy(dtype="float64")
    unc = pd.to_numeric(universe["board_uncertainty_penalty"], errors="coerce").fillna(0.5).to_numpy(dtype="float64")
    node_penalty = pd.to_numeric(universe.get("board_instability_penalty"), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    node_values = probs - float(board_objective_lambda_unc) * unc - node_penalty
    order = np.argsort(-node_values)
    universe = universe.iloc[order].reset_index(drop=True)
    probs = probs[order]
    unc = unc[order]
    node_penalty = node_penalty[order]
    players = players[order]
    targets = targets[order]
    games = games[order]
    directions = directions[order]
    scripts = scripts[order]

    dep = _build_pairwise_dependency_matrix(
        universe,
        same_game_weight=float(board_objective_corr_same_game),
        same_player_weight=float(board_objective_corr_same_player),
        same_target_weight=float(board_objective_corr_same_target),
        same_direction_weight=float(board_objective_corr_same_direction),
        same_script_cluster_weight=float(board_objective_corr_same_script_cluster),
    )
    node_values = probs - float(board_objective_lambda_unc) * unc - node_penalty
    prefix = np.concatenate(([0.0], np.cumsum(node_values)))

    max_nodes = max(1000, int(board_objective_max_search_nodes))
    node_counter = {"count": 0, "truncated": False}
    best_score = -np.inf
    best_indices: list[int] = []

    # Warm start with deterministic capped top-node picks.
    warm_selected: list[int] = []
    warm_player_counts: dict[str, int] = {}
    warm_target_counts: dict[str, int] = {}
    warm_game_counts: dict[str, int] = {}
    warm_script_counts: dict[str, int] = {}
    for idx in range(n):
        if len(warm_selected) >= k:
            break
        if not _is_candidate_feasible(
            idx,
            players=players,
            targets=targets,
            games=games,
            scripts=scripts,
            player_counts=warm_player_counts,
            target_counts=warm_target_counts,
            game_counts=warm_game_counts,
            script_counts=warm_script_counts,
            target_caps=target_caps,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
        ):
            continue
        warm_selected.append(idx)
        player = str(players[idx])
        target = str(targets[idx])
        game = str(games[idx])
        script = str(scripts[idx])
        warm_player_counts[player] = warm_player_counts.get(player, 0) + 1
        warm_target_counts[target] = warm_target_counts.get(target, 0) + 1
        warm_game_counts[game] = warm_game_counts.get(game, 0) + 1
        if script:
            warm_script_counts[script] = warm_script_counts.get(script, 0) + 1
    if len(warm_selected) == k:
        best_indices = warm_selected[:]
        best_score = _score_selected_indices(
            best_indices,
            probs=probs,
            unc=unc,
            node_penalty=node_penalty,
            dep_matrix=dep,
            targets=targets,
            directions=directions,
            games=games,
            lambda_corr=float(board_objective_lambda_corr),
            lambda_conc=float(board_objective_lambda_conc),
            lambda_unc=float(board_objective_lambda_unc),
        )

    def dfs(
        pos: int,
        selected: list[int],
        node_sum: float,
        pair_penalty: float,
        player_counts: dict[str, int],
        target_counts: dict[str, int],
        game_counts: dict[str, int],
        script_counts: dict[str, int],
    ) -> None:
        nonlocal best_score, best_indices
        if node_counter["truncated"]:
            return
        node_counter["count"] += 1
        if node_counter["count"] > max_nodes:
            node_counter["truncated"] = True
            return

        selected_count = len(selected)
        remaining_needed = k - selected_count
        remaining_rows = n - pos
        if remaining_needed <= 0:
            score = float(node_sum) - float(board_objective_lambda_corr) * float(pair_penalty) - float(board_objective_lambda_conc) * _compute_concentration_penalty(
                selected,
                targets=targets,
                directions=directions,
                games=games,
            )
            if score > best_score:
                best_score = score
                best_indices = selected[:]
            return
        if remaining_rows < remaining_needed:
            return

        # Optimistic upper bound ignores pair/concentration penalties.
        upper = float(node_sum) + float(prefix[min(n, pos + remaining_needed)] - prefix[pos])
        if upper <= best_score + 1e-12:
            return

        idx = pos
        if _is_candidate_feasible(
            idx,
            players=players,
            targets=targets,
            games=games,
            scripts=scripts,
            player_counts=player_counts,
            target_counts=target_counts,
            game_counts=game_counts,
            script_counts=script_counts,
            target_caps=target_caps,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
        ):
            player = str(players[idx])
            target = str(targets[idx])
            game = str(games[idx])
            script = str(scripts[idx])

            add_pair_penalty = 0.0
            for existing in selected:
                add_pair_penalty += float(dep[idx, existing])

            player_counts[player] = player_counts.get(player, 0) + 1
            target_counts[target] = target_counts.get(target, 0) + 1
            game_counts[game] = game_counts.get(game, 0) + 1
            if script:
                script_counts[script] = script_counts.get(script, 0) + 1
            selected.append(idx)

            dfs(
                pos + 1,
                selected,
                node_sum=float(node_sum + node_values[idx]),
                pair_penalty=float(pair_penalty + add_pair_penalty),
                player_counts=player_counts,
                target_counts=target_counts,
                game_counts=game_counts,
                script_counts=script_counts,
            )

            selected.pop()
            player_counts[player] -= 1
            if player_counts[player] <= 0:
                del player_counts[player]
            target_counts[target] -= 1
            if target_counts[target] <= 0:
                del target_counts[target]
            game_counts[game] -= 1
            if game_counts[game] <= 0:
                del game_counts[game]
            if script:
                script_counts[script] = script_counts.get(script, 0) - 1
                if script_counts[script] <= 0:
                    script_counts.pop(script, None)

        dfs(
            pos + 1,
            selected,
            node_sum=node_sum,
            pair_penalty=pair_penalty,
            player_counts=player_counts,
            target_counts=target_counts,
            game_counts=game_counts,
            script_counts=script_counts,
        )

    dfs(
        pos=0,
        selected=[],
        node_sum=0.0,
        pair_penalty=0.0,
        player_counts={},
        target_counts={},
        game_counts={},
        script_counts={},
    )

    if not best_indices:
        ranked_fallback = base.sort_values(["board_play_win_prob", "ev_adjusted", "abs_edge"], ascending=[False, False, False]).copy()
        fallback = _apply_portfolio_caps(
            ranked_fallback,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_target=max_plays_per_target,
            max_total_plays=k,
            max_target_plays=max_target_plays,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
        )
        fallback["board_objective_search_truncated"] = bool(node_counter["truncated"])
        fallback["board_objective_solver_mode"] = "fallback_capped_rank"
        fallback["board_objective_score"] = np.nan
        fallback["board_objective_swap_applied"] = False
        fallback["board_objective_instability_enabled"] = bool(instability_active)
        fallback["board_objective_instability_veto_count"] = int(instability_veto_count)
        fallback["board_objective_dynamic_size_enabled"] = bool(board_objective_dynamic_size_enabled)
        fallback["board_objective_dynamic_day_score"] = float(dynamic_day_score) if np.isfinite(dynamic_day_score) else np.nan
        fallback["board_objective_dynamic_cutoff_separation"] = float(dynamic_cutoff_separation) if np.isfinite(dynamic_cutoff_separation) else np.nan
        fallback["board_objective_dynamic_shrink"] = int(dynamic_shrink)
        fallback["board_objective_dynamic_target_size"] = int(dynamic_target_size)
        fallback["board_objective_target_size_requested"] = int(board_size)
        return fallback

    selected_board = universe.iloc[best_indices].copy()
    selected_board["board_objective_search_truncated"] = bool(node_counter["truncated"])
    selected_board["board_objective_solver_mode"] = "exact_branch_and_bound"

    # Swap optimization pass against an append pool from outside the selected board.
    current = selected_board.copy()
    current["_swap_member"] = True
    swap_pool_cap = max(0, int(board_objective_swap_candidates))
    if swap_pool_cap > 0:
        selected_sources = set(pd.to_numeric(current["_source_index"], errors="coerce").fillna(-1).astype(int).tolist())
        swap_pool = base.loc[~base["_source_index"].isin(selected_sources)].copy()
        if not swap_pool.empty:
            swap_pool = swap_pool.sort_values(["board_play_win_prob", "ev_adjusted", "abs_edge", "expected_win_rate"], ascending=[False, False, False, False]).head(swap_pool_cap).copy()
        swap_applied = False
        for _ in range(max(0, int(board_objective_swap_rounds))):
            if swap_pool.empty:
                break
            best_trial_gain = 0.0
            best_trial_board: pd.DataFrame | None = None
            current_score = _score_selected_indices(
                list(range(len(current))),
                probs=pd.to_numeric(current["board_play_win_prob"], errors="coerce").fillna(0.5).to_numpy(dtype="float64"),
                unc=pd.to_numeric(current["board_uncertainty_penalty"], errors="coerce").fillna(0.5).to_numpy(dtype="float64"),
                node_penalty=pd.to_numeric(current.get("board_instability_penalty"), errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
                dep_matrix=_build_pairwise_dependency_matrix(
                    current,
                    same_game_weight=float(board_objective_corr_same_game),
                    same_player_weight=float(board_objective_corr_same_player),
                    same_target_weight=float(board_objective_corr_same_target),
                    same_direction_weight=float(board_objective_corr_same_direction),
                    same_script_cluster_weight=float(board_objective_corr_same_script_cluster),
                ),
                targets=_clean_key_component(current.get("target", pd.Series("", index=current.index))).to_numpy(dtype=str),
                directions=_clean_key_component(current.get("direction", pd.Series("", index=current.index))).to_numpy(dtype=str),
                games=_clean_key_component(current.get("game_key", pd.Series("", index=current.index))).to_numpy(dtype=str),
                lambda_corr=float(board_objective_lambda_corr),
                lambda_conc=float(board_objective_lambda_conc),
                lambda_unc=float(board_objective_lambda_unc),
            )
            for remove_idx in current.index.tolist():
                keep = current.drop(index=remove_idx).copy()
                for _, add_row in swap_pool.iterrows():
                    add_source = int(pd.to_numeric(pd.Series([add_row.get("_source_index", -1)]), errors="coerce").fillna(-1).iloc[0])
                    if add_source in set(pd.to_numeric(keep["_source_index"], errors="coerce").fillna(-1).astype(int).tolist()):
                        continue
                    trial = pd.concat([keep, add_row.to_frame().T], axis=0, ignore_index=True)
                    trial_ranked = trial.sort_values(["board_play_win_prob", "ev_adjusted", "abs_edge"], ascending=[False, False, False]).copy()
                    capped = _apply_portfolio_caps(
                        trial_ranked,
                        max_plays_per_player=max_plays_per_player,
                        max_plays_per_target=max_plays_per_target,
                        max_total_plays=k,
                        max_target_plays=max_target_plays,
                        max_plays_per_game=max_plays_per_game,
                        max_plays_per_script_cluster=max_plays_per_script_cluster,
                    )
                    if len(capped) != k:
                        continue
                    trial_score = _score_selected_indices(
                        list(range(len(capped))),
                        probs=pd.to_numeric(capped["board_play_win_prob"], errors="coerce").fillna(0.5).to_numpy(dtype="float64"),
                        unc=pd.to_numeric(capped["board_uncertainty_penalty"], errors="coerce").fillna(0.5).to_numpy(dtype="float64"),
                        node_penalty=pd.to_numeric(capped.get("board_instability_penalty"), errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
                        dep_matrix=_build_pairwise_dependency_matrix(
                            capped,
                            same_game_weight=float(board_objective_corr_same_game),
                            same_player_weight=float(board_objective_corr_same_player),
                            same_target_weight=float(board_objective_corr_same_target),
                            same_direction_weight=float(board_objective_corr_same_direction),
                            same_script_cluster_weight=float(board_objective_corr_same_script_cluster),
                        ),
                        targets=_clean_key_component(capped.get("target", pd.Series("", index=capped.index))).to_numpy(dtype=str),
                        directions=_clean_key_component(capped.get("direction", pd.Series("", index=capped.index))).to_numpy(dtype=str),
                        games=_clean_key_component(capped.get("game_key", pd.Series("", index=capped.index))).to_numpy(dtype=str),
                        lambda_corr=float(board_objective_lambda_corr),
                        lambda_conc=float(board_objective_lambda_conc),
                        lambda_unc=float(board_objective_lambda_unc),
                    )
                    gain = float(trial_score - current_score)
                    if gain > best_trial_gain + 1e-12:
                        best_trial_gain = gain
                        best_trial_board = capped
            if best_trial_board is None:
                break
            current = best_trial_board.copy()
            swap_applied = True
        current["board_objective_swap_applied"] = bool(swap_applied)
    else:
        current["board_objective_swap_applied"] = False

    final = current.sort_values(["board_play_win_prob", "ev_adjusted", "abs_edge"], ascending=[False, False, False]).copy()
    caps_relaxed = False

    # Keep fixed-K behavior when feasible by filling any residual deficit from the full candidate pool.
    if len(final) < k:
        selected_rows = [row.to_dict() for _, row in final.iterrows()]
        seen_indices = set(pd.to_numeric(final.get("_source_index"), errors="coerce").fillna(-1).astype(int).tolist())
        player_counts, target_counts, game_counts, script_cluster_counts = _selection_counters_from_rows(selected_rows)
        full_caps = _resolve_target_caps(base, max_plays_per_target=max_plays_per_target, max_target_plays=max_target_plays)
        filler_ranked = base.sort_values(["board_play_win_prob", "ev_adjusted", "abs_edge", "expected_win_rate"], ascending=[False, False, False, False]).copy()
        _append_rows_with_caps(
            filler_ranked,
            selected_rows,
            seen_indices,
            player_counts,
            target_counts,
            game_counts,
            script_cluster_counts,
            full_caps,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
            max_total_plays=k,
        )
        final = pd.DataFrame.from_records(selected_rows) if selected_rows else final.iloc[0:0].copy()
        if not final.empty:
            if "board_objective_swap_applied" in final.columns:
                final["board_objective_swap_applied"] = pd.to_numeric(final["board_objective_swap_applied"], errors="coerce").fillna(0).astype(bool)
            else:
                final["board_objective_swap_applied"] = False

    # Mirror baseline behavior: if min-board target is unmet, relax target-family caps.
    target_size = min(requested_min, k) if requested_min > 0 else 0
    if target_size > 0 and len(final) < target_size:
        selected_rows = [row.to_dict() for _, row in final.iterrows()]
        seen_indices = set(pd.to_numeric(final.get("_source_index"), errors="coerce").fillna(-1).astype(int).tolist())
        player_counts, target_counts, game_counts, script_cluster_counts = _selection_counters_from_rows(selected_rows)
        ranked_base = base.sort_values(["board_play_win_prob", "ev_adjusted", "abs_edge", "expected_win_rate"], ascending=[False, False, False, False]).copy()
        relaxed_caps: dict[str, int] = {}
        added = _append_rows_with_caps(
            ranked_base,
            selected_rows,
            seen_indices,
            player_counts,
            target_counts,
            game_counts,
            script_cluster_counts,
            relaxed_caps,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
            max_total_plays=target_size,
            max_new_rows=target_size - len(selected_rows),
        )
        if added > 0:
            caps_relaxed = True
            final = pd.DataFrame.from_records(selected_rows)
            if "board_objective_swap_applied" not in final.columns:
                final["board_objective_swap_applied"] = False
            final["board_objective_solver_mode"] = final.get("board_objective_solver_mode", "exact_branch_and_bound")
            final["board_objective_search_truncated"] = pd.to_numeric(final.get("board_objective_search_truncated"), errors="coerce").fillna(0).astype(bool)

    final["board_objective_score"] = _score_selected_indices(
        list(range(len(final))),
        probs=pd.to_numeric(final["board_play_win_prob"], errors="coerce").fillna(0.5).to_numpy(dtype="float64"),
        unc=pd.to_numeric(final["board_uncertainty_penalty"], errors="coerce").fillna(0.5).to_numpy(dtype="float64"),
        node_penalty=pd.to_numeric(final.get("board_instability_penalty"), errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
        dep_matrix=_build_pairwise_dependency_matrix(
            final,
            same_game_weight=float(board_objective_corr_same_game),
            same_player_weight=float(board_objective_corr_same_player),
            same_target_weight=float(board_objective_corr_same_target),
            same_direction_weight=float(board_objective_corr_same_direction),
            same_script_cluster_weight=float(board_objective_corr_same_script_cluster),
        ),
        targets=_clean_key_component(final.get("target", pd.Series("", index=final.index))).to_numpy(dtype=str),
        directions=_clean_key_component(final.get("direction", pd.Series("", index=final.index))).to_numpy(dtype=str),
        games=_clean_key_component(final.get("game_key", pd.Series("", index=final.index))).to_numpy(dtype=str),
        lambda_corr=float(board_objective_lambda_corr),
        lambda_conc=float(board_objective_lambda_conc),
        lambda_unc=float(board_objective_lambda_unc),
    )
    final["board_objective_instability_enabled"] = bool(instability_active)
    final["board_objective_instability_veto_count"] = int(instability_veto_count)
    final["board_objective_dynamic_size_enabled"] = bool(board_objective_dynamic_size_enabled)
    final["board_objective_dynamic_day_score"] = float(dynamic_day_score) if np.isfinite(dynamic_day_score) else np.nan
    final["board_objective_dynamic_cutoff_separation"] = float(dynamic_cutoff_separation) if np.isfinite(dynamic_cutoff_separation) else np.nan
    final["board_objective_dynamic_shrink"] = int(dynamic_shrink)
    final["board_objective_dynamic_target_size"] = int(dynamic_target_size)
    final["board_objective_target_size_requested"] = int(board_size)
    final["board_caps_relaxed"] = bool(caps_relaxed)
    return final


def _append_rows_with_caps(
    ranked: pd.DataFrame,
    selected_rows: list[dict],
    seen_indices: set,
    player_counts: dict[str, int],
    target_counts: dict[str, int],
    game_counts: dict[str, int],
    script_cluster_counts: dict[str, int],
    caps: dict[str, int],
    max_plays_per_player: int,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
    max_total_plays: int,
    max_new_rows: int | None = None,
) -> int:
    if ranked.empty:
        return 0
    if max_new_rows is not None and int(max_new_rows) <= 0:
        return 0

    added = 0
    for row_index, row in ranked.iterrows():
        if max_total_plays > 0 and len(selected_rows) >= int(max_total_plays):
            break
        if max_new_rows is not None and added >= int(max_new_rows):
            break
        if row_index in seen_indices:
            continue

        player = str(row.get("player", ""))
        target = str(row.get("target", ""))
        game_key = str(row.get("game_key", ""))
        script_cluster = _normalize_script_cluster(row.get("script_cluster_id", ""))

        if max_plays_per_player > 0 and player_counts.get(player, 0) >= int(max_plays_per_player):
            continue
        target_cap = int(caps.get(target, 0))
        if target_cap > 0 and target_counts.get(target, 0) >= target_cap:
            continue
        if max_plays_per_game > 0 and game_counts.get(game_key, 0) >= int(max_plays_per_game):
            continue
        if (
            max_plays_per_script_cluster > 0
            and script_cluster
            and script_cluster_counts.get(script_cluster, 0) >= int(max_plays_per_script_cluster)
        ):
            continue

        selected_rows.append(row.to_dict())
        seen_indices.add(row_index)
        player_counts[player] = player_counts.get(player, 0) + 1
        target_counts[target] = target_counts.get(target, 0) + 1
        game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if script_cluster:
            script_cluster_counts[script_cluster] = script_cluster_counts.get(script_cluster, 0) + 1
        added += 1
    return added


def _selection_counters_from_rows(selected_rows: list[dict]) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    player_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    script_cluster_counts: dict[str, int] = {}
    for row in selected_rows:
        player = str(row.get("player", ""))
        target = str(row.get("target", ""))
        game_key = str(row.get("game_key", ""))
        script_cluster = _normalize_script_cluster(row.get("script_cluster_id", ""))
        player_counts[player] = player_counts.get(player, 0) + 1
        target_counts[target] = target_counts.get(target, 0) + 1
        game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if script_cluster:
            script_cluster_counts[script_cluster] = script_cluster_counts.get(script_cluster, 0) + 1
    return player_counts, target_counts, game_counts, script_cluster_counts


def _select_edge_append_shadow_board(
    candidates: pd.DataFrame,
    max_plays_per_player: int,
    max_plays_per_target: int,
    max_total_plays: int,
    max_target_plays: dict[str, int] | None,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
    append_agreement_min: int,
    append_edge_percentile_min: float,
    append_max_extra_plays: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    base_size = int(max_total_plays) if max_total_plays > 0 else int(len(candidates))
    if base_size <= 0:
        return candidates.iloc[0:0].copy()

    working = candidates.copy()
    working["_source_index"] = working.index

    # 1) Base board is pure edge, fully capped.
    edge_ranked = working.sort_values(["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).copy()
    base_board = _apply_portfolio_caps(
        edge_ranked,
        max_plays_per_player=max_plays_per_player,
        max_plays_per_target=max_plays_per_target,
        max_total_plays=base_size,
        max_target_plays=max_target_plays,
        max_plays_per_game=max_plays_per_game,
        max_plays_per_script_cluster=max_plays_per_script_cluster,
    )
    if base_board.empty:
        return base_board

    base_indices = set(pd.to_numeric(base_board["_source_index"], errors="coerce").dropna().astype(int).tolist())
    base_board = base_board.copy()
    base_board["append_shadow_added"] = False

    extra_cap = max(0, int(append_max_extra_plays))
    if extra_cap <= 0:
        base_board["append_anchor_member"] = 1
        base_board["append_agreement_count"] = np.nan
        base_board["append_edge_percentile"] = np.nan
        base_board["append_sources"] = ""
        return base_board

    # 2) Build strict append-only candidate gates.
    overfetch = int(np.clip(3 * base_size, 1, len(working)))
    idx_e = set(working.sort_values(["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).head(overfetch).index.tolist())
    idx_t = set(
        working.sort_values(["thompson_ev", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False, False])
        .head(overfetch)
        .index.tolist()
    )
    idx_v = set(working.sort_values(["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False]).head(overfetch).index.tolist())

    agreement = working.index.to_series().map(lambda idx: int(idx in idx_e) + int(idx in idx_t) + int(idx in idx_v)).astype(int)
    edge_pct = pd.to_numeric(working["abs_edge"], errors="coerce").fillna(0.0).rank(method="average", pct=True)

    working["append_agreement_count"] = agreement
    working["append_edge_percentile"] = edge_pct
    working["append_sources"] = working.apply(
        lambda row: ",".join(
            part
            for part, enabled in (
                ("E", bool(row.name in idx_e)),
                ("T", bool(row.name in idx_t)),
                ("V", bool(row.name in idx_v)),
            )
            if enabled
        ),
        axis=1,
    )
    quality_mask = (
        (_numeric_series(working, "market_books", 0.0) >= 4.0)
        & (_numeric_series(working, "history_rows", 0.0) >= 35.0)
        & (_numeric_series(working, "final_confidence", 0.0) >= 0.03)
    )
    append_mask = (
        (~working.index.isin(base_indices))
        & (working["append_agreement_count"] >= int(max(1, append_agreement_min)))
        & (working["append_edge_percentile"] >= float(np.clip(append_edge_percentile_min, 0.0, 1.0)))
        & quality_mask
    )
    append_ranked = working.loc[append_mask].sort_values(
        ["append_agreement_count", "append_edge_percentile", "edge", "expected_win_rate", "final_confidence"],
        ascending=[False, False, False, False, False],
    ).copy()

    selected_rows = [row.to_dict() for _, row in base_board.iterrows()]
    seen_indices = set(base_indices)
    player_counts, target_counts, game_counts, script_cluster_counts = _selection_counters_from_rows(selected_rows)

    caps = _resolve_target_caps(working, max_plays_per_target=max_plays_per_target, max_target_plays=max_target_plays)
    # Append mode is intentionally additive; widen target caps by extra_cap.
    widened_caps = {target: (int(cap) + extra_cap if int(cap) > 0 else 0) for target, cap in caps.items()}

    _append_rows_with_caps(
        append_ranked,
        selected_rows,
        seen_indices,
        player_counts,
        target_counts,
        game_counts,
        script_cluster_counts,
        widened_caps,
        max_plays_per_player=max_plays_per_player,
        max_plays_per_game=max_plays_per_game,
        max_plays_per_script_cluster=max_plays_per_script_cluster,
        max_total_plays=base_size + extra_cap,
        max_new_rows=extra_cap,
    )

    out = pd.DataFrame.from_records(selected_rows) if selected_rows else working.iloc[0:0].copy()
    if out.empty:
        return out
    out["_source_index"] = pd.to_numeric(out.get("_source_index"), errors="coerce").fillna(-1).astype(int)
    out["append_shadow_added"] = ~out["_source_index"].isin(base_indices)
    out["append_anchor_member"] = (~out["append_shadow_added"]).astype(int)
    out.loc[~out["append_shadow_added"], "append_sources"] = ""
    out.loc[~out["append_shadow_added"], "append_agreement_count"] = np.nan
    out.loc[~out["append_shadow_added"], "append_edge_percentile"] = np.nan
    return out


def _select_set_theory_board(
    candidates: pd.DataFrame,
    max_plays_per_player: int,
    max_plays_per_target: int,
    max_total_plays: int,
    max_target_plays: dict[str, int] | None,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    board_size = int(max_total_plays) if max_total_plays > 0 else int(len(candidates))
    if board_size <= 0:
        return candidates.iloc[0:0].copy()

    # Edge is the anchor universe. Thompson/EV are used as confirmation overlays.
    overfetch = int(np.clip(3 * board_size, 1, len(candidates)))
    edge_ranked = candidates.sort_values(["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).copy()
    thompson_ranked = candidates.sort_values(
        ["thompson_ev", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False, False]
    ).copy()
    ev_ranked = candidates.sort_values(["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False]).copy()

    edge_idx = set(edge_ranked.head(overfetch).index.tolist())
    thompson_idx = set(thompson_ranked.head(overfetch).index.tolist())
    ev_idx = set(ev_ranked.head(overfetch).index.tolist())

    scored = candidates.loc[candidates.index.isin(edge_idx)].copy()
    if scored.empty:
        return scored
    scored["in_edge_set"] = True
    scored["in_thompson_set"] = scored.index.isin(thompson_idx)
    scored["in_ev_set"] = scored.index.isin(ev_idx)
    scored["agreement_count"] = 1 + scored["in_thompson_set"].astype(int) + scored["in_ev_set"].astype(int)
    scored["set_sources"] = scored.apply(
        lambda row: ",".join(
            part
            for part, enabled in (
                ("E", bool(row.get("in_edge_set"))),
                ("T", bool(row.get("in_thompson_set"))),
                ("V", bool(row.get("in_ev_set"))),
            )
            if enabled
        ),
        axis=1,
    )

    scored["set_group"] = "anchor_fallback"
    scored.loc[scored["in_thompson_set"] | scored["in_ev_set"], "set_group"] = "strong_expansion"
    scored.loc[scored["in_thompson_set"] & scored["in_ev_set"], "set_group"] = "core"
    scored["set_strength"] = scored["set_group"].map({"core": 3, "strong_expansion": 2, "anchor_fallback": 1}).fillna(0).astype(int)

    scored["z_edge"] = _zscore_series(scored["edge"])
    scored["z_expected_win_rate"] = _zscore_series(scored["expected_win_rate"])
    scored["z_ev_adjusted"] = _zscore_series(scored["ev_adjusted"])
    scored["consensus_score"] = (
        0.45 * scored["z_edge"]
        + 0.20 * scored["z_expected_win_rate"]
        + 0.20 * scored["z_ev_adjusted"]
        + 0.15 * (scored["agreement_count"] - 1.0)
    )

    sort_consensus = ["consensus_score", "agreement_count", "edge", "expected_win_rate", "ev_adjusted", "abs_edge", "final_confidence"]
    core_ranked = scored.loc[scored["set_group"].eq("core")].sort_values(sort_consensus, ascending=[False] * len(sort_consensus)).copy()
    strong_ranked = scored.loc[scored["set_group"].eq("strong_expansion")].sort_values(sort_consensus, ascending=[False] * len(sort_consensus)).copy()
    fallback_ranked = scored.loc[scored["set_group"].eq("anchor_fallback")].sort_values(
        ["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]
    ).copy()

    selected_rows: list[dict] = []
    seen_indices: set = set()
    player_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    script_cluster_counts: dict[str, int] = {}
    caps = _resolve_target_caps(scored, max_plays_per_target=max_plays_per_target, max_target_plays=max_target_plays)

    _append_rows_with_caps(
        core_ranked,
        selected_rows,
        seen_indices,
        player_counts,
        target_counts,
        game_counts,
        script_cluster_counts,
        caps,
        max_plays_per_player=max_plays_per_player,
        max_plays_per_game=max_plays_per_game,
        max_plays_per_script_cluster=max_plays_per_script_cluster,
        max_total_plays=board_size,
    )
    _append_rows_with_caps(
        strong_ranked,
        selected_rows,
        seen_indices,
        player_counts,
        target_counts,
        game_counts,
        script_cluster_counts,
        caps,
        max_plays_per_player=max_plays_per_player,
        max_plays_per_game=max_plays_per_game,
        max_plays_per_script_cluster=max_plays_per_script_cluster,
        max_total_plays=board_size,
    )

    if len(selected_rows) < board_size:
        _append_rows_with_caps(
            fallback_ranked,
            selected_rows,
            seen_indices,
            player_counts,
            target_counts,
            game_counts,
            script_cluster_counts,
            caps,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
            max_total_plays=board_size,
            max_new_rows=board_size - len(selected_rows),
        )

    if not selected_rows:
        return scored.iloc[0:0].copy()
    return pd.DataFrame.from_records(selected_rows)


def _apply_portfolio_caps(
    ranked: pd.DataFrame,
    max_plays_per_player: int,
    max_plays_per_target: int,
    max_total_plays: int,
    max_target_plays: dict[str, int] | None,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    selected_rows: list[dict] = []
    player_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    script_cluster_counts: dict[str, int] = {}

    caps = _resolve_target_caps(ranked, max_plays_per_target=max_plays_per_target, max_target_plays=max_target_plays)

    for _, row in ranked.iterrows():
        if max_total_plays > 0 and len(selected_rows) >= int(max_total_plays):
            break

        player = str(row.get("player", ""))
        target = str(row.get("target", ""))
        game_key = str(row.get("game_key", ""))
        script_cluster = _normalize_script_cluster(row.get("script_cluster_id", ""))

        if max_plays_per_player > 0 and player_counts.get(player, 0) >= int(max_plays_per_player):
            continue
        target_cap = int(caps.get(target, 0))
        if target_cap > 0 and target_counts.get(target, 0) >= target_cap:
            continue
        if max_plays_per_game > 0 and game_counts.get(game_key, 0) >= int(max_plays_per_game):
            continue
        if (
            max_plays_per_script_cluster > 0
            and script_cluster
            and script_cluster_counts.get(script_cluster, 0) >= int(max_plays_per_script_cluster)
        ):
            continue

        selected_rows.append(row.to_dict())
        player_counts[player] = player_counts.get(player, 0) + 1
        target_counts[target] = target_counts.get(target, 0) + 1
        game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if script_cluster:
            script_cluster_counts[script_cluster] = script_cluster_counts.get(script_cluster, 0) + 1

    if not selected_rows:
        return ranked.iloc[0:0].copy()
    return pd.DataFrame.from_records(selected_rows)


def compute_final_board(
    plays: pd.DataFrame,
    american_odds: int = -110,
    min_ev: float = 0.0,
    min_final_confidence: float = 0.02,
    min_recommendation: str = "consider",
    selection_mode: str = "thompson_ev",
    ranking_mode: str = "ev_adjusted",
    max_plays_per_player: int = 1,
    max_plays_per_target: int = 8,
    max_total_plays: int = 20,
    min_board_plays: int = 0,
    max_target_plays: dict[str, int] | None = None,
    max_plays_per_game: int = 2,
    max_plays_per_script_cluster: int = 2,
    non_pts_min_gap_percentile: float = 0.90,
    edge_adjust_k: float = 0.30,
    thompson_temperature: float = 1.0,
    thompson_seed: int = 17,
    min_bet_win_rate: float = 0.57,
    medium_bet_win_rate: float = 0.60,
    full_bet_win_rate: float = 0.65,
    medium_tier_percentile: float = 0.80,
    strong_tier_percentile: float = 0.90,
    elite_tier_percentile: float = 0.95,
    small_bet_fraction: float = 0.005,
    medium_bet_fraction: float = 0.010,
    full_bet_fraction: float = 0.015,
    max_bet_fraction: float = 0.02,
    max_total_bet_fraction: float = 0.05,
    sizing_method: str = "tiered_probability",
    flat_bet_fraction: float = 0.0,
    coarse_low_bet_fraction: float = 0.003,
    coarse_mid_bet_fraction: float = 0.005,
    coarse_high_bet_fraction: float = 0.007,
    coarse_high_max_share: float = 0.30,
    coarse_mid_max_share: float = 0.50,
    coarse_high_max_plays: int = 0,
    coarse_mid_max_plays: int = 0,
    coarse_score_alpha_uncertainty: float = 0.18,
    coarse_score_beta_dependency: float = 0.12,
    coarse_score_gamma_support: float = 0.08,
    coarse_score_model: str = "legacy",
    coarse_score_delta_prob_weight: float = 0.0,
    coarse_score_ev_weight: float = 0.0,
    coarse_score_risk_weight: float = 0.0,
    coarse_score_recency_weight: float = 0.0,
    staking_bucket_model_payload: dict | None = None,
    staking_bucket_model_month: str | None = None,
    staking_bucket_model_min_rows: int = 0,
    belief_uncertainty_lower: float = BELIEF_UNCERTAINTY_LOWER,
    belief_uncertainty_upper: float = BELIEF_UNCERTAINTY_UPPER,
    append_agreement_min: int = 3,
    append_edge_percentile_min: float = 0.90,
    append_max_extra_plays: int = 3,
    board_objective_overfetch: float = 4.0,
    board_objective_candidate_limit: int = 36,
    board_objective_max_search_nodes: int = 750000,
    board_objective_lambda_corr: float = 0.12,
    board_objective_lambda_conc: float = 0.07,
    board_objective_lambda_unc: float = 0.06,
    board_objective_corr_same_game: float = 0.65,
    board_objective_corr_same_player: float = 1.0,
    board_objective_corr_same_target: float = 0.15,
    board_objective_corr_same_direction: float = 0.05,
    board_objective_corr_same_script_cluster: float = 0.30,
    board_objective_swap_candidates: int = 18,
    board_objective_swap_rounds: int = 2,
    board_objective_instability_enabled: bool = False,
    board_objective_lambda_shadow_disagreement: float = 0.0,
    board_objective_lambda_segment_weakness: float = 0.0,
    board_objective_instability_near_cutoff_window: int = 3,
    board_objective_instability_top_protected: int = 3,
    board_objective_instability_veto_enabled: bool = False,
    board_objective_instability_veto_quantile: float = 0.85,
    board_objective_dynamic_size_enabled: bool = False,
    board_objective_dynamic_size_max_shrink: int = 0,
    board_objective_dynamic_size_trigger: float = 0.62,
    max_history_staleness_days: int = 0,
    min_recency_factor: float = 0.0,
    selected_board_calibrator: dict | None = None,
    selected_board_calibration_month: str | None = None,
    learned_gate_payload: dict | None = None,
    learned_gate_month: str | None = None,
    learned_gate_min_rows: int = 0,
) -> pd.DataFrame:
    out = plays.copy()
    if out.empty:
        return out

    payout = american_profit_per_unit(american_odds)
    out["expected_win_rate"] = pd.to_numeric(out["expected_win_rate"], errors="coerce").fillna(0.5)
    out["expected_push_rate"] = _numeric_series(out, "expected_push_rate", 0.0).clip(lower=0.0, upper=1.0)
    if "expected_loss_rate" in out.columns:
        out["expected_loss_rate"] = _numeric_series(out, "expected_loss_rate", 0.0)
    else:
        out["expected_loss_rate"] = np.clip(1.0 - out["expected_win_rate"] - out["expected_push_rate"], 0.0, 1.0)
    out["selector_expected_win_rate"] = out["expected_win_rate"].clip(lower=0.0, upper=1.0)
    out["gap_percentile"] = pd.to_numeric(out["gap_percentile"], errors="coerce").fillna(0.0)
    out["belief_uncertainty"] = _numeric_series(out, "belief_uncertainty", 1.0)
    normalized_belief = normalize_belief_uncertainty(
        out["belief_uncertainty"],
        default=1.0,
        lower=float(belief_uncertainty_lower),
        upper=float(belief_uncertainty_upper),
    )
    belief_conf = belief_confidence_factor(
        out["belief_uncertainty"],
        default=1.0,
        lower=float(belief_uncertainty_lower),
        upper=float(belief_uncertainty_upper),
    )
    if "belief_uncertainty_normalized" in out.columns:
        out["belief_uncertainty_normalized"] = pd.to_numeric(out["belief_uncertainty_normalized"], errors="coerce").fillna(normalized_belief)
    else:
        out["belief_uncertainty_normalized"] = normalized_belief
    if "belief_confidence_factor" in out.columns:
        out["belief_confidence_factor"] = pd.to_numeric(out["belief_confidence_factor"], errors="coerce").fillna(belief_conf)
    else:
        out["belief_confidence_factor"] = belief_conf
    out["feasibility"] = _numeric_series(out, "feasibility", 0.0)
    out["abs_edge"] = _numeric_series(out, "abs_edge", 0.0)
    out["edge"] = _numeric_series(out, "edge", 0.0)
    out["posterior_alpha"] = _numeric_series(out, "posterior_alpha", 1.0)
    out["posterior_beta"] = _numeric_series(out, "posterior_beta", 1.0)
    out["posterior_variance"] = _numeric_series(out, "posterior_variance", 0.25)

    out["game_key"] = _build_game_key(out)
    out["market_prior_win_rate"] = 0.5
    base_confidence = out["gap_percentile"] * out["belief_confidence_factor"] * np.clip(out["feasibility"], 0.0, None)
    confidence_blend = np.clip(pd.to_numeric(base_confidence, errors="coerce").fillna(0.0), 0.0, 1.0)
    uncertainty_penalty = np.clip(np.sqrt(np.clip(out["posterior_variance"], 0.0, 1.0)) * 0.6, 0.0, 0.45)
    out["calibration_blend_weight"] = np.clip(confidence_blend * (1.0 - uncertainty_penalty), 0.10, 0.95)
    out["calibrated_win_rate"] = (
        out["calibration_blend_weight"] * out["expected_win_rate"]
        + (1.0 - out["calibration_blend_weight"]) * out["market_prior_win_rate"]
    )
    out["expected_win_rate"] = out["calibrated_win_rate"].clip(lower=0.0, upper=1.0 - out["expected_push_rate"])
    out["expected_loss_rate"] = np.clip(1.0 - out["expected_win_rate"] - out["expected_push_rate"], 0.0, 1.0)
    out["p_calibrated"] = out["expected_win_rate"]
    out["board_play_win_prob"] = out["p_calibrated"]

    out["ev"] = out["expected_win_rate"] * payout - out["expected_loss_rate"]
    out["final_confidence"] = base_confidence
    out["recommendation_rank"] = out["recommendation"].map(recommendation_rank)
    edge_baseline = out.groupby("target")["abs_edge"].transform(lambda s: s.median() if len(s) else 1.0).replace(0.0, 1.0)
    out["edge_scale"] = (out["abs_edge"] / edge_baseline).clip(lower=0.50, upper=2.50)
    out["ev_adjusted"] = out["ev"] * (1.0 + float(edge_adjust_k) * (out["edge_scale"] - 1.0))
    out["ranking_mode"] = str(selection_mode or ranking_mode)

    temp = max(float(thompson_temperature), 1e-6)
    out["thompson_alpha"] = out["posterior_alpha"] / temp
    out["thompson_beta"] = out["posterior_beta"] / temp

    thompson_conditional: list[float] = []
    for _, sample_row in out.iterrows():
        seed = _stable_seed_from_row(sample_row, int(thompson_seed))
        rng = np.random.default_rng(seed)
        alpha = max(0.1, float(sample_row["thompson_alpha"]))
        beta = max(0.1, float(sample_row["thompson_beta"]))
        thompson_conditional.append(float(rng.beta(alpha, beta)))
    out["thompson_conditional_win_rate"] = thompson_conditional
    resolved_share = np.clip(1.0 - out["expected_push_rate"], 0.0, 1.0)
    out["thompson_win_rate"] = np.clip(out["thompson_conditional_win_rate"] * resolved_share, 0.0, resolved_share)
    out["thompson_loss_rate"] = np.clip(resolved_share - out["thompson_win_rate"], 0.0, 1.0)
    out["thompson_ev"] = out["thompson_win_rate"] * payout - out["thompson_loss_rate"]

    if "conditional_eligible_for_board" in out.columns:
        eligible_mask = pd.to_numeric(out["conditional_eligible_for_board"], errors="coerce").fillna(0).astype(bool)
        out = out.loc[eligible_mask].copy()
        if out.empty:
            return out

    min_recency = float(min_recency_factor)
    if min_recency > 0.0 and "recency_factor" in out.columns:
        out["recency_factor"] = _numeric_series(out, "recency_factor", 0.0)
        out = out.loc[out["recency_factor"] >= min_recency].copy()
        if out.empty:
            return out

    staleness_cap = int(max_history_staleness_days)
    if staleness_cap > 0 and "market_date" in out.columns and "last_history_date" in out.columns:
        market_dates = pd.to_datetime(out["market_date"], errors="coerce")
        history_dates = pd.to_datetime(out["last_history_date"], errors="coerce")
        staleness_days = (market_dates - history_dates).dt.days
        out["history_staleness_days"] = staleness_days
        out = out.loc[staleness_days.isna() | (staleness_days <= staleness_cap)].copy()
        if out.empty:
            return out

    requested_min_board = max(0, int(min_board_plays))
    if max_total_plays > 0:
        requested_min_board = min(requested_min_board, int(max_total_plays))

    out = out.loc[out["recommendation_rank"] <= minimum_recommendation_rank(min_recommendation)].copy()
    out = out.loc[out["final_confidence"] >= float(min_final_confidence)].copy()
    out = out.loc[(out["target"] == "PTS") | (out["gap_percentile"] >= float(non_pts_min_gap_percentile))].copy()
    if out.empty:
        return out

    effective_min_ev = float(min_ev)
    if requested_min_board > 0:
        ev_sorted = pd.to_numeric(out["ev"], errors="coerce").fillna(-np.inf).sort_values(ascending=False).reset_index(drop=True)
        if not ev_sorted.empty:
            floor_index = max(0, min(int(requested_min_board), len(ev_sorted)) - 1)
            adaptive_floor = float(ev_sorted.iloc[floor_index])
            if np.isfinite(adaptive_floor):
                effective_min_ev = min(effective_min_ev, adaptive_floor)
    out = out.loc[pd.to_numeric(out["ev"], errors="coerce") >= float(effective_min_ev)].copy()
    if out.empty:
        return out
    out["ev_gate_effective_min_ev"] = float(effective_min_ev)
    out["ev_gate_relaxed"] = bool(effective_min_ev < float(min_ev))
    gate_backfill_pool = out.iloc[0:0].copy()
    if isinstance(learned_gate_payload, dict):
        gate_frame = pd.DataFrame(
            {
                "expected_win_rate": pd.to_numeric(out.get("expected_win_rate"), errors="coerce").fillna(0.5),
                "target": out.get("target", pd.Series("", index=out.index)),
                "direction": out.get("direction", pd.Series("", index=out.index)),
                "market_date": out.get("market_date", pd.Series("", index=out.index)),
            },
            index=out.index,
        )
        gate_pass_mask, gate_threshold, gate_source, gate_month, gate_details = apply_learned_pool_gate_fn(
            gate_frame,
            payload=learned_gate_payload,
            run_date_hint=learned_gate_month,
            prob_col="expected_win_rate",
            target_col="target",
            direction_col="direction",
        )
        gate_pass_mask = pd.to_numeric(gate_pass_mask, errors="coerce").fillna(0).astype(bool)
        out["learned_gate_threshold"] = pd.to_numeric(gate_threshold, errors="coerce").fillna(float("-inf"))
        out["learned_gate_source"] = gate_source.astype(str)
        out["learned_gate_month"] = str(gate_month or "")
        out["learned_gate_pass"] = gate_pass_mask
        out["learned_gate_pass_rows"] = int(gate_pass_mask.sum())
        out["learned_gate_enabled"] = bool(gate_details.get("enabled", True))
        gate_required_rows = max(1, int(learned_gate_min_rows) if learned_gate_min_rows is not None else 1)
        out["learned_gate_required_rows"] = int(gate_required_rows)
        enforce_gate = bool(gate_pass_mask.sum() >= int(gate_required_rows))
        out["learned_gate_enforced"] = bool(enforce_gate)
        out["learned_gate_blocked_reason"] = "" if enforce_gate else "insufficient_pass_rows"
        if enforce_gate:
            gate_backfill_pool = out.loc[~gate_pass_mask].copy()
            out = out.loc[gate_pass_mask].copy()
            if out.empty:
                return out
    else:
        out["learned_gate_enabled"] = False
        out["learned_gate_enforced"] = False
        out["learned_gate_pass"] = True
        out["learned_gate_pass_rows"] = int(len(out))
        out["learned_gate_required_rows"] = 0
        out["learned_gate_blocked_reason"] = ""

    effective_mode = str(selection_mode or ranking_mode)
    rank_columns = ["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    caps_already_applied = False
    if effective_mode == "xgb_ltr" and "xgb_ltr_score" in out.columns:
        out["xgb_ltr_score"] = pd.to_numeric(out["xgb_ltr_score"], errors="coerce").fillna(-1.0)
        rank_columns = ["xgb_ltr_score", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    elif effective_mode == "robust_reranker" and "robust_reranker_prob" in out.columns:
        out["robust_reranker_prob"] = pd.to_numeric(out["robust_reranker_prob"], errors="coerce").fillna(-1.0)
        out["robust_reranker_blend_raw"] = _numeric_series(out, "robust_reranker_blend_raw", -1.0)
        rank_columns = ["robust_reranker_prob", "robust_reranker_blend_raw", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    elif effective_mode == "edge":
        rank_columns = ["edge", "abs_edge", "expected_win_rate", "final_confidence"]
    elif effective_mode == "abs_edge":
        rank_columns = ["abs_edge", "ev_adjusted", "expected_win_rate", "final_confidence"]
    elif effective_mode == "thompson_ev":
        rank_columns = ["thompson_ev", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    elif effective_mode == "set_theory":
        out = _select_set_theory_board(
            out,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_target=max_plays_per_target,
            max_total_plays=max_total_plays,
            max_target_plays=max_target_plays,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
        )
        caps_already_applied = True
        rank_columns = ["set_strength", "consensus_score", "agreement_count", "expected_win_rate", "ev_adjusted", "abs_edge"]
    elif effective_mode == "edge_append_shadow":
        out = _select_edge_append_shadow_board(
            out,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_target=max_plays_per_target,
            max_total_plays=max_total_plays,
            max_target_plays=max_target_plays,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
            append_agreement_min=append_agreement_min,
            append_edge_percentile_min=append_edge_percentile_min,
            append_max_extra_plays=append_max_extra_plays,
        )
        caps_already_applied = True
        rank_columns = ["append_anchor_member", "edge", "abs_edge", "expected_win_rate", "final_confidence"]
    elif effective_mode == "board_objective":
        out = _select_board_objective_board(
            out,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_target=max_plays_per_target,
            max_total_plays=max_total_plays,
            max_target_plays=max_target_plays,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
            min_board_plays=min_board_plays,
            board_objective_overfetch=board_objective_overfetch,
            board_objective_candidate_limit=board_objective_candidate_limit,
            board_objective_max_search_nodes=board_objective_max_search_nodes,
            board_objective_lambda_corr=board_objective_lambda_corr,
            board_objective_lambda_conc=board_objective_lambda_conc,
            board_objective_lambda_unc=board_objective_lambda_unc,
            board_objective_corr_same_game=board_objective_corr_same_game,
            board_objective_corr_same_player=board_objective_corr_same_player,
            board_objective_corr_same_target=board_objective_corr_same_target,
            board_objective_corr_same_direction=board_objective_corr_same_direction,
            board_objective_corr_same_script_cluster=board_objective_corr_same_script_cluster,
            board_objective_swap_candidates=board_objective_swap_candidates,
            board_objective_swap_rounds=board_objective_swap_rounds,
            payout_per_unit=float(payout),
            staking_bucket_model_payload=staking_bucket_model_payload,
            staking_bucket_model_month=staking_bucket_model_month,
            staking_bucket_model_min_rows=int(staking_bucket_model_min_rows),
            board_objective_instability_enabled=bool(board_objective_instability_enabled),
            board_objective_lambda_shadow_disagreement=float(board_objective_lambda_shadow_disagreement),
            board_objective_lambda_segment_weakness=float(board_objective_lambda_segment_weakness),
            board_objective_instability_near_cutoff_window=int(board_objective_instability_near_cutoff_window),
            board_objective_instability_top_protected=int(board_objective_instability_top_protected),
            board_objective_instability_veto_enabled=bool(board_objective_instability_veto_enabled),
            board_objective_instability_veto_quantile=float(board_objective_instability_veto_quantile),
            board_objective_dynamic_size_enabled=bool(board_objective_dynamic_size_enabled),
            board_objective_dynamic_size_max_shrink=int(board_objective_dynamic_size_max_shrink),
            board_objective_dynamic_size_trigger=float(board_objective_dynamic_size_trigger),
        )
        caps_already_applied = True
        rank_columns = ["board_play_win_prob", "ev_adjusted", "abs_edge", "final_confidence"]

    if not caps_already_applied:
        ranked_pool = out.sort_values(rank_columns, ascending=[False] * len(rank_columns)).copy()
        out = _apply_portfolio_caps(
            ranked_pool,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_target=max_plays_per_target,
            max_total_plays=max_total_plays,
            max_target_plays=max_target_plays,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
        )
        caps_relaxed = False
        if requested_min_board > 0 and len(out) < requested_min_board:
            relaxed_target_caps = None
            relaxed_per_target_cap = 0
            relaxed_out = _apply_portfolio_caps(
                ranked_pool,
                max_plays_per_player=max_plays_per_player,
                max_plays_per_target=relaxed_per_target_cap,
                max_total_plays=max_total_plays,
                max_target_plays=relaxed_target_caps,
                max_plays_per_game=max_plays_per_game,
                max_plays_per_script_cluster=max_plays_per_script_cluster,
            )
            if len(relaxed_out) > len(out):
                out = relaxed_out
                caps_relaxed = True
        if not out.empty:
            out["board_caps_relaxed"] = bool(caps_relaxed)
    else:
        if out.empty:
            return out
        missing_rank_cols = [column for column in rank_columns if column not in out.columns]
        if missing_rank_cols:
            return out.iloc[0:0].copy()
        out = out.sort_values(rank_columns, ascending=[False] * len(rank_columns)).copy()
        if "board_caps_relaxed" in out.columns:
            out["board_caps_relaxed"] = pd.to_numeric(out["board_caps_relaxed"], errors="coerce").fillna(0).astype(bool)
        else:
            out["board_caps_relaxed"] = False
    if out.empty:
        return out

    # Single canonical published probability for downstream sizing/UI/diagnostics.
    if "board_play_win_prob" in out.columns:
        canonical = pd.to_numeric(out["board_play_win_prob"], errors="coerce").fillna(
            pd.to_numeric(out["expected_win_rate"], errors="coerce").fillna(0.5)
        )
    else:
        canonical = pd.to_numeric(out["expected_win_rate"], errors="coerce").fillna(0.5)
    out["selected_board_prob_raw"] = canonical.astype("float64")
    calibration_frame = pd.DataFrame(
        {
            "target": out.get("target", pd.Series("", index=out.index)),
            "direction": out.get("direction", pd.Series("", index=out.index)),
            "board_play_win_prob": out["selected_board_prob_raw"],
            "market_date": out.get("market_date", pd.Series("", index=out.index)),
        },
        index=out.index,
    )
    calibrated_probs, calibration_source, calibration_month = apply_selected_board_calibration_fn(
        calibration_frame,
        payload=selected_board_calibrator,
        run_date_hint=selected_board_calibration_month,
        prob_col="board_play_win_prob",
        target_col="target",
        direction_col="direction",
    )
    out["selected_board_calibration_source"] = calibration_source.reindex(out.index).fillna("identity")
    out["selected_board_calibration_month"] = str(calibration_month or "")
    canonical = pd.to_numeric(calibrated_probs, errors="coerce").fillna(out["selected_board_prob_raw"])
    out["p_calibrated"] = canonical.clip(lower=0.0, upper=1.0 - out["expected_push_rate"])
    out["board_play_win_prob"] = out["p_calibrated"]
    out["expected_win_rate"] = out["p_calibrated"]
    out["expected_loss_rate"] = np.clip(1.0 - out["expected_win_rate"] - out["expected_push_rate"], 0.0, 1.0)
    out["ev"] = out["expected_win_rate"] * payout - out["expected_loss_rate"]
    out["ev_adjusted"] = out["ev"] * (1.0 + float(edge_adjust_k) * (out["edge_scale"] - 1.0))

    effective_sizing_method = str(sizing_method or "tiered_probability").strip().lower()
    if effective_sizing_method == "flat_fraction":
        sized_out = out.copy()
        per_play_fraction = float(flat_bet_fraction)
        if per_play_fraction <= 0.0:
            per_play_fraction = float(small_bet_fraction)
        per_play_fraction = float(np.clip(per_play_fraction, 0.0, max_bet_fraction))
        sized_out["allocation_tier"] = "flat"
        sized_out["allocation_action"] = "flat"
        sized_out["allocation_action_from_tier"] = "flat"
        sized_out["allocation_action_from_probability"] = "flat"
        sized_out["allocation_action_level"] = 1 if per_play_fraction > 0.0 else 0
        sized_out["bet_fraction_raw"] = per_play_fraction
        total_raw_fraction = per_play_fraction * float(len(sized_out))
        scale = 1.0
        if total_raw_fraction > 0.0 and float(max_total_bet_fraction) > 0.0:
            scale = min(1.0, float(max_total_bet_fraction) / total_raw_fraction)
        sized_out["bet_fraction_scale"] = float(scale)
        sized_out["bet_fraction"] = pd.to_numeric(sized_out["bet_fraction_raw"], errors="coerce").fillna(0.0) * float(scale)
    elif effective_sizing_method == "coarse_bucket":
        sized_out = _apply_coarse_bucket_sizing(
            out,
            max_bet_fraction=float(max_bet_fraction),
            max_total_bet_fraction=float(max_total_bet_fraction),
            low_bet_fraction=float(coarse_low_bet_fraction),
            mid_bet_fraction=float(coarse_mid_bet_fraction),
            high_bet_fraction=float(coarse_high_bet_fraction),
            high_max_share=float(coarse_high_max_share),
            mid_max_share=float(coarse_mid_max_share),
            high_max_plays=int(coarse_high_max_plays),
            mid_max_plays=int(coarse_mid_max_plays),
            score_alpha_uncertainty=float(coarse_score_alpha_uncertainty),
            score_beta_dependency=float(coarse_score_beta_dependency),
            score_gamma_support=float(coarse_score_gamma_support),
            score_model=str(coarse_score_model),
            score_delta_prob_weight=float(coarse_score_delta_prob_weight),
            score_ev_weight=float(coarse_score_ev_weight),
            score_risk_weight=float(coarse_score_risk_weight),
            score_recency_weight=float(coarse_score_recency_weight),
            payout_per_unit=float(payout),
            staking_bucket_model_payload=staking_bucket_model_payload,
            staking_bucket_model_month=staking_bucket_model_month,
            staking_bucket_model_min_rows=int(staking_bucket_model_min_rows),
            same_game_weight=float(board_objective_corr_same_game),
            same_player_weight=float(board_objective_corr_same_player),
            same_target_weight=float(board_objective_corr_same_target),
            same_direction_weight=float(board_objective_corr_same_direction),
            same_script_cluster_weight=float(board_objective_corr_same_script_cluster),
        )
    else:
        sized_out = apply_tiered_bet_sizing(
            out,
            expected_win_rate_col="expected_win_rate",
            gap_percentile_col="gap_percentile",
            min_bet_win_rate=min_bet_win_rate,
            medium_bet_win_rate=medium_bet_win_rate,
            full_bet_win_rate=full_bet_win_rate,
            medium_tier_percentile=medium_tier_percentile,
            strong_tier_percentile=strong_tier_percentile,
            elite_tier_percentile=elite_tier_percentile,
            small_bet_fraction=small_bet_fraction,
            medium_bet_fraction=medium_bet_fraction,
            full_bet_fraction=full_bet_fraction,
            max_bet_fraction=max_bet_fraction,
            max_total_bet_fraction=max_total_bet_fraction,
        )
    sized_out["sizing_method"] = effective_sizing_method
    sized_out["expected_profit_fraction"] = pd.to_numeric(sized_out["bet_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(
        sized_out["ev"], errors="coerce"
    ).fillna(0.0)
    active_mask = pd.to_numeric(sized_out["bet_fraction"], errors="coerce").fillna(0.0) > 0.0
    if active_mask.any():
        out = sized_out.loc[active_mask].copy()
    else:
        # Fallback: keep a small fractional allocation on the best-ranked plays
        # so the board remains actionable when strict tier gates reject all rows.
        fallback_fraction = float(np.clip(small_bet_fraction, 0.0, max_bet_fraction))
        if fallback_fraction <= 0.0:
            return sized_out.iloc[0:0].copy()
        out = sized_out.head(max_total_plays if max_total_plays > 0 else len(sized_out)).copy()
        out["allocation_tier"] = "fallback_small"
        out["allocation_action"] = "fallback_small"
        out["bet_fraction_raw"] = fallback_fraction
        out["bet_fraction_scale"] = 1.0
        out["bet_fraction"] = fallback_fraction
        out["expected_profit_fraction"] = pd.to_numeric(out["bet_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(
            out["ev"], errors="coerce"
        ).fillna(0.0)

    if requested_min_board > 0 and len(out) < requested_min_board:
        target_size = min(requested_min_board, int(max_total_plays) if int(max_total_plays) > 0 else requested_min_board)
        deficit = max(0, int(target_size) - int(len(out)))
        if deficit > 0:
            remaining = sized_out.loc[~sized_out.index.isin(out.index)].copy()
            filler = remaining.head(deficit).copy()
            if not filler.empty:
                fallback_fraction = float(np.clip(small_bet_fraction, 0.0, max_bet_fraction))
                filler["allocation_tier"] = "fallback_small"
                filler["allocation_action"] = "fallback_small"
                filler["bet_fraction_raw"] = fallback_fraction
                filler["bet_fraction_scale"] = 1.0
                filler["bet_fraction"] = fallback_fraction
                filler["expected_profit_fraction"] = pd.to_numeric(filler["bet_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(
                    filler["ev"], errors="coerce"
                ).fillna(0.0)
                filler["learned_gate_override_fill"] = False
                out = pd.concat([out, filler], axis=0, ignore_index=False)
                deficit = max(0, int(target_size) - int(len(out)))
            if deficit > 0 and not gate_backfill_pool.empty:
                gate_remaining = gate_backfill_pool.loc[~gate_backfill_pool.index.isin(out.index)].copy()
                if not gate_remaining.empty:
                    gate_remaining = gate_remaining.sort_values(rank_columns, ascending=[False] * len(rank_columns)).head(deficit).copy()
                    fallback_fraction = float(np.clip(small_bet_fraction, 0.0, max_bet_fraction))
                    gate_remaining["allocation_tier"] = "fallback_small"
                    gate_remaining["allocation_action"] = "fallback_small"
                    gate_remaining["bet_fraction_raw"] = fallback_fraction
                    gate_remaining["bet_fraction_scale"] = 1.0
                    gate_remaining["bet_fraction"] = fallback_fraction
                    gate_remaining["expected_profit_fraction"] = pd.to_numeric(gate_remaining["bet_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(
                        gate_remaining["ev"], errors="coerce"
                    ).fillna(0.0)
                    gate_remaining["learned_gate_override_fill"] = True
                    out = pd.concat([out, gate_remaining], axis=0, ignore_index=False)

    if max_total_bet_fraction > 0 and float(pd.to_numeric(out.get("bet_fraction"), errors="coerce").fillna(0.0).sum()) > float(max_total_bet_fraction):
        scale = float(max_total_bet_fraction) / float(pd.to_numeric(out.get("bet_fraction"), errors="coerce").fillna(0.0).sum())
        out["bet_fraction"] = pd.to_numeric(out.get("bet_fraction"), errors="coerce").fillna(0.0) * scale
        if "bet_fraction_scale" in out.columns:
            out["bet_fraction_scale"] = pd.to_numeric(out["bet_fraction_scale"], errors="coerce").fillna(1.0) * scale
        out["expected_profit_fraction"] = pd.to_numeric(out["bet_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(out["ev"], errors="coerce").fillna(0.0)
    if "learned_gate_override_fill" not in out.columns:
        out["learned_gate_override_fill"] = False
    else:
        out["learned_gate_override_fill"] = pd.to_numeric(out["learned_gate_override_fill"], errors="coerce").fillna(0).astype(bool)
    out = out.sort_values(rank_columns, ascending=[False] * len(rank_columns)).copy()
    out["selected_rank"] = np.arange(1, len(out) + 1)
    if "_source_index" in out.columns:
        out = out.drop(columns=["_source_index"])
    out = out.drop(columns=["recommendation_rank"])
    out = out.reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Selector CSV not found: {csv_path}")

    plays = pd.read_csv(csv_path)
    learned_gate_payload = None
    learned_gate_summary: dict[str, object] = {"enabled": False, "reason": "disabled_flag"}
    if args.enable_learned_gate:
        learned_gate_path = args.learned_gate_json.resolve()
        if learned_gate_path.exists():
            try:
                learned_gate_payload = json.loads(learned_gate_path.read_text(encoding="utf-8"))
                month_count = 0
                if isinstance(learned_gate_payload, dict):
                    months = learned_gate_payload.get("months", {})
                    if isinstance(months, dict):
                        month_count = int(len(months))
                learned_gate_summary = {
                    "enabled": True,
                    "path": str(learned_gate_path),
                    "months": month_count,
                    "version": int(learned_gate_payload.get("version", 0)) if isinstance(learned_gate_payload, dict) else 0,
                }
            except Exception as exc:
                learned_gate_payload = None
                learned_gate_summary = {
                    "enabled": False,
                    "reason": "load_error",
                    "path": str(learned_gate_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            learned_gate_summary = {
                "enabled": False,
                "reason": "missing_file",
                "path": str(learned_gate_path),
            }
    staking_bucket_model_payload = None
    staking_bucket_model_summary: dict[str, object] = {"enabled": False, "reason": "disabled_flag"}
    if not args.disable_staking_bucket_model:
        staking_bucket_model_path = args.staking_bucket_model_json.resolve()
        if staking_bucket_model_path.exists():
            try:
                staking_bucket_model_payload = json.loads(staking_bucket_model_path.read_text(encoding="utf-8"))
                month_count = 0
                if isinstance(staking_bucket_model_payload, dict):
                    months = staking_bucket_model_payload.get("months", {})
                    if isinstance(months, dict):
                        month_count = int(len(months))
                staking_bucket_model_summary = {
                    "enabled": True,
                    "path": str(staking_bucket_model_path),
                    "months": month_count,
                    "version": int(staking_bucket_model_payload.get("version", 0)) if isinstance(staking_bucket_model_payload, dict) else 0,
                }
            except Exception as exc:
                staking_bucket_model_payload = None
                staking_bucket_model_summary = {
                    "enabled": False,
                    "reason": "load_error",
                    "path": str(staking_bucket_model_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            staking_bucket_model_summary = {
                "enabled": False,
                "reason": "missing_file",
                "path": str(staking_bucket_model_path),
            }

    final_board = compute_final_board(
        plays,
        american_odds=args.american_odds,
        min_ev=args.min_ev,
        min_final_confidence=args.min_final_confidence,
        min_recommendation=args.min_recommendation,
        selection_mode=args.selection_mode,
        ranking_mode=args.selection_mode,
        max_plays_per_player=args.max_plays_per_player,
        max_plays_per_target=args.max_plays_per_target,
        max_total_plays=args.max_total_plays,
        min_board_plays=args.min_board_plays,
        max_target_plays={"PTS": args.max_pts_plays, "TRB": args.max_trb_plays, "AST": args.max_ast_plays},
        max_plays_per_game=args.max_plays_per_game,
        max_plays_per_script_cluster=args.max_plays_per_script_cluster,
        non_pts_min_gap_percentile=args.non_pts_min_gap_percentile,
        edge_adjust_k=args.edge_adjust_k,
        thompson_temperature=args.thompson_temperature,
        thompson_seed=args.thompson_seed,
        min_bet_win_rate=args.min_bet_win_rate,
        medium_bet_win_rate=args.medium_bet_win_rate,
        full_bet_win_rate=args.full_bet_win_rate,
        medium_tier_percentile=args.medium_tier_percentile,
        strong_tier_percentile=args.strong_tier_percentile,
        elite_tier_percentile=args.elite_tier_percentile,
        small_bet_fraction=args.small_bet_fraction,
        medium_bet_fraction=args.medium_bet_fraction,
        full_bet_fraction=args.full_bet_fraction,
        max_bet_fraction=args.max_bet_fraction,
        max_total_bet_fraction=args.max_total_bet_fraction,
        sizing_method=args.sizing_method,
        flat_bet_fraction=args.flat_bet_fraction,
        coarse_low_bet_fraction=args.coarse_low_bet_fraction,
        coarse_mid_bet_fraction=args.coarse_mid_bet_fraction,
        coarse_high_bet_fraction=args.coarse_high_bet_fraction,
        coarse_high_max_share=args.coarse_high_max_share,
        coarse_mid_max_share=args.coarse_mid_max_share,
        coarse_high_max_plays=args.coarse_high_max_plays,
        coarse_mid_max_plays=args.coarse_mid_max_plays,
        coarse_score_alpha_uncertainty=args.coarse_score_alpha_uncertainty,
        coarse_score_beta_dependency=args.coarse_score_beta_dependency,
        coarse_score_gamma_support=args.coarse_score_gamma_support,
        coarse_score_model=args.coarse_score_model,
        coarse_score_delta_prob_weight=args.coarse_score_delta_prob_weight,
        coarse_score_ev_weight=args.coarse_score_ev_weight,
        coarse_score_risk_weight=args.coarse_score_risk_weight,
        coarse_score_recency_weight=args.coarse_score_recency_weight,
        staking_bucket_model_payload=staking_bucket_model_payload,
        staking_bucket_model_month=args.staking_bucket_model_month,
        staking_bucket_model_min_rows=args.staking_bucket_model_min_rows,
        belief_uncertainty_lower=args.belief_uncertainty_lower,
        belief_uncertainty_upper=args.belief_uncertainty_upper,
        append_agreement_min=args.append_agreement_min,
        append_edge_percentile_min=args.append_edge_percentile_min,
        append_max_extra_plays=args.append_max_extra_plays,
        board_objective_overfetch=args.board_objective_overfetch,
        board_objective_candidate_limit=args.board_objective_candidate_limit,
        board_objective_max_search_nodes=args.board_objective_max_search_nodes,
        board_objective_lambda_corr=args.board_objective_lambda_corr,
        board_objective_lambda_conc=args.board_objective_lambda_conc,
        board_objective_lambda_unc=args.board_objective_lambda_unc,
        board_objective_corr_same_game=args.board_objective_corr_same_game,
        board_objective_corr_same_player=args.board_objective_corr_same_player,
        board_objective_corr_same_target=args.board_objective_corr_same_target,
        board_objective_corr_same_direction=args.board_objective_corr_same_direction,
        board_objective_corr_same_script_cluster=args.board_objective_corr_same_script_cluster,
        board_objective_swap_candidates=args.board_objective_swap_candidates,
        board_objective_swap_rounds=args.board_objective_swap_rounds,
        board_objective_instability_enabled=args.board_objective_instability_enabled,
        board_objective_lambda_shadow_disagreement=args.board_objective_lambda_shadow_disagreement,
        board_objective_lambda_segment_weakness=args.board_objective_lambda_segment_weakness,
        board_objective_instability_near_cutoff_window=args.board_objective_instability_near_cutoff_window,
        board_objective_instability_top_protected=args.board_objective_instability_top_protected,
        board_objective_instability_veto_enabled=args.board_objective_instability_veto_enabled,
        board_objective_instability_veto_quantile=args.board_objective_instability_veto_quantile,
        board_objective_dynamic_size_enabled=args.board_objective_dynamic_size_enabled,
        board_objective_dynamic_size_max_shrink=args.board_objective_dynamic_size_max_shrink,
        board_objective_dynamic_size_trigger=args.board_objective_dynamic_size_trigger,
        max_history_staleness_days=args.max_history_staleness_days,
        min_recency_factor=args.min_recency_factor,
        learned_gate_payload=learned_gate_payload,
        learned_gate_month=args.learned_gate_month,
        learned_gate_min_rows=args.learned_gate_min_rows,
    )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    final_board.to_csv(args.csv_out, index=False)

    payload = {
        "source_csv": str(csv_path),
        "rows_in": int(len(plays)),
        "rows_out": int(len(final_board)),
        "american_odds": int(args.american_odds),
        "min_ev": float(args.min_ev),
        "min_final_confidence": float(args.min_final_confidence),
        "min_recommendation": args.min_recommendation,
        "selection_mode": args.selection_mode,
        "max_plays_per_player": int(args.max_plays_per_player),
        "max_plays_per_target": int(args.max_plays_per_target),
        "max_plays_per_game": int(args.max_plays_per_game),
        "max_plays_per_script_cluster": int(args.max_plays_per_script_cluster),
        "max_pts_plays": int(args.max_pts_plays),
        "max_trb_plays": int(args.max_trb_plays),
        "max_ast_plays": int(args.max_ast_plays),
        "max_total_plays": int(args.max_total_plays),
        "min_board_plays": int(args.min_board_plays),
        "non_pts_min_gap_percentile": float(args.non_pts_min_gap_percentile),
        "edge_adjust_k": float(args.edge_adjust_k),
        "thompson_temperature": float(args.thompson_temperature),
        "thompson_seed": int(args.thompson_seed),
        "min_bet_win_rate": float(args.min_bet_win_rate),
        "medium_bet_win_rate": float(args.medium_bet_win_rate),
        "full_bet_win_rate": float(args.full_bet_win_rate),
        "medium_tier_percentile": float(args.medium_tier_percentile),
        "strong_tier_percentile": float(args.strong_tier_percentile),
        "elite_tier_percentile": float(args.elite_tier_percentile),
        "small_bet_fraction": float(args.small_bet_fraction),
        "medium_bet_fraction": float(args.medium_bet_fraction),
        "full_bet_fraction": float(args.full_bet_fraction),
        "max_bet_fraction": float(args.max_bet_fraction),
        "max_total_bet_fraction": float(args.max_total_bet_fraction),
        "sizing_method": str(args.sizing_method),
        "flat_bet_fraction": float(args.flat_bet_fraction),
        "coarse_low_bet_fraction": float(args.coarse_low_bet_fraction),
        "coarse_mid_bet_fraction": float(args.coarse_mid_bet_fraction),
        "coarse_high_bet_fraction": float(args.coarse_high_bet_fraction),
        "coarse_high_max_share": float(args.coarse_high_max_share),
        "coarse_mid_max_share": float(args.coarse_mid_max_share),
        "coarse_high_max_plays": int(args.coarse_high_max_plays),
        "coarse_mid_max_plays": int(args.coarse_mid_max_plays),
        "coarse_score_alpha_uncertainty": float(args.coarse_score_alpha_uncertainty),
        "coarse_score_beta_dependency": float(args.coarse_score_beta_dependency),
        "coarse_score_gamma_support": float(args.coarse_score_gamma_support),
        "coarse_score_model": str(args.coarse_score_model),
        "coarse_score_delta_prob_weight": float(args.coarse_score_delta_prob_weight),
        "coarse_score_ev_weight": float(args.coarse_score_ev_weight),
        "coarse_score_risk_weight": float(args.coarse_score_risk_weight),
        "coarse_score_recency_weight": float(args.coarse_score_recency_weight),
        "staking_bucket_model": staking_bucket_model_summary,
        "staking_bucket_model_month": str(args.staking_bucket_model_month or ""),
        "staking_bucket_model_min_rows": int(args.staking_bucket_model_min_rows),
        "belief_uncertainty_lower": float(args.belief_uncertainty_lower),
        "belief_uncertainty_upper": float(args.belief_uncertainty_upper),
        "append_agreement_min": int(args.append_agreement_min),
        "append_edge_percentile_min": float(args.append_edge_percentile_min),
        "append_max_extra_plays": int(args.append_max_extra_plays),
        "board_objective_overfetch": float(args.board_objective_overfetch),
        "board_objective_candidate_limit": int(args.board_objective_candidate_limit),
        "board_objective_max_search_nodes": int(args.board_objective_max_search_nodes),
        "board_objective_lambda_corr": float(args.board_objective_lambda_corr),
        "board_objective_lambda_conc": float(args.board_objective_lambda_conc),
        "board_objective_lambda_unc": float(args.board_objective_lambda_unc),
        "board_objective_corr_same_game": float(args.board_objective_corr_same_game),
        "board_objective_corr_same_player": float(args.board_objective_corr_same_player),
        "board_objective_corr_same_target": float(args.board_objective_corr_same_target),
        "board_objective_corr_same_direction": float(args.board_objective_corr_same_direction),
        "board_objective_corr_same_script_cluster": float(args.board_objective_corr_same_script_cluster),
        "board_objective_swap_candidates": int(args.board_objective_swap_candidates),
        "board_objective_swap_rounds": int(args.board_objective_swap_rounds),
        "board_objective_instability_enabled": bool(args.board_objective_instability_enabled),
        "board_objective_lambda_shadow_disagreement": float(args.board_objective_lambda_shadow_disagreement),
        "board_objective_lambda_segment_weakness": float(args.board_objective_lambda_segment_weakness),
        "board_objective_instability_near_cutoff_window": int(args.board_objective_instability_near_cutoff_window),
        "board_objective_instability_top_protected": int(args.board_objective_instability_top_protected),
        "board_objective_instability_veto_enabled": bool(args.board_objective_instability_veto_enabled),
        "board_objective_instability_veto_quantile": float(args.board_objective_instability_veto_quantile),
        "board_objective_dynamic_size_enabled": bool(args.board_objective_dynamic_size_enabled),
        "board_objective_dynamic_size_max_shrink": int(args.board_objective_dynamic_size_max_shrink),
        "board_objective_dynamic_size_trigger": float(args.board_objective_dynamic_size_trigger),
        "max_history_staleness_days": int(args.max_history_staleness_days),
        "min_recency_factor": float(args.min_recency_factor),
        "learned_gate": learned_gate_summary,
        "learned_gate_min_rows": int(args.learned_gate_min_rows),
        "learned_gate_month": str(args.learned_gate_month or ""),
        "top_plays": final_board.head(20).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("FINAL MARKET PLAY BOARD")
    print("=" * 90)
    print(f"Input rows:   {len(plays)}")
    print(f"Output rows:  {len(final_board)}")
    print(f"CSV:          {args.csv_out}")
    print(f"JSON:         {args.json_out}")
    if args.enable_learned_gate:
        print(f"Learned gate: {learned_gate_summary}")
    if not args.disable_staking_bucket_model:
        print(f"Staking bucket model: {staking_bucket_model_summary}")
    if not final_board.empty:
        show_cols = [
            "player",
            "target",
            "direction",
            "prediction",
            "market_line",
            "abs_edge",
            "expected_win_rate",
            "expected_push_rate",
            "ev",
            "ev_adjusted",
            "thompson_ev",
            "final_confidence",
            "selected_rank",
            "allocation_tier",
            "allocation_action",
            "bet_fraction",
            "recommendation",
        ]
        print("\nTop final plays:")
        print(final_board[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
