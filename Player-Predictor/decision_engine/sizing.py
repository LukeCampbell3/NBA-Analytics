from __future__ import annotations

import numpy as np
import pandas as pd

from .gating import StrategyConfig


def american_profit_per_unit(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def compute_kelly_fraction(expected_win_rate: float, payout_per_unit: float, expected_push_rate: float = 0.0) -> float:
    p_win = float(np.clip(expected_win_rate, 0.0, 1.0))
    p_push = float(np.clip(expected_push_rate, 0.0, 1.0 - p_win))
    p_loss = max(0.0, 1.0 - p_win - p_push)
    if payout_per_unit <= 0.0:
        return 0.0
    full_kelly = (payout_per_unit * p_win - p_loss) / payout_per_unit
    return max(0.0, float(full_kelly))


def percentile_size_multiplier(gap_percentile: pd.Series, config: StrategyConfig) -> pd.Series:
    pct = pd.to_numeric(gap_percentile, errors="coerce").fillna(0.0)
    start = float(config.edge_scale_start_percentile)
    span = max(float(config.edge_scale_span), 1e-9)
    lift = float(config.edge_scale_lift)
    max_multiplier = float(config.edge_scale_max_multiplier)
    scaled = 1.0 + lift * ((pct - start) / span).clip(lower=0.0)
    return scaled.clip(lower=1.0, upper=max_multiplier)


def percentile_tier_from_series(
    gap_percentile: pd.Series,
    medium_tier_percentile: float,
    strong_tier_percentile: float,
    elite_tier_percentile: float,
) -> pd.Series:
    pct = pd.to_numeric(gap_percentile, errors="coerce").fillna(0.0)
    return pd.Series(
        np.select(
            [
                pct >= float(elite_tier_percentile),
                pct >= float(strong_tier_percentile),
                pct >= float(medium_tier_percentile),
            ],
            ["elite", "strong", "medium"],
            default="weak",
        ),
        index=pct.index,
        dtype="object",
    )


def tier_action_level_from_series(percentile_tier: pd.Series) -> pd.Series:
    return percentile_tier.map({"weak": 0, "medium": 1, "strong": 2, "elite": 3}).fillna(0).astype(int)


def probability_action_level_from_series(
    expected_win_rate: pd.Series,
    min_bet_win_rate: float,
    medium_bet_win_rate: float,
    full_bet_win_rate: float,
) -> pd.Series:
    rate = pd.to_numeric(expected_win_rate, errors="coerce").fillna(0.0)
    return pd.Series(
        np.select(
            [
                rate >= float(full_bet_win_rate),
                rate >= float(medium_bet_win_rate),
                rate >= float(min_bet_win_rate),
            ],
            [3, 2, 1],
            default=0,
        ),
        index=rate.index,
        dtype="int64",
    )


def action_label_from_level(level: pd.Series) -> pd.Series:
    return level.map({0: "no_bet", 1: "small", 2: "medium", 3: "full"}).fillna("no_bet")


def _minmax_scale(series: pd.Series, default: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(float(default)).astype("float64")
    if values.empty:
        return values
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    if span <= 1e-9:
        return pd.Series(float(default), index=values.index, dtype="float64")
    return ((values - lo) / span).clip(lower=0.0, upper=1.0).astype("float64")


def bet_fraction_from_action_level(
    action_level: pd.Series,
    small_bet_fraction: float,
    medium_bet_fraction: float,
    full_bet_fraction: float,
    max_bet_fraction: float,
) -> pd.Series:
    fractions = action_level.map(
        {
            0: 0.0,
            1: float(small_bet_fraction),
            2: float(medium_bet_fraction),
            3: float(full_bet_fraction),
        }
    ).fillna(0.0)
    return pd.to_numeric(fractions, errors="coerce").fillna(0.0).clip(lower=0.0, upper=float(max_bet_fraction))


def apply_tiered_bet_sizing(
    df: pd.DataFrame,
    expected_win_rate_col: str,
    gap_percentile_col: str,
    min_bet_win_rate: float,
    medium_bet_win_rate: float,
    full_bet_win_rate: float,
    medium_tier_percentile: float,
    strong_tier_percentile: float,
    elite_tier_percentile: float,
    small_bet_fraction: float,
    medium_bet_fraction: float,
    full_bet_fraction: float,
    max_bet_fraction: float,
    max_total_bet_fraction: float,
) -> pd.DataFrame:
    out = df.copy()
    out["allocation_tier"] = percentile_tier_from_series(
        out.get(gap_percentile_col),
        medium_tier_percentile=medium_tier_percentile,
        strong_tier_percentile=strong_tier_percentile,
        elite_tier_percentile=elite_tier_percentile,
    )
    tier_levels = tier_action_level_from_series(out["allocation_tier"])
    probability_levels = probability_action_level_from_series(
        out.get(expected_win_rate_col),
        min_bet_win_rate=min_bet_win_rate,
        medium_bet_win_rate=medium_bet_win_rate,
        full_bet_win_rate=full_bet_win_rate,
    )
    out["allocation_action_from_tier"] = action_label_from_level(tier_levels)
    out["allocation_action_from_probability"] = action_label_from_level(probability_levels)
    out["allocation_action_level"] = np.minimum(tier_levels, probability_levels)
    out["allocation_action"] = action_label_from_level(out["allocation_action_level"])
    out["bet_fraction_raw"] = bet_fraction_from_action_level(
        out["allocation_action_level"],
        small_bet_fraction=small_bet_fraction,
        medium_bet_fraction=medium_bet_fraction,
        full_bet_fraction=full_bet_fraction,
        max_bet_fraction=max_bet_fraction,
    )

    total_fraction = float(pd.to_numeric(out["bet_fraction_raw"], errors="coerce").fillna(0.0).sum())
    scale = 1.0
    if total_fraction > 0.0 and float(max_total_bet_fraction) > 0.0:
        scale = min(1.0, float(max_total_bet_fraction) / total_fraction)
    out["bet_fraction_scale"] = scale
    out["bet_fraction"] = pd.to_numeric(out["bet_fraction_raw"], errors="coerce").fillna(0.0) * scale
    return out


def size_bets(selected_df: pd.DataFrame, bankroll: float, config: StrategyConfig) -> pd.DataFrame:
    if selected_df.empty:
        out = selected_df.copy()
        out["stake"] = 0.0
        out["bet_fraction"] = 0.0
        out["raw_kelly_fraction"] = 0.0
        out["payout_per_unit"] = american_profit_per_unit(config.american_odds)
        out["size_multiplier"] = 1.0
        return out

    payout = american_profit_per_unit(config.american_odds)
    out = selected_df.copy()
    out["payout_per_unit"] = payout

    raw_kelly = out.apply(
        lambda row: compute_kelly_fraction(
            expected_win_rate=float(row.get("expected_win_rate", 0.0)),
            payout_per_unit=payout,
            expected_push_rate=float(row.get("expected_push_rate", 0.0)),
        ),
        axis=1,
    ).astype(float)
    out["raw_kelly_fraction"] = raw_kelly

    if config.sizing_method == "flat":
        out["bet_fraction"] = 0.0
        out["stake"] = float(config.flat_stake)
        out["size_multiplier"] = 1.0
        return out

    if config.sizing_method == "tiered_probability":
        out = apply_tiered_bet_sizing(
            out,
            expected_win_rate_col="expected_win_rate",
            gap_percentile_col="gap_percentile",
            min_bet_win_rate=config.min_bet_win_rate,
            medium_bet_win_rate=config.medium_bet_win_rate,
            full_bet_win_rate=config.full_bet_win_rate,
            medium_tier_percentile=config.medium_tier_percentile,
            strong_tier_percentile=config.strong_tier_percentile,
            elite_tier_percentile=config.elite_tier_percentile,
            small_bet_fraction=config.small_bet_fraction,
            medium_bet_fraction=config.medium_bet_fraction,
            full_bet_fraction=config.full_bet_fraction,
            max_bet_fraction=config.max_bet_fraction,
            max_total_bet_fraction=config.max_total_bet_fraction,
        )
        out["size_multiplier"] = 1.0
        out["stake"] = bankroll * pd.to_numeric(out["bet_fraction"], errors="coerce").fillna(0.0)
        return out

    if config.sizing_method == "coarse_bucket":
        score_model = str(getattr(config, "coarse_score_model", "legacy")).strip().lower()
        if score_model not in {"legacy", "stake_score_v1", "stake_model_v2"}:
            score_model = "legacy"
        score_prob = pd.to_numeric(out.get("expected_win_rate"), errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
        expected_push = pd.to_numeric(out.get("expected_push_rate"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=0.95)
        non_push = (1.0 - expected_push).clip(lower=0.05, upper=1.0)
        cond_prob = (score_prob / non_push).clip(lower=0.0, upper=1.0)
        break_even_cond = 1.0 / (1.0 + max(float(payout), 1e-9))
        delta_prob = (cond_prob - float(break_even_cond)).clip(lower=-1.0, upper=1.0)
        delta_prob_strength = _minmax_scale(delta_prob, default=0.5)

        score_unc = pd.to_numeric(out.get("belief_uncertainty_normalized"), errors="coerce")
        if score_unc.isna().all():
            score_unc = np.sqrt(pd.to_numeric(out.get("posterior_variance"), errors="coerce").fillna(0.25).clip(lower=0.0, upper=1.0))
        score_unc = pd.to_numeric(score_unc, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)

        support = np.log1p(pd.to_numeric(out.get("history_rows"), errors="coerce").fillna(0.0).clip(lower=0.0))
        support_norm = _minmax_scale(support, default=0.0)

        ev_source = pd.to_numeric(out.get("ev_adjusted", out.get("ev")), errors="coerce").fillna(0.0)
        ev_strength = _minmax_scale(ev_source, default=0.5)
        recency_norm = pd.to_numeric(out.get("recency_factor"), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
        recency_norm = _minmax_scale(recency_norm, default=0.5)

        risk_penalty = pd.to_numeric(out.get("risk_penalty"), errors="coerce").fillna(0.0).clip(lower=0.0)
        volatility = pd.to_numeric(out.get("volatility_score"), errors="coerce").fillna(0.0).clip(lower=0.0)
        spike_probability = pd.to_numeric(out.get("spike_probability"), errors="coerce").fillna(0.0).clip(lower=0.0)
        tail_imbalance = pd.to_numeric(out.get("tail_imbalance"), errors="coerce").fillna(0.0).abs()
        risk_component = (
            0.45 * _minmax_scale(risk_penalty, default=0.0)
            + 0.30 * _minmax_scale(volatility, default=0.0)
            + 0.15 * _minmax_scale(spike_probability, default=0.0)
            + 0.10 * _minmax_scale(tail_imbalance, default=0.0)
        )

        score = (
            score_prob
            - float(getattr(config, "coarse_score_alpha_uncertainty", 0.18)) * score_unc
            + float(getattr(config, "coarse_score_gamma_support", 0.08)) * support_norm
        )
        if score_model == "stake_model_v2":
            # Offline simulation path has no external month payload; treat expected
            # win-probability signal as the learned-model proxy while preserving
            # the same conservative penalties.
            score = (
                score_prob
                - float(getattr(config, "coarse_score_alpha_uncertainty", 0.18)) * score_unc
                - float(getattr(config, "coarse_score_beta_dependency", 0.12)) * 0.0
                + float(getattr(config, "coarse_score_gamma_support", 0.08)) * support_norm
            )
        if score_model == "stake_score_v1":
            score = (
                score
                + float(getattr(config, "coarse_score_delta_prob_weight", 0.0)) * delta_prob_strength
                + float(getattr(config, "coarse_score_ev_weight", 0.0)) * ev_strength
                - float(getattr(config, "coarse_score_risk_weight", 0.0)) * risk_component
                + float(getattr(config, "coarse_score_recency_weight", 0.0)) * recency_norm
            )
        out["coarse_score_model"] = score_model
        out["coarse_score_delta_prob_strength"] = pd.to_numeric(delta_prob_strength, errors="coerce").fillna(0.5)
        out["coarse_score_ev_strength"] = pd.to_numeric(ev_strength, errors="coerce").fillna(0.5)
        out["coarse_score_risk"] = pd.to_numeric(risk_component, errors="coerce").fillna(0.0)
        out["coarse_score_recency"] = pd.to_numeric(recency_norm, errors="coerce").fillna(0.5)
        out["coarse_score"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
        out = out.sort_values(["coarse_score", "expected_win_rate", "ev"], ascending=[False, False, False]).copy()

        n_rows = int(len(out))
        high_cap = int(np.floor(float(getattr(config, "coarse_high_max_share", 0.30)) * float(n_rows)))
        mid_cap = int(np.floor(float(getattr(config, "coarse_mid_max_share", 0.50)) * float(n_rows)))
        high_max_plays = int(getattr(config, "coarse_high_max_plays", 0))
        mid_max_plays = int(getattr(config, "coarse_mid_max_plays", 0))
        if high_max_plays > 0:
            high_cap = min(high_cap, high_max_plays)
        high_cap = max(0, min(high_cap, n_rows))
        if mid_max_plays > 0:
            mid_cap = min(mid_cap, mid_max_plays)
        mid_cap = max(0, min(mid_cap, max(0, n_rows - high_cap)))

        out["coarse_bucket"] = "low"
        if high_cap > 0:
            out.iloc[:high_cap, out.columns.get_loc("coarse_bucket")] = "high"
        if mid_cap > 0:
            out.iloc[high_cap : high_cap + mid_cap, out.columns.get_loc("coarse_bucket")] = "mid"

        bucket_to_fraction = {
            "low": float(getattr(config, "coarse_low_bet_fraction", 0.003)),
            "mid": float(getattr(config, "coarse_mid_bet_fraction", 0.005)),
            "high": float(getattr(config, "coarse_high_bet_fraction", 0.007)),
        }
        out["allocation_tier"] = "coarse_" + out["coarse_bucket"].astype(str)
        out["allocation_action"] = out["allocation_tier"]
        out["allocation_action_level"] = out["coarse_bucket"].map({"low": 1, "mid": 2, "high": 3}).fillna(1).astype(int)
        out["bet_fraction_raw"] = pd.to_numeric(out["coarse_bucket"].map(bucket_to_fraction), errors="coerce").fillna(0.0)
        out["bet_fraction_raw"] = out["bet_fraction_raw"].clip(lower=0.0, upper=float(config.max_bet_fraction))

        total_raw = float(pd.to_numeric(out["bet_fraction_raw"], errors="coerce").fillna(0.0).sum())
        scale = 1.0
        if total_raw > 0.0 and float(config.max_total_bet_fraction) > 0.0:
            scale = min(1.0, float(config.max_total_bet_fraction) / total_raw)
        out["bet_fraction_scale"] = float(scale)
        out["bet_fraction"] = pd.to_numeric(out["bet_fraction_raw"], errors="coerce").fillna(0.0) * float(scale)
        out["size_multiplier"] = 1.0
        out["stake"] = bankroll * pd.to_numeric(out["bet_fraction"], errors="coerce").fillna(0.0)
        return out

    size_multiplier = percentile_size_multiplier(out.get("gap_percentile"), config)
    out["size_multiplier"] = size_multiplier

    if config.sizing_method == "base_fraction":
        scaled_fraction = pd.Series(float(config.base_bet_fraction), index=out.index, dtype="float64") * size_multiplier
    else:
        scaled_fraction = raw_kelly * float(config.kelly_fraction) * size_multiplier

    scaled_fraction = scaled_fraction.clip(lower=float(config.min_bet_fraction), upper=float(config.max_bet_fraction))
    out["bet_fraction"] = scaled_fraction
    out["stake"] = bankroll * scaled_fraction
    return out


def settle_bets(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        out = selected_df.copy()
        out["expected_profit"] = 0.0
        out["profit"] = 0.0
        out["roi"] = 0.0
        return out

    out = selected_df.copy()
    out["expected_profit"] = out["stake"] * pd.to_numeric(out["ev"], errors="coerce").fillna(0.0)
    result = out["result"].astype(str).str.lower()
    payout = pd.to_numeric(out["payout_per_unit"], errors="coerce").fillna(0.0)
    out["profit"] = np.select(
        [result.eq("win"), result.eq("push")],
        [out["stake"] * payout, 0.0],
        default=-out["stake"],
    )
    out["roi"] = np.where(out["stake"] > 0.0, out["profit"] / out["stake"], 0.0)
    return out
