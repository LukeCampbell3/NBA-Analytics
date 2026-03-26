#!/usr/bin/env python3
"""
Structured LSTM trainer.

This keeps the LSTM as the sequence core, but makes the latent state explicit:
- observability encoder with residual / volatility context
- slow-state branch from player identity + longer-horizon summaries
- predictive bottleneck
- belief-state uncertainty branch
- stable environment branch
- baseline-anchored normal + tail correction heads
- spike and feasibility auxiliary heads
"""

import os
import sys
import json
import io
import contextlib
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    GaussianNoise,
    Concatenate,
    Lambda,
    LayerNormalization,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    Add,
    GlobalMaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from unified_moe_trainer import UnifiedMoETrainer
from improved_stacking_trainer import prepare_gbm_features_v2

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


PRODUCTION_BEST_MAE = 4.441093564966795


def gather_feature(x, idx, width=1):
    if idx < 0:
        shape = tf.shape(x)
        if width == 1:
            return tf.zeros([shape[0], shape[1], 1], dtype=x.dtype)
        return tf.zeros([shape[0], shape[1], width], dtype=x.dtype)
    if width == 1:
        return x[:, :, idx:idx + 1]
    return x[:, :, idx:idx + width]


def gather_last(x, idx, width=1):
    if idx < 0:
        shape = tf.shape(x)
        if width == 1:
            return tf.zeros([shape[0], 1], dtype=x.dtype)
        return tf.zeros([shape[0], width], dtype=x.dtype)
    if width == 1:
        return x[:, -1, idx:idx + 1]
    return x[:, -1, idx:idx + width]


def build_feature_spec(feature_columns):
    mapping = {name: idx for idx, name in enumerate(feature_columns)}
    return {
        "player_idx": 0,
        "team_idx": 1,
        "opp_idx": 2,
        "pts_idx": mapping.get("PTS", -1),
        "trb_idx": mapping.get("TRB", -1),
        "ast_idx": mapping.get("AST", -1),
        "pts_lag_idx": mapping.get("PTS_lag1", -1),
        "trb_lag_idx": mapping.get("TRB_lag1", -1),
        "ast_lag_idx": mapping.get("AST_lag1", -1),
        "pts_roll_idx": mapping.get("PTS_rolling_avg", -1),
        "trb_roll_idx": mapping.get("TRB_rolling_avg", -1),
        "ast_roll_idx": mapping.get("AST_rolling_avg", -1),
        "pts_std_idx": mapping.get("PTS_rolling_std", -1),
        "trb_std_idx": mapping.get("TRB_rolling_std", -1),
        "ast_std_idx": mapping.get("AST_rolling_std", -1),
        "market_pts_idx": mapping.get("Market_PTS", -1),
        "market_trb_idx": mapping.get("Market_TRB", -1),
        "market_ast_idx": mapping.get("Market_AST", -1),
        "market_pts_gap_idx": mapping.get("PTS_market_gap", -1),
        "market_trb_gap_idx": mapping.get("TRB_market_gap", -1),
        "market_ast_gap_idx": mapping.get("AST_market_gap", -1),
        "market_pts_conf_idx": mapping.get("Market_PTS_consensus_conf", -1),
        "market_trb_conf_idx": mapping.get("Market_TRB_consensus_conf", -1),
        "market_ast_conf_idx": mapping.get("Market_AST_consensus_conf", -1),
        "market_pts_quality_idx": mapping.get("Market_PTS_source_quality", -1),
        "market_trb_quality_idx": mapping.get("Market_TRB_source_quality", -1),
        "market_ast_quality_idx": mapping.get("Market_AST_source_quality", -1),
        "market_pts_lean_idx": mapping.get("Market_PTS_price_lean", -1),
        "market_trb_lean_idx": mapping.get("Market_TRB_price_lean", -1),
        "market_ast_lean_idx": mapping.get("Market_AST_price_lean", -1),
        "mp_idx": mapping.get("MP", -1),
        "mp_roll_idx": mapping.get("MP_rolling_avg", -1),
        "usg_idx": mapping.get("USG%", -1),
        "usg_roll_idx": mapping.get("USG%_rolling_avg", -1),
        "rest_days_idx": mapping.get("Rest_Days", -1),
        "did_not_play_idx": mapping.get("Did_Not_Play", -1),
        "opp_dfrtg_idx": mapping.get("oppDfRtg_3", -1),
        "mp_trend_idx": mapping.get("MP_trend", -1),
        "high_mp_idx": mapping.get("High_MP_Flag", -1),
        "fga_trend_idx": mapping.get("FGA_trend", -1),
        "ast_trend_idx": mapping.get("AST_trend", -1),
        "ast_var_idx": mapping.get("AST_variance", -1),
        "usg_ast_ratio_trend_idx": mapping.get("USG_AST_ratio_trend", -1),
        "high_playmaker_idx": mapping.get("High_Playmaker_Flag", -1),
        "year_idx": mapping.get("Year", -1),
        "month_sin_idx": mapping.get("Month_sin", -1),
        "month_cos_idx": mapping.get("Month_cos", -1),
    }


def build_structured_lstm(seq_len, n_features, n_targets, feature_spec, counts, seed=42, config=None):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    cfg = config or {}
    obs_units = cfg.get("obs_units", 96)
    lstm_units = cfg.get("lstm_units", 96)
    lstm2_units = cfg.get("lstm2_units", 64)
    bottleneck_dim = cfg.get("bottleneck_dim", 48)
    belief_dim = cfg.get("belief_dim", 24)
    slow_dim = cfg.get("slow_dim", 32)
    env_dim = cfg.get("env_dim", 32)
    dense_dim = cfg.get("dense_dim", 96)
    dropout = cfg.get("dropout", 0.25)
    noise_std = cfg.get("noise_std", 0.03)
    l2_reg = cfg.get("l2_reg", 6e-5)

    seq_input = Input(shape=(seq_len, n_features), name="seq_input")
    base_input = Input(shape=(n_targets,), name="base_input")

    player_ids = Lambda(lambda x: tf.cast(x[:, -1, feature_spec["player_idx"]], tf.int32), name="player_ids")(seq_input)
    team_ids = Lambda(lambda x: tf.cast(x[:, -1, feature_spec["team_idx"]], tf.int32), name="team_ids")(seq_input)
    opp_ids = Lambda(lambda x: tf.cast(x[:, -1, feature_spec["opp_idx"]], tf.int32), name="opp_ids")(seq_input)

    player_embed = Embedding(max(1, counts["players"]), 16, name="player_embed")(player_ids)
    team_embed = Embedding(max(1, counts["teams"]), 8, name="team_embed")(team_ids)
    opp_embed = Embedding(max(1, counts["opponents"]), 8, name="opp_embed")(opp_ids)

    pts_proxy = gather_feature(seq_input, feature_spec["pts_idx"] if feature_spec["pts_idx"] >= 0 else feature_spec["pts_lag_idx"])
    trb_proxy = gather_feature(seq_input, feature_spec["trb_idx"] if feature_spec["trb_idx"] >= 0 else feature_spec["trb_lag_idx"])
    ast_proxy = gather_feature(seq_input, feature_spec["ast_idx"] if feature_spec["ast_idx"] >= 0 else feature_spec["ast_lag_idx"])

    stat_hist = Lambda(
        lambda xs: tf.concat(xs, axis=-1),
        name="stat_hist",
    )([pts_proxy, trb_proxy, ast_proxy])
    baseline_hist = Lambda(
        lambda x: tf.concat(
            [
                gather_feature(x, feature_spec["pts_roll_idx"]),
                gather_feature(x, feature_spec["trb_roll_idx"]),
                gather_feature(x, feature_spec["ast_roll_idx"]),
            ],
            axis=-1,
        ),
        name="baseline_hist",
    )(seq_input)
    residual_hist = Lambda(lambda xs: xs[0] - xs[1], name="residual_hist")([stat_hist, baseline_hist])

    pts_vol = gather_feature(seq_input, feature_spec["pts_std_idx"]) if feature_spec["pts_std_idx"] >= 0 else Lambda(
        lambda xs: tf.abs(xs[0] - xs[1]),
        name="pts_vol_proxy",
    )([pts_proxy, gather_feature(seq_input, feature_spec["pts_roll_idx"])])
    trb_vol = gather_feature(seq_input, feature_spec["trb_std_idx"]) if feature_spec["trb_std_idx"] >= 0 else Lambda(
        lambda xs: tf.abs(xs[0] - xs[1]),
        name="trb_vol_proxy",
    )([trb_proxy, gather_feature(seq_input, feature_spec["trb_roll_idx"])])
    ast_vol = gather_feature(seq_input, feature_spec["ast_std_idx"]) if feature_spec["ast_std_idx"] >= 0 else Lambda(
        lambda xs: tf.maximum(tf.abs(xs[0] - xs[1]), xs[2]),
        name="ast_vol_proxy",
    )([
        ast_proxy,
        gather_feature(seq_input, feature_spec["ast_roll_idx"]),
        gather_feature(seq_input, feature_spec["ast_var_idx"]),
    ])
    volatility_hist = Lambda(
        lambda xs: tf.concat(xs, axis=-1),
        name="volatility_hist",
    )([pts_vol, trb_vol, ast_vol])
    usage_hist = Lambda(
        lambda x: tf.concat(
            [
                gather_feature(x, feature_spec["mp_idx"]),
                gather_feature(x, feature_spec["mp_roll_idx"]),
                gather_feature(x, feature_spec["usg_idx"]),
                gather_feature(x, feature_spec["usg_roll_idx"]),
            ],
            axis=-1,
        ),
        name="usage_hist",
    )(seq_input)
    market_pts_idx = feature_spec.get("market_pts_idx", -1)
    market_trb_idx = feature_spec.get("market_trb_idx", -1)
    market_ast_idx = feature_spec.get("market_ast_idx", -1)
    market_pts_gap_idx = feature_spec.get("market_pts_gap_idx", -1)
    market_trb_gap_idx = feature_spec.get("market_trb_gap_idx", -1)
    market_ast_gap_idx = feature_spec.get("market_ast_gap_idx", -1)
    market_pts_conf_idx = feature_spec.get("market_pts_conf_idx", -1)
    market_trb_conf_idx = feature_spec.get("market_trb_conf_idx", -1)
    market_ast_conf_idx = feature_spec.get("market_ast_conf_idx", -1)
    market_pts_quality_idx = feature_spec.get("market_pts_quality_idx", -1)
    market_trb_quality_idx = feature_spec.get("market_trb_quality_idx", -1)
    market_ast_quality_idx = feature_spec.get("market_ast_quality_idx", -1)
    market_pts_lean_idx = feature_spec.get("market_pts_lean_idx", -1)
    market_trb_lean_idx = feature_spec.get("market_trb_lean_idx", -1)
    market_ast_lean_idx = feature_spec.get("market_ast_lean_idx", -1)
    market_hist = Lambda(
        lambda x: tf.concat(
            [
                gather_feature(x, market_pts_idx),
                gather_feature(x, market_trb_idx),
                gather_feature(x, market_ast_idx),
                gather_feature(x, market_pts_gap_idx),
                gather_feature(x, market_trb_gap_idx),
                gather_feature(x, market_ast_gap_idx),
                gather_feature(x, market_pts_conf_idx),
                gather_feature(x, market_trb_conf_idx),
                gather_feature(x, market_ast_conf_idx),
                gather_feature(x, market_pts_quality_idx),
                gather_feature(x, market_trb_quality_idx),
                gather_feature(x, market_ast_quality_idx),
                gather_feature(x, market_pts_lean_idx),
                gather_feature(x, market_trb_lean_idx),
                gather_feature(x, market_ast_lean_idx),
            ],
            axis=-1,
        ),
        name="market_hist",
    )(seq_input)

    obs_input = Concatenate(name="observability_concat")(
        [seq_input[:, :, 3:], stat_hist, baseline_hist, residual_hist, volatility_hist, usage_hist, market_hist]
    )
    obs = GaussianNoise(noise_std, name="obs_noise")(obs_input)
    obs = Dense(
        obs_units,
        activation="swish",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="obs_dense",
    )(obs)
    obs = LayerNormalization(name="obs_ln")(obs)
    obs = Dropout(dropout, name="obs_drop")(obs)

    x = LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=dropout * 0.5,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="lstm1",
    )(obs)
    x = LayerNormalization(name="lstm1_ln")(x)
    x2 = LSTM(
        lstm2_units,
        return_sequences=True,
        dropout=dropout * 0.9,
        recurrent_dropout=dropout * 0.4,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="lstm2",
    )(x)
    x2 = LayerNormalization(name="lstm2_ln")(x2)

    attn = MultiHeadAttention(num_heads=2, key_dim=max(8, lstm2_units // 4), dropout=dropout * 0.5, name="temporal_attn")(
        x2, x2
    )
    x2 = Add(name="attn_skip")([x2, attn])
    x2 = LayerNormalization(name="attn_ln")(x2)

    last_hidden = Lambda(lambda z: z[:, -1, :], name="last_hidden")(x2)
    avg_hidden = GlobalAveragePooling1D(name="avg_hidden")(x2)
    residual_summary = GlobalAveragePooling1D(name="residual_summary")(residual_hist)
    vol_summary = GlobalAveragePooling1D(name="vol_summary")(volatility_hist)
    usage_summary = GlobalAveragePooling1D(name="usage_summary")(usage_hist)
    market_summary = GlobalAveragePooling1D(name="market_summary")(market_hist)
    baseline_summary = Lambda(lambda x: x[:, -1, :], name="baseline_summary")(baseline_hist)
    year_state = Lambda(lambda x: gather_last(x, feature_spec["year_idx"]), name="year_state")(seq_input)
    month_state = Lambda(
        lambda x: tf.concat(
            [gather_last(x, feature_spec["month_sin_idx"]), gather_last(x, feature_spec["month_cos_idx"])],
            axis=-1,
        ),
        name="month_state",
    )(seq_input)

    slow_in = Concatenate(name="slow_in")(
        [player_embed, team_embed, opp_embed, baseline_summary, vol_summary, usage_summary, market_summary, year_state, month_state]
    )
    slow_mode = Dense(slow_dim, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="slow_dense")(slow_in)
    slow_mode = LayerNormalization(name="slow_ln")(slow_mode)

    bottleneck_in = Concatenate(name="bottleneck_in")([last_hidden, avg_hidden, residual_summary])
    bottleneck = Dense(
        bottleneck_dim,
        activation="swish",
        kernel_regularizer=regularizers.l2(l2_reg),
        activity_regularizer=regularizers.l2(cfg.get("bottleneck_reg", 5e-5)),
        name="predictive_bottleneck",
    )(bottleneck_in)
    bottleneck = Dropout(dropout, name="bottleneck_drop")(bottleneck)

    belief_mu = Dense(belief_dim, activation=None, kernel_regularizer=regularizers.l2(l2_reg), name="belief_mu")(last_hidden)
    belief_logvar = Dense(belief_dim, activation=None, kernel_regularizer=regularizers.l2(l2_reg), name="belief_logvar")(last_hidden)
    belief_std = Lambda(lambda x: tf.nn.softplus(x) + 1e-3, name="belief_std")(belief_logvar)

    env_in = Concatenate(name="env_in")([slow_mode, avg_hidden, vol_summary, usage_summary, market_summary, month_state])
    stable_env = Dense(env_dim, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="stable_env")(env_in)
    stable_env = LayerNormalization(name="stable_env_ln")(stable_env)

    regime_context = Concatenate(name="regime_context")([z if "z" in locals() else last_hidden, residual_summary, vol_summary, usage_summary, market_summary, stable_env])
    role_shift_prob = Dense(
        dense_dim // 3,
        activation="swish",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="role_shift_fc",
    )(regime_context)
    role_shift_prob = Dense(1, activation="sigmoid", name="role_shift_prob")(role_shift_prob)

    volatility_regime_prob = Dense(
        dense_dim // 3,
        activation="swish",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="volatility_regime_fc",
    )(regime_context)
    volatility_regime_prob = Dense(1, activation="sigmoid", name="volatility_regime_prob")(volatility_regime_prob)

    context_pressure_prob = Dense(
        dense_dim // 3,
        activation="swish",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="context_pressure_fc",
    )(regime_context)
    context_pressure_prob = Dense(1, activation="sigmoid", name="context_pressure_prob")(context_pressure_prob)

    z = Concatenate(name="latent_state")(
        [
            last_hidden,
            avg_hidden,
            bottleneck,
            belief_mu,
            belief_std,
            slow_mode,
            stable_env,
            market_summary,
            role_shift_prob,
            volatility_regime_prob,
            context_pressure_prob,
            base_input,
        ]
    )
    z = Dense(dense_dim, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="latent_dense")(z)
    z = LayerNormalization(name="latent_ln")(z)
    z = Dropout(dropout, name="latent_drop")(z)

    delta_normal = Dense(dense_dim // 2, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="delta_normal_fc")(z)
    delta_normal = Dense(n_targets, activation=None, name="delta_normal")(delta_normal)

    delta_tail = Dense(dense_dim // 2, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="delta_tail_fc")(z)
    delta_tail = Dense(n_targets, activation="tanh", name="delta_tail_raw")(delta_tail)
    delta_tail = Lambda(lambda x: x * 2.5, name="delta_tail")(delta_tail)

    spike_in = Concatenate(name="spike_in")([z, residual_summary, vol_summary, belief_std])
    spike_prob = Dense(dense_dim // 2, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="spike_fc")(spike_in)
    spike_prob = Dense(n_targets, activation="sigmoid", name="spike_prob")(spike_prob)

    sigma_in = Concatenate(name="sigma_in")([z, vol_summary, belief_std])
    sigma = Dense(dense_dim // 2, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="sigma_fc")(sigma_in)
    sigma = Dense(n_targets, activation="softplus", name="sigma_raw")(sigma)
    sigma = Lambda(lambda x: x + 5e-2, name="sigma")(sigma)

    feasibility_in = Concatenate(name="feasibility_in")([z, vol_summary, usage_summary])
    feasibility = Dense(dense_dim // 3, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="feasibility_fc")(feasibility_in)
    feasibility = Dense(1, activation="sigmoid", name="feasibility")(feasibility)

    delta = Lambda(lambda xs: xs[0] + xs[1] * xs[2], name="delta_pred")([delta_normal, spike_prob, delta_tail])

    return Model(
        inputs=[seq_input, base_input],
        outputs={
            "delta": delta,
            "delta_normal": delta_normal,
            "delta_tail": delta_tail,
            "sigma": sigma,
            "spike_prob": spike_prob,
            "feasibility": feasibility,
            "bottleneck": bottleneck,
            "last_hidden": last_hidden,
            "avg_hidden": avg_hidden,
            "belief_mu": belief_mu,
            "belief_std": belief_std,
            "slow_mode": slow_mode,
            "stable_env": stable_env,
            "role_shift_prob": role_shift_prob,
            "volatility_regime_prob": volatility_regime_prob,
            "context_pressure_prob": context_pressure_prob,
            "latent_dense": z,
        },
        name=f"structured_lstm_s{seed}",
    )


def build_pts_embedding_model(seq_len, n_features, feature_spec, counts, seed=42, config=None):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    cfg = config or {}
    obs_units = cfg.get("pts_obs_units", 64)
    lstm_units = cfg.get("pts_lstm_units", 64)
    slow_dim = cfg.get("pts_slow_dim", 24)
    embed_dim = cfg.get("pts_embed_dim", 24)
    dropout = cfg.get("pts_dropout", 0.20)
    l2_reg = cfg.get("pts_l2_reg", 5e-5)

    seq_input = Input(shape=(seq_len, n_features), name="pts_seq_input")
    base_input = Input(shape=(1,), name="pts_base_input")

    player_ids = Lambda(lambda x: tf.cast(x[:, -1, feature_spec["player_idx"]], tf.int32), name="pts_player_ids")(seq_input)
    team_ids = Lambda(lambda x: tf.cast(x[:, -1, feature_spec["team_idx"]], tf.int32), name="pts_team_ids")(seq_input)
    opp_ids = Lambda(lambda x: tf.cast(x[:, -1, feature_spec["opp_idx"]], tf.int32), name="pts_opp_ids")(seq_input)

    player_embed = Embedding(max(1, counts["players"]), 12, name="pts_player_embed")(player_ids)
    team_embed = Embedding(max(1, counts["teams"]), 6, name="pts_team_embed")(team_ids)
    opp_embed = Embedding(max(1, counts["opponents"]), 6, name="pts_opp_embed")(opp_ids)

    pts_hist_idx = feature_spec["pts_idx"] if feature_spec["pts_idx"] >= 0 else feature_spec["pts_lag_idx"]
    pts_hist = Lambda(lambda x: gather_feature(x, pts_hist_idx), name="pts_hist_only")(seq_input)
    pts_base_hist = Lambda(lambda x: gather_feature(x, feature_spec["pts_roll_idx"]), name="pts_base_hist")(seq_input)
    pts_resid = Lambda(lambda xs: xs[0] - xs[1], name="pts_resid_hist")([pts_hist, pts_base_hist])
    if feature_spec["pts_std_idx"] >= 0:
        pts_vol = Lambda(lambda x: gather_feature(x, feature_spec["pts_std_idx"]), name="pts_vol_hist")(seq_input)
    else:
        pts_vol = Lambda(lambda xs: tf.abs(xs[0] - xs[1]), name="pts_vol_hist")([pts_hist, pts_base_hist])
    usage_hist = Lambda(
        lambda x: tf.concat(
            [
                gather_feature(x, feature_spec["mp_idx"]),
                gather_feature(x, feature_spec["mp_roll_idx"]),
                gather_feature(x, feature_spec["usg_idx"]),
                gather_feature(x, feature_spec["usg_roll_idx"]),
            ],
            axis=-1,
        ),
        name="pts_usage_hist",
    )(seq_input)
    pts_market_idx = feature_spec.get("market_pts_idx", -1)
    pts_market_gap_idx = feature_spec.get("market_pts_gap_idx", -1)
    pts_market_conf_idx = feature_spec.get("market_pts_conf_idx", -1)
    pts_market_quality_idx = feature_spec.get("market_pts_quality_idx", -1)
    pts_market_lean_idx = feature_spec.get("market_pts_lean_idx", -1)
    market_hist = Lambda(
        lambda x: tf.concat(
            [
                gather_feature(x, pts_market_idx),
                gather_feature(x, pts_market_gap_idx),
                gather_feature(x, pts_market_conf_idx),
                gather_feature(x, pts_market_quality_idx),
                gather_feature(x, pts_market_lean_idx),
            ],
            axis=-1,
        ),
        name="pts_market_hist",
    )(seq_input)

    obs = Concatenate(name="pts_obs_concat")([seq_input[:, :, 3:], pts_hist, pts_base_hist, pts_resid, pts_vol, usage_hist, market_hist])
    obs = GaussianNoise(cfg.get("pts_noise_std", 0.02), name="pts_obs_noise")(obs)
    obs = Dense(obs_units, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_obs_dense")(obs)
    obs = LayerNormalization(name="pts_obs_ln")(obs)
    obs = Dropout(dropout, name="pts_obs_drop")(obs)

    x = LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=dropout * 0.4,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="pts_lstm",
    )(obs)
    x = LayerNormalization(name="pts_lstm_ln")(x)
    attn = MultiHeadAttention(num_heads=2, key_dim=max(8, lstm_units // 4), dropout=dropout * 0.5, name="pts_attn")(x, x)
    x = Add(name="pts_attn_skip")([x, attn])
    x = LayerNormalization(name="pts_attn_ln")(x)

    short_state = Lambda(lambda z: tf.reduce_mean(z[:, -3:, :], axis=1), name="pts_short_state")(x)
    medium_state = GlobalAveragePooling1D(name="pts_medium_state")(x)
    max_state = GlobalMaxPooling1D(name="pts_max_state")(x)
    resid_state = GlobalAveragePooling1D(name="pts_resid_state")(pts_resid)
    vol_state = GlobalAveragePooling1D(name="pts_vol_state")(pts_vol)
    usage_state = GlobalAveragePooling1D(name="pts_usage_state")(usage_hist)
    market_state = GlobalAveragePooling1D(name="pts_market_state")(market_hist)
    trend_state = Lambda(
        lambda x: tf.reduce_mean(x[:, -3:, :], axis=1) - tf.reduce_mean(x[:, :3, :], axis=1),
        name="pts_trend_state",
    )(pts_hist)
    trend_resid_state = Lambda(
        lambda x: tf.reduce_mean(x[:, -3:, :], axis=1) - tf.reduce_mean(x[:, :3, :], axis=1),
        name="pts_trend_resid_state",
    )(pts_resid)
    slow_in = Concatenate(name="pts_slow_in")([player_embed, team_embed, opp_embed, usage_state, market_state, base_input])
    slow_state = Dense(slow_dim, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_slow_dense")(slow_in)
    slow_state = LayerNormalization(name="pts_slow_ln")(slow_state)

    latent = Concatenate(
        name="pts_latent_concat"
    )([short_state, medium_state, max_state, resid_state, vol_state, usage_state, market_state, trend_state, trend_resid_state, slow_state, base_input])
    latent = Dense(48, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_latent_dense")(latent)
    latent = LayerNormalization(name="pts_latent_ln")(latent)
    latent = Dropout(dropout, name="pts_latent_drop")(latent)

    state_context = Concatenate(
        name="pts_state_context"
    )([latent, short_state, resid_state, vol_state, usage_state, market_state, trend_state, trend_resid_state, slow_state, base_input])
    state_context = Dense(64, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_state_context_dense")(state_context)
    state_context = LayerNormalization(name="pts_state_context_ln")(state_context)

    pts_continuation = Dense(24, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_continuation_fc")(state_context)
    pts_continuation = Dense(1, activation="sigmoid", name="pts_continuation_prob")(pts_continuation)

    pts_reversion = Dense(24, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_reversion_fc")(state_context)
    pts_reversion = Dense(1, activation="sigmoid", name="pts_reversion_prob")(pts_reversion)

    pts_opportunity = Dense(24, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_opportunity_fc")(state_context)
    pts_opportunity = Dense(1, activation="sigmoid", name="pts_opportunity_jump_prob")(pts_opportunity)

    pts_trend_trust = Dense(24, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_trend_trust_fc")(state_context)
    pts_trend_trust = Dense(1, activation="sigmoid", name="pts_trend_trust")(pts_trend_trust)

    pts_baseline_trust = Dense(24, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_baseline_trust_fc")(state_context)
    pts_baseline_trust = Dense(1, activation="sigmoid", name="pts_baseline_trust")(pts_baseline_trust)

    pts_elasticity = Dense(24, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_elasticity_fc")(state_context)
    pts_elasticity = Dense(1, activation="sigmoid", name="pts_elasticity")(pts_elasticity)

    pts_downside = Dense(24, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="pts_downside_fc")(state_context)
    pts_downside = Dense(1, activation="sigmoid", name="pts_downside_risk")(pts_downside)

    pts_embedding_in = Concatenate(
        name="pts_embedding_in"
    )([
        state_context,
        pts_continuation,
        pts_reversion,
        pts_opportunity,
        pts_trend_trust,
        pts_baseline_trust,
        pts_elasticity,
        pts_downside,
    ])
    pts_embedding = Dense(embed_dim, activation="tanh", kernel_regularizer=regularizers.l2(l2_reg), name="pts_embedding")(pts_embedding_in)

    pts_spike = Dense(1, activation="sigmoid", name="pts_spike")(pts_embedding)
    pts_above = Dense(1, activation="sigmoid", name="pts_above_baseline")(pts_embedding)
    pts_volatility = Dense(1, activation="softplus", name="pts_volatility")(pts_embedding)
    pts_delta_in = Concatenate(
        name="pts_delta_in"
    )([
        pts_embedding,
        pts_continuation,
        pts_reversion,
        pts_opportunity,
        pts_trend_trust,
        pts_baseline_trust,
        pts_elasticity,
        pts_downside,
        pts_spike,
        pts_volatility,
    ])
    pts_delta = Dense(1, activation=None, name="pts_delta")(pts_delta_in)
    pts_spike_gate = Lambda(
        lambda xs: tf.sigmoid(
            4.0
            * (
                0.34 * xs[0]
                + 0.18 * xs[1]
                + 0.14 * xs[2]
                + 0.12 * xs[3]
                + 0.10 * xs[4]
                - 0.18 * xs[5]
                - 0.42
            )
        ),
        name="pts_spike_gate",
    )([
        pts_spike,
        pts_opportunity,
        pts_continuation,
        pts_elasticity,
        pts_trend_trust,
        pts_downside,
    ])
    pts_spike_delta = Lambda(
        lambda xs: xs[0] * xs[1],
        name="pts_spike_delta",
    )([pts_delta, pts_spike_gate])
    pts_normal_delta = Lambda(
        lambda xs: xs[0] - xs[1],
        name="pts_normal_delta",
    )([pts_delta, pts_spike_delta])

    return Model(
        inputs=[seq_input, base_input],
        outputs={
            "pts_embedding": pts_embedding,
            "pts_delta": pts_delta,
            "pts_normal_delta": pts_normal_delta,
            "pts_spike_delta": pts_spike_delta,
            "pts_spike_gate": pts_spike_gate,
            "pts_spike": pts_spike,
            "pts_above_baseline": pts_above,
            "pts_volatility": pts_volatility,
            "pts_continuation_prob": pts_continuation,
            "pts_reversion_prob": pts_reversion,
            "pts_opportunity_jump_prob": pts_opportunity,
            "pts_trend_trust": pts_trend_trust,
            "pts_baseline_trust": pts_baseline_trust,
            "pts_elasticity": pts_elasticity,
            "pts_downside_risk": pts_downside,
        },
        name=f"pts_embedding_s{seed}",
    )


class StructuredLSTMTrainingModel(tf.keras.Model):
    def __init__(self, net, feature_spec, regression_weight=1.0, nll_weight=0.25, sigma_weight=0.08,
                 spike_weight=0.20, feasibility_weight=0.08, bottleneck_weight=0.002,
                 role_regime_weight=0.04, volatility_regime_weight=0.04, context_regime_weight=0.03):
        super().__init__()
        self.net = net
        self.feature_spec = feature_spec
        self.regression_weight = float(regression_weight)
        self.nll_weight = float(nll_weight)
        self.sigma_weight = float(sigma_weight)
        self.spike_weight = float(spike_weight)
        self.feasibility_weight = float(feasibility_weight)
        self.bottleneck_weight = float(bottleneck_weight)
        self.role_regime_weight = float(role_regime_weight)
        self.volatility_regime_weight = float(volatility_regime_weight)
        self.context_regime_weight = float(context_regime_weight)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.delta_mae_tracker = tf.keras.metrics.Mean(name="delta_mae")
        self.spike_rate_tracker = tf.keras.metrics.Mean(name="spike_rate")
        self.feasibility_tracker = tf.keras.metrics.Mean(name="feasibility_rate")
        self.role_shift_tracker = tf.keras.metrics.Mean(name="role_shift_rate")
        self.volatility_regime_tracker = tf.keras.metrics.Mean(name="volatility_regime_rate")
        self.context_pressure_tracker = tf.keras.metrics.Mean(name="context_pressure_rate")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.delta_mae_tracker,
            self.spike_rate_tracker,
            self.feasibility_tracker,
            self.role_shift_tracker,
            self.volatility_regime_tracker,
            self.context_pressure_tracker,
        ]

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)["delta"]

    def _recent_seq_vol(self, x_seq):
        pieces = [
            gather_feature(x_seq, self.feature_spec["pts_idx"]),
            gather_feature(x_seq, self.feature_spec["trb_idx"]),
            gather_feature(x_seq, self.feature_spec["ast_idx"]),
        ]
        stats = tf.concat(pieces, axis=-1)
        return tf.math.reduce_std(stats, axis=1)

    def _labels(self, delta_true, x_seq):
        abs_delta = tf.abs(delta_true)
        mean_abs = tf.reduce_mean(abs_delta, axis=0, keepdims=True)
        std_abs = tf.math.reduce_std(abs_delta, axis=0, keepdims=True)
        spike_labels = tf.cast(abs_delta > (mean_abs + 1.1 * std_abs), tf.float32)

        seq_vol = self._recent_seq_vol(x_seq)
        vol_mean = tf.reduce_mean(seq_vol, axis=0, keepdims=True)
        high_vol = tf.cast(seq_vol > vol_mean, tf.float32)
        predictable = 1.0 - tf.cast(
            tf.logical_or(
                tf.reduce_max(spike_labels, axis=1, keepdims=True) > 0.0,
                tf.reduce_mean(high_vol, axis=1, keepdims=True) > 0.5,
            ),
            tf.float32,
        )

        mp_trend = tf.abs(gather_last(x_seq, self.feature_spec["mp_trend_idx"]))
        high_mp = gather_last(x_seq, self.feature_spec["high_mp_idx"])
        fga_trend = tf.abs(gather_last(x_seq, self.feature_spec["fga_trend_idx"]))
        ast_trend = tf.abs(gather_last(x_seq, self.feature_spec["ast_trend_idx"]))
        usg_ast_ratio_trend = tf.abs(gather_last(x_seq, self.feature_spec["usg_ast_ratio_trend_idx"]))
        high_playmaker = gather_last(x_seq, self.feature_spec["high_playmaker_idx"])
        ast_variance = gather_last(x_seq, self.feature_spec["ast_var_idx"])
        rest_days = gather_last(x_seq, self.feature_spec["rest_days_idx"])
        did_not_play = gather_last(x_seq, self.feature_spec["did_not_play_idx"])
        opp_dfrtg = gather_last(x_seq, self.feature_spec["opp_dfrtg_idx"])

        role_shift_label = tf.cast(
            tf.logical_or(
                tf.reduce_max(
                    tf.concat(
                        [
                            high_mp,
                            high_playmaker,
                            tf.cast(mp_trend > tf.reduce_mean(mp_trend), tf.float32),
                            tf.cast(fga_trend > tf.reduce_mean(fga_trend), tf.float32),
                            tf.cast(usg_ast_ratio_trend > tf.reduce_mean(usg_ast_ratio_trend), tf.float32),
                        ],
                        axis=1,
                    ),
                    axis=1,
                    keepdims=True,
                ) > 0.5,
                tf.reduce_max(
                    tf.cast(ast_trend > (tf.reduce_mean(ast_trend) + 0.5 * tf.math.reduce_std(ast_trend)), tf.float32),
                    axis=1,
                    keepdims=True,
                ) > 0.0,
            ),
            tf.float32,
        )

        volatility_regime_label = tf.cast(
            tf.logical_or(
                tf.reduce_max(spike_labels, axis=1, keepdims=True) > 0.0,
                tf.logical_or(
                    tf.reduce_max(
                        tf.cast(ast_variance > tf.reduce_mean(ast_variance), tf.float32),
                        axis=1,
                        keepdims=True,
                    ) > 0.0,
                    tf.reduce_mean(high_vol, axis=1, keepdims=True) > 0.5,
                ),
            ),
            tf.float32,
        )

        low_rest = tf.cast(rest_days <= 1.0, tf.float32)
        strong_defense = tf.cast(opp_dfrtg < tf.reduce_mean(opp_dfrtg), tf.float32)
        context_pressure_label = tf.cast(
            tf.logical_or(
                tf.reduce_max(
                    tf.concat([low_rest, did_not_play, strong_defense], axis=1),
                    axis=1,
                    keepdims=True,
                ) > 0.0,
                tf.reduce_mean(high_vol, axis=1, keepdims=True) > 0.5,
            ),
            tf.float32,
        )
        return spike_labels, predictable, role_shift_label, volatility_regime_label, context_pressure_label

    def _compute_losses(self, x_seq, base, delta_true, training):
        outputs = self.net([x_seq, base], training=training)
        delta_pred = outputs["delta"]
        sigma = outputs["sigma"]
        spike_prob = outputs["spike_prob"]
        feasibility = outputs["feasibility"]
        bottleneck = outputs["bottleneck"]
        role_shift_prob = outputs["role_shift_prob"]
        volatility_regime_prob = outputs["volatility_regime_prob"]
        context_pressure_prob = outputs["context_pressure_prob"]

        spike_labels, feasible_labels, role_shift_labels, volatility_regime_labels, context_pressure_labels = self._labels(delta_true, x_seq)
        error = delta_true - delta_pred
        abs_error = tf.abs(error)

        huber = tf.reduce_mean(tf.keras.losses.huber(delta_true, delta_pred, delta=1.25))
        mse = tf.reduce_mean(tf.square(error))
        regression_loss = 0.65 * huber + 0.35 * mse

        nll = 0.5 * tf.square(error / sigma) + tf.math.log(sigma)
        nll_loss = tf.reduce_mean(nll)
        sigma_cal = tf.reduce_mean(tf.abs(abs_error - sigma))
        spike_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(spike_labels, spike_prob))
        feasibility_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(feasible_labels, feasibility))
        role_regime_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(role_shift_labels, role_shift_prob))
        volatility_regime_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(volatility_regime_labels, volatility_regime_prob))
        context_regime_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(context_pressure_labels, context_pressure_prob))
        bottleneck_loss = tf.reduce_mean(tf.square(bottleneck))

        total = (
            self.regression_weight * regression_loss
            + self.nll_weight * nll_loss
            + self.sigma_weight * sigma_cal
            + self.spike_weight * spike_loss
            + self.feasibility_weight * feasibility_loss
            + self.role_regime_weight * role_regime_loss
            + self.volatility_regime_weight * volatility_regime_loss
            + self.context_regime_weight * context_regime_loss
            + self.bottleneck_weight * bottleneck_loss
        )
        if self.net.losses:
            total += tf.add_n(self.net.losses)

        metrics = {
            "delta_mae": tf.reduce_mean(abs_error),
            "spike_rate": tf.reduce_mean(spike_labels),
            "feasibility_rate": tf.reduce_mean(feasible_labels),
            "role_shift_rate": tf.reduce_mean(role_shift_labels),
            "volatility_regime_rate": tf.reduce_mean(volatility_regime_labels),
            "context_pressure_rate": tf.reduce_mean(context_pressure_labels),
        }
        return total, metrics

    def train_step(self, data):
        (x_seq, base), delta_true = data
        delta_true = tf.cast(delta_true, tf.float32)
        with tf.GradientTape() as tape:
            total, metrics = self._compute_losses(x_seq, base, delta_true, training=True)
        grads = tape.gradient(total, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        self.loss_tracker.update_state(total)
        self.delta_mae_tracker.update_state(metrics["delta_mae"])
        self.spike_rate_tracker.update_state(metrics["spike_rate"])
        self.feasibility_tracker.update_state(metrics["feasibility_rate"])
        self.role_shift_tracker.update_state(metrics["role_shift_rate"])
        self.volatility_regime_tracker.update_state(metrics["volatility_regime_rate"])
        self.context_pressure_tracker.update_state(metrics["context_pressure_rate"])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (x_seq, base), delta_true = data
        delta_true = tf.cast(delta_true, tf.float32)
        total, metrics = self._compute_losses(x_seq, base, delta_true, training=False)
        self.loss_tracker.update_state(total)
        self.delta_mae_tracker.update_state(metrics["delta_mae"])
        self.spike_rate_tracker.update_state(metrics["spike_rate"])
        self.feasibility_tracker.update_state(metrics["feasibility_rate"])
        self.role_shift_tracker.update_state(metrics["role_shift_rate"])
        self.volatility_regime_tracker.update_state(metrics["volatility_regime_rate"])
        self.context_pressure_tracker.update_state(metrics["context_pressure_rate"])
        return {m.name: m.result() for m in self.metrics}


class PTSEmbeddingTrainingModel(tf.keras.Model):
    def __init__(
        self,
        net,
        feature_spec,
        spike_weight=0.15,
        above_weight=0.10,
        vol_weight=0.08,
        embedding_weight=0.002,
        continuation_weight=0.08,
        reversion_weight=0.08,
        opportunity_weight=0.10,
        trust_weight=0.06,
        elasticity_weight=0.08,
        downside_weight=0.06,
        normal_split_weight=0.10,
        spike_split_weight=0.12,
        gate_weight=0.08,
    ):
        super().__init__()
        self.net = net
        self.feature_spec = feature_spec
        self.spike_weight = float(spike_weight)
        self.above_weight = float(above_weight)
        self.vol_weight = float(vol_weight)
        self.embedding_weight = float(embedding_weight)
        self.continuation_weight = float(continuation_weight)
        self.reversion_weight = float(reversion_weight)
        self.opportunity_weight = float(opportunity_weight)
        self.trust_weight = float(trust_weight)
        self.elasticity_weight = float(elasticity_weight)
        self.downside_weight = float(downside_weight)
        self.normal_split_weight = float(normal_split_weight)
        self.spike_split_weight = float(spike_split_weight)
        self.gate_weight = float(gate_weight)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.delta_mae_tracker = tf.keras.metrics.Mean(name="delta_mae")
        self.spike_rate_tracker = tf.keras.metrics.Mean(name="spike_rate")

    @property
    def metrics(self):
        return [self.loss_tracker, self.delta_mae_tracker, self.spike_rate_tracker]

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)["pts_delta"]

    def _labels(self, x_seq, y_delta):
        pts_idx = self.feature_spec["pts_idx"] if self.feature_spec["pts_idx"] >= 0 else self.feature_spec["pts_lag_idx"]
        pts_hist = gather_feature(x_seq, pts_idx)
        pts_base_hist = gather_feature(x_seq, self.feature_spec["pts_roll_idx"])
        pts_resid = pts_hist - pts_base_hist

        recent_resid = tf.reduce_mean(pts_resid[:, -3:, :], axis=1)
        long_resid = tf.reduce_mean(pts_resid, axis=1)
        recent_trend = tf.reduce_mean(pts_hist[:, -3:, :], axis=1) - tf.reduce_mean(pts_hist[:, :3, :], axis=1)
        recent_vol = tf.math.reduce_std(pts_hist, axis=1)

        abs_recent_resid = tf.abs(recent_resid)
        resid_cutoff = tf.reduce_mean(abs_recent_resid) + 0.25 * tf.math.reduce_std(abs_recent_resid)
        high_resid = abs_recent_resid > resid_cutoff

        continuation_label = tf.cast(
            tf.logical_and(
                tf.sign(y_delta) == tf.sign(recent_resid),
                high_resid,
            ),
            tf.float32,
        )
        reversion_label = tf.cast(
            tf.logical_and(
                tf.sign(y_delta) == -tf.sign(recent_resid),
                high_resid,
            ),
            tf.float32,
        )

        mp_trend = gather_last(x_seq, self.feature_spec["mp_trend_idx"])
        fga_trend = gather_last(x_seq, self.feature_spec["fga_trend_idx"])
        usg = gather_last(x_seq, self.feature_spec["usg_idx"])
        usg_roll = gather_last(x_seq, self.feature_spec["usg_roll_idx"])
        high_mp = gather_last(x_seq, self.feature_spec["high_mp_idx"])
        rest_days = gather_last(x_seq, self.feature_spec["rest_days_idx"])
        did_not_play = gather_last(x_seq, self.feature_spec["did_not_play_idx"])
        opp_dfrtg = gather_last(x_seq, self.feature_spec["opp_dfrtg_idx"])

        opportunity_signal = tf.concat(
            [
                tf.cast(mp_trend > tf.reduce_mean(mp_trend), tf.float32),
                tf.cast(fga_trend > tf.reduce_mean(fga_trend), tf.float32),
                tf.cast(usg > usg_roll, tf.float32),
                high_mp,
            ],
            axis=1,
        )
        opportunity_label = tf.cast(tf.reduce_mean(opportunity_signal, axis=1, keepdims=True) > 0.5, tf.float32)

        trend_trust_label = tf.cast(
            tf.logical_and(
                tf.abs(recent_trend) > tf.reduce_mean(tf.abs(recent_trend)),
                recent_vol < (tf.reduce_mean(recent_vol) + 0.5 * tf.math.reduce_std(recent_vol)),
            ),
            tf.float32,
        )
        baseline_trust_label = tf.cast(
            tf.abs(long_resid) < (tf.reduce_mean(tf.abs(long_resid)) + 0.25 * tf.math.reduce_std(tf.abs(long_resid))),
            tf.float32,
        )
        elasticity_label = tf.cast(
            tf.logical_or(
                opportunity_label > 0.5,
                tf.abs(y_delta) > (tf.reduce_mean(tf.abs(y_delta)) + tf.math.reduce_std(tf.abs(y_delta))),
            ),
            tf.float32,
        )
        downside_label = tf.cast(
            tf.logical_or(
                y_delta < 0.0,
                tf.reduce_max(
                    tf.concat(
                        [
                            tf.cast(rest_days <= 1.0, tf.float32),
                            did_not_play,
                            tf.cast(opp_dfrtg < tf.reduce_mean(opp_dfrtg), tf.float32),
                        ],
                        axis=1,
                    ),
                    axis=1,
                    keepdims=True,
                ) > 0.0,
            ),
            tf.float32,
        )
        return (
            continuation_label,
            reversion_label,
            opportunity_label,
            trend_trust_label,
            baseline_trust_label,
            elasticity_label,
            downside_label,
        )

    @staticmethod
    def _split_residual_targets(y_delta):
        abs_delta = tf.abs(y_delta)
        spike_cutoff = tf.reduce_mean(abs_delta) + 0.75 * tf.math.reduce_std(abs_delta)
        spike_mask = tf.cast(abs_delta > spike_cutoff, tf.float32)
        spike_target = tf.sign(y_delta) * tf.nn.relu(abs_delta - spike_cutoff)
        normal_target = y_delta - spike_target
        return normal_target, spike_target, spike_mask

    def _compute(self, x_seq, x_base, y_delta, training):
        outputs = self.net([x_seq, x_base], training=training)
        delta_pred = outputs["pts_delta"]
        normal_delta_pred = outputs["pts_normal_delta"]
        spike_delta_pred = outputs["pts_spike_delta"]
        spike_gate = outputs["pts_spike_gate"]
        spike_prob = outputs["pts_spike"]
        above_prob = outputs["pts_above_baseline"]
        vol_pred = outputs["pts_volatility"]
        embedding = outputs["pts_embedding"]
        continuation_prob = outputs["pts_continuation_prob"]
        reversion_prob = outputs["pts_reversion_prob"]
        opportunity_prob = outputs["pts_opportunity_jump_prob"]
        trend_trust = outputs["pts_trend_trust"]
        baseline_trust = outputs["pts_baseline_trust"]
        elasticity = outputs["pts_elasticity"]
        downside_risk = outputs["pts_downside_risk"]

        error = y_delta - delta_pred
        abs_error = tf.abs(error)
        huber = tf.reduce_mean(tf.keras.losses.huber(y_delta, delta_pred, delta=1.1))
        mse = tf.reduce_mean(tf.square(error))
        spike_label = tf.cast(tf.abs(y_delta) > (tf.reduce_mean(tf.abs(y_delta)) + tf.math.reduce_std(tf.abs(y_delta))), tf.float32)
        normal_target, spike_target, spike_gate_label = self._split_residual_targets(y_delta)
        above_label = tf.cast(y_delta > 0.0, tf.float32)
        vol_target = tf.abs(y_delta - tf.reduce_mean(y_delta))
        (
            continuation_label,
            reversion_label,
            opportunity_label,
            trend_trust_label,
            baseline_trust_label,
            elasticity_label,
            downside_label,
        ) = self._labels(x_seq, y_delta)

        spike_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(spike_label, spike_prob))
        above_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(above_label, above_prob))
        vol_loss = tf.reduce_mean(tf.square(vol_target - vol_pred))
        normal_split_loss = tf.reduce_mean(tf.keras.losses.huber(normal_target, normal_delta_pred, delta=0.9))
        gate_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(spike_gate_label, spike_gate))
        spike_weights = 0.25 + spike_gate_label
        spike_split_loss = tf.reduce_sum(spike_weights * tf.square(spike_target - spike_delta_pred)) / (
            tf.reduce_sum(spike_weights) + 1e-6
        )
        continuation_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(continuation_label, continuation_prob))
        reversion_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(reversion_label, reversion_prob))
        opportunity_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(opportunity_label, opportunity_prob))
        trend_trust_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(trend_trust_label, trend_trust))
        baseline_trust_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(baseline_trust_label, baseline_trust))
        elasticity_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(elasticity_label, elasticity))
        downside_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(downside_label, downside_risk))
        embed_loss = tf.reduce_mean(tf.square(embedding))
        total = (
            0.65 * huber
            + 0.35 * mse
            + self.spike_weight * spike_loss
            + self.above_weight * above_loss
            + self.vol_weight * vol_loss
            + self.continuation_weight * continuation_loss
            + self.reversion_weight * reversion_loss
            + self.opportunity_weight * opportunity_loss
            + self.trust_weight * (trend_trust_loss + baseline_trust_loss)
            + self.elasticity_weight * elasticity_loss
            + self.downside_weight * downside_loss
            + self.normal_split_weight * normal_split_loss
            + self.spike_split_weight * spike_split_loss
            + self.gate_weight * gate_loss
            + self.embedding_weight * embed_loss
        )
        if self.net.losses:
            total += tf.add_n(self.net.losses)
        return total, tf.reduce_mean(abs_error), tf.reduce_mean(spike_label)

    def train_step(self, data):
        (x_seq, x_base), y_delta = data
        with tf.GradientTape() as tape:
            total, delta_mae, spike_rate = self._compute(tf.cast(x_seq, tf.float32), tf.cast(x_base, tf.float32), tf.cast(y_delta, tf.float32), True)
        grads = tape.gradient(total, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        self.loss_tracker.update_state(total)
        self.delta_mae_tracker.update_state(delta_mae)
        self.spike_rate_tracker.update_state(spike_rate)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (x_seq, x_base), y_delta = data
        total, delta_mae, spike_rate = self._compute(tf.cast(x_seq, tf.float32), tf.cast(x_base, tf.float32), tf.cast(y_delta, tf.float32), False)
        self.loss_tracker.update_state(total)
        self.delta_mae_tracker.update_state(delta_mae)
        self.spike_rate_tracker.update_state(spike_rate)
        return {m.name: m.result() for m in self.metrics}


class CosineAnnealingWarmRestarts(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=5, max_lr=6e-4, min_lr=1e-6, t0=35, t_mult=2):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.t0 = t0
        self.t_mult = t_mult

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / max(1, self.warmup_epochs))
        else:
            t = epoch - self.warmup_epochs
            current = self.t0
            while t >= current:
                t -= current
                current = int(current * self.t_mult)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * t / max(1, current)))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, float(lr))


class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, swa_start=30, swa_freq=4):
        super().__init__()
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.swa_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.swa_start or (epoch - self.swa_start) % self.swa_freq != 0:
            return
        weights = self.model.get_weights()
        if self.swa_weights is None:
            self.swa_weights = [np.copy(w) for w in weights]
        else:
            for i, w in enumerate(weights):
                self.swa_weights[i] = (self.swa_weights[i] * self.swa_count + w) / (self.swa_count + 1)
        self.swa_count += 1

    def apply_swa(self):
        if self.swa_weights is not None and self.swa_count > 1:
            self.model.set_weights(self.swa_weights)
            return True
        return False


class R2MonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, b_val, scaler_y, y_val_raw, target_names):
        super().__init__()
        self.X_val = X_val
        self.b_val = b_val
        self.scaler_y = scaler_y
        self.y_val_raw = y_val_raw
        self.target_names = target_names
        self.best_avg = 999.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 != 0:
            return
        delta_pred = self.model.predict([self.X_val, self.b_val], verbose=0, batch_size=512)
        pred_orig = self.scaler_y.inverse_transform(self.b_val + delta_pred)
        maes, r2s = [], []
        for i, target in enumerate(self.target_names):
            maes.append(mean_absolute_error(self.y_val_raw[:, i], pred_orig[:, i]))
            r2s.append(r2_score(self.y_val_raw[:, i], pred_orig[:, i]))
        avg = float(np.mean(maes))
        if avg < self.best_avg:
            self.best_avg = avg
            self.best_epoch = epoch + 1
        parts = " | ".join(f"{t}: MAE={m:.3f} R2={r:.3f}" for t, m, r in zip(self.target_names, maes, r2s))
        print(f"\n  [Epoch {epoch + 1}] {parts} | Avg={avg:.3f} (best={self.best_avg:.3f}@{self.best_epoch})")


def train_single(X_train, b_train, delta_train, X_val, b_val, delta_val, y_val_raw,
                 target_names, feature_spec, counts, seq_len, n_features, n_targets,
                 seed=42, config=None, scaler_y=None):
    cfg = dict(config or {})
    epochs = cfg.get("epochs", 120)
    batch_size = cfg.get("batch_size", 96)
    max_lr = cfg.get("max_lr", 5e-4)

    net = build_structured_lstm(
        seq_len=seq_len,
        n_features=n_features,
        n_targets=n_targets,
        feature_spec=feature_spec,
        counts=counts,
        seed=seed,
        config=cfg,
    )
    model = StructuredLSTMTrainingModel(
        net,
        feature_spec=feature_spec,
        regression_weight=cfg.get("regression_weight", 1.0),
        nll_weight=cfg.get("nll_weight", 0.25),
        sigma_weight=cfg.get("sigma_weight", 0.08),
        spike_weight=cfg.get("spike_weight", 0.20),
        feasibility_weight=cfg.get("feasibility_weight", 0.08),
        bottleneck_weight=cfg.get("bottleneck_weight", 0.002),
        role_regime_weight=cfg.get("role_regime_weight", 0.04),
        volatility_regime_weight=cfg.get("volatility_regime_weight", 0.04),
        context_regime_weight=cfg.get("context_regime_weight", 0.03),
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(max_lr), clipnorm=1.0))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=24, restore_best_weights=True, verbose=1),
        CosineAnnealingWarmRestarts(warmup_epochs=6, max_lr=max_lr, min_lr=1e-6, t0=35, t_mult=2),
        SWACallback(swa_start=28, swa_freq=4),
    ]
    if scaler_y is not None:
        callbacks.append(R2MonitorCallback(X_val, b_val, scaler_y, y_val_raw, target_names))

    history = model.fit(
        [X_train, b_train],
        delta_train,
        validation_data=([X_val, b_val], delta_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    best_val = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(best_val) + 1
    print(f"  Best val_loss: {best_val:.5f} at epoch {best_epoch}")

    swa_cb = next((cb for cb in callbacks if isinstance(cb, SWACallback)), None)
    es_weights = model.get_weights()
    es_val = best_val
    if swa_cb and swa_cb.apply_swa():
        swa_val = model.evaluate([X_val, b_val], delta_val, verbose=0)[0]
        print(f"  SWA val_loss: {swa_val:.5f} vs ES val_loss: {es_val:.5f}")
        if swa_val >= es_val:
            model.set_weights(es_weights)
        else:
            best_val = swa_val

    return model, history, best_val


def get_ensemble_variants():
    return [
        {"lstm_units": 96, "lstm2_units": 64, "dense_dim": 96, "dropout": 0.25, "obs_units": 96},
        {"lstm_units": 112, "lstm2_units": 72, "dense_dim": 112, "dropout": 0.22, "obs_units": 112},
        {"lstm_units": 88, "lstm2_units": 56, "dense_dim": 88, "dropout": 0.28, "obs_units": 88},
    ]


def train_ensemble(X_train, b_train, delta_train, X_val, b_val, delta_val, y_val_raw,
                   target_names, feature_spec, counts, seq_len, n_features, n_targets,
                   config=None, scaler_y=None, n_models=3):
    models = []
    val_losses = []
    variants = get_ensemble_variants()

    for i in range(n_models):
        seed = 42 + i * 19
        print(f"\n{'=' * 60}")
        print(f"  ENSEMBLE MEMBER {i + 1}/{n_models} (seed={seed})")
        print(f"{'=' * 60}")
        cfg = dict(config or {})
        cfg.update(variants[i % len(variants)])

        model, _history, best_val = train_single(
            X_train,
            b_train,
            delta_train,
            X_val,
            b_val,
            delta_val,
            y_val_raw,
            target_names,
            feature_spec,
            counts,
            seq_len,
            n_features,
            n_targets,
            seed=seed,
            config=cfg,
            scaler_y=scaler_y,
        )
        models.append(model)
        val_losses.append(best_val)

    return models, val_losses


def weighted_ensemble_predict(models, val_losses, X, baselines):
    weights = 1.0 / (np.array(val_losses) + 1e-8)
    weights /= weights.sum()
    preds = [_run_inference_forward(m, X, baselines)["delta"] for m in models]
    return sum(p * w for p, w in zip(preds, weights))


def _run_inference_forward(model_or_net, X, baselines):
    net = getattr(model_or_net, "net", model_or_net)
    forward = getattr(model_or_net, "_inference_forward", None)
    if forward is not None:
        outputs = forward(X, baselines)
    else:
        outputs = net([X, baselines], training=False)
    return {
        key: value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        for key, value in outputs.items()
    }


def weighted_ensemble_diagnostics(models, val_losses, X, baselines):
    weights = 1.0 / (np.array(val_losses) + 1e-8)
    weights /= weights.sum()
    block_names = [
        "delta",
        "delta_normal",
        "delta_tail",
        "sigma",
        "spike_prob",
        "feasibility",
        "bottleneck",
        "belief_mu",
        "belief_std",
        "slow_mode",
        "stable_env",
        "role_shift_prob",
        "volatility_regime_prob",
        "context_pressure_prob",
    ]
    blended = {name: None for name in block_names}
    for model, weight in zip(models, weights):
        outputs = _run_inference_forward(model, X, baselines)
        for name in block_names:
            arr = outputs[name]
            blended[name] = arr * weight if blended[name] is None else blended[name] + arr * weight
    return blended


def weighted_ensemble_latent_export(models, val_losses, X, baselines):
    diagnostics = weighted_ensemble_diagnostics(models, val_losses, X, baselines)
    return {
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


def build_structured_latent_feature_matrix(latent_outputs):
    return np.hstack([
        latent_outputs["bottleneck"],
        latent_outputs["belief_mu"],
        latent_outputs["belief_std"],
        latent_outputs["slow_mode"],
        latent_outputs["stable_env"],
        latent_outputs["sigma"],
        latent_outputs["spike_prob"],
        latent_outputs["feasibility"],
        latent_outputs["role_shift_prob"],
        latent_outputs["volatility_regime_prob"],
        latent_outputs["context_pressure_prob"],
    ])


def get_structured_latent_block_dims(latent_outputs):
    ordered_blocks = [
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
    return {name: int(latent_outputs[name].shape[1]) for name in ordered_blocks}


def get_pts_state_block_dims(pts_outputs):
    ordered_blocks = [
        "pts_embedding",
        "pts_normal_delta",
        "pts_spike_delta",
        "pts_spike_gate",
        "pts_spike",
        "pts_above_baseline",
        "pts_volatility",
        "pts_continuation_prob",
        "pts_reversion_prob",
        "pts_opportunity_jump_prob",
        "pts_trend_trust",
        "pts_baseline_trust",
        "pts_elasticity",
        "pts_downside_risk",
    ]
    return {name: int(pts_outputs[name].shape[1]) for name in ordered_blocks}


def summarize_latent_environment(diagnostics, target_names):
    summary = {
        "global": {
            "slow_mode_mean_abs": float(np.mean(np.abs(diagnostics["slow_mode"]))),
            "stable_env_mean_abs": float(np.mean(np.abs(diagnostics["stable_env"]))),
            "belief_std_mean": float(np.mean(diagnostics["belief_std"])),
            "feasibility_mean": float(np.mean(diagnostics["feasibility"])),
            "role_shift_mean": float(np.mean(diagnostics["role_shift_prob"])),
            "volatility_regime_mean": float(np.mean(diagnostics["volatility_regime_prob"])),
            "context_pressure_mean": float(np.mean(diagnostics["context_pressure_prob"])),
        },
        "targets": {},
    }
    for idx, target in enumerate(target_names):
        summary["targets"][target] = {
            "delta_normal_mean_abs": float(np.mean(np.abs(diagnostics["delta_normal"][:, idx]))),
            "delta_tail_mean_abs": float(np.mean(np.abs(diagnostics["delta_tail"][:, idx]))),
            "spike_prob_mean": float(np.mean(diagnostics["spike_prob"][:, idx])),
            "sigma_mean": float(np.mean(diagnostics["sigma"][:, idx])),
        }
    return summary


def train_pts_embedding_branch(X_train, b_train, delta_train, X_val, b_val, delta_val, feature_spec, counts, config=None):
    cfg = dict(config or {})
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    pts_train = delta_train[:, :1]
    pts_val = delta_val[:, :1]
    base_train = b_train[:, :1]
    base_val = b_val[:, :1]

    net = build_pts_embedding_model(seq_len, n_features, feature_spec, counts, seed=777, config=cfg)
    model = PTSEmbeddingTrainingModel(
        net,
        feature_spec=feature_spec,
        spike_weight=cfg.get("pts_spike_weight", 0.18),
        above_weight=cfg.get("pts_above_weight", 0.10),
        vol_weight=cfg.get("pts_vol_weight", 0.08),
        embedding_weight=cfg.get("pts_embedding_weight", 0.002),
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.get("pts_lr", 4e-4), clipnorm=1.0))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True, verbose=1),
    ]
    print("\n" + "=" * 80)
    print("PTS EMBEDDING BRANCH")
    print("=" * 80)
    model.fit(
        [X_train, base_train],
        pts_train,
        validation_data=([X_val, base_val], pts_val),
        epochs=cfg.get("pts_epochs", 70),
        batch_size=cfg.get("pts_batch_size", 96),
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    train_outputs = net.predict([X_train, base_train], verbose=0, batch_size=512)
    val_outputs = net.predict([X_val, base_val], verbose=0, batch_size=512)
    return model, train_outputs, val_outputs


def build_pts_state_feature_matrix(pts_outputs):
    return np.hstack([
        pts_outputs["pts_embedding"],
        pts_outputs["pts_normal_delta"],
        pts_outputs["pts_spike_delta"],
        pts_outputs["pts_spike_gate"],
        pts_outputs["pts_spike"],
        pts_outputs["pts_above_baseline"],
        pts_outputs["pts_volatility"],
        pts_outputs["pts_continuation_prob"],
        pts_outputs["pts_reversion_prob"],
        pts_outputs["pts_opportunity_jump_prob"],
        pts_outputs["pts_trend_trust"],
        pts_outputs["pts_baseline_trust"],
        pts_outputs["pts_elasticity"],
        pts_outputs["pts_downside_risk"],
    ])


def _block_importance_summary(importances, start_idx, block_dims):
    result = {}
    cursor = int(start_idx)
    total = float(np.sum(importances)) + 1e-8
    for name, width in block_dims.items():
        block_value = float(np.sum(importances[cursor:cursor + width]))
        result[name] = {
            "importance": block_value,
            "share": float(block_value / total),
        }
        cursor += width
    return result


def audit_latent_feature_usage(cb_models, cb_model_info, feature_sets, latent_block_dims, pts_block_dims=None, extra_feature_block_dims=None):
    v2_dim = int(feature_sets["v2"][0].shape[1])
    v3_dim = int(feature_sets["v3"][0].shape[1])
    pts_block_dims = pts_block_dims or {}
    extra_feature_block_dims = extra_feature_block_dims or {}
    audit = {}

    for bundle, info in zip(cb_models, cb_model_info):
        target = info["target"]
        members = bundle["members"] if isinstance(bundle, dict) else [{
            "model": bundle,
            "feature_version": info.get("feature_version", "v2"),
            "candidate": info.get("candidate", "single_model"),
        }]
        weights = np.array(bundle.get("weights", []), dtype=np.float32) if isinstance(bundle, dict) else np.array([1.0], dtype=np.float32)
        if weights.size != len(members) or float(weights.sum()) <= 0:
            weights = np.full(len(members), 1.0 / max(1, len(members)), dtype=np.float32)
        else:
            weights = weights / weights.sum()

        member_summaries = []
        base_share = 0.0
        latent_share = 0.0
        pts_share = 0.0
        block_totals = {}

        for member, weight in zip(members, weights):
            model = member["model"]
            feature_version = member["feature_version"]
            if feature_version in feature_sets:
                X_ref = feature_sets[feature_version][1]
            elif feature_version.endswith("v2"):
                X_ref = feature_sets["v2"][1]
            else:
                X_ref = feature_sets["v3"][1]
            importances = np.asarray(model.get_feature_importance(type="FeatureImportance", data=None), dtype=np.float64)
            if importances.shape[0] != X_ref.shape[1]:
                if importances.shape[0] > X_ref.shape[1]:
                    importances = importances[: X_ref.shape[1]]
                else:
                    importances = np.pad(importances, (0, X_ref.shape[1] - importances.shape[0]))
            total = float(np.sum(importances)) + 1e-8
            if feature_version == "latent_v2":
                base_dim = v2_dim
                block_summary = _block_importance_summary(importances, base_dim, latent_block_dims)
                base_component = float(np.sum(importances[:base_dim]) / total)
                latent_component = float(np.sum(importances[base_dim:]) / total)
                pts_component = 0.0
            elif feature_version == "latent_v3":
                base_dim = v3_dim
                block_summary = _block_importance_summary(importances, base_dim, latent_block_dims)
                base_component = float(np.sum(importances[:base_dim]) / total)
                latent_component = float(np.sum(importances[base_dim:]) / total)
                pts_component = 0.0
            elif feature_version == "pts_v2":
                base_dim = v2_dim
                block_summary = _block_importance_summary(importances, base_dim, pts_block_dims)
                base_component = float(np.sum(importances[:base_dim]) / total)
                latent_component = 0.0
                pts_component = float(np.sum(importances[base_dim:]) / total)
            elif feature_version == "pts_v3":
                base_dim = v3_dim
                block_summary = _block_importance_summary(importances, base_dim, pts_block_dims)
                base_component = float(np.sum(importances[:base_dim]) / total)
                latent_component = 0.0
                pts_component = float(np.sum(importances[base_dim:]) / total)
            elif feature_version in extra_feature_block_dims:
                base_dim = v2_dim if feature_version.endswith("v2") else v3_dim
                block_summary = _block_importance_summary(importances, base_dim, extra_feature_block_dims[feature_version])
                base_component = float(np.sum(importances[:base_dim]) / total)
                latent_component = float(np.sum(importances[base_dim:]) / total)
                pts_component = 0.0
            else:
                block_summary = {}
                base_component = 1.0
                latent_component = 0.0
                pts_component = 0.0

            base_share += float(weight * base_component)
            latent_share += float(weight * latent_component)
            pts_share += float(weight * pts_component)
            for name, payload in block_summary.items():
                block_totals[name] = block_totals.get(name, 0.0) + float(weight * payload["share"])
            member_summaries.append({
                "candidate": member["candidate"],
                "feature_version": feature_version,
                "weight": float(weight),
                "base_share": float(base_component),
                "latent_share": float(latent_component),
                "pts_state_share": float(pts_component),
                "block_shares": {name: float(payload["share"]) for name, payload in block_summary.items()},
            })

        audit[target] = {
            "base_feature_share": float(base_share),
            "latent_feature_share": float(latent_share),
            "pts_state_feature_share": float(pts_share),
            "block_share_summary": {name: float(value) for name, value in sorted(block_totals.items(), key=lambda item: item[1], reverse=True)},
            "members": member_summaries,
        }

    return audit


def prepare_gbm_features_v3(X_seq, baselines):
    n, sl, _nf = X_seq.shape
    last = X_seq[:, -1, :]
    seq_mean = X_seq.mean(axis=1)
    seq_std = X_seq.std(axis=1)
    trend = X_seq[:, -3:, :].mean(axis=1) - X_seq[:, :3, :].mean(axis=1) if sl >= 6 else np.zeros_like(last)
    last3 = X_seq[:, -3:, :].reshape(n, -1)
    seq_min = X_seq.min(axis=1)
    seq_max = X_seq.max(axis=1)
    seq_range = seq_max - seq_min
    momentum = last - seq_mean
    recent_std = X_seq[:, -3:, :].std(axis=1) if sl >= 3 else np.zeros_like(last)
    imm_delta = X_seq[:, -1, :] - X_seq[:, -2, :] if sl >= 2 else np.zeros_like(last)
    q25 = np.quantile(X_seq, 0.25, axis=1)
    q75 = np.quantile(X_seq, 0.75, axis=1)
    iqr = q75 - q25
    median = np.median(X_seq, axis=1)
    last_vs_median = last - median
    weights = np.exp(np.linspace(-1, 0, sl))
    weights /= weights.sum()
    ewm = np.tensordot(weights, X_seq, axes=([0], [1]))
    mid_mom = X_seq[:, -2, :] - X_seq[:, : max(1, sl - 2), :].mean(axis=1)
    accel = momentum - mid_mom
    return np.hstack([
        last, seq_mean, seq_std, trend, last3,
        seq_min, seq_max, seq_range, momentum, recent_std, imm_delta,
        q25, q75, iqr, median, last_vs_median, ewm, accel,
        baselines,
    ])


def create_shared_trainer():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        trainer = UnifiedMoETrainer()
    return trainer


def evaluate_predictions(pred_orig, y_val_orig, b_val_orig, target_names, header):
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)
    total_mae = 0.0
    metrics = {}
    for i, target in enumerate(target_names):
        mae = mean_absolute_error(y_val_orig[:, i], pred_orig[:, i])
        r2 = r2_score(y_val_orig[:, i], pred_orig[:, i])
        baseline_mae = mean_absolute_error(y_val_orig[:, i], b_val_orig[:, i])
        improvement = baseline_mae - mae
        pct = (improvement / baseline_mae * 100.0) if baseline_mae > 0 else 0.0
        total_mae += mae
        metrics[target] = {
            "mae": float(mae),
            "r2": float(r2),
            "baseline_mae": float(baseline_mae),
            "improvement": float(improvement),
            "improvement_pct": float(pct),
        }
        print(f"\n{target}:")
        print(f"  MAE:          {mae:.4f}")
        print(f"  R2:           {r2:.4f}")
        print(f"  Baseline MAE: {baseline_mae:.4f}")
        print(f"  Improvement:  {improvement:+.4f} ({pct:+.1f}%)")
    avg_mae = total_mae / len(target_names)
    print(f"\n{'=' * 60}")
    print(f"  OVERALL AVG MAE: {avg_mae:.4f}")
    print(f"{'=' * 60}")
    return avg_mae, metrics


def evaluate_rolling_windows(pred_orig, y_true_orig, target_names, header, n_windows=4):
    n_samples = len(y_true_orig)
    if n_samples < max(40, n_windows * 8):
        return {
            "n_windows": 0,
            "window_size": int(n_samples),
            "targets": {},
        }

    window_size = max(1, n_samples // n_windows)
    summary = {
        "n_windows": int(n_windows),
        "window_size": int(window_size),
        "targets": {},
    }
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)
    for t_idx, target in enumerate(target_names):
        maes = []
        for w_idx in range(n_windows):
            start = w_idx * window_size
            end = n_samples if w_idx == n_windows - 1 else min(n_samples, (w_idx + 1) * window_size)
            mae = mean_absolute_error(y_true_orig[start:end, t_idx], pred_orig[start:end, t_idx])
            maes.append(float(mae))
        summary["targets"][target] = {
            "window_maes": maes,
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
            "max_mae": float(np.max(maes)),
            "min_mae": float(np.min(maes)),
        }
        print(
            f"  {target}: mean={np.mean(maes):.4f} std={np.std(maes):.4f} "
            f"min={np.min(maes):.4f} max={np.max(maes):.4f} windows={', '.join(f'{m:.4f}' for m in maes)}"
        )
    return summary


def rolling_window_stats_1d(pred, truth, n_windows=4):
    n_samples = len(truth)
    if n_samples == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    window_size = max(1, n_samples // n_windows)
    maes = []
    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = n_samples if w_idx == n_windows - 1 else min(n_samples, (w_idx + 1) * window_size)
        maes.append(float(mean_absolute_error(truth[start:end], pred[start:end])))
    return {
        "mean": float(np.mean(maes)),
        "std": float(np.std(maes)),
        "max": float(np.max(maes)),
        "min": float(np.min(maes)),
    }


def _fixed_pts_ablation_params(seed):
    return dict(
        loss_function="MAE",
        iterations=1000,
        learning_rate=0.023,
        depth=8,
        l2_leaf_reg=3.0,
        min_data_in_leaf=24,
        subsample=0.76,
        colsample_bylevel=0.72,
        early_stopping_rounds=65,
        verbose=0,
        random_seed=int(seed),
    )


def _evaluate_pts_candidate_mae(X_train, pts_delta_train, X_val, pts_delta_val, b_val, y_val_orig, scaler_y, params):
    model = CatBoostRegressor(**params)
    model.fit(X_train, pts_delta_train, eval_set=(X_val, pts_delta_val), verbose=0)
    pred_delta = np.asarray(model.predict(X_val), dtype=np.float32)
    pred_scaled = np.zeros_like(b_val, dtype=np.float32)
    pred_scaled[:, 0] = pred_delta + b_val[:, 0]
    pred_orig = scaler_y.inverse_transform(pred_scaled)[:, 0]
    true_orig = y_val_orig[:, 0]
    mae = float(mean_absolute_error(true_orig, pred_orig))
    r2 = float(r2_score(true_orig, pred_orig))
    rolling = rolling_window_stats_1d(pred_orig, true_orig, n_windows=4)
    return model, pred_delta, mae, r2, rolling


def _get_pts_latent_block_groups():
    return {
        "belief_mu": ["belief_mu"],
        "belief_std": ["belief_std"],
        "belief_pair": ["belief_mu", "belief_std"],
        "bottleneck": ["bottleneck"],
        "stable_env": ["stable_env"],
        "slow_mode": ["slow_mode"],
        "feasibility": ["feasibility"],
        "sigma": ["sigma"],
        "spike_prob": ["spike_prob"],
        "role_shift_prob": ["role_shift_prob"],
        "volatility_regime_prob": ["volatility_regime_prob"],
        "context_pressure_prob": ["context_pressure_prob"],
        "regime_only": ["role_shift_prob", "volatility_regime_prob", "context_pressure_prob"],
        "state_only": ["bottleneck", "belief_mu", "belief_std", "slow_mode", "stable_env", "feasibility"],
        "risk_only": ["sigma", "spike_prob"],
        "state_plus_risk": [
            "bottleneck", "belief_mu", "belief_std", "slow_mode", "stable_env", "feasibility", "sigma", "spike_prob",
        ],
        "state_plus_regime": [
            "bottleneck", "belief_mu", "belief_std", "slow_mode", "stable_env", "feasibility",
            "role_shift_prob", "volatility_regime_prob", "context_pressure_prob",
        ],
        "belief_pair_plus_bottleneck": ["belief_mu", "belief_std", "bottleneck"],
        "belief_pair_plus_bottleneck_plus_feasibility": ["belief_mu", "belief_std", "bottleneck", "feasibility"],
        "belief_pair_plus_bottleneck_plus_feasibility_plus_stable_env": [
            "belief_mu", "belief_std", "bottleneck", "feasibility", "stable_env",
        ],
        "all_latent": [
            "bottleneck", "belief_mu", "belief_std", "slow_mode", "stable_env",
            "sigma", "spike_prob", "feasibility",
            "role_shift_prob", "volatility_regime_prob", "context_pressure_prob",
        ],
    }


def run_pts_latent_ablation(
    X_train,
    b_train,
    delta_train,
    X_val,
    b_val,
    delta_val,
    y_val_orig,
    scaler_y,
    structured_train_latents,
    structured_val_latents,
    config=None,
):
    cfg = dict(config or {})
    seeds = cfg.get("pts_ablation_seeds", [842])
    if isinstance(seeds, int):
        seeds = [seeds]
    seeds = [int(seed) for seed in seeds]

    base_sets = {
        "v2": (prepare_gbm_features_v2(X_train, b_train), prepare_gbm_features_v2(X_val, b_val)),
        "v3": (prepare_gbm_features_v3(X_train, b_train), prepare_gbm_features_v3(X_val, b_val)),
    }

    pts_delta_train = delta_train[:, 0]
    pts_delta_val = delta_val[:, 0]
    base_rows = []
    for base_name, (Xtr, Xva) in base_sets.items():
        maes = []
        r2s = []
        for seed in seeds:
            _model, _pred_delta, mae, r2, _rolling = _evaluate_pts_candidate_mae(
                Xtr,
                pts_delta_train,
                Xva,
                pts_delta_val,
                b_val,
                y_val_orig,
                scaler_y,
                _fixed_pts_ablation_params(seed),
            )
            maes.append(mae)
            r2s.append(r2)
        base_rows.append({
            "name": f"base_{base_name}",
            "base_name": base_name,
            "block_names": [],
            "n_features": int(Xtr.shape[1]),
            "mae": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "r2": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        })
    base_rows.sort(key=lambda item: item["mae"])
    best_base = base_rows[0]
    best_base_name = best_base["base_name"]
    X_base_train, X_base_val = base_sets[best_base_name]

    rows = list(base_rows)
    best_subset = {
        "name": best_base["name"],
        "block_names": [],
        "mae": float(best_base["mae"]),
        "feature_version": best_base_name,
        "train": X_base_train,
        "val": X_base_val,
    }

    for group_name, block_names in _get_pts_latent_block_groups().items():
        Xtr = np.hstack([X_base_train, np.hstack([structured_train_latents[name] for name in block_names])])
        Xva = np.hstack([X_base_val, np.hstack([structured_val_latents[name] for name in block_names])])
        maes = []
        r2s = []
        for seed in seeds:
            _model, _pred_delta, mae, r2, _rolling = _evaluate_pts_candidate_mae(
                Xtr,
                pts_delta_train,
                Xva,
                pts_delta_val,
                b_val,
                y_val_orig,
                scaler_y,
                _fixed_pts_ablation_params(seed),
            )
            maes.append(mae)
            r2s.append(r2)
        row = {
            "name": f"{best_base_name}+{group_name}",
            "base_name": best_base_name,
            "block_names": list(block_names),
            "n_features": int(Xtr.shape[1]),
            "mae": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "r2": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "delta_vs_best_base": float(np.mean(maes) - best_base["mae"]),
        }
        rows.append(row)
        if row["mae"] + 1e-8 < best_subset["mae"]:
            best_subset = {
                "name": row["name"],
                "block_names": list(block_names),
                "mae": float(row["mae"]),
                "feature_version": f"pts_ablate_{best_base_name}",
                "train": Xtr,
                "val": Xva,
            }

    baseline_pred_orig = scaler_y.inverse_transform(b_val)[:, 0]
    baseline_mae = float(mean_absolute_error(y_val_orig[:, 0], baseline_pred_orig))
    baseline_r2 = float(r2_score(y_val_orig[:, 0], baseline_pred_orig))
    rows.append({
        "name": "baseline_anchor",
        "base_name": "baseline",
        "block_names": [],
        "n_features": 0,
        "mae": baseline_mae,
        "mae_std": 0.0,
        "r2": baseline_r2,
        "r2_std": 0.0,
        "delta_vs_best_base": float(baseline_mae - best_base["mae"]),
    })
    rows = sorted(rows, key=lambda item: item["mae"])

    print("\n" + "=" * 80)
    print("PTS LATENT ABLATION")
    print("=" * 80)
    print(f"  Seeds: {seeds}")
    print(f"  Best base family: {best_base_name} | MAE={best_base['mae']:.4f} | R2={best_base['r2']:.4f}")
    for row in rows[:12]:
        print(
            f"  {row['name']:44s} MAE={row['mae']:.4f}±{row['mae_std']:.4f} "
            f"R2={row['r2']:.4f}±{row['r2_std']:.4f} "
            f"delta={row.get('delta_vs_best_base', 0.0):+.4f} n={row['n_features']}"
        )

    ablation_payload = {
        "seeds": seeds,
        "best_base_name": best_base_name,
        "best_base_mae": float(best_base["mae"]),
        "best_base_r2": float(best_base["r2"]),
        "best_subset_name": best_subset["name"],
        "best_subset_blocks": list(best_subset["block_names"]),
        "best_subset_mae": float(best_subset["mae"]),
        "baseline_anchor_mae": baseline_mae,
        "baseline_anchor_r2": baseline_r2,
        "rows": rows,
    }
    feature_pair = (best_subset["train"], best_subset["val"])
    return ablation_payload, (best_subset["feature_version"], feature_pair, list(best_subset["block_names"]))


def train_catboost_delta(X_train, b_train, delta_train, X_val, b_val, delta_val, y_val_orig, scaler_y, target_names,
                         pts_augmented=None, latent_augmented=None, structured_latent_pair=None, config=None):
    if CatBoostRegressor is None:
        raise ImportError("catboost is required for improved_lstm stacking mode")

    feature_sets = {
        "v2": (prepare_gbm_features_v2(X_train, b_train), prepare_gbm_features_v2(X_val, b_val)),
        "v3": (prepare_gbm_features_v3(X_train, b_train), prepare_gbm_features_v3(X_val, b_val)),
    }
    if latent_augmented is not None:
        feature_sets["latent_v2"] = latent_augmented["v2"]
        feature_sets["latent_v3"] = latent_augmented["v3"]
    pts_latent_ablation = None
    pts_ablate_feature_version = None
    if structured_latent_pair is not None:
        pts_latent_ablation, pts_ablate_feature_spec = run_pts_latent_ablation(
            X_train,
            b_train,
            delta_train,
            X_val,
            b_val,
            delta_val,
            y_val_orig,
            scaler_y,
            structured_latent_pair[0],
            structured_latent_pair[1],
            config=config,
        )
        pts_ablate_feature_version, pts_ablate_feature_pair, pts_ablate_blocks = pts_ablate_feature_spec
        if pts_ablate_blocks:
            feature_sets[pts_ablate_feature_version] = pts_ablate_feature_pair
        else:
            pts_ablate_feature_version = None

    models = []
    delta_pred = np.zeros_like(delta_val)
    model_info = []
    recency_weight_train = np.exp(np.linspace(-1.0, 0.0, len(X_train))).astype(np.float32)
    recency_weight_train = recency_weight_train / np.mean(recency_weight_train)
    print("\n" + "=" * 80)
    print("CATBOOST DELTA MODEL")
    print("=" * 80)
    dim_parts = [f"v2={feature_sets['v2'][0].shape[1]}", f"v3={feature_sets['v3'][0].shape[1]}"]
    if "latent_v2" in feature_sets:
        dim_parts.extend([
            f"latent_v2={feature_sets['latent_v2'][0].shape[1]}",
            f"latent_v3={feature_sets['latent_v3'][0].shape[1]}",
        ])
    for key, value in feature_sets.items():
        if key.startswith("pts_ablate_"):
            dim_parts.append(f"{key}={value[0].shape[1]}")
    print("GBM feature dims: " + ", ".join(dim_parts))

    def resolve_feature_pair(feature_version):
        if feature_version == "pts_v2":
            return pts_augmented["v2"]
        if feature_version == "pts_v3":
            return pts_augmented["v3"]
        return feature_sets[feature_version]

    def fit_positive_weights(candidate_matrix, target_delta):
        weights, *_ = np.linalg.lstsq(candidate_matrix, target_delta, rcond=None)
        weights = np.clip(weights.astype(np.float64), 0.0, None)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-8:
            return None
        weights /= weights.sum()
        return weights.astype(np.float32)

    def candidate_score(mae, rolling_stats):
        return float(mae + 0.025 * rolling_stats["std"] + 0.01 * max(0.0, rolling_stats["max"] - mae))

    for t_idx, target in enumerate(target_names):
        candidate_specs = [
            ("base_v2", "v2", dict(
                loss_function="MAE", iterations=700, learning_rate=0.035,
                depth=6, l2_leaf_reg=1.5, min_data_in_leaf=20,
                subsample=0.8, colsample_bylevel=0.8, early_stopping_rounds=40,
                verbose=0, random_seed=42 + t_idx,
            )),
            ("stable_v3", "v3", dict(
                loss_function="MAE", iterations=800, learning_rate=0.03,
                depth=6, l2_leaf_reg=2.0, min_data_in_leaf=24,
                subsample=0.8, colsample_bylevel=0.8, early_stopping_rounds=45,
                verbose=0, random_seed=142 + t_idx,
            )),
            ("base_v2_s2", "v2", dict(
                loss_function="MAE", iterations=900, learning_rate=0.028,
                depth=7, l2_leaf_reg=2.5, min_data_in_leaf=18,
                subsample=0.82, colsample_bylevel=0.78, early_stopping_rounds=55,
                verbose=0, random_seed=1042 + t_idx,
            )),
        ]
        if latent_augmented is not None:
            candidate_specs.extend([
                ("latent_v2", "latent_v2", dict(
                    loss_function="MAE", iterations=850, learning_rate=0.03,
                    depth=6, l2_leaf_reg=2.0, min_data_in_leaf=22,
                    subsample=0.8, colsample_bylevel=0.75, early_stopping_rounds=50,
                    verbose=0, random_seed=642 + t_idx,
                )),
                ("latent_v3", "latent_v3", dict(
                    loss_function="MAE", iterations=950, learning_rate=0.025,
                    depth=7, l2_leaf_reg=2.5, min_data_in_leaf=24,
                    subsample=0.78, colsample_bylevel=0.75, early_stopping_rounds=55,
                    verbose=0, random_seed=742 + t_idx,
                )),
                ("latent_v2_s2", "latent_v2", dict(
                    loss_function="MAE", iterations=1000, learning_rate=0.024,
                    depth=7, l2_leaf_reg=3.0, min_data_in_leaf=20,
                    subsample=0.8, colsample_bylevel=0.72, early_stopping_rounds=60,
                    verbose=0, random_seed=1742 + t_idx,
                )),
                ("latent_v3_s2", "latent_v3", dict(
                    loss_function="MAE", iterations=1100, learning_rate=0.022,
                    depth=8, l2_leaf_reg=3.5, min_data_in_leaf=24,
                    subsample=0.76, colsample_bylevel=0.70, early_stopping_rounds=65,
                    verbose=0, random_seed=2742 + t_idx,
                )),
                ("lossguide_latent_v3", "latent_v3", dict(
                    loss_function="MAE", iterations=900, learning_rate=0.028,
                    depth=8, grow_policy="Lossguide", max_leaves=64,
                    l2_leaf_reg=3.0, min_data_in_leaf=20,
                    subsample=0.8, colsample_bylevel=0.72, early_stopping_rounds=60,
                    verbose=0, random_seed=3742 + t_idx,
                )),
                ("robust_latent_v2", "latent_v2", dict(
                    loss_function="MAE", iterations=950, learning_rate=0.026,
                    depth=7, l2_leaf_reg=3.5, min_data_in_leaf=24,
                    bootstrap_type="Bayesian", bagging_temperature=0.8,
                    random_strength=1.5, model_shrink_rate=0.02, model_shrink_mode="Constant",
                    colsample_bylevel=0.72, early_stopping_rounds=60,
                    verbose=0, random_seed=4742 + t_idx,
                )),
                ("robust_latent_v3", "latent_v3", dict(
                    loss_function="MAE", iterations=1050, learning_rate=0.022,
                    depth=8, l2_leaf_reg=4.0, min_data_in_leaf=26,
                    bootstrap_type="Bayesian", bagging_temperature=1.0,
                    random_strength=1.8, model_shrink_rate=0.025, model_shrink_mode="Constant",
                    colsample_bylevel=0.70, early_stopping_rounds=70,
                    verbose=0, random_seed=5742 + t_idx,
                )),
                ("recency_latent_v2", "latent_v2", dict(
                    loss_function="MAE", iterations=950, learning_rate=0.026,
                    depth=7, l2_leaf_reg=3.0, min_data_in_leaf=22,
                    subsample=0.80, colsample_bylevel=0.72, early_stopping_rounds=60,
                    verbose=0, random_seed=6742 + t_idx,
                ), {"sample_weight": recency_weight_train}),
                ("recency_latent_v3", "latent_v3", dict(
                    loss_function="MAE", iterations=1100, learning_rate=0.021,
                    depth=8, l2_leaf_reg=3.8, min_data_in_leaf=24,
                    subsample=0.78, colsample_bylevel=0.70, early_stopping_rounds=70,
                    verbose=0, random_seed=7742 + t_idx,
                ), {"sample_weight": recency_weight_train}),
            ])
        if t_idx == 0:
            candidate_specs.extend([
                ("pts_deep_v3", "v3", dict(
                    loss_function="MAE", iterations=1100, learning_rate=0.025,
                    depth=8, l2_leaf_reg=3.0, min_data_in_leaf=28,
                    subsample=0.75, colsample_bylevel=0.75, early_stopping_rounds=60,
                    verbose=0, random_seed=242,
                )),
                ("pts_wide_v3", "v3", dict(
                    loss_function="RMSE", iterations=950, learning_rate=0.03,
                    depth=7, l2_leaf_reg=1.0, min_data_in_leaf=16,
                    subsample=0.85, colsample_bylevel=0.85, early_stopping_rounds=55,
                    verbose=0, random_seed=342,
                )),
            ])
            if latent_augmented is not None:
                candidate_specs.extend([
                    ("pts_latent_v2", "latent_v2", dict(
                        loss_function="MAE", iterations=1000, learning_rate=0.025,
                        depth=7, l2_leaf_reg=2.5, min_data_in_leaf=20,
                        subsample=0.78, colsample_bylevel=0.75, early_stopping_rounds=60,
                        verbose=0, random_seed=842,
                    )),
                    ("pts_latent_v3", "latent_v3", dict(
                        loss_function="MAE", iterations=1100, learning_rate=0.022,
                    depth=8, l2_leaf_reg=3.0, min_data_in_leaf=24,
                    subsample=0.75, colsample_bylevel=0.72, early_stopping_rounds=65,
                    verbose=0, random_seed=942,
                )),
                ("pts_latent_v2_s2", "latent_v2", dict(
                    loss_function="MAE", iterations=1100, learning_rate=0.022,
                    depth=8, l2_leaf_reg=3.0, min_data_in_leaf=22,
                    subsample=0.76, colsample_bylevel=0.72, early_stopping_rounds=70,
                    verbose=0, random_seed=1842,
                )),
                ("pts_latent_v3_s2", "latent_v3", dict(
                    loss_function="MAE", iterations=1200, learning_rate=0.020,
                    depth=8, l2_leaf_reg=3.5, min_data_in_leaf=24,
                    subsample=0.74, colsample_bylevel=0.70, early_stopping_rounds=75,
                    verbose=0, random_seed=1942,
                )),
                ("pts_robust_latent_v3", "latent_v3", dict(
                    loss_function="MAE", iterations=1250, learning_rate=0.019,
                    depth=8, l2_leaf_reg=4.2, min_data_in_leaf=28,
                    bootstrap_type="Bayesian", bagging_temperature=1.2,
                    random_strength=2.0, model_shrink_rate=0.03, model_shrink_mode="Constant",
                    colsample_bylevel=0.68, early_stopping_rounds=80,
                    verbose=0, random_seed=2942,
                )),
                ("pts_recency_latent_v3", "latent_v3", dict(
                    loss_function="MAE", iterations=1300, learning_rate=0.018,
                    depth=8, l2_leaf_reg=4.0, min_data_in_leaf=26,
                    subsample=0.76, colsample_bylevel=0.68, early_stopping_rounds=85,
                    verbose=0, random_seed=3942,
                ), {"sample_weight": recency_weight_train}),
                ])
            if pts_augmented is not None:
                candidate_specs.extend([
                    ("pts_embed_v2", "pts_v2", dict(
                        loss_function="MAE", iterations=900, learning_rate=0.03,
                        depth=6, l2_leaf_reg=1.5, min_data_in_leaf=18,
                        subsample=0.8, colsample_bylevel=0.8, early_stopping_rounds=50,
                        verbose=0, random_seed=442,
                    )),
                    ("pts_embed_v3", "pts_v3", dict(
                        loss_function="MAE", iterations=1000, learning_rate=0.025,
                        depth=7, l2_leaf_reg=2.5, min_data_in_leaf=20,
                        subsample=0.78, colsample_bylevel=0.78, early_stopping_rounds=55,
                        verbose=0, random_seed=542,
                    )),
                ])
            if pts_ablate_feature_version is not None:
                candidate_specs.extend([
                    ("pts_ablate_core", pts_ablate_feature_version, dict(
                        loss_function="MAE", iterations=1050, learning_rate=0.023,
                        depth=8, l2_leaf_reg=3.0, min_data_in_leaf=24,
                        subsample=0.76, colsample_bylevel=0.72, early_stopping_rounds=65,
                        verbose=0, random_seed=4942,
                    )),
                    ("pts_ablate_robust", pts_ablate_feature_version, dict(
                        loss_function="MAE", iterations=1150, learning_rate=0.021,
                        depth=8, l2_leaf_reg=3.8, min_data_in_leaf=26,
                        bootstrap_type="Bayesian", bagging_temperature=1.0,
                        random_strength=1.8, model_shrink_rate=0.025, model_shrink_mode="Constant",
                        colsample_bylevel=0.70, early_stopping_rounds=72,
                        verbose=0, random_seed=5942,
                    )),
                ])

        best_model_bundle = None
        best_feature_version = None
        best_delta = None
        best_mae = float("inf")
        best_score = float("inf")
        best_name = None
        candidate_records = []

        for spec in candidate_specs:
            if len(spec) == 3:
                candidate_name, feature_version, params = spec
                fit_kwargs = {}
            elif len(spec) == 4:
                candidate_name, feature_version, params, fit_kwargs = spec
            else:
                raise ValueError(f"Unexpected candidate spec format: {spec}")
            X_gbm_train, X_gbm_val = resolve_feature_pair(feature_version)
            model = CatBoostRegressor(**params)
            model.fit(X_gbm_train, delta_train[:, t_idx], eval_set=(X_gbm_val, delta_val[:, t_idx]), verbose=0, **fit_kwargs)
            candidate_delta = model.predict(X_gbm_val)
            full_delta = delta_pred.copy()
            full_delta[:, t_idx] = candidate_delta
            pred_orig = scaler_y.inverse_transform(b_val + full_delta)
            mae = mean_absolute_error(y_val_orig[:, t_idx], pred_orig[:, t_idx])
            rolling_stats = rolling_window_stats_1d(pred_orig[:, t_idx], y_val_orig[:, t_idx], n_windows=4)
            score = candidate_score(mae, rolling_stats)
            candidate_records.append({
                "name": candidate_name,
                "feature_version": feature_version,
                "model": model,
                "delta": candidate_delta,
                "mae": float(mae),
                "score": float(score),
                "rolling_stats": rolling_stats,
                "best_iteration": int(model.best_iteration_),
            })
            if score + 1e-8 < best_score or (abs(score - best_score) <= 1e-8 and mae < best_mae):
                best_mae = mae
                best_score = score
                best_model_bundle = {
                    "members": [{
                        "model": model,
                        "feature_version": feature_version,
                        "candidate": candidate_name,
                        "best_iteration": int(model.best_iteration_),
                    }],
                    "weights": [1.0],
                    "ensemble_size": 1,
                }
                best_feature_version = feature_version
                best_delta = candidate_delta
                best_name = candidate_name

        candidate_records.sort(key=lambda item: (item["score"], item["mae"]))
        for top_k in (2, 3, 4):
            if len(candidate_records) < top_k:
                continue
            top_records = candidate_records[:top_k]
            candidate_matrix = np.column_stack([rec["delta"] for rec in top_records])

            blend_variants = [
                (f"top{top_k}_ensemble", np.full(top_k, 1.0 / top_k, dtype=np.float32)),
            ]
            weighted = fit_positive_weights(candidate_matrix, delta_val[:, t_idx])
            if weighted is not None:
                blend_variants.append((f"weighted_top{top_k}_ensemble", weighted))

            for blend_name, weights in blend_variants:
                blended_delta = candidate_matrix @ weights
                full_delta = delta_pred.copy()
                full_delta[:, t_idx] = blended_delta
                pred_orig = scaler_y.inverse_transform(b_val + full_delta)
                mae = mean_absolute_error(y_val_orig[:, t_idx], pred_orig[:, t_idx])
                rolling_stats = rolling_window_stats_1d(pred_orig[:, t_idx], y_val_orig[:, t_idx], n_windows=4)
                score = candidate_score(mae, rolling_stats)
                if score + 1e-8 < best_score or (abs(score - best_score) <= 1e-8 and mae + 1e-6 < best_mae):
                    best_mae = float(mae)
                    best_score = float(score)
                    best_delta = blended_delta
                    best_name = blend_name
                    best_feature_version = "+".join(rec["feature_version"] for rec in top_records)
                    best_model_bundle = {
                        "members": [
                            {
                                "model": rec["model"],
                                "feature_version": rec["feature_version"],
                                "candidate": rec["name"],
                                "best_iteration": rec["best_iteration"],
                            }
                            for rec in top_records
                        ],
                        "weights": [float(w) for w in weights],
                        "ensemble_size": top_k,
                    }

        delta_pred[:, t_idx] = best_delta
        models.append(best_model_bundle)
        member_feature_versions = [member["feature_version"] for member in best_model_bundle["members"]]
        member_candidates = [member["candidate"] for member in best_model_bundle["members"]]
        model_info.append({
            "target": target,
            "feature_version": best_feature_version,
            "candidate": best_name,
            "feature_versions": member_feature_versions,
            "candidates": member_candidates,
            "weights": [float(w) for w in best_model_bundle.get("weights", [1.0])],
            "ensemble_size": int(best_model_bundle["ensemble_size"]),
            "best_iteration": int(np.mean([member["best_iteration"] for member in best_model_bundle["members"]])),
            "mae": float(best_mae),
            "selection_score": float(best_score),
        })
        print(
            f"  {target}: MAE={best_mae:.4f}, feature={best_feature_version}, "
            f"config={best_name}, members={best_model_bundle['ensemble_size']}, score={best_score:.4f}"
        )

    return models, delta_pred, feature_sets, model_info, pts_latent_ablation


def train_meta_blend(lstm_delta, cb_delta, baselines_val, delta_val, target_names):
    meta_models = []
    meta_delta = np.zeros_like(delta_val)
    print("\n" + "=" * 80)
    print("RIDGE META-BLEND")
    print("=" * 80)
    for t_idx, target in enumerate(target_names):
        X_meta = np.hstack([
            lstm_delta[:, t_idx:t_idx + 1],
            cb_delta[:, t_idx:t_idx + 1],
            baselines_val,
        ])
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_meta, delta_val[:, t_idx])
        meta_delta[:, t_idx] = ridge.predict(X_meta)
        meta_models.append(ridge)
        print(
            f"  {target}: scaled_MAE={np.mean(np.abs(meta_delta[:, t_idx] - delta_val[:, t_idx])):.4f} | "
            f"coefs={ridge.coef_[:2]}"
        )
    return meta_models, meta_delta


def select_per_target_best(predictions_by_method, y_val_orig, target_names):
    chosen_methods = []
    combined_pred = np.zeros_like(next(iter(predictions_by_method.values())))
    per_target_summary = {}

    print("\n" + "=" * 80)
    print("PER-TARGET METHOD SELECTION")
    print("=" * 80)
    for t_idx, target in enumerate(target_names):
        target_scores = {
            method: mean_absolute_error(y_val_orig[:, t_idx], preds[:, t_idx])
            for method, preds in predictions_by_method.items()
        }
        best_method = min(target_scores, key=target_scores.get)
        chosen_methods.append(best_method)
        combined_pred[:, t_idx] = predictions_by_method[best_method][:, t_idx]
        per_target_summary[target] = {
            "best_method": best_method,
            "method_maes": {name: float(score) for name, score in target_scores.items()},
        }
        ordered = ", ".join(f"{name}={score:.4f}" for name, score in sorted(target_scores.items(), key=lambda item: item[1]))
        print(f"  {target}: {best_method} | {ordered}")

    return combined_pred, chosen_methods, per_target_summary


def main(epochs_override=None, batch_size_override=None, save_only=False):
    print("=" * 80)
    print("STRUCTURED LSTM STACK")
    print("  Structured LSTM + CatBoost delta + Ridge meta-blend")
    print("=" * 80)

    trainer = create_shared_trainer()
    X, baselines, y, _df = trainer.prepare_data()

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    b_train, b_val = baselines[:split_idx], baselines[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    delta_train = y_train - b_train
    delta_val = y_val - b_val
    y_val_orig = trainer.scaler_y.inverse_transform(y_val)
    b_val_orig = trainer.scaler_y.inverse_transform(b_val)

    feature_spec = build_feature_spec(trainer.feature_columns)
    counts = {
        "players": len(getattr(trainer, "player_mapping", {})) or int(np.max(X[:, :, 0])) + 1,
        "teams": len(getattr(trainer, "team_mapping", {})) or int(np.max(X[:, :, 1])) + 1,
        "opponents": len(getattr(trainer, "opponent_mapping", {})) or int(np.max(X[:, :, 2])) + 1,
    }

    n_targets = y_train.shape[1]
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    config = {
        "epochs": 120,
        "batch_size": 96,
        "max_lr": 5e-4,
        "obs_units": 96,
        "lstm_units": 96,
        "lstm2_units": 64,
        "bottleneck_dim": 48,
        "belief_dim": 24,
        "slow_dim": 32,
        "env_dim": 32,
        "dense_dim": 96,
        "dropout": 0.25,
        "noise_std": 0.03,
        "l2_reg": 6e-5,
        "regression_weight": 1.0,
        "nll_weight": 0.25,
        "sigma_weight": 0.08,
        "spike_weight": 0.20,
        "feasibility_weight": 0.08,
        "bottleneck_weight": 0.002,
        # Keep regime supervision active, but lighter than the first pass so it
        # shapes the latent without pulling the mean-prediction objective too hard.
        "role_regime_weight": 0.04,
        "volatility_regime_weight": 0.04,
        "context_regime_weight": 0.03,
        "bottleneck_reg": 5e-5,
    }
    if epochs_override is not None:
        config["epochs"] = int(epochs_override)
    if batch_size_override is not None:
        config["batch_size"] = int(batch_size_override)

    n_models = 3
    models, val_losses = train_ensemble(
        X_train,
        b_train,
        delta_train,
        X_val,
        b_val,
        delta_val,
        y_val_orig,
        trainer.target_columns,
        feature_spec,
        counts,
        seq_len,
        n_features,
        n_targets,
        config=config,
        scaler_y=trainer.scaler_y,
        n_models=n_models,
    )

    structured_train_latents = weighted_ensemble_latent_export(models, val_losses, X_train, b_train)
    structured_val_latents = weighted_ensemble_latent_export(models, val_losses, X_val, b_val)
    structured_val_diagnostics = weighted_ensemble_diagnostics(models, val_losses, X_val, b_val)
    latent_feature_train = build_structured_latent_feature_matrix(structured_train_latents)
    latent_feature_val = build_structured_latent_feature_matrix(structured_val_latents)

    pts_branch, pts_train_outputs, pts_val_outputs = train_pts_embedding_branch(
        X_train,
        b_train,
        delta_train,
        X_val,
        b_val,
        delta_val,
        feature_spec,
        counts,
        config=config,
    )
    pts_state_train = build_pts_state_feature_matrix(pts_train_outputs)
    pts_state_val = build_pts_state_feature_matrix(pts_val_outputs)
    pts_augmented = {
        "v2": (
            np.hstack([prepare_gbm_features_v2(X_train, b_train), pts_state_train]),
            np.hstack([prepare_gbm_features_v2(X_val, b_val), pts_state_val]),
        ),
        "v3": (
            np.hstack([prepare_gbm_features_v3(X_train, b_train), pts_state_train]),
            np.hstack([prepare_gbm_features_v3(X_val, b_val), pts_state_val]),
        ),
    }
    latent_augmented = {
        "v2": (
            np.hstack([prepare_gbm_features_v2(X_train, b_train), latent_feature_train]),
            np.hstack([prepare_gbm_features_v2(X_val, b_val), latent_feature_val]),
        ),
        "v3": (
            np.hstack([prepare_gbm_features_v3(X_train, b_train), latent_feature_train]),
            np.hstack([prepare_gbm_features_v3(X_val, b_val), latent_feature_val]),
        ),
    }
    latent_block_dims = get_structured_latent_block_dims(structured_val_latents)
    pts_state_block_dims = get_pts_state_block_dims(pts_val_outputs)

    structured_delta = weighted_ensemble_predict(models, val_losses, X_val, b_val)

    cb_models, cb_delta, feature_sets, cb_model_info, pts_latent_ablation = train_catboost_delta(
        X_train,
        b_train,
        delta_train,
        X_val,
        b_val,
        delta_val,
        y_val_orig,
        trainer.scaler_y,
        trainer.target_columns,
        pts_augmented=pts_augmented,
        latent_augmented=latent_augmented,
        structured_latent_pair=(structured_train_latents, structured_val_latents),
        config=config,
    )
    extra_feature_block_dims = {}
    if pts_latent_ablation and pts_latent_ablation.get("best_subset_blocks"):
        pts_ablate_key = f"pts_ablate_{pts_latent_ablation['best_base_name']}"
        extra_feature_block_dims[pts_ablate_key] = {
            block: latent_block_dims[block]
            for block in pts_latent_ablation["best_subset_blocks"]
            if block in latent_block_dims
        }
    latent_head_audit = audit_latent_feature_usage(
        cb_models,
        cb_model_info,
        feature_sets,
        latent_block_dims=latent_block_dims,
        pts_block_dims=pts_state_block_dims,
        extra_feature_block_dims=extra_feature_block_dims,
    )
    print("\n" + "=" * 80)
    print("LATENT HEAD AUDIT")
    print("=" * 80)
    for target, payload in latent_head_audit.items():
        top_blocks = ", ".join(
            f"{name}={share:.3f}" for name, share in list(payload["block_share_summary"].items())[:5]
        ) or "none"
        print(
            f"  {target}: base_share={payload['base_feature_share']:.3f} "
            f"latent_share={payload['latent_feature_share']:.3f} "
            f"pts_state_share={payload['pts_state_feature_share']:.3f} | {top_blocks}"
        )
    cb_avg_mae = float(np.mean([info["mae"] for info in cb_model_info]))
    cb_metrics = {
        info["target"]: {
            "mae": float(info["mae"]),
            "feature_version": info["feature_version"],
            "candidate": info["candidate"],
        }
        for info in cb_model_info
    }
    cb_rolling_summary = None
    meta_models = []
    targetwise_methods = ["catboost_delta"] * len(trainer.target_columns)
    targetwise_summary = {
        target: {"best_method": "catboost_delta", "method_maes": {"catboost_delta": float(info["mae"])}}
        for target, info in zip(trainer.target_columns, cb_model_info)
    }
    structured_avg_mae = None
    structured_metrics = {}
    meta_avg_mae = None
    meta_metrics = {}
    targetwise_avg_mae = cb_avg_mae
    targetwise_metrics = {}
    best_method = "catboost_delta"
    best_avg_mae = cb_avg_mae
    promoted_to_production = False

    if save_only:
        print("\n" + "=" * 80)
        print("SAVE-ONLY MODE")
        print("=" * 80)
        print(f"  catboost_delta avg_mae: {cb_avg_mae:.4f}")
        print("  Skipped post-train validation comparison and production promotion.")
    else:
        structured_pred_orig = trainer.scaler_y.inverse_transform(b_val + structured_delta)
        structured_avg_mae, structured_metrics = evaluate_predictions(
            structured_pred_orig,
            y_val_orig,
            b_val_orig,
            trainer.target_columns,
            "STRUCTURED LSTM ENSEMBLE",
        )

        cb_pred_orig = trainer.scaler_y.inverse_transform(b_val + cb_delta)
        cb_avg_mae, cb_metrics = evaluate_predictions(
            cb_pred_orig,
            y_val_orig,
            b_val_orig,
            trainer.target_columns,
            "CATBOOST DELTA",
        )
        cb_rolling_summary = evaluate_rolling_windows(
            cb_pred_orig,
            y_val_orig,
            trainer.target_columns,
            "CATBOOST DELTA ROLLING-WINDOW ROBUSTNESS",
            n_windows=4,
        )

        meta_models, meta_delta = train_meta_blend(
            structured_delta,
            cb_delta,
            b_val,
            delta_val,
            trainer.target_columns,
        )
        meta_pred_orig = trainer.scaler_y.inverse_transform(b_val + meta_delta)
        meta_avg_mae, meta_metrics = evaluate_predictions(
            meta_pred_orig,
            y_val_orig,
            b_val_orig,
            trainer.target_columns,
            "STRUCTURED LSTM + CATBOOST META-BLEND",
        )

        targetwise_pred_orig, targetwise_methods, targetwise_summary = select_per_target_best(
            {
                "structured_lstm_ensemble": structured_pred_orig,
                "catboost_delta": cb_pred_orig,
                "structured_lstm_catboost_blend": meta_pred_orig,
            },
            y_val_orig,
            trainer.target_columns,
        )
        targetwise_avg_mae, targetwise_metrics = evaluate_predictions(
            targetwise_pred_orig,
            y_val_orig,
            b_val_orig,
            trainer.target_columns,
            "TARGET-WISE BEST COMPOSITE",
        )

        approaches = {
            "structured_lstm_ensemble": (structured_avg_mae, structured_metrics, structured_delta),
            "catboost_delta": (cb_avg_mae, cb_metrics, cb_delta),
            "structured_lstm_catboost_blend": (meta_avg_mae, meta_metrics, meta_delta),
            "targetwise_best_composite": (targetwise_avg_mae, targetwise_metrics, None),
        }
        best_method = min(approaches, key=lambda name: approaches[name][0])
        best_avg_mae, _best_metrics, _best_delta = approaches[best_method]

        production_path = Path("model/production_structured_lstm_stack.json")
        current_production_mae = PRODUCTION_BEST_MAE
        if production_path.exists():
            try:
                production_payload = json.loads(production_path.read_text(encoding="utf-8"))
                current_production_mae = float(production_payload.get("avg_mae", current_production_mae))
            except Exception:
                current_production_mae = PRODUCTION_BEST_MAE

        print("\n" + "=" * 80)
        print("METHOD COMPARISON")
        print("=" * 80)
        for name, (avg_mae, _metrics, _delta) in sorted(approaches.items(), key=lambda item: item[1][0]):
            marker = " <-- BEST" if name == best_method else ""
            print(f"  {name:32s}: {avg_mae:.4f}{marker}")
        print(f"  current_best_production          : {current_production_mae:.4f}")
        promoted_to_production = best_avg_mae < current_production_mae
        if promoted_to_production:
            print(f"\nNEW BEST: {best_method} improved by {current_production_mae - best_avg_mae:.4f}")
        else:
            print(f"\nBest method this run: {best_method} (needs {best_avg_mae - current_production_mae:.4f} to beat production)")

    os.makedirs("model", exist_ok=True)
    run_id = f"lstm_v7_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("model") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        model.save_weights(run_dir / f"lstm_v7_member_{i}.weights.h5")
    pts_branch.save_weights(run_dir / "lstm_v7_pts_branch.weights.h5")
    joblib.dump(cb_models, run_dir / "lstm_v7_catboost_models.pkl")
    if meta_models:
        joblib.dump(meta_models, run_dir / "lstm_v7_meta_models.pkl")
    joblib.dump(trainer.scaler_x, run_dir / "lstm_v7_scaler_x.pkl")
    joblib.dump(trainer.scaler_y, run_dir / "lstm_v7_scaler_y.pkl")

    run_artifact_paths = {
        "lstm_weights": [str((run_dir / f"lstm_v7_member_{i}.weights.h5").as_posix()) for i in range(n_models)],
        "pts_branch_weights": str((run_dir / "lstm_v7_pts_branch.weights.h5").as_posix()),
        "catboost_models": str((run_dir / "lstm_v7_catboost_models.pkl").as_posix()),
        "scaler_x": str((run_dir / "lstm_v7_scaler_x.pkl").as_posix()),
        "scaler_y": str((run_dir / "lstm_v7_scaler_y.pkl").as_posix()),
        "metadata": str((run_dir / "lstm_v7_metadata.json").as_posix()),
    }
    if meta_models:
        run_artifact_paths["meta_models"] = str((run_dir / "lstm_v7_meta_models.pkl").as_posix())

    metadata = {
        "model_type": "structured_lstm_stack",
        "best_method": best_method,
        "promoted_to_production": promoted_to_production,
        "n_models": n_models,
        "seq_len": seq_len,
        "n_features": n_features,
        "n_targets": n_targets,
        "target_columns": trainer.target_columns,
        "baseline_features": trainer.baseline_features,
        "feature_columns": trainer.feature_columns,
        "feature_spec": feature_spec,
        "seeds": [42 + i * 19 for i in range(n_models)],
        "ensemble_member_configs": [dict(config, **variant) for variant in get_ensemble_variants()[:n_models]],
        "val_losses": [float(v) for v in val_losses],
        "gbm_feature_dims": {name: int(values[0].shape[1]) for name, values in feature_sets.items()},
        "pts_embedding_dim": int(pts_val_outputs["pts_embedding"].shape[1]),
        "pts_state_feature_dim": int(pts_state_val.shape[1]),
        "structured_latent_dim": int(latent_feature_val.shape[1]),
        "structured_latent_blocks": {
            key: int(value.shape[1]) if len(value.shape) > 1 else 1
            for key, value in structured_val_latents.items()
        },
        "pts_state_blocks": pts_state_block_dims,
        "counts": counts,
        "player_mapping": getattr(trainer, "player_mapping", {}),
        "team_mapping": getattr(trainer, "team_mapping", {}),
        "opponent_mapping": getattr(trainer, "opponent_mapping", {}),
        "latent_environment_summary": summarize_latent_environment(structured_val_diagnostics, trainer.target_columns),
        "catboost_model_info": cb_model_info,
        "latent_head_audit": latent_head_audit,
        "pts_latent_ablation": pts_latent_ablation,
        "targetwise_methods": targetwise_methods,
        "targetwise_summary": targetwise_summary,
        "config": config,
        "results": {
            "structured_lstm_ensemble": structured_metrics,
            "catboost_delta": cb_metrics,
            "structured_lstm_catboost_blend": meta_metrics,
            "targetwise_best_composite": targetwise_metrics,
        },
        "avg_mae": float(best_avg_mae),
        "run_id": run_id,
        "artifact_paths": run_artifact_paths,
        "save_only": bool(save_only),
    }
    if cb_rolling_summary is not None:
        metadata["rolling_window_validation"] = {
            "catboost_delta": cb_rolling_summary,
        }
    with open("model/lstm_v7_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    (run_dir / "lstm_v7_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latest_manifest = {
        "run_id": run_id,
        "avg_mae": float(best_avg_mae),
        "best_method": best_method,
        "artifact_paths": run_artifact_paths,
    }
    Path("model/latest_structured_lstm_stack.json").write_text(json.dumps(latest_manifest, indent=2), encoding="utf-8")

    if not save_only:
        best_ever_path = Path("model/lstm_v7_best_ever.json")
        best_ever_payload = None
        if best_ever_path.exists():
            try:
                best_ever_payload = json.loads(best_ever_path.read_text(encoding="utf-8"))
            except Exception:
                best_ever_payload = None
        if best_ever_payload is None or float(best_ever_payload.get("avg_mae", 999.0)) > float(best_avg_mae):
            best_ever_payload = {
                "avg_mae": float(best_avg_mae),
                "best_method": best_method,
                "run_id": run_id,
                "target_columns": trainer.target_columns,
                "catboost_model_info": cb_model_info,
                "artifact_paths": run_artifact_paths,
                "rolling_window_validation": {
                    "catboost_delta": cb_rolling_summary,
                },
            }
            best_ever_path.write_text(json.dumps(best_ever_payload, indent=2), encoding="utf-8")
            print("Best-ever registry updated: model/lstm_v7_best_ever.json")

    if not save_only and promoted_to_production:
        production_metadata = {
            "model_type": "structured_lstm_stack_targetwise",
            "avg_mae": float(best_avg_mae),
            "best_method": best_method,
            "target_columns": trainer.target_columns,
            "targetwise_methods": targetwise_methods,
            "run_id": run_id,
            "artifact_paths": run_artifact_paths,
        }
        with open("model/production_structured_lstm_stack.json", "w", encoding="utf-8") as f:
            json.dump(production_metadata, f, indent=2)
        print("\nProduction candidate promoted: model/production_structured_lstm_stack.json")

    print(f"\nArtifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
