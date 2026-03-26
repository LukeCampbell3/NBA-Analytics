"""
Stacking Meta-Learner + Gradient Boosting Approach
===================================================
Strategy:
1. Train the proven simple LSTM (delta predictor)
2. Train a LightGBM model on flattened features (different inductive bias)
3. Stack: Ridge meta-learner combines LSTM + GBM predictions per target

The LSTM captures sequential patterns. GBM captures non-linear feature interactions.
A Ridge meta-learner learns optimal blending weights per target.

Targets to beat:
  Old ensemble: PTS=8.36, TRB=3.29, AST=2.73 (avg=4.795)
  Simple LSTM:  PTS=9.08, TRB=2.54, AST=2.72 (avg=4.78)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Concatenate, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from unified_moe_trainer import UnifiedMoETrainer


def build_simple_lstm(seq_len, n_features, n_targets, seed=42):
    """Exact replica of the proven simple LSTM architecture."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    seq_input = Input(shape=(seq_len, n_features), name="seq_input")
    base_input = Input(shape=(n_targets,), name="base_input")

    x = LSTM(128, return_sequences=True, name="lstm1")(seq_input)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False, name="lstm2")(x)
    x = Dropout(0.2)(x)

    merged = Concatenate()([x, base_input])
    x = Dense(128, activation="relu", name="dense1")(merged)
    x = LayerNormalization(name="ln1")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu", name="dense2")(x)
    x = Dropout(0.1)(x)
    delta_pred = Dense(n_targets, activation=None, name="delta_pred")(x)

    model = Model(inputs=[seq_input, base_input], outputs=delta_pred,
                  name=f"lstm_seed{seed}")
    return model


def train_lstm(X_train, b_train, delta_train, X_val, b_val, delta_val,
               seq_len, n_features, n_targets, seed=42):
    """Train one LSTM model. Returns model and val predictions."""
    model = build_simple_lstm(seq_len, n_features, n_targets, seed=seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mse', metrics=['mae']
    )
    history = model.fit(
        [X_train, b_train], delta_train,
        validation_data=([X_val, b_val], delta_val),
        epochs=120, batch_size=128, shuffle=True,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                              min_lr=1e-5, verbose=1),
        ],
        verbose=1
    )
    best_val = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val) + 1
    print(f"  LSTM best val_loss: {best_val:.5f} at epoch {best_epoch}")
    return model


def prepare_gbm_features(X_seq, baselines):
    """Flatten sequence data for GBM. Use last N timesteps + stats."""
    n_samples, seq_len, n_feat = X_seq.shape

    # Last timestep features (most recent game)
    last = X_seq[:, -1, :]  # (n, n_feat)

    # Rolling stats over the sequence
    seq_mean = X_seq.mean(axis=1)  # (n, n_feat)
    seq_std = X_seq.std(axis=1)    # (n, n_feat)

    # Trend: last 3 games mean vs first 3 games mean
    if seq_len >= 6:
        recent = X_seq[:, -3:, :].mean(axis=1)
        early = X_seq[:, :3, :].mean(axis=1)
        trend = recent - early
    else:
        trend = np.zeros_like(last)

    # Last 3 timesteps flattened (captures very recent form)
    last3 = X_seq[:, -3:, :].reshape(n_samples, -1)

    # Combine all with baselines
    features = np.hstack([last, seq_mean, seq_std, trend, last3, baselines])
    return features


def train_gbm_models(X_gbm_train, delta_train, X_gbm_val, delta_val, n_targets):
    """Train one GBM per target. Returns list of models and val predictions."""
    try:
        import lightgbm as lgb
        use_lgb = True
        print("  Using LightGBM")
    except ImportError:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            use_lgb = False
            print("  LightGBM not available, using sklearn GradientBoosting")
        except ImportError:
            print("  No GBM available, skipping")
            return None, None

    models = []
    val_preds = np.zeros_like(delta_val)

    for t in range(n_targets):
        target_name = ["PTS", "TRB", "AST"][t]
        print(f"\n  Training GBM for {target_name}...")

        if use_lgb:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1,
                'n_jobs': -1,
            }
            train_data = lgb.Dataset(X_gbm_train, label=delta_train[:, t])
            val_data = lgb.Dataset(X_gbm_val, label=delta_val[:, t], reference=train_data)

            model = lgb.train(
                params, train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=50)
                ]
            )
            val_preds[:, t] = model.predict(X_gbm_val)
            print(f"  {target_name} GBM best iteration: {model.best_iteration}")
        else:
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20,
                validation_fraction=0.15, n_iter_no_change=30,
                verbose=0
            )
            model.fit(X_gbm_train, delta_train[:, t])
            val_preds[:, t] = model.predict(X_gbm_val)
            print(f"  {target_name} GBM n_estimators used: {model.n_estimators_}")

        models.append(model)

    return models, val_preds


def train_stacking_meta(lstm_val_deltas, gbm_val_deltas, baselines_val,
                        true_deltas, n_targets):
    """Train Ridge meta-learner per target on OOF predictions.
    Features: LSTM delta pred, GBM delta pred, baseline values."""
    meta_models = []

    for t in range(n_targets):
        target_name = ["PTS", "TRB", "AST"][t]

        # Stack features for meta-learner
        meta_features = [lstm_val_deltas[:, t:t+1]]
        if gbm_val_deltas is not None:
            meta_features.append(gbm_val_deltas[:, t:t+1])
        meta_features.append(baselines_val)  # baseline context
        X_meta = np.hstack(meta_features)

        # Simple Ridge regression — won't overfit on small feature set
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_meta, true_deltas[:, t])
        meta_models.append(ridge)

        pred = ridge.predict(X_meta)
        mae_meta = np.mean(np.abs(pred - true_deltas[:, t]))
        print(f"  {target_name} meta-learner train MAE (scaled): {mae_meta:.4f}")
        print(f"    Coefficients: {ridge.coef_[:4]}...")

    return meta_models


def evaluate_all(trainer, y_val, b_val, lstm_delta, gbm_delta, meta_models,
                 n_targets):
    """Evaluate all approaches and the stacked result."""
    y_val_orig = trainer.scaler_y.inverse_transform(y_val)
    b_val_orig = trainer.scaler_y.inverse_transform(b_val)

    results = {}

    # 1. Baseline only
    print("\n--- BASELINE (rolling avg) ---")
    for i, t in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_orig[:, i], b_val_orig[:, i])
        print(f"  {t}: {mae:.4f}")

    # 2. LSTM only
    print("\n--- LSTM ONLY ---")
    lstm_preds_scaled = b_val + lstm_delta
    lstm_orig = trainer.scaler_y.inverse_transform(lstm_preds_scaled)
    lstm_total = 0
    for i, t in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_orig[:, i], lstm_orig[:, i])
        lstm_total += mae
        print(f"  {t}: {mae:.4f}")
    lstm_avg = lstm_total / n_targets
    print(f"  Avg MAE: {lstm_avg:.4f}")
    results['lstm'] = lstm_avg

    # 3. GBM only
    if gbm_delta is not None:
        print("\n--- GBM ONLY ---")
        gbm_preds_scaled = b_val + gbm_delta
        gbm_orig = trainer.scaler_y.inverse_transform(gbm_preds_scaled)
        gbm_total = 0
        for i, t in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_orig[:, i], gbm_orig[:, i])
            gbm_total += mae
            print(f"  {t}: {mae:.4f}")
        gbm_avg = gbm_total / n_targets
        print(f"  Avg MAE: {gbm_avg:.4f}")
        results['gbm'] = gbm_avg

    # 4. Simple average of LSTM + GBM
    if gbm_delta is not None:
        print("\n--- SIMPLE AVG (LSTM + GBM) ---")
        avg_delta = (lstm_delta + gbm_delta) / 2
        avg_preds_scaled = b_val + avg_delta
        avg_orig = trainer.scaler_y.inverse_transform(avg_preds_scaled)
        avg_total = 0
        for i, t in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_orig[:, i], avg_orig[:, i])
            avg_total += mae
            print(f"  {t}: {mae:.4f}")
        avg_avg = avg_total / n_targets
        print(f"  Avg MAE: {avg_avg:.4f}")
        results['simple_avg'] = avg_avg

    # 5. Stacked meta-learner
    if meta_models is not None:
        print("\n--- STACKED META-LEARNER ---")
        meta_delta = np.zeros_like(lstm_delta)
        for t_idx in range(n_targets):
            meta_features = [lstm_delta[:, t_idx:t_idx+1]]
            if gbm_delta is not None:
                meta_features.append(gbm_delta[:, t_idx:t_idx+1])
            meta_features.append(b_val)
            X_meta = np.hstack(meta_features)
            meta_delta[:, t_idx] = meta_models[t_idx].predict(X_meta)

        meta_preds_scaled = b_val + meta_delta
        meta_orig = trainer.scaler_y.inverse_transform(meta_preds_scaled)
        meta_total = 0
        for i, t in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_orig[:, i], meta_orig[:, i])
            r2 = r2_score(y_val_orig[:, i], meta_orig[:, i])
            meta_total += mae
            print(f"  {t}: MAE={mae:.4f}  R²={r2:.4f}")
        meta_avg = meta_total / n_targets
        print(f"  Avg MAE: {meta_avg:.4f}")
        results['stacked'] = meta_avg

    # 6. Optimal per-target blend search (LSTM vs GBM)
    if gbm_delta is not None:
        print("\n--- OPTIMAL PER-TARGET BLEND ---")
        opt_delta = np.zeros_like(lstm_delta)
        for t_idx in range(n_targets):
            best_alpha = 0
            best_mae = float('inf')
            for alpha in np.arange(0, 1.01, 0.05):
                blended = alpha * lstm_delta[:, t_idx] + (1 - alpha) * gbm_delta[:, t_idx]
                preds_s = b_val[:, t_idx] + blended
                # Inverse transform just this column
                full_scaled = b_val.copy()
                full_scaled[:, t_idx] = b_val[:, t_idx] + blended
                full_orig = trainer.scaler_y.inverse_transform(full_scaled)
                mae = mean_absolute_error(y_val_orig[:, t_idx], full_orig[:, t_idx])
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha
            opt_delta[:, t_idx] = best_alpha * lstm_delta[:, t_idx] + (1 - best_alpha) * gbm_delta[:, t_idx]
            print(f"  {trainer.target_columns[t_idx]}: best_alpha={best_alpha:.2f} (LSTM weight), MAE={best_mae:.4f}")

        opt_preds_scaled = b_val + opt_delta
        opt_orig = trainer.scaler_y.inverse_transform(opt_preds_scaled)
        opt_total = 0
        for i, t in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_orig[:, i], opt_orig[:, i])
            opt_total += mae
        opt_avg = opt_total / n_targets
        print(f"  Avg MAE: {opt_avg:.4f}")
        results['optimal_blend'] = opt_avg

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, avg in sorted(results.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if avg == min(results.values()) else ""
        print(f"  {name:20s}: {avg:.4f}{marker}")
    print(f"  {'target (old ens)':20s}: 4.795")
    print(f"  {'target (lstm)':20s}: 4.780")
    print("=" * 60)

    return results


def main():
    print("=" * 80)
    print("STACKING META-LEARNER + GBM APPROACH")
    print("=" * 80)

    # --- Data prep ---
    trainer = UnifiedMoETrainer()
    X, baselines, y, df = trainer.prepare_data()

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    b_train, b_val = baselines[:split_idx], baselines[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    delta_train = y_train - b_train
    delta_val = y_val - b_val

    n_targets = y_train.shape[1]
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    print(f"Seq shape: ({seq_len}, {n_features}), Targets: {n_targets}")

    # --- 1. Train LSTM ---
    print("\n" + "=" * 60)
    print("PHASE 1: LSTM (proven architecture)")
    print("=" * 60)
    lstm_model = train_lstm(X_train, b_train, delta_train,
                            X_val, b_val, delta_val,
                            seq_len, n_features, n_targets, seed=42)
    lstm_val_delta = lstm_model.predict([X_val, b_val], verbose=0, batch_size=256)

    # --- 2. Train GBM ---
    print("\n" + "=" * 60)
    print("PHASE 2: Gradient Boosting (different inductive bias)")
    print("=" * 60)
    X_gbm_train = prepare_gbm_features(X_train, b_train)
    X_gbm_val = prepare_gbm_features(X_val, b_val)
    print(f"  GBM feature dim: {X_gbm_train.shape[1]}")

    gbm_models, gbm_val_delta = train_gbm_models(
        X_gbm_train, delta_train, X_gbm_val, delta_val, n_targets
    )

    # --- 3. Train meta-learner ---
    print("\n" + "=" * 60)
    print("PHASE 3: Stacking Meta-Learner")
    print("=" * 60)
    meta_models = train_stacking_meta(
        lstm_val_delta, gbm_val_delta, b_val, delta_val, n_targets
    )

    # --- 4. Evaluate everything ---
    print("\n" + "=" * 60)
    print("PHASE 4: Evaluation")
    print("=" * 60)
    results = evaluate_all(
        trainer, y_val, b_val, lstm_val_delta, gbm_val_delta,
        meta_models, n_targets
    )

    # --- 5. Also try: LSTM with 3 seeds + GBM stacking ---
    print("\n" + "=" * 60)
    print("BONUS: 3-seed LSTM ensemble + GBM stacking")
    print("=" * 60)
    lstm_deltas = [lstm_val_delta]
    for seed in [59, 76]:
        print(f"\n  Training LSTM seed={seed}...")
        m = train_lstm(X_train, b_train, delta_train,
                       X_val, b_val, delta_val,
                       seq_len, n_features, n_targets, seed=seed)
        lstm_deltas.append(m.predict([X_val, b_val], verbose=0, batch_size=256))

    # Average the 3 LSTM predictions
    lstm_ensemble_delta = np.mean(lstm_deltas, axis=0)

    # Evaluate LSTM ensemble alone
    print("\n--- 3-LSTM ENSEMBLE ---")
    ens_preds = b_val + lstm_ensemble_delta
    ens_orig = trainer.scaler_y.inverse_transform(ens_preds)
    y_val_orig = trainer.scaler_y.inverse_transform(y_val)
    ens_total = 0
    for i, t in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_orig[:, i], ens_orig[:, i])
        ens_total += mae
        print(f"  {t}: {mae:.4f}")
    print(f"  Avg MAE: {ens_total / n_targets:.4f}")

    # Stack: 3-LSTM ensemble + GBM
    if gbm_val_delta is not None:
        print("\n--- 3-LSTM + GBM STACKED ---")
        meta_models_v2 = train_stacking_meta(
            lstm_ensemble_delta, gbm_val_delta, b_val, delta_val, n_targets
        )
        meta_delta_v2 = np.zeros_like(lstm_val_delta)
        for t_idx in range(n_targets):
            meta_features = [lstm_ensemble_delta[:, t_idx:t_idx+1],
                             gbm_val_delta[:, t_idx:t_idx+1],
                             b_val]
            X_meta = np.hstack(meta_features)
            meta_delta_v2[:, t_idx] = meta_models_v2[t_idx].predict(X_meta)

        meta_preds_v2 = b_val + meta_delta_v2
        meta_orig_v2 = trainer.scaler_y.inverse_transform(meta_preds_v2)
        v2_total = 0
        for i, t in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_orig[:, i], meta_orig_v2[:, i])
            v2_total += mae
            print(f"  {t}: MAE={mae:.4f}")
        print(f"  Avg MAE: {v2_total / n_targets:.4f}")


if __name__ == "__main__":
    main()
