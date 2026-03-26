"""
Enhanced LSTM Ensemble Trainer v4
Strategy: Replicate the EXACT simple LSTM that got PTS=9.08, TRB=2.54, AST=2.72
but train 5 copies with different random seeds. Average predictions.

No fancy tricks — just seed diversity + ensemble averaging.
The simple LSTM architecture is proven. Ensemble should reduce variance.

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
    Input, Dense, LSTM, Dropout, Concatenate, LayerNormalization,
    GaussianNoise
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score

from unified_moe_trainer import UnifiedMoETrainer


def build_simple_lstm(seq_len, n_features, n_targets, seed=42, variant=0):
    """Exact replica of UnifiedMoETrainer.build_simple_lstm_model()
    but without the dummy MoE outputs. Variant adds slight diversity."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    seq_input = Input(shape=(seq_len, n_features), name="seq_input")
    base_input = Input(shape=(n_targets,), name="base_input")

    # Slight architectural diversity per variant
    lstm1_units = [128, 128, 96, 128, 128, 96, 128][variant % 7]
    lstm2_units = [64, 64, 64, 48, 64, 48, 64][variant % 7]
    drop_rate = [0.2, 0.25, 0.2, 0.2, 0.15, 0.25, 0.2][variant % 7]

    # LSTM encoder
    x = LSTM(lstm1_units, return_sequences=True, name="lstm1")(seq_input)
    x = Dropout(drop_rate)(x)
    x = LSTM(lstm2_units, return_sequences=False, name="lstm2")(x)
    x = Dropout(drop_rate)(x)

    # Combine with baseline
    merged = Concatenate()([x, base_input])

    # Dense head
    x = Dense(128, activation="relu", name="dense1")(merged)
    x = LayerNormalization(name="ln1")(x)
    x = Dropout(drop_rate)(x)
    x = Dense(64, activation="relu", name="dense2")(x)
    x = Dropout(0.1)(x)
    delta_pred = Dense(n_targets, activation=None, name="delta_pred")(x)

    model = Model(inputs=[seq_input, base_input], outputs=delta_pred,
                  name=f"lstm_seed{seed}_v{variant}")
    return model


def train_ensemble(X_train, b_train, delta_train, X_val, b_val, delta_val,
                   seq_len, n_features, n_targets, n_models=5):
    """Train N identical LSTM models with different seeds. Plain MSE loss."""
    models = []
    val_losses = []

    for i in range(n_models):
        seed = 42 + i * 17
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE MEMBER {i+1}/{n_models}  (seed={seed})")
        print(f"{'='*60}")

        model = build_simple_lstm(seq_len, n_features, n_targets, seed=seed, variant=i)
        if i == 0:
            model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss='mse',
            metrics=['mae']
        )

        history = model.fit(
            [X_train, b_train], delta_train,
            validation_data=([X_val, b_val], delta_val),
            epochs=120,
            batch_size=128,
            shuffle=True,  # different shuffle order per epoch + seed
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=20,
                              restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=8,
                    min_lr=1e-5, verbose=1
                ),
            ],
            verbose=1
        )

        best_val = min(history.history['val_loss'])
        best_epoch = history.history['val_loss'].index(best_val) + 1
        print(f"  Member {i+1} best val_loss: {best_val:.5f} at epoch {best_epoch}")

        models.append(model)
        val_losses.append(best_val)

    return models, val_losses


def weighted_ensemble_predict(models, val_losses, X, baselines):
    """Weighted average: lower val_loss = higher weight."""
    losses = np.array(val_losses)
    weights = 1.0 / (losses + 1e-8)
    weights = weights / weights.sum()
    print(f"  Ensemble weights: {[f'{w:.3f}' for w in weights]}")

    preds = [m.predict([X, baselines], verbose=0, batch_size=256) for m in models]
    result = np.zeros_like(preds[0])
    for p, w in zip(preds, weights):
        result += p * w
    return result


def main():
    print("=" * 80)
    print("SIMPLE LSTM ENSEMBLE v4")
    print("Same proven architecture x5 seeds, plain MSE, ensemble average")
    print("=" * 80)

    trainer = UnifiedMoETrainer()
    X, baselines, y, df = trainer.prepare_data()

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    b_train, b_val = baselines[:split_idx], baselines[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    delta_train = y_train - b_train
    delta_val = y_val - b_val

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    n_targets = y_train.shape[1]
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    N_MODELS = 7
    models, val_losses = train_ensemble(
        X_train, b_train, delta_train,
        X_val, b_val, delta_val,
        seq_len, n_features, n_targets,
        n_models=N_MODELS
    )

    # ---- Results ----
    y_val_orig = trainer.scaler_y.inverse_transform(y_val)
    b_val_orig = trainer.scaler_y.inverse_transform(b_val)

    print("\n" + "=" * 80)
    print("INDIVIDUAL MEMBERS")
    print("=" * 80)
    for i, m in enumerate(models):
        dp = m.predict([X_val, b_val], verbose=0)
        ps = b_val + dp
        po = trainer.scaler_y.inverse_transform(ps)
        maes = [mean_absolute_error(y_val_orig[:, j], po[:, j])
                for j in range(n_targets)]
        print(f"  Member {i+1}: PTS={maes[0]:.3f}  TRB={maes[1]:.3f}  "
              f"AST={maes[2]:.3f}  avg={np.mean(maes):.3f}")

    # Weighted ensemble
    print("\n" + "=" * 80)
    print("WEIGHTED ENSEMBLE")
    print("=" * 80)
    delta_pred = weighted_ensemble_predict(models, val_losses, X_val, b_val)
    preds_scaled = b_val + delta_pred
    pred_orig = trainer.scaler_y.inverse_transform(preds_scaled)

    total_mae = 0
    for i, target in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_orig[:, i], pred_orig[:, i])
        r2 = r2_score(y_val_orig[:, i], pred_orig[:, i])
        baseline_mae = mean_absolute_error(y_val_orig[:, i], b_val_orig[:, i])
        improvement = baseline_mae - mae
        pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
        total_mae += mae
        print(f"\n{target}:")
        print(f"  Ensemble MAE: {mae:.4f}")
        print(f"  Ensemble R²:  {r2:.4f}")
        print(f"  Baseline MAE: {baseline_mae:.4f}")
        print(f"  Improvement:  {improvement:+.4f} ({pct:+.1f}%)")

    avg_mae = total_mae / len(trainer.target_columns)
    print(f"\n{'='*60}")
    print(f"  OVERALL AVG MAE: {avg_mae:.4f}")
    print(f"  Target to beat:  4.795 (old ensemble)")
    print(f"  Simple LSTM:     4.78")
    print(f"{'='*60}")

    # Simple average
    print("\n--- Simple average ensemble ---")
    preds_list = [m.predict([X_val, b_val], verbose=0, batch_size=256) for m in models]
    delta_simple = np.mean(preds_list, axis=0)
    ps_simple = b_val + delta_simple
    po_simple = trainer.scaler_y.inverse_transform(ps_simple)
    for i, target in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_orig[:, i], po_simple[:, i])
        print(f"  {target} MAE: {mae:.4f}")

    # Top-K ensemble (best 3 members by val_loss)
    print("\n--- Top-3 ensemble (best members only) ---")
    top_k = 3
    sorted_idx = np.argsort(val_losses)[:top_k]
    top_preds = [preds_list[i] for i in sorted_idx]
    top_losses = [val_losses[i] for i in sorted_idx]
    top_weights = 1.0 / (np.array(top_losses) + 1e-8)
    top_weights = top_weights / top_weights.sum()
    print(f"  Using members: {[i+1 for i in sorted_idx]}")
    print(f"  Weights: {[f'{w:.3f}' for w in top_weights]}")
    delta_top = np.zeros_like(preds_list[0])
    for p, w in zip(top_preds, top_weights):
        delta_top += p * w
    ps_top = b_val + delta_top
    po_top = trainer.scaler_y.inverse_transform(ps_top)
    top_total = 0
    for i, target in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_orig[:, i], po_top[:, i])
        top_total += mae
        print(f"  {target} MAE: {mae:.4f}")
    print(f"  Top-3 avg MAE: {top_total / len(trainer.target_columns):.4f}")

    # Save
    import joblib, json
    os.makedirs("model", exist_ok=True)
    for i, m in enumerate(models):
        m.save_weights(f"model/ensemble_member_{i}.weights.h5")
    joblib.dump(trainer.scaler_x, 'model/ensemble_scaler_x.pkl')
    joblib.dump(trainer.scaler_y, 'model/ensemble_scaler_y.pkl')

    meta = {
        'model_type': 'simple_lstm_ensemble_v4',
        'n_models': N_MODELS,
        'seq_len': seq_len,
        'n_features': n_features,
        'n_targets': n_targets,
        'target_columns': trainer.target_columns,
        'baseline_features': trainer.baseline_features,
        'feature_columns': trainer.feature_columns,
        'seeds': [42 + i * 17 for i in range(N_MODELS)],
        'val_losses': [float(v) for v in val_losses],
    }
    with open('model/ensemble_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nEnsemble saved to model/")


if __name__ == "__main__":
    main()
