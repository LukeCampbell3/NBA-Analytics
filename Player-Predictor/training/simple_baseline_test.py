"""
Quick sanity check: Simple LSTM + Dense model (no MoE) to see if the
delta-correction approach can beat the old ensemble baseline.
Reuses data loading from UnifiedMoETrainer.
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

from unified_moe_trainer import UnifiedMoETrainer


def build_simple_model(seq_len, n_features, n_targets):
    """Simple LSTM + Dense model for delta prediction with attention"""
    seq_input = Input(shape=(seq_len, n_features), name="seq_input")
    base_input = Input(shape=(n_targets,), name="base_input")
    
    # Bidirectional LSTM encoder
    x = tf.keras.layers.Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        name="bilstm1"
    )(seq_input)
    x = Dropout(0.25)(x)
    x = tf.keras.layers.Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        name="bilstm2"
    )(x)
    x = Dropout(0.2)(x)
    
    # Simple attention pooling
    attn_weights = Dense(1, activation="tanh", name="attn_score")(x)  # [B, T, 1]
    attn_weights = tf.keras.layers.Softmax(axis=1, name="attn_softmax")(attn_weights)
    x = tf.reduce_sum(x * attn_weights, axis=1)  # [B, D]
    
    # Combine with baseline
    combined = Concatenate()([x, base_input])
    
    # Dense head
    x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(combined)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    delta_pred = Dense(n_targets, activation=None, name="delta_pred")(x)
    
    model = Model(inputs=[seq_input, base_input], outputs=delta_pred)
    return model


def main():
    print("=" * 80)
    print("SIMPLE BASELINE TEST (LSTM + Dense, no MoE)")
    print("=" * 80)
    
    # Use UnifiedMoETrainer just for data loading
    trainer = UnifiedMoETrainer()
    X, baselines, y, df = trainer.prepare_data()
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    b_train, b_val = baselines[:split_idx], baselines[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Compute delta targets
    delta_train = y_train - b_train
    delta_val = y_val - b_val
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    print(f"Delta train - mean: {np.mean(delta_train, axis=0)}, std: {np.std(delta_train, axis=0)}")
    
    # Build simple model
    model = build_simple_model(
        seq_len=X_train.shape[1],
        n_features=X_train.shape[2],
        n_targets=y_train.shape[1]
    )
    model.summary()
    
    # Compile — MSE loss on delta
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mse',
        metrics=['mae']
    )
    
    # Train
    history = model.fit(
        [X_train, b_train], delta_train,
        validation_data=([X_val, b_val], delta_val),
        epochs=120,
        batch_size=128,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        ],
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    delta_pred = model.predict([X_val, b_val], verbose=0)
    preds_scaled = b_val + delta_pred
    
    print(f"Delta predicted - mean: {np.mean(delta_pred, axis=0)}, std: {np.std(delta_pred, axis=0)}")
    print(f"Delta true - mean: {np.mean(delta_val, axis=0)}, std: {np.std(delta_val, axis=0)}")
    
    # Inverse transform
    pred_original = trainer.scaler_y.inverse_transform(preds_scaled)
    y_val_original = trainer.scaler_y.inverse_transform(y_val)
    b_val_original = trainer.scaler_y.inverse_transform(b_val)
    
    for i, target in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_original[:, i], pred_original[:, i])
        r2 = r2_score(y_val_original[:, i], pred_original[:, i])
        baseline_mae = mean_absolute_error(y_val_original[:, i], b_val_original[:, i])
        improvement = baseline_mae - mae
        pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
        
        print(f"\n{target}:")
        print(f"  Model MAE:    {mae:.4f}")
        print(f"  Model R²:     {r2:.4f}")
        print(f"  Baseline MAE: {baseline_mae:.4f}")
        print(f"  Improvement:  {improvement:+.4f} ({pct:+.1f}%)")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
