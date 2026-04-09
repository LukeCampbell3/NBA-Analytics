#!/usr/bin/env python3
"""
Unified training entry point for the cleaned repository.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "training"))

from hybrid_spike_moe_trainer import HybridSpikeMoETrainer
from improved_baseline_trainer import ImprovedBaselineTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified NBA Player Prediction Trainer")
    parser.add_argument(
        "--mode",
        type=str,
        default="improved_lstm",
        choices=["improved_lstm", "baseline", "moe", "ensemble"],
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length")
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Train and save artifacts without post-train validation reporting",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"UNIFIED TRAINER - Mode: {args.mode.upper()}")
    print("=" * 80)

    if args.mode == "improved_lstm":
        print("\nTraining Structured LSTM Stack...")
        print("   - Structured latent LSTM with slow-state / bottleneck / uncertainty heads")
        print("   - CatBoost delta model on shared validation split")
        print("   - Ridge meta-blend chooses the best validation path")
        print("   - Baseline+delta architecture with benchmark comparison\n")

        from improved_lstm_v7 import main as train_v7

        train_v7(
            epochs_override=args.epochs,
            batch_size_override=args.batch_size,
            save_only=args.save_only,
        )

    elif args.mode == "baseline":
        print("\nTraining Improved Baseline Model...")
        trainer = ImprovedBaselineTrainer()
        if args.epochs is not None:
            trainer.config["epochs"] = args.epochs
        if args.batch_size is not None:
            trainer.config["batch_size"] = args.batch_size
        trainer.config["seq_len"] = args.seq_len
        trainer.train()

    elif args.mode == "moe":
        print("\nTraining Hybrid Spike-Driver MoE Model...")
        trainer = HybridSpikeMoETrainer(ensemble_size=1)
        if args.epochs is not None:
            trainer.config["epochs"] = args.epochs
        if args.batch_size is not None:
            trainer.config["batch_size"] = args.batch_size
        trainer.config["seq_len"] = args.seq_len
        trainer.train()

    elif args.mode == "ensemble":
        print("\nTraining Ensemble of 3 MoE Models...")
        trainer = HybridSpikeMoETrainer(ensemble_size=3)
        if args.epochs is not None:
            trainer.config["epochs"] = args.epochs
        if args.batch_size is not None:
            trainer.config["batch_size"] = args.batch_size
        trainer.config["seq_len"] = args.seq_len
        trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("\nRecommended mode: --mode improved_lstm")


if __name__ == "__main__":
    main()
