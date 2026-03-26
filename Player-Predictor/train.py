#!/usr/bin/env python3
"""
UNIFIED TRAINER - Simplified Entry Point

This file imports and orchestrates the existing trainer components.
Run with: python train.py [--mode moe|baseline|ensemble|accurate|simple]
"""

import sys
import argparse
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent / "training"))

from hybrid_spike_moe_trainer import HybridSpikeMoETrainer
from improved_baseline_trainer import ImprovedBaselineTrainer


def main():
    parser = argparse.ArgumentParser(description="Unified NBA Player Prediction Trainer")
    parser.add_argument("--mode", type=str, default="simple", 
                       choices=["moe", "baseline", "ensemble", "accurate", "simple", "improved_lstm"],
                       help="Training mode")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length")
    parser.add_argument("--save-only", action="store_true", help="Train and save artifacts without post-train validation reporting")
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"UNIFIED TRAINER - Mode: {args.mode.upper()}")
    print("="*80)
    
    if args.mode == "improved_lstm":
        print("\n🎯 Training Structured LSTM Stack...")
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
        
    elif args.mode == "simple":
        print("\n🎯 Training SIMPLE ACCURATE Model (RECOMMENDED)...")
        print("   - Simple LSTM architecture")
        print("   - R² optimization")
        print("   - No MoE complexity")
        print("   - Expected: R² > 0.3, MAE < 4.0\n")
        
        from simple_accurate_trainer import SimpleAccurateTrainer
        trainer = SimpleAccurateTrainer()
        if args.epochs:
            trainer.config["epochs"] = args.epochs
        if args.batch_size:
            trainer.config["batch_size"] = args.batch_size
        trainer.train()
        
    elif args.mode == "accurate":
        print("\n🎯 Training ACCURACY-IMPROVED Model...")
        print("   - Simplified MoE (6 experts)")
        print("   - R² loss + entropy fixes")
        print("   - Expected: R² > 0.2, MAE < 4.2\n")
        
        from accuracy_fixes import AccuracyImprovedTrainer
        trainer = AccuracyImprovedTrainer()
        if args.epochs:
            trainer.config["epochs"] = args.epochs
        if args.batch_size:
            trainer.config["batch_size"] = args.batch_size
        trainer.train()
        
    elif args.mode == "moe":
        print("\n🔧 Training Hybrid Spike-Driver MoE Model...")
        print("   - 11 experts (8 regular + 3 spike)")
        print("   - Emergency fixes applied")
        print("   - Expected: Balanced experts, R² > 0.15\n")
        
        trainer = HybridSpikeMoETrainer(ensemble_size=1)
        if args.epochs:
            trainer.config["epochs"] = args.epochs
        if args.batch_size:
            trainer.config["batch_size"] = args.batch_size
        trainer.config["seq_len"] = args.seq_len
        trainer.train()
        
    elif args.mode == "baseline":
        print("\n🔧 Training Improved Baseline Model...")
        print("   - No MoE, single model")
        print("   - Baseline + delta architecture\n")
        
        trainer = ImprovedBaselineTrainer()
        if args.epochs:
            trainer.config["epochs"] = args.epochs
        if args.batch_size:
            trainer.config["batch_size"] = args.batch_size
        trainer.config["seq_len"] = args.seq_len
        trainer.train()
        
    elif args.mode == "ensemble":
        print("\n🔧 Training Ensemble of 3 MoE Models...")
        print("   - 3 models with different strategies")
        print("   - Ensemble aggregation\n")
        
        trainer = HybridSpikeMoETrainer(ensemble_size=3)
        if args.epochs:
            trainer.config["epochs"] = args.epochs
        if args.batch_size:
            trainer.config["batch_size"] = args.batch_size
        trainer.config["seq_len"] = args.seq_len
        trainer.train()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    if args.mode == "improved_lstm" and args.save_only:
        print("  1. Run validation: python scripts\\validate_structured_stack.py")
        print("  2. Optional inference check: python inference.py")
    else:
        print("  1. Run evaluation: python inference.py")
        print("  2. Check results: inference/evaluation_results.json")
    print("\nRecommended mode for accuracy: --mode improved_lstm")


if __name__ == "__main__":
    main()
