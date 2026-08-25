"""Runs Stage 2-4 (spec section 33): dense baseline, Switch Top-1, Top-2
MoE, at matched compute (same attention depth/hidden/heads/data/steps;
only the FFN type in the last two of four blocks differs). Freezes each
result to reports/ + manifests/ before moving to the next stage.

CPU-only, no GPU in this environment (see reports/INVENTORY.md) --
step/steps counts are deliberately reduced-scale, not full training; wall
time and throughput are measured and reported honestly rather than
fabricated (spec section 59).

Run: python -m sports.universal_model.train.run_baselines
"""
from __future__ import annotations

import json
from pathlib import Path

from sports.universal_model.train.checkpoints import save_checkpoint
from sports.universal_model.train.config import TrainConfig
from sports.universal_model.train.trainer import train_model

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
CHECKPOINTS_DIR = MANIFESTS_DIR / "checkpoints"

SHARED = dict(hidden_dim=96, n_heads=4, ffn_mult=4, dropout=0.1, lr=3e-4, batch_size=64, steps=3000, eval_every=500, alpha=0.5, seed=1234)

CONFIGS = {
    "dense_baseline": TrainConfig(name="dense_baseline", block_types=["dense", "dense", "dense", "dense"], **SHARED),
    "switch_baseline": TrainConfig(name="switch_baseline", block_types=["dense", "dense", "switch", "switch"], n_experts=8, **SHARED),
    "top2_moe": TrainConfig(name="top2_moe", block_types=["dense", "dense", "top2_moe", "top2_moe"], n_experts=8, **SHARED),
}


def _strip_model(history: list[dict]) -> list[dict]:
    return history  # already JSON-serializable (no tensors retained)


def run_all() -> dict:
    summary = {}
    for name, config in CONFIGS.items():
        print(f"=== training {name} ===")
        result = train_model(config)
        (MANIFESTS_DIR / f"{name}_config.json").write_text(json.dumps(config.to_dict(), indent=2))
        report = {
            "name": name,
            "config": config.to_dict(),
            "history": _strip_model(result["history"]),
            "final_select_metrics": result["final_select_metrics"],
            "total_params": result["total_params"],
            "active_params": result["active_params"],
            "wall_time_sec": result["wall_time_sec"],
            "examples_per_sec": result["examples_per_sec"],
            "sampler_effective_contribution": result["sampler_effective_contribution"],
        }
        (REPORTS_DIR / f"{name}_results.json").write_text(json.dumps(report, indent=2))
        save_checkpoint(CHECKPOINTS_DIR / f"{name}.pt", result["model"], result["optimizer"], config)
        summary[name] = {
            "final_micro_brier": result["final_select_metrics"]["micro_classification"]["brier"],
            "final_micro_log_loss": result["final_select_metrics"]["micro_classification"]["log_loss"],
            "final_micro_auc": result["final_select_metrics"]["micro_classification"]["auc"],
            "worst_sport_brier": result["final_select_metrics"]["worst_sport_brier"],
            "total_params": result["total_params"],
            "active_params": result["active_params"],
            "wall_time_sec": result["wall_time_sec"],
            "examples_per_sec": result["examples_per_sec"],
        }
        print(json.dumps(summary[name], indent=2))
    (REPORTS_DIR / "baseline_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_all()
