"""Revision 2 baselines (post-FINAL_REPORT optimization pass): same three
architectures as run_baselines.py, retrained with the upgraded 2-layer MLP
heads (model/heads.py) and a larger shared config, specifically to try to
lower loss/MAE relative to the original run. Writes to *_v2-suffixed
files so the original (already-reported) run_baselines.py artifacts are
never overwritten -- both are kept, and FINAL_REPORT.md is updated with
an explicit before/after revision section rather than silently replacing
the original numbers.

Run: python -m sports.universal_model.train.run_baselines_v2
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

# hidden_dim 96->128 and steps 3000->3500: real capacity + training-budget
# increase, on top of the upgraded heads -- both intended to reduce
# loss/MAE, not just architecture-search busywork.
SHARED = dict(hidden_dim=128, n_heads=4, ffn_mult=4, dropout=0.1, lr=3e-4, batch_size=64, steps=3500, eval_every=700, alpha=0.5, seed=1234)

CONFIGS = {
    "dense_baseline_v2": TrainConfig(name="dense_baseline_v2", block_types=["dense", "dense", "dense", "dense"], **SHARED),
    "switch_baseline_v2": TrainConfig(name="switch_baseline_v2", block_types=["dense", "dense", "switch", "switch"], n_experts=8, **SHARED),
    "top2_moe_v2": TrainConfig(name="top2_moe_v2", block_types=["dense", "dense", "top2_moe", "top2_moe"], n_experts=8, **SHARED),
}


def run_all() -> dict:
    summary = {}
    for name, config in CONFIGS.items():
        print(f"=== training {name} ===")
        result = train_model(config)
        (MANIFESTS_DIR / f"{name}_config.json").write_text(json.dumps(config.to_dict(), indent=2))
        report = {
            "name": name,
            "config": config.to_dict(),
            "history": result["history"],
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
            "regression_mae": result["final_select_metrics"]["regression"]["mae"],
            "total_params": result["total_params"],
            "active_params": result["active_params"],
            "wall_time_sec": result["wall_time_sec"],
            "examples_per_sec": result["examples_per_sec"],
        }
        print(json.dumps(summary[name], indent=2))
    (REPORTS_DIR / "baseline_comparison_summary_v2.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_all()
