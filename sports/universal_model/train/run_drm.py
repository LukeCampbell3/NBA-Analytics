"""Stage 5 (spec section 33): bounded DRM structural development, starting
from the trained Top-2 MoE checkpoint, using DERIVE/SELECT only.

Run: python -m sports.universal_model.train.run_drm
"""
from __future__ import annotations

import json
from pathlib import Path

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.drm_controller.controller import run_drm_development
from sports.universal_model.train.checkpoints import load_checkpoint, save_checkpoint
from sports.universal_model.train.config import TrainConfig

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    model, payload = load_checkpoint(MANIFESTS_DIR / "checkpoints" / "top2_moe.pt")
    config = TrainConfig(**payload["train_config"])

    derive = UniversalDataset(split="DERIVE")
    select = UniversalDataset(split="SELECT")

    pre_drm_config = dict(model.config)

    budget = run_drm_development(
        model,
        config,
        derive,
        select,
        n_cycles=3,
        max_mutation_attempts_per_cycle=3,
        finetune_steps=300,
    )

    report = budget.to_report()
    report["pre_drm_model_config"] = pre_drm_config
    report["post_drm_model_config"] = model.config
    report["post_drm_total_params"] = model.total_parameters()
    report["post_drm_active_params"] = model.active_parameters_per_token()
    (REPORTS_DIR / "drm_mutation_history.json").write_text(json.dumps(report, indent=2))

    (MANIFESTS_DIR / "drm_final_config.json").write_text(
        json.dumps(
            {
                "model_config": model.config,
                "committed_mutations": [r.to_dict() for r in budget.history if r.status == "PERMANENT"],
                "total_committed": report["committed_count"],
                "total_rejected": report["rejected_count"],
            },
            indent=2,
        )
    )
    save_checkpoint(MANIFESTS_DIR / "checkpoints" / "drm_final.pt", model, None, config, extra={"drm_report": report})
    print(json.dumps({k: v for k, v in report.items() if k != "mutations"}, indent=2))
    print(f"committed={report['committed_count']} rejected={report['rejected_count']}")


if __name__ == "__main__":
    main()
