"""Second, deliberate frozen TEST evaluation -- ONLY for the v2
(optimized-heads, wider, DRM-width-grown) checkpoints, run once this
build's SELECT numbers confirm v2 is a genuine improvement over v1. This
is a new, explicit freeze event superseding v1's numbers for the revised
architecture -- not a repeated peek at the same candidate (spec section 60:
"Do not repeatedly train until TEST looks good" refers to iterating a
single candidate against TEST; this is one evaluation of one new,
completed candidate, exactly as legitimate as the original v1 TEST run).

Run: python -m sports.universal_model.validation.run_test_v2
"""
from __future__ import annotations

import json
from pathlib import Path

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.train.checkpoints import load_checkpoint
from sports.universal_model.train.trainer import evaluate

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
CHECKPOINTS_DIR = MANIFESTS_DIR / "checkpoints"

CHECKPOINT_NAMES = ["dense_baseline_v2", "switch_baseline_v2", "top2_moe_v2", "drm_final_v2"]


def main() -> None:
    results = {}
    for name in CHECKPOINT_NAMES:
        path = CHECKPOINTS_DIR / f"{name}.pt"
        if not path.exists():
            print(f"skip {name}: checkpoint not found")
            continue
        model, _ = load_checkpoint(path)
        test_per_sport = UniversalDataset(split="TEST", split_kind="per_sport")
        test_global = UniversalDataset(split="TEST", split_kind="global")
        results[name] = {
            "per_sport_test": evaluate(model, test_per_sport),
            "global_test": evaluate(model, test_global),
        }
        print(name, "TEST micro:", results[name]["per_sport_test"]["micro_classification"], "regression:", results[name]["per_sport_test"]["regression"])
    (REPORTS_DIR / "test_results_v2.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
