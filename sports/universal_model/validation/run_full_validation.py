"""Stage 7 + validation battery (spec section 33 stage 7; sections
11/45/46/49): the ONE frozen TEST evaluation for every already-trained/
DRM-developed checkpoint, plus the transfer, small-data, negative-transfer,
and ablation studies. TEST is touched exactly once, in this single script
execution -- every number derived from TEST below comes from this one run,
not from repeated exploratory peeking.

Run: python -m sports.universal_model.validation.run_full_validation
"""
from __future__ import annotations

import json
from pathlib import Path

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.train.checkpoints import load_checkpoint
from sports.universal_model.train.config import TrainConfig
from sports.universal_model.train.trainer import evaluate
from sports.universal_model.validation.ablations import run_input_ablations, run_router_balance_ablation
from sports.universal_model.validation.calibration import reliability_curve
from sports.universal_model.validation.transfer import (
    leave_one_sport_out,
    negative_transfer_audit,
    small_data_regime_test,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
CHECKPOINTS_DIR = MANIFESTS_DIR / "checkpoints"

CHECKPOINT_NAMES = ["dense_baseline", "switch_baseline", "top2_moe", "drm_final"]


def run_test_once() -> dict:
    results = {}
    for name in CHECKPOINT_NAMES:
        path = CHECKPOINTS_DIR / f"{name}.pt"
        if not path.exists():
            continue
        model, payload = load_checkpoint(path)
        test_per_sport = UniversalDataset(split="TEST", split_kind="per_sport")
        test_global = UniversalDataset(split="TEST", split_kind="global")
        results[name] = {
            "per_sport_test": evaluate(model, test_per_sport),
            "global_test": evaluate(model, test_global),
        }
    return results


def main() -> None:
    print("=== TEST (touched once) ===")
    test_results = run_test_once()
    (REPORTS_DIR / "test_results.json").write_text(json.dumps(test_results, indent=2))
    for name, r in test_results.items():
        print(name, "per_sport TEST micro:", r["per_sport_test"]["micro_classification"])

    print("=== leave-one-sport-out (2-sport scope, disclosed) ===")
    base_config = TrainConfig(name="loso_base", hidden_dim=96, n_heads=4, block_types=["dense", "dense", "dense", "dense"], steps=2000, eval_every=1000, batch_size=64)
    loso = leave_one_sport_out(base_config, all_sports=["mlb", "nfl"])
    (REPORTS_DIR / "leave_one_sport_out_results.json").write_text(json.dumps(loso, indent=2))
    print(json.dumps({s: r["final_select_metrics"]["micro_classification"] for s, r in loso.items()}, indent=2))

    print("=== negative transfer audit ===")
    dense_model, _ = load_checkpoint(CHECKPOINTS_DIR / "dense_baseline.pt")
    neg_transfer = negative_transfer_audit(loso, dense_model, all_sports=["mlb", "nfl"])
    (REPORTS_DIR / "negative_transfer_audit.json").write_text(json.dumps(neg_transfer, indent=2))
    print(json.dumps(neg_transfer, indent=2, default=str))

    print("=== small-data regime test (NFL) ===")
    small_data = small_data_regime_test(base_config, sport="nfl", fractions=[1.0, 0.5, 0.25, 0.1])
    (REPORTS_DIR / "small_data_transfer_results.json").write_text(json.dumps(small_data, indent=2))
    print(json.dumps({k: v["final_select_metrics"]["regression"] for k, v in small_data.items()}, indent=2))

    print("=== ablations A/B/C (input masking, TEST) ===")
    drm_model, _ = load_checkpoint(CHECKPOINTS_DIR / "drm_final.pt")
    test_per_sport = UniversalDataset(split="TEST", split_kind="per_sport")
    ablation_abc = run_input_ablations(drm_model, test_per_sport)
    print(json.dumps(ablation_abc, indent=2))

    print("=== ablation F: router balance loss removed (real retrain) ===")
    moe_config = TrainConfig(name="top2_moe_ablation", hidden_dim=96, n_heads=4, block_types=["dense", "dense", "top2_moe", "top2_moe"], n_experts=8, steps=3000, eval_every=1000, batch_size=64)
    ablation_f = run_router_balance_ablation(moe_config)
    print(json.dumps(ablation_f["final_select_metrics"]["micro_classification"], indent=2))

    (REPORTS_DIR / "ablation_report.json").write_text(
        json.dumps({"A_B_C_input_masking": ablation_abc, "F_router_balance_removed": ablation_f}, indent=2)
    )

    print("=== calibration report (TEST, drm_final) ===")
    calib = reliability_curve(drm_model, test_per_sport)
    (REPORTS_DIR / "calibration_report.json").write_text(json.dumps(calib, indent=2))
    print("ece_overall:", calib["ece_overall"], "priced n:", calib["priced_subset"]["n"])


if __name__ == "__main__":
    main()
