from __future__ import annotations

"""Runs the 2x2 real-data ablation (DEVELOPMENT_STAMPS only) and the
price-independent joint-probability calibration check, saves results to
reports/. See REPORT.md for the write-up and manifest.py for the frozen
conclusion."""

import json
from pathlib import Path

from sports.mlb.research.h_over_ranker.data_windows import DEVELOPMENT_STAMPS

from .ablation import VARIANTS, run_variant
from .calibration_check import calibration_by_decile, joint_probability_calibration
from .observation_universe import build_observation_universe

OUTPUT_DIR = Path(__file__).resolve().parent / "reports"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 2x2 real-data ablation (DEVELOPMENT_STAMPS, TEST_STAMPS never read) ===")
    variant_summary = []
    for name, cfg in VARIANTS.items():
        result = run_variant(name, cfg["state"], cfg["pair_filter"])
        n_pairs = len(result.all_pairs)
        n_priced = int(result.all_pairs["d_s"].notna().sum()) if n_pairs else 0
        class_counts = result.all_pairs["pair_class"].value_counts().to_dict() if n_pairs else {}
        print(f"{name}: state={cfg['state']} filter={cfg['pair_filter']} pairs={n_pairs} priced={n_priced} classes={class_counts}")
        print(f"  risk_certificate: {result.risk_certificate.status}")
        variant_summary.append(
            {
                "variant": name,
                "state": cfg["state"],
                "pair_filter": cfg["pair_filter"],
                "n_pairs": n_pairs,
                "n_priced_pairs": n_priced,
                "pair_class_counts": class_counts,
                "risk_certificate_status": result.risk_certificate.status,
                "n_action_days": int((result.action_decisions["action"] == "ACT").sum()) if not result.action_decisions.empty else 0,
            }
        )
        if not result.all_pairs.empty:
            result.all_pairs.to_csv(OUTPUT_DIR / f"{name}_pairs.csv", index=False)
        if not result.action_decisions.empty:
            result.action_decisions.to_csv(OUTPUT_DIR / f"{name}_action_decisions.csv", index=False)

    with open(OUTPUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(variant_summary, f, indent=2)

    print("\n=== price-independent joint-probability calibration (narrow vs broad) ===")
    calibration_summary = {}
    for state in ("narrow", "broad"):
        universe = build_observation_universe(DEVELOPMENT_STAMPS, mode=state)
        calib = joint_probability_calibration(universe)
        by_decile = calibration_by_decile(calib)
        by_decile.to_csv(OUTPUT_DIR / f"calibration_{state}_by_decile.csv", index=False)
        gap = float(calib["p_joint"].mean() - calib["both_win"].mean()) if len(calib) else float("nan")
        print(f"{state}: n_pairs={len(calib)} n_days={calib['date'].nunique() if len(calib) else 0} "
              f"mean_pred={calib['p_joint'].mean():.4f} actual={calib['both_win'].mean():.4f} gap={gap:+.4f}")
        calibration_summary[state] = {
            "n_pairs": int(len(calib)),
            "n_days": int(calib["date"].nunique()) if len(calib) else 0,
            "mean_predicted_p_joint": float(calib["p_joint"].mean()) if len(calib) else None,
            "actual_both_win_rate": float(calib["both_win"].mean()) if len(calib) else None,
            "calibration_gap": gap,
        }
        for same_game, part in calib.groupby("same_game"):
            key = f"{state}_same_game_{same_game}"
            calibration_summary[key] = {
                "n_pairs": int(len(part)),
                "mean_predicted_p_joint": float(part["p_joint"].mean()),
                "actual_both_win_rate": float(part["both_win"].mean()),
                "calibration_gap": float(part["p_joint"].mean() - part["both_win"].mean()),
            }

    with open(OUTPUT_DIR / "calibration_summary.json", "w") as f:
        json.dump(calibration_summary, f, indent=2)


if __name__ == "__main__":
    main()
