from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .path_conditioned_backtest import chronological_path_conditioned_replay


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronological NBA path-conditioned joint-outcome replay"
    )
    parser.add_argument("--reservoir", type=Path, required=True)
    parser.add_argument("--path-features", type=Path, required=True)
    parser.add_argument("--path-certificate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-label", default="real_path_chronological")
    parser.add_argument("--initial-history", type=Path)
    parser.add_argument("--initial-path-history", type=Path)
    parser.add_argument("--initial-calibration", type=Path)
    parser.add_argument("--risk-target", type=float)
    return parser


def main() -> int:
    args = _parser().parse_args()
    reservoir = pd.read_csv(args.reservoir, low_memory=False)
    paths = pd.read_csv(args.path_features, low_memory=False)
    path_certificate = json.loads(args.path_certificate.read_text(encoding="utf-8"))
    initial_history = (
        pd.read_csv(args.initial_history, low_memory=False) if args.initial_history else None
    )
    initial_paths = (
        pd.read_csv(args.initial_path_history, low_memory=False)
        if args.initial_path_history
        else None
    )
    initial_calibration = (
        json.loads(args.initial_calibration.read_text(encoding="utf-8"))
        if args.initial_calibration
        else None
    )
    result = chronological_path_conditioned_replay(
        reservoir,
        paths,
        path_certificate=path_certificate,
        block_label=args.block_label,
        initial_history=initial_history,
        initial_path_history=initial_paths,
        initial_calibration_scores=initial_calibration,
        risk_target=args.risk_target,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(args.output_dir / "path_conditioned_world_decisions.csv", index=False)
    result.proof_trajectories.to_csv(
        args.output_dir / "fixed_parlay_proof_trajectories.csv", index=False
    )
    result.checkpoint_evidence.to_csv(
        args.output_dir / "path_checkpoint_evidence.csv", index=False
    )
    result.candidate_evidence.to_csv(
        args.output_dir / "path_candidate_evidence.csv", index=False
    )
    _write_json(args.output_dir / "path_conditioned_outcome_set_report.json", result.report)
    _write_json(args.output_dir / "selective_risk_report.json", result.selective_risk_report)
    _write_json(args.output_dir / "path_ablation_report.json", result.ablation_report)
    _write_json(
        args.output_dir / "path_calibration_scores.json",
        {key: list(values) for key, values in result.calibration_scores.items()},
    )
    print(json.dumps(_json_ready(result.report), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
