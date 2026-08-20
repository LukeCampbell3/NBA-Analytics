from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import load_data_proc_history, replay_frozen_selector
from .binary_path_audit import run_binary_path_sensitivity_audit
from .confirmation import chronological_confirmation
from .dataset import build_frozen_dataset, write_frozen_dataset
from .freeze import build_freeze_manifest
from .outcome_set_backtest import (
    chronological_outcome_set_replay,
    combine_outcome_set_replays,
)
from .research_replay import replay_master_research_ledger
from .survival_backtest import (
    build_transfer_reservoir,
    chronological_survival_replay,
    combine_survival_replays,
)
from .synthetic_audit import run_null_power_audit


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPOSITORY_QUOTES = (
    REPO_ROOT
    / "Player-Predictor"
    / "data"
    / "market_odds"
    / "nba"
    / "v9_6_sequence"
    / "market_snapshot_sequence.csv"
)


def _portable_source_path(path: Path) -> str:
    resolved = path.resolve()
    for base in (REPO_ROOT, REPO_ROOT.parent):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.name


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NBA conditional-chain V1.1 research pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    repository = subparsers.add_parser("repository-audit")
    repository.add_argument("--quotes", type=Path, default=DEFAULT_REPOSITORY_QUOTES)
    repository.add_argument("--output", type=Path, required=True)

    dataset = subparsers.add_parser("build-dataset")
    dataset.add_argument("--quotes", type=Path, nargs="+", required=True)
    dataset.add_argument("--outcomes", type=Path)
    dataset.add_argument("--output-dir", type=Path, required=True)

    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--settled-features", type=Path, required=True)
    confirm.add_argument("--output-dir", type=Path, required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)

    synthetic = subparsers.add_parser("synthetic-audit")
    synthetic.add_argument("--simulations", type=int, default=20)
    synthetic.add_argument("--events", type=int, default=80)
    synthetic.add_argument("--output", type=Path, required=True)

    replay = subparsers.add_parser("backtest-selector")
    replay.add_argument("--candidate-pool", type=Path, required=True)
    replay.add_argument("--data-proc-dir", type=Path)
    replay.add_argument("--output-dir", type=Path, required=True)

    master_replay = subparsers.add_parser("backtest-master")
    master_replay.add_argument("--master-ledger", type=Path, required=True)
    master_replay.add_argument("--holdout-start", default="2026-02-11")
    master_replay.add_argument("--output-dir", type=Path, required=True)

    survival_replay = subparsers.add_parser("backtest-survival")
    survival_replay.add_argument("--research-reservoir", type=Path, required=True)
    survival_replay.add_argument("--transfer-candidate-pool", type=Path, required=True)
    survival_replay.add_argument("--data-proc-dir", type=Path, required=True)
    survival_replay.add_argument("--warmup-slates", type=int, default=20)
    survival_replay.add_argument("--output-dir", type=Path, required=True)

    outcome_set_replay = subparsers.add_parser("backtest-outcome-set")
    outcome_set_replay.add_argument("--research-reservoir", type=Path, required=True)
    outcome_set_replay.add_argument("--transfer-reservoir", type=Path, required=True)
    outcome_set_replay.add_argument("--output-dir", type=Path, required=True)

    binary_path_audit = subparsers.add_parser("binary-path-audit")
    binary_path_audit.add_argument("--aps-threshold", type=float, default=0.90)
    binary_path_audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "repository-audit":
        result = build_frozen_dataset([args.quotes])
        report = dict(result.manifest)
        report["quality_rows"] = result.path_result.quality_ledger.to_dict(
            orient="records"
        )
        report["freeze"] = build_freeze_manifest()
        _write_json(args.output, report)
        print(
            json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.command == "build-dataset":
        result = build_frozen_dataset(args.quotes, outcome_path=args.outcomes)
        write_frozen_dataset(result, args.output_dir)
        print(
            json.dumps(
                _json_ready(result.manifest), indent=2, sort_keys=True, allow_nan=False
            )
        )
        return 0
    if args.command == "confirm":
        features = pd.read_csv(args.settled_features, low_memory=False)
        result = chronological_confirmation(features)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.player_predictions.to_csv(
            args.output_dir / "path_player_predictions.csv", index=False
        )
        result.event_evaluations.to_csv(
            args.output_dir / "path_event_evaluations.csv", index=False
        )
        _write_json(args.output_dir / "path_confirmation_report.json", result.report)
        print(
            json.dumps(
                _json_ready(result.report), indent=2, sort_keys=True, allow_nan=False
            )
        )
        return 0
    if args.command == "freeze":
        manifest = build_freeze_manifest()
        _write_json(args.output, manifest)
        print(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.command == "synthetic-audit":
        report = run_null_power_audit(simulations=args.simulations, events=args.events)
        _write_json(args.output, report)
        print(
            json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.command == "backtest-selector":
        pool = pd.read_csv(args.candidate_pool, low_memory=False)
        history = (
            load_data_proc_history(args.data_proc_dir) if args.data_proc_dir else None
        )
        decisions, report = replay_frozen_selector(pool, historical_actuals=history)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        decisions.to_csv(args.output_dir / "frozen_selector_decisions.csv", index=False)
        _write_json(args.output_dir / "frozen_selector_backtest.json", report)
        print(
            json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.command == "backtest-master":
        source = pd.read_csv(args.master_ledger, low_memory=False)
        result = replay_master_research_ledger(
            source,
            reported_holdout_start=args.holdout_start,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.reservoir.to_csv(
            args.output_dir / "frozen_reservoir_replay.csv", index=False
        )
        result.slate_decisions.to_csv(
            args.output_dir / "frozen_slate_decisions.csv", index=False
        )
        report = dict(result.report)
        report["source_manifest"] = {
            "path": _portable_source_path(args.master_ledger),
            "sha256": hashlib.sha256(args.master_ledger.read_bytes()).hexdigest(),
            "rows": int(len(source)),
        }
        _write_json(args.output_dir / "frozen_research_replay.json", report)
        print(
            json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.command == "backtest-survival":
        research_reservoir = pd.read_csv(args.research_reservoir, low_memory=False)
        transfer_pool = pd.read_csv(args.transfer_candidate_pool, low_memory=False)
        actual_history = load_data_proc_history(args.data_proc_dir)
        transfer_reservoir = build_transfer_reservoir(transfer_pool, actual_history)
        research_replay = chronological_survival_replay(
            research_reservoir,
            block_label="historical_expanding",
            warmup_slates=args.warmup_slates,
        )
        transfer_replay = chronological_survival_replay(
            transfer_reservoir,
            block_label="cross_version_transfer",
            initial_history=research_reservoir,
            warmup_slates=0,
        )
        combined = combine_survival_replays([research_replay, transfer_replay])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        transfer_reservoir.to_csv(
            args.output_dir / "transfer_reservoir_replay.csv", index=False
        )
        combined.decisions.to_csv(
            args.output_dir / "survival_policy_decisions.csv", index=False
        )
        combined.selected_legs.to_csv(
            args.output_dir / "survival_policy_selected_legs.csv", index=False
        )
        _write_json(args.output_dir / "survival_policy_backtest.json", combined.report)
        print(
            json.dumps(
                _json_ready(combined.report), indent=2, sort_keys=True, allow_nan=False
            )
        )
        return 0
    if args.command == "backtest-outcome-set":
        research_reservoir = pd.read_csv(args.research_reservoir, low_memory=False)
        transfer_reservoir = pd.read_csv(args.transfer_reservoir, low_memory=False)
        research_replay = chronological_outcome_set_replay(
            research_reservoir,
            block_label="historical_expanding",
        )
        transfer_replay = chronological_outcome_set_replay(
            transfer_reservoir,
            block_label="cross_version_transfer",
            initial_history=research_reservoir,
            initial_calibration_scores=research_replay.calibration_scores,
        )
        combined = combine_outcome_set_replays([research_replay, transfer_replay])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        combined.decisions.to_csv(
            args.output_dir / "binary_outcome_set_decisions.csv", index=False
        )
        report = dict(combined.report)
        report["source_manifests"] = {
            "research_reservoir": {
                "path": _portable_source_path(args.research_reservoir),
                "sha256": hashlib.sha256(
                    args.research_reservoir.read_bytes()
                ).hexdigest(),
                "rows": int(len(research_reservoir)),
            },
            "transfer_reservoir": {
                "path": _portable_source_path(args.transfer_reservoir),
                "sha256": hashlib.sha256(
                    args.transfer_reservoir.read_bytes()
                ).hexdigest(),
                "rows": int(len(transfer_reservoir)),
            },
        }
        report["freeze_manifest"] = build_freeze_manifest()
        _write_json(args.output_dir / "binary_outcome_set_backtest.json", report)
        print(
            json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False)
        )
        return 0
    if args.command == "binary-path-audit":
        report = run_binary_path_sensitivity_audit(
            aps_threshold=args.aps_threshold,
        )
        report["freeze_manifest"] = build_freeze_manifest()
        _write_json(args.output, report)
        print(
            json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False)
        )
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
