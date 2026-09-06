#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.advanced.game_conditioned_moe import EXPERT_NAMES, MODEL_VERSION, SCHEMA_VERSION, TARGETS, build_expert_state
from sports.mlb.advanced.schema import AdvancedCandidateContext
from sports.mlb.advanced.sequential_pa_model import simulate_hitter_market
from sports.mlb.scripts import fit_game_conditioned_hitter_moe as base
from sports.mlb.scripts import fit_game_conditioned_hitter_moe_nonregression as nonreg
from sports.mlb.scripts.collect_historical_game_conditioned_outcomes import (
    DEFAULT_LEDGER,
    EVIDENCE_CLASS as OUTCOME_EVIDENCE_CLASS,
    SCHEMA_VERSION as OUTCOME_SCHEMA_VERSION,
    load_outcome_ledger,
    lookup_outcome,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "mlb_game_conditioned_outcome_joined_validation.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "mlb_game_conditioned_outcome_joined_validation.md"
DEFAULT_MODEL = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "game_conditioned_hitter_moe_outcome_joined_shadow_v3.json"

TARGET_SPECS = {
    "H": (0.5, "H", "H_OVER_0_5"),
    "TB": (1.5, "TB", "TB_OVER_1_5"),
    "HR": (0.5, "HR", "HR_OVER_0_5"),
}

CURRENT_GAME_REALIZED_OR_DERIVED_COLUMNS = (
    "H", "TB", "HR", "R", "RBI", "PA", "AB",
    "H_market_gap", "TB_market_gap", "HR_market_gap", "R_market_gap", "RBI_market_gap",
    "H_rolling_avg", "TB_rolling_avg", "HR_rolling_avg", "R_rolling_avg", "RBI_rolling_avg",
)


def _mask_current_game(row: pd.Series) -> pd.Series:
    masked = row.copy()
    for column in CURRENT_GAME_REALIZED_OR_DERIVED_COLUMNS:
        if column in masked.index:
            masked[column] = np.nan
    return masked


def _candidate_rank(date: str, game_id: str, player_id: int) -> str:
    return hashlib.sha256(f"{date}|{game_id}|{player_id}".encode("utf-8")).hexdigest()


def _discover_candidates(
    data_root: Path,
    *,
    season: int,
    min_history: int,
    ledger: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    stats = {"files_seen": 0, "hitter_files": 0, "eligible_rows": 0, "ledger_matches": 0, "missing_outcome": 0}
    for path in sorted(data_root.glob(f"*/{season}_processed_processed.csv")):
        stats["files_seen"] += 1
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {"Date", "Player", "Player_MLBAM_ID", "Player_Type", "Game_ID"})
        except Exception:
            continue
        if frame.empty or str(frame.iloc[0].get("Player_Type") or "").strip().lower() != "hitter":
            continue
        stats["hitter_files"] += 1
        frame["_date"] = pd.to_datetime(frame.get("Date"), errors="coerce")
        frame = frame.loc[frame["_date"].notna()].sort_values("_date").reset_index(drop=True)
        if len(frame) <= min_history:
            continue
        fallback_player_id = stats["hitter_files"]
        player_id = int(base.finite(frame.iloc[-1].get("Player_MLBAM_ID"), fallback_player_id) or fallback_player_id)
        player_name = str(frame.iloc[0].get("Player") or path.parent.name).replace("_", " ")
        for idx in range(min_history, len(frame)):
            stats["eligible_rows"] += 1
            row = frame.iloc[idx]
            game_id = str(row.get("Game_ID") or "").strip()
            date = row["_date"].date().isoformat()
            if not game_id or lookup_outcome(ledger, season=season, game_id=game_id, player_id=player_id) is None:
                stats["missing_outcome"] += 1
                continue
            stats["ledger_matches"] += 1
            candidates.append({
                "path": str(path),
                "idx": idx,
                "date": date,
                "game_id": game_id,
                "player_id": player_id,
                "player": player_name,
                "rank": _candidate_rank(date, game_id, player_id),
            })
    return candidates, stats


def _collect_joined_examples(
    data_root: Path,
    *,
    outcome_ledger_path: Path,
    season: int,
    max_games: int,
    trials: int,
    min_history: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger = load_outcome_ledger(outcome_ledger_path)
    candidates, discovery = _discover_candidates(data_root, season=season, min_history=min_history, ledger=ledger)
    candidates = sorted(candidates, key=lambda item: item["rank"])
    if max_games > 0:
        candidates = candidates[:max_games]
    by_path: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_path.setdefault(candidate["path"], []).append(candidate)

    examples: list[dict[str, Any]] = []
    joined_games = 0
    identity_mismatches = 0
    for raw_path, selected in sorted(by_path.items()):
        path = Path(raw_path)
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        frame["_date"] = pd.to_datetime(frame.get("Date"), errors="coerce")
        frame = frame.loc[frame["_date"].notna()].sort_values("_date").reset_index(drop=True)
        for candidate in sorted(selected, key=lambda item: item["idx"]):
            idx = int(candidate["idx"])
            if idx >= len(frame):
                continue
            raw_row = frame.iloc[idx]
            label = lookup_outcome(
                ledger,
                season=season,
                game_id=candidate["game_id"],
                player_id=int(candidate["player_id"]),
            )
            if label is None or str(label.get("date")) != candidate["date"]:
                identity_mismatches += 1
                continue

            # This is the critical boundary: history may contain prior outcomes,
            # but the current game row is stripped of realized/target-derived data.
            row = _mask_current_game(raw_row)
            history = frame.iloc[:idx]
            as_of_date = candidate["date"]
            player_id = int(candidate["player_id"])
            player_name = str(candidate["player"])
            batter = base.prior_batter_profile(history, player_id=player_id, player_name=player_name, as_of_date=as_of_date)
            pitcher = base.prior_pitcher_proxy(row, as_of_date=as_of_date)
            batting_order = int(base.finite(row.get("Batting_Order"), 6) or 6)
            team_runs = base.finite(row.get("Team_Expected_Runs"), base.finite(row.get("Expected_Team_Runs")))
            context = AdvancedCandidateContext(
                game_id=str(candidate["game_id"]),
                run_date=as_of_date,
                batter=batter,
                pitcher=pitcher,
                direct_matchup=None,
                batting_order=batting_order,
                is_home=str(row.get("Is_Home") or "0").strip().lower() in {"1", "true", "yes"},
                team_expected_runs=team_runs,
                park_factor=float(base.finite(row.get("Park_Factor"), 1.0) or 1.0),
                defense_residual=0.0,
                defense_status="HISTORICAL_AVERAGE_CONTEXT_ONLY",
                data_freshness_status="FRESH",
                missing_components=(
                    "HISTORICAL_FULL_LIVE_PITCH_CONTEXT_NOT_PRESERVED",
                    "HISTORICAL_DIRECT_MATCHUP_STATE_NOT_EXACTLY_REPLAYABLE",
                    "HISTORICAL_DEFENSE_STATE_NOT_EXACTLY_REPLAYABLE",
                ),
                temperature_f=base.finite(row.get("Temperature_F"), base.finite(row.get("Temp_F"), base.finite(row.get("Temperature")))),
            )

            added = False
            for target in TARGETS:
                line, realized_name, outcome_name = TARGET_SPECS[target]
                outcome = label["outcomes"].get(outcome_name)
                actual = label["realized"].get(realized_name)
                if outcome not in {0, 1} or actual is None:
                    continue
                sequential = simulate_hitter_market(context, target=target, market_line=line, trials=trials)
                # Do not use current-row market_gap/rolling fields. The historical
                # diagnostic prior is the strict-history structural model only.
                prior = max(1e-5, min(1.0 - 1e-5, float(sequential.raw_structural_probability)))
                state = build_expert_state(context, sequential, target=target, pitch_compatibility_score=0.0)
                examples.append({
                    "date": as_of_date,
                    "game_id": str(candidate["game_id"]),
                    "player_id": player_id,
                    "player": player_name,
                    "target": target,
                    "actual": float(actual),
                    "outcome": int(outcome),
                    "outcome_sha256": str(label["outcome_sha256"]),
                    "outcome_evidence_class": str(label["evidence_class"]),
                    "prior_probability": prior,
                    "prior_source": "STRICT_HISTORY_SEQUENTIAL_STRUCTURAL",
                    "sequential_probability": float(sequential.raw_structural_probability),
                    "sequential_uncertainty": float(sequential.uncertainty),
                    "evidence_strength": float(state.evidence_strength),
                    "features": {name: float(state.effective_features[name]) for name in EXPERT_NAMES},
                })
                added = True
            if added:
                joined_games += 1

    join = {
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "outcome_evidence_class": OUTCOME_EVIDENCE_CLASS,
        "outcome_ledger_rows": len(ledger),
        "discovery": discovery,
        "sampled_hitter_games": len(candidates),
        "joined_hitter_games": joined_games,
        "joined_target_examples": len(examples),
        "identity_mismatches": identity_mismatches,
        "join_key": ["season", "game_id", "player_id"],
        "outcomes_read_from_feature_row": False,
        "current_game_realized_columns_masked": True,
        "current_game_target_derived_columns_masked": True,
        "prior_source": "STRICT_HISTORY_SEQUENTIAL_STRUCTURAL",
        "current_row_market_gap_or_rolling_used_in_prior": False,
    }
    return sorted(examples, key=lambda row: (row["date"], row["player"], row["target"])), join


def _markdown(payload: dict[str, Any]) -> str:
    join = payload["outcome_join"]
    lines = [
        "# MLB Game-Conditioned MoE — Outcome-Joined Historical Validation",
        "",
        f"Model: `{payload['model_version']}`",
        "",
        f"Sampled hitter-games: **{join['sampled_hitter_games']:,}**; joined: **{join['joined_hitter_games']:,}**; target examples: **{join['joined_target_examples']:,}**.",
        "",
        "The current game outcome is resolved only through the separate hash-verified outcome ledger. Current-game H/TB/HR/PA/AB and target-derived rolling/gap fields are masked before feature construction. Historical prior outcomes remain available only through strictly earlier games.",
        "",
        "| Target | Fit rows | OOF rows | Folds pass | Prior Brier | Candidate | Brier gain | Prior LL | Candidate | LL gain | Diagnostic NR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for target in TARGETS:
        val = payload["targets"][target]["validation"]
        f = lambda x: "n/a" if x is None else f"{float(x):.5f}"
        lines.append(
            f"| {target} | {val.get('fit_rows', 0):,} | {val.get('validation_rows', 0):,} | "
            f"{val.get('folds_both_improved', 0)}/{val.get('fold_count', 0)} | {f(val.get('prior_brier'))} | "
            f"{f(val.get('candidate_brier'))} | {f(val.get('brier_gain'))} | {f(val.get('prior_log_loss'))} | "
            f"{f(val.get('candidate_log_loss'))} | {f(val.get('logloss_gain'))} | "
            f"{val.get('diagnostic_non_regression_gate_passed', False)} |"
        )
    lines += [
        "",
        "This remains diagnostic evidence, not production authority: exact live pitch compatibility, BvP process state, handedness splits, weather, and defense still require snapshot-backed train/serve parity.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--outcome-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--max-games", type=int, default=6000)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--min-history", type=int, default=20)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--ridge", type=float, default=1.5)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, join = _collect_joined_examples(
        args.data_root,
        outcome_ledger_path=args.outcome_ledger,
        season=args.season,
        max_games=args.max_games,
        trials=args.trials,
        min_history=args.min_history,
    )
    if not rows:
        raise SystemExit("no outcome-joined historical examples")
    targets = {
        target: nonreg._target_fit(rows, target=target, ridge=args.ridge, folds=args.folds, min_train_rows=args.min_train_rows)
        for target in TARGETS
    }
    game_keys = {(row["date"], row["game_id"], row["player_id"]) for row in rows}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "training_status": "FITTED_OUTCOME_JOINED_STRICT_HISTORY_SHADOW",
        "evidence_class": "OUTCOME_JOINED_STRICT_HISTORY_DIAGNOSTIC_NOT_CERTIFICATION",
        "season": args.season,
        "rows": len(rows),
        "games": len(game_keys),
        "players": len({row["player_id"] for row in rows}),
        "dates": len({row["date"] for row in rows}),
        "architecture": "strict_history_structural_prior_plus_game_conditioned_residual_moe",
        "outcome_join": join,
        "validation_design": "deterministic_cross_corpus_hash_sample_plus_expanding_window_strictly_prior_dates",
        "train_serve_feature_parity_proven": False,
        "training_feature_contract": {
            "parity_proven": False,
            "outcome_join_after_feature_boundary": True,
            "current_game_realized_columns_masked": list(CURRENT_GAME_REALIZED_OR_DERIVED_COLUMNS),
            "current_row_rolling_or_market_gap_prior": False,
            "required_evidence_source_for_authority": "mlb_game_conditioned_pregame_snapshot_v1",
            "live_not_exactly_replayable_features": list(nonreg.LIVE_FEATURES_REQUIRING_EXACT_REPLAY),
        },
        "controls": {
            "max_games": args.max_games,
            "trials": args.trials,
            "min_history": args.min_history,
            "folds": args.folds,
            "min_train_rows": args.min_train_rows,
            "ridge": args.ridge,
        },
        "targets": targets,
        "positive_authority": False,
        "negative_authority_allowed": False,
        "statistical_gate_passed": False,
        "promotion_rule": "Outcome-joined historical lift is diagnostic. Production authority remains fail-closed until exact snapshot-backed train/serve parity and certification thresholds pass.",
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
