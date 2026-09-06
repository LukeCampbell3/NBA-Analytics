#!/usr/bin/env python3
"""Idempotent production wiring for the sequential PA H/TB model.

This v2 patcher checks for the exact replacement text rather than embedding
marker comments inside YAML literal blocks. It is intended for the one-time
bootstrap workflow and is safe to rerun.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def replace_once(path: Path, old: str, new: str, *, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched {path.relative_to(REPO)}: {label}")
        return False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one anchor for {label}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path.relative_to(REPO)}: {label}")
    return True


def patch_selector() -> None:
    path = REPO / "sports/mlb/scripts/select_high_precision_predictions.py"
    replace_once(
        path,
        '''    calibrated_hit_probability = min(
        MAX_CALIBRATED_PROBABILITY,
        validated_graded_hit_rate * max(0.0, 1.0 - push_probability),
    )
    (
        historical_market_availability_key,
''',
        '''    calibrated_hit_probability = min(
        MAX_CALIBRATED_PROBABILITY,
        validated_graded_hit_rate * max(0.0, 1.0 - push_probability),
    )

    # H/TB OVER rows with complete advanced evidence are now driven by the
    # sequential nightly distribution. Until independently calibrated, the new
    # model has negative authority only: it may lower/veto legacy confidence,
    # never increase it.
    sequential_status = str(row.get("Sequential_PA_Status", "")).strip().upper()
    if target in {"H", "TB"} and direction == "OVER" and sequential_status == "READY":
        sequential_raw = to_float(row.get("Sequential_PA_Raw_Probability"), default=float("nan"))
        sequential_calibrated = to_float(row.get("Sequential_PA_Calibrated_Probability"), default=float("nan"))
        sequential_usable = to_float(row.get("Sequential_PA_Usable_Probability"), default=float("nan"))
        if all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (sequential_raw, sequential_calibrated, sequential_usable)):
            legacy_calibrated_probability = calibrated_hit_probability
            legacy_graded_probability = validated_graded_hit_rate
            model_hit_probability = sequential_raw
            model_graded_hit_rate = sequential_raw
            push_probability = 0.0
            calibrated_hit_probability = min(
                legacy_calibrated_probability,
                sequential_calibrated,
                sequential_usable,
            )
            validated_graded_hit_rate = min(legacy_graded_probability, sequential_usable)
            row["Probability_Model_Version"] = str(row.get("Sequential_PA_Model_Version") or "sequential_pa_contact_model_v1")
            row["Probability_Authority"] = "NEGATIVE_AUTHORITY_UNTIL_INDEPENDENT_ADVANCED_MODEL_CALIBRATION"
    (
        historical_market_availability_key,
''',
        label="sequential probability authority",
    )
    replace_once(
        path,
        '''                (
                    candidate.historically_calibrated_hit_probability,
                    candidate.hit_probability_calibration_status,
                    candidate.hit_probability_calibration_support,
                ) = apply_hit_probability_calibration(candidate.model_hit_probability, hit_probability_calibration)
                candidate.final_hit_probability = (
                    min(candidate.calibrated_hit_probability, candidate.historically_calibrated_hit_probability)
                    if candidate.historically_calibrated_hit_probability is not None
                    else candidate.calibrated_hit_probability
                )
                candidates.append(candidate)
''',
        '''                sequential_ready = (
                    candidate.target in {"H", "TB"}
                    and candidate.direction == "OVER"
                    and str(candidate.raw.get("Sequential_PA_Status", "")).strip().upper() == "READY"
                )
                sequential_usable = to_float(candidate.raw.get("Sequential_PA_Usable_Probability"), default=float("nan"))
                if sequential_ready and math.isfinite(sequential_usable) and 0.0 <= sequential_usable <= 1.0:
                    # The frozen isotonic curve was fitted against the legacy
                    # Poisson probability. Do not apply it to a different model
                    # class. The sequential model's own uncertainty haircut is
                    # downward-only until a locked recalibrator is validated.
                    candidate.historically_calibrated_hit_probability = None
                    candidate.hit_probability_calibration_status = str(
                        candidate.raw.get("Sequential_PA_Calibration_Status")
                        or "UNCALIBRATED_NEGATIVE_AUTHORITY_ONLY"
                    )
                    candidate.hit_probability_calibration_support = int(
                        1000 * clamp01(to_float(candidate.raw.get("Sequential_PA_Support"), default=0.0))
                    )
                    candidate.final_hit_probability = min(candidate.calibrated_hit_probability, sequential_usable)
                else:
                    (
                        candidate.historically_calibrated_hit_probability,
                        candidate.hit_probability_calibration_status,
                        candidate.hit_probability_calibration_support,
                    ) = apply_hit_probability_calibration(candidate.model_hit_probability, hit_probability_calibration)
                    candidate.final_hit_probability = (
                        min(candidate.calibrated_hit_probability, candidate.historically_calibrated_hit_probability)
                        if candidate.historically_calibrated_hit_probability is not None
                        else candidate.calibrated_hit_probability
                    )
                candidates.append(candidate)
''',
        label="advanced calibration guard",
    )
    replace_once(
        path,
        '''        "Final_Hit_Probability",
    ]
''',
        '''        "Final_Hit_Probability",
        "Player_MLBAM_ID",
        "Sequential_Batting_Order",
        "Market_Over_Price_Time",
        "Market_Under_Price_Time",
        "Probability_Model_Version",
        "Probability_Authority",
        "Sequential_PA_Model_Version",
        "Sequential_PA_Status",
        "Sequential_PA_Raw_Probability",
        "Sequential_PA_Calibrated_Probability",
        "Sequential_PA_Usable_Probability",
        "Sequential_PA_Probability_LCB",
        "Sequential_PA_Probability_SE",
        "Sequential_PA_Uncertainty",
        "Sequential_PA_Support",
        "Sequential_PA_Support_Status",
        "Sequential_PA_Calibration_Status",
        "Sequential_PA_Expected_PA",
        "Sequential_PA_Expected_AB",
        "Sequential_PA_Expected_H",
        "Sequential_PA_Expected_TB",
        "Sequential_PA_P_H_0",
        "Sequential_PA_P_H_1",
        "Sequential_PA_P_H_GE_2",
        "Sequential_PA_P_TB_0",
        "Sequential_PA_P_TB_1",
        "Sequential_PA_P_TB_GE_2",
        "Sequential_PA_P_HR_GE_1",
        "Sequential_PA_Uncertainty_Components",
        "Sequential_PA_Diagnostics",
    ]
''',
        label="selected CSV advanced columns",
    )
    replace_once(
        path,
        '''                    "Final_Hit_Probability": f"{candidate.final_hit_probability:.6f}",
                }
''',
        '''                    "Final_Hit_Probability": f"{candidate.final_hit_probability:.6f}",
                    "Player_MLBAM_ID": candidate.raw.get("Player_MLBAM_ID", ""),
                    "Sequential_Batting_Order": candidate.raw.get("Sequential_Batting_Order", ""),
                    "Market_Over_Price_Time": candidate.raw.get("Market_Over_Price_Time", ""),
                    "Market_Under_Price_Time": candidate.raw.get("Market_Under_Price_Time", ""),
                    "Probability_Model_Version": candidate.raw.get("Probability_Model_Version", ""),
                    "Probability_Authority": candidate.raw.get("Probability_Authority", ""),
                    "Sequential_PA_Model_Version": candidate.raw.get("Sequential_PA_Model_Version", ""),
                    "Sequential_PA_Status": candidate.raw.get("Sequential_PA_Status", ""),
                    "Sequential_PA_Raw_Probability": candidate.raw.get("Sequential_PA_Raw_Probability", ""),
                    "Sequential_PA_Calibrated_Probability": candidate.raw.get("Sequential_PA_Calibrated_Probability", ""),
                    "Sequential_PA_Usable_Probability": candidate.raw.get("Sequential_PA_Usable_Probability", ""),
                    "Sequential_PA_Probability_LCB": candidate.raw.get("Sequential_PA_Probability_LCB", ""),
                    "Sequential_PA_Probability_SE": candidate.raw.get("Sequential_PA_Probability_SE", ""),
                    "Sequential_PA_Uncertainty": candidate.raw.get("Sequential_PA_Uncertainty", ""),
                    "Sequential_PA_Support": candidate.raw.get("Sequential_PA_Support", ""),
                    "Sequential_PA_Support_Status": candidate.raw.get("Sequential_PA_Support_Status", ""),
                    "Sequential_PA_Calibration_Status": candidate.raw.get("Sequential_PA_Calibration_Status", ""),
                    "Sequential_PA_Expected_PA": candidate.raw.get("Sequential_PA_Expected_PA", ""),
                    "Sequential_PA_Expected_AB": candidate.raw.get("Sequential_PA_Expected_AB", ""),
                    "Sequential_PA_Expected_H": candidate.raw.get("Sequential_PA_Expected_H", ""),
                    "Sequential_PA_Expected_TB": candidate.raw.get("Sequential_PA_Expected_TB", ""),
                    "Sequential_PA_P_H_0": candidate.raw.get("Sequential_PA_P_H_0", ""),
                    "Sequential_PA_P_H_1": candidate.raw.get("Sequential_PA_P_H_1", ""),
                    "Sequential_PA_P_H_GE_2": candidate.raw.get("Sequential_PA_P_H_GE_2", ""),
                    "Sequential_PA_P_TB_0": candidate.raw.get("Sequential_PA_P_TB_0", ""),
                    "Sequential_PA_P_TB_1": candidate.raw.get("Sequential_PA_P_TB_1", ""),
                    "Sequential_PA_P_TB_GE_2": candidate.raw.get("Sequential_PA_P_TB_GE_2", ""),
                    "Sequential_PA_P_HR_GE_1": candidate.raw.get("Sequential_PA_P_HR_GE_1", ""),
                    "Sequential_PA_Uncertainty_Components": candidate.raw.get("Sequential_PA_Uncertainty_Components", ""),
                    "Sequential_PA_Diagnostics": candidate.raw.get("Sequential_PA_Diagnostics", ""),
                }
''',
        label="selected CSV advanced values",
    )


def patch_orchestrator() -> None:
    path = REPO / "sports/site/pipeline/run_daily_predictions.py"
    replace_once(
        path,
        '''MLB_GENERATOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "generate_daily_prediction_pool.py"
MLB_GOVERNANCE_CAPTURE = REPO_ROOT / "sports" / "mlb" / "governance" / "capture_complete_slate.py"
''',
        '''MLB_GENERATOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "generate_daily_prediction_pool.py"
# Incremental Statcast/FanGraphs enrichment + sequential nightly H/TB model.
MLB_SEQUENTIAL_PA_ENRICHER = REPO_ROOT / "sports" / "mlb" / "scripts" / "run_sequential_pa_hitter_model.py"
MLB_GOVERNANCE_CAPTURE = REPO_ROOT / "sports" / "mlb" / "governance" / "capture_complete_slate.py"
''',
        label="orchestrator constant",
    )
    replace_once(
        path,
        '''    mlb_dist_json = Path(getattr(args, "private_output_dir", DEFAULT_PRIVATE_OUTPUT_DIR)).resolve() / "mlb" / "data" / "daily_predictions.json"

    pool_digits = "".join(char for char in pool_csv.stem if char.isdigit())
''',
        '''    # Enrich the exact current raw pool before any selector/parlay
    # consumes it. A source outage leaves the advanced model unavailable and
    # preserves the legacy fallback; it never silently reuses stale evidence.
    if MLB_SEQUENTIAL_PA_ENRICHER.exists():
        sequential_run_date = resolve_effective_run_date(args.run_date).isoformat()
        try:
            run_step(
                "Refresh MLB Advanced H/TB Data + Sequential PA Model",
                [args.python, str(MLB_SEQUENTIAL_PA_ENRICHER), "--pool-csv", str(pool_csv), "--run-date", sequential_run_date],
            )
        except Exception as exc:
            print(
                "[warning] MLB sequential-PA advanced model unavailable; H/TB remains on the existing calibrated fallback. "
                f"{format_step_failure(exc)}"
            )

    mlb_dist_json = Path(getattr(args, "private_output_dir", DEFAULT_PRIVATE_OUTPUT_DIR)).resolve() / "mlb" / "data" / "daily_predictions.json"

    pool_digits = "".join(char for char in pool_csv.stem if char.isdigit())
''',
        label="orchestrator advanced step",
    )


def patch_exporter() -> None:
    path = REPO / "sports/mlb/scripts/export_web_prediction_payload.py"
    replace_once(
        path,
        '''def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def valid_american_price(value: object) -> float | None:
''',
        '''def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _finite_or_none(value: object) -> float | None:
    result = to_float(value, default=float("nan"))
    return result if math.isfinite(result) else None


def _sequential_v21_uncertainty(row: dict[str, str], lineup_status: str) -> dict[str, float] | None:
    if str(row.get("Sequential_PA_Status", "")).strip().upper() != "READY":
        return None
    source = _json_dict(row.get("Sequential_PA_Uncertainty_Components")) or {}
    def risk(name: str, default: float = 0.0) -> float:
        return max(0.0, min(1.0, to_float(source.get(name), default=default)))
    return {
        "model": 0.025 * max(risk("batter_sample"), risk("pitcher_sample")),
        "calibration": 0.030,
        "market_disagreement": 0.020,
        "player_role": 0.005,
        "lineup": 0.0 if lineup_status == "confirmed" else 0.050,
        "opportunity": 0.020 * risk("expected_pa", 0.25),
        "data_support": 0.025 * max(risk("contact_quality_missing"), risk("advanced_pitching_missing")),
        "distribution_shift": 0.020 * risk("data_freshness"),
    }


def valid_american_price(value: object) -> float | None:
''',
        label="export helper and V2.1 uncertainty mapping",
    )
    replace_once(
        path,
        '''                "matchup_network_adjustment": to_float(row.get("Matchup_Network_Adjustment")),
                "player_mlbam_id": resolved_player_id,
''',
        '''                "matchup_network_adjustment": to_float(row.get("Matchup_Network_Adjustment")),
                "model_version": row.get("Probability_Model_Version") or row.get("Sequential_PA_Model_Version") or row.get("Matchup_Network_Version", ""),
                "probability_authority": row.get("Probability_Authority", ""),
                "raw_structural_probability": _finite_or_none(row.get("Sequential_PA_Raw_Probability")),
                "calibrated_probability": _finite_or_none(row.get("Sequential_PA_Calibrated_Probability")),
                "usable_probability": _finite_or_none(row.get("Sequential_PA_Usable_Probability")),
                "probability_lcb": _finite_or_none(row.get("Sequential_PA_Probability_LCB")),
                "uncertainty": _finite_or_none(row.get("Sequential_PA_Uncertainty")),
                "uncertainty_components": _sequential_v21_uncertainty(row, lineup_status),
                "support_status": row.get("Sequential_PA_Support_Status", ""),
                "ood_status": "IN_SUPPORT" if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" and str(row.get("Sequential_PA_Support_Status", "")).strip().upper() == "SUPPORTED" else "UNMEASURED",
                "expected_plate_appearances": _finite_or_none(row.get("Sequential_PA_Expected_PA")),
                "expected_at_bats": _finite_or_none(row.get("Sequential_PA_Expected_AB")),
                "expected_hits_sequential": _finite_or_none(row.get("Sequential_PA_Expected_H")),
                "expected_tb_sequential": _finite_or_none(row.get("Sequential_PA_Expected_TB")),
                "p_h_0": _finite_or_none(row.get("Sequential_PA_P_H_0")),
                "sequential_pa_status": row.get("Sequential_PA_Status", ""),
                "sequential_pa_calibration_status": row.get("Sequential_PA_Calibration_Status", ""),
                "selected_side_price_time": row.get("Market_Over_Price_Time" if direction == "OVER" else "Market_Under_Price_Time", ""),
                "player_status": "ACTIVE" if lineup_status == "confirmed" else "UNKNOWN",
                "player_mlbam_id": resolved_player_id,
''',
        label="export advanced fields",
    )


def patch_unified_adapter() -> None:
    path = REPO / "sports/mlb/unified/adapters.py"
    replace_once(
        path,
        '''    final_probability = play.get("final_hit_probability")
    price = play.get("selected_side_price", play.get("american_price"))
''',
        '''    final_probability = play.get("final_hit_probability")
    structural_probability = play.get("raw_structural_probability")
    if structural_probability is None:
        structural_probability = play.get("model_hit_probability")
    calibrated_probability = play.get("calibrated_probability")
    if calibrated_probability is None:
        calibrated_probability = final_probability
    usable_probability = play.get("usable_probability")
    uncertainty_components = play.get("uncertainty_components")
    uncertainty = play.get("uncertainty")
    if uncertainty is None and final_probability is not None and not uncertainty_components:
        uncertainty = 0.0
    price = play.get("selected_side_price", play.get("american_price"))
''',
        label="canonical advanced probability setup",
    )
    replace_once(
        path,
        '''        structural_probability=play.get("model_hit_probability"), market_conditioned_probability=None,
        raw_probability=play.get("estimated_hit_probability"), calibrated_probability=final_probability,
        uncertainty=uncertainty, usable_probability=None, support_status=support,
''',
        '''        structural_probability=structural_probability,
        market_conditioned_probability=play.get("market_conditioned_probability"),
        raw_probability=play.get("estimated_hit_probability"), calibrated_probability=calibrated_probability,
        uncertainty=uncertainty, usable_probability=usable_probability,
        support_status=(str(play.get("support_status") or "").upper() or support),
''',
        label="canonical advanced probability values",
    )
    replace_once(
        path,
        '''            "uncertainty_components": play.get("uncertainty_components"),
            "model_version": play.get("model_version") or play.get("matchup_network_version"),
''',
        '''            "uncertainty_components": uncertainty_components,
            "expected_plate_appearances": play.get("expected_plate_appearances"),
            "pa_probability_3_plus": play.get("pa_probability_3_plus"),
            "settlement_identity": f"{play.get('game_id')}:{play.get('player_mlbam_id') or play.get('player_id') or play.get('player')}:game",
            "model_version": play.get("model_version") or play.get("matchup_network_version"),
''',
        label="canonical source payload",
    )


def patch_pinned_runner() -> None:
    path = REPO / "sports/mlb/scripts/run_pinned_v2_1_daily_shadow.py"
    replace_once(
        path,
        '''        _run(["git", "worktree", "add", "--detach", str(worktree), V21_COMMIT])
        EVIDENCE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        _run(
''',
        '''        _run(["git", "worktree", "add", "--detach", str(worktree), V21_COMMIT])
        # Preserve the frozen policy/scoring implementation, but overlay the
        # current adapter/contract modules so newly resolved canonical facts
        # actually reach that policy. This repairs plumbing, not thresholds.
        for relative in (
            Path("sports/mlb/unified/adapters.py"),
            Path("sports/mlb/unified/candidate_contract.py"),
        ):
            source = REPO_ROOT / relative
            target = worktree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        EVIDENCE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        _run(
''',
        label="V2.1 canonical plumbing overlay",
    )


def patch_main_workflow() -> None:
    path = REPO / ".github/workflows/mlb-predictions.yml"
    replace_once(
        path,
        '''      - 'sports/mlb/unified/**'
      - 'sports/mlb/scripts/run_unified_mlb_shadow.py'
''',
        '''      - 'sports/mlb/unified/**'
      - 'sports/mlb/advanced/**'
      - 'sports/mlb/scripts/run_sequential_pa_hitter_model.py'
      - 'sports/mlb/tests/test_sequential_pa_model.py'
      - 'sports/mlb/tests/test_advanced_mlb_profiles.py'
      - 'sports/mlb/requirements-advanced.txt'
      - 'sports/mlb/scripts/run_unified_mlb_shadow.py'
''',
        label="workflow trigger paths",
    )
    replace_once(
        path,
        '''            requirements.txt
            sports/mlb/requirements-same-game.txt
''',
        '''            requirements.txt
            sports/mlb/requirements-same-game.txt
            sports/mlb/requirements-advanced.txt
''',
        label="workflow dependency cache",
    )
    replace_once(
        path,
        '''          python -m pip install -r requirements.txt
          python -m pip install -r sports/mlb/requirements-same-game.txt
''',
        '''          python -m pip install -r requirements.txt
          python -m pip install -r sports/mlb/requirements-same-game.txt
          python -m pip install -r sports/mlb/requirements-advanced.txt
''',
        label="workflow advanced install",
    )
    replace_once(
        path,
        '''            sports/mlb/data/predictions/calibration/*.json
          key: mlb-daily-v4-${{ runner.os }}-${{ hashFiles('Player-Predictor/scripts/update_mlb_processed_data.py', 'sports/mlb/decision_engine/matchup_network.py', 'sports/mlb/scripts/backtest_latent_daily_pools.py') }}-${{ steps.run-config.outputs.run_stamp }}-${{ github.run_id }}
''',
        '''            sports/mlb/data/predictions/calibration/*.json
            sports/mlb/data/advanced
          key: mlb-daily-v4-${{ runner.os }}-${{ hashFiles('Player-Predictor/scripts/update_mlb_processed_data.py', 'sports/mlb/decision_engine/matchup_network.py', 'sports/mlb/advanced/**', 'sports/mlb/requirements-advanced.txt', 'sports/mlb/scripts/backtest_latent_daily_pools.py') }}-${{ steps.run-config.outputs.run_stamp }}-${{ github.run_id }}
''',
        label="workflow advanced cache",
    )
    replace_once(
        path,
        '''            sports/mlb/tests/test_validate_v4_live_players.py \\
            tests/test_run_daily_predictions.py \\
            -q
''',
        '''            sports/mlb/tests/test_validate_v4_live_players.py \\
            sports/mlb/tests/test_sequential_pa_model.py \\
            sports/mlb/tests/test_advanced_mlb_profiles.py \\
            tests/test_run_daily_predictions.py \\
            -q
''',
        label="workflow tests",
    )
    replace_once(
        path,
        '''            artifacts/mlb_slate_integrity_audit.json \\
            docs/mlb_v2_daily_evidence.md \\
''',
        '''            artifacts/mlb_slate_integrity_audit.json \\
            artifacts/mlb_sequential_pa_model_validation.json \\
            artifacts/mlb_sequential_pa_model_validation.md \\
            sports/mlb/web/data/sequential_pa_hitter_predictions.json \\
            docs/mlb_v2_daily_evidence.md \\
''',
        label="workflow staging",
    )
    replace_once(
        path,
        '''            sports/mlb/web/data/mlb_engine_manifest.json
            sports/mlb/data/predictions/backtests/latent_daily_pool_replay_2026.json
''',
        '''            sports/mlb/web/data/mlb_engine_manifest.json
            sports/mlb/web/data/sequential_pa_hitter_predictions.json
            artifacts/mlb_sequential_pa_model_validation.json
            artifacts/mlb_sequential_pa_model_validation.md
            sports/mlb/data/predictions/backtests/latent_daily_pool_replay_2026.json
''',
        label="workflow artifacts",
    )


def patch_consistency_workflow() -> None:
    path = REPO / ".github/workflows/mlb-frontend-consistency.yml"
    replace_once(
        path,
        '''            requirements.txt
            sports/mlb/requirements-same-game.txt
''',
        '''            requirements.txt
            sports/mlb/requirements-same-game.txt
            sports/mlb/requirements-advanced.txt
''',
        label="consistency dependency cache",
    )
    replace_once(
        path,
        '''          python -m pip install -r requirements.txt
          python -m pip install -r sports/mlb/requirements-same-game.txt
''',
        '''          python -m pip install -r requirements.txt
          python -m pip install -r sports/mlb/requirements-same-game.txt
          python -m pip install -r sports/mlb/requirements-advanced.txt
''',
        label="consistency advanced install",
    )
    replace_once(
        path,
        '''            sports/mlb/data/predictions/calibration/*.json
          key: mlb-frontend-sync-${{ runner.os }}-${{ hashFiles('Player-Predictor/scripts/update_mlb_processed_data.py', 'sports/mlb/decision_engine/matchup_network.py', 'sports/mlb/scripts/backtest_latent_daily_pools.py') }}-${{ steps.run-config.outputs.run_stamp }}-${{ github.run_id }}
''',
        '''            sports/mlb/data/predictions/calibration/*.json
            sports/mlb/data/advanced
          key: mlb-frontend-sync-${{ runner.os }}-${{ hashFiles('Player-Predictor/scripts/update_mlb_processed_data.py', 'sports/mlb/decision_engine/matchup_network.py', 'sports/mlb/advanced/**', 'sports/mlb/requirements-advanced.txt', 'sports/mlb/scripts/backtest_latent_daily_pools.py') }}-${{ steps.run-config.outputs.run_stamp }}-${{ github.run_id }}
''',
        label="consistency advanced cache",
    )
    replace_once(
        path,
        '''              "high_hit_parlay_predictions.json",
          )
''',
        '''              "high_hit_parlay_predictions.json",
              "sequential_pa_hitter_predictions.json",
          )
''',
        label="consistency product validation",
    )
    replace_once(
        path,
        '''            sports/mlb/web/data/high_hit_parlay_predictions.json \\
            dist/mlb \\
''',
        '''            sports/mlb/web/data/high_hit_parlay_predictions.json \\
            sports/mlb/web/data/sequential_pa_hitter_predictions.json \\
            dist/mlb \\
''',
        label="consistency staging",
    )
    replace_once(
        path,
        '''            dist/mlb/data/high_hit_parlay_predictions.json
            dist/mlb/predictions.js
''',
        '''            dist/mlb/data/high_hit_parlay_predictions.json
            dist/mlb/data/sequential_pa_hitter_predictions.json
            dist/mlb/predictions.js
''',
        label="consistency artifacts",
    )


def main() -> int:
    patch_selector()
    patch_orchestrator()
    patch_exporter()
    patch_unified_adapter()
    patch_pinned_runner()
    patch_main_workflow()
    patch_consistency_workflow()
    print("sequential PA production wiring v2 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
