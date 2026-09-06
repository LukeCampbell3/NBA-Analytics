#!/usr/bin/env python3
"""Idempotently wire the sequential-PA H/TB model into MLB production.

This script exists because the affected orchestrator/selector/workflow files are
large and long-lived.  It applies narrowly-scoped, assertion-backed replacements
and exits non-zero if an expected source anchor drifts.  The bootstrap workflow
runs it once and commits the resulting ordinary source changes; normal daily
runs do NOT execute this patcher.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MARKER = "SEQUENTIAL_PA_CONTACT_MODEL_V1_WIRING"


def replace_once(path: Path, old: str, new: str, *, marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one patch anchor for {marker}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path.relative_to(REPO)}: {marker}")
    return True


def patch_selector() -> None:
    path = REPO / "sports/mlb/scripts/select_high_precision_predictions.py"

    old = '''    calibrated_hit_probability = min(
        MAX_CALIBRATED_PROBABILITY,
        validated_graded_hit_rate * max(0.0, 1.0 - push_probability),
    )
    (
        historical_market_availability_key,
'''
    new = '''    calibrated_hit_probability = min(
        MAX_CALIBRATED_PROBABILITY,
        validated_graded_hit_rate * max(0.0, 1.0 - push_probability),
    )

    # SEQUENTIAL_PA_CONTACT_MODEL_V1_WIRING: H/TB OVER probabilities may be
    # replaced by a fully modeled sequential-PA distribution only when the
    # advanced-data row is READY.  Until the new model earns independent
    # calibration authority it is strictly negative authority: it may lower
    # the legacy probability/EV and veto a candidate, never raise either.
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
            push_probability = 0.0  # H 0.5 / TB 1.5 are half-lines in this path.
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
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONTACT_MODEL_V1_WIRING")

    old = '''                (
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
'''
    new = '''                # SEQUENTIAL_PA_ADVANCED_CALIBRATION_GUARD: the historical
                # isotonic curve was fitted against the old Poisson point-
                # projection probability.  Applying it to the new structural
                # probability would be a model-class mismatch.  READY H/TB
                # rows therefore retain their own downward-only uncertainty
                # adjustment; legacy rows continue through the frozen curve.
                sequential_ready = (
                    candidate.target in {"H", "TB"}
                    and candidate.direction == "OVER"
                    and str(candidate.raw.get("Sequential_PA_Status", "")).strip().upper() == "READY"
                )
                sequential_usable = to_float(candidate.raw.get("Sequential_PA_Usable_Probability"), default=float("nan"))
                if sequential_ready and math.isfinite(sequential_usable) and 0.0 <= sequential_usable <= 1.0:
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
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_ADVANCED_CALIBRATION_GUARD")

    old = '''        "Final_Hit_Probability",
    ]
'''
    new = '''        "Final_Hit_Probability",
        # SEQUENTIAL_PA_OUTPUT_COLUMNS
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
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_OUTPUT_COLUMNS")

    old = '''                    "Final_Hit_Probability": f"{candidate.final_hit_probability:.6f}",
                }
'''
    new = '''                    "Final_Hit_Probability": f"{candidate.final_hit_probability:.6f}",
                    # SEQUENTIAL_PA_OUTPUT_VALUES: preserve raw/calibrated/
                    # usable/LCB probability stages and diagnostic evidence all
                    # the way into web/unified adapters.
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
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_OUTPUT_VALUES")


def patch_orchestrator() -> None:
    path = REPO / "sports/site/pipeline/run_daily_predictions.py"
    old = '''MLB_GENERATOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "generate_daily_prediction_pool.py"
MLB_GOVERNANCE_CAPTURE = REPO_ROOT / "sports" / "mlb" / "governance" / "capture_complete_slate.py"
'''
    new = '''MLB_GENERATOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "generate_daily_prediction_pool.py"
# SEQUENTIAL_PA_CONTACT_MODEL_V1_ORCHESTRATOR: incremental Statcast/FanGraphs
# refresh + nightly H/TB distribution enrichment. Failure is explicit and
# falls back to the existing legacy H/TB probability path; it never fabricates
# advanced inputs or weakens publication gates.
MLB_SEQUENTIAL_PA_ENRICHER = REPO_ROOT / "sports" / "mlb" / "scripts" / "run_sequential_pa_hitter_model.py"
MLB_GOVERNANCE_CAPTURE = REPO_ROOT / "sports" / "mlb" / "governance" / "capture_complete_slate.py"
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONTACT_MODEL_V1_ORCHESTRATOR")

    old = '''    mlb_dist_json = Path(getattr(args, "private_output_dir", DEFAULT_PRIVATE_OUTPUT_DIR)).resolve() / "mlb" / "data" / "daily_predictions.json"

    pool_digits = "".join(char for char in pool_csv.stem if char.isdigit())
'''
    new = '''    # SEQUENTIAL_PA_CONTACT_MODEL_V1_DAILY_STEP: enrich the exact raw pool
    # before any selector/parlay reads it. Advanced refresh failures are
    # visible and leave Sequential_PA_Status unavailable, which makes the
    # selector retain its legacy fallback rather than silently using stale data.
    if MLB_SEQUENTIAL_PA_ENRICHER.exists():
        sequential_run_date = resolve_effective_run_date(args.run_date).isoformat()
        try:
            run_step(
                "Refresh MLB Advanced H/TB Data + Sequential PA Model",
                [
                    args.python,
                    str(MLB_SEQUENTIAL_PA_ENRICHER),
                    "--pool-csv",
                    str(pool_csv),
                    "--run-date",
                    sequential_run_date,
                ],
            )
        except Exception as exc:  # advanced model is additive until certified
            print(
                "[warning] MLB sequential-PA advanced model unavailable; "
                "H/TB remains on the existing calibrated fallback for this run. "
                f"{format_step_failure(exc)}"
            )

    mlb_dist_json = Path(getattr(args, "private_output_dir", DEFAULT_PRIVATE_OUTPUT_DIR)).resolve() / "mlb" / "data" / "daily_predictions.json"

    pool_digits = "".join(char for char in pool_csv.stem if char.isdigit())
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONTACT_MODEL_V1_DAILY_STEP")


def patch_exporter() -> None:
    path = REPO / "sports/mlb/scripts/export_web_prediction_payload.py"
    old = '''def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def valid_american_price(value: object) -> float | None:
'''
    new = '''def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json_dict(value: object) -> dict[str, object] | None:
    # SEQUENTIAL_PA_EXPORT_JSON_HELPER
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


def _sequential_v21_uncertainty(row: dict[str, str], lineup_status: str) -> dict[str, float] | None:
    """Translate v1 risk indicators into V2.1 probability-uncertainty units.

    The sequential model's stored components are normalized risk indicators,
    not percentage-point standard errors.  V2.1 expects eight bounded
    probability-uncertainty contributions whose RSS is compared with its frozen
    0.10 maximum.  This mapping is fixed plumbing, never fitted to outcomes.
    """
    if str(row.get("Sequential_PA_Status", "")).strip().upper() != "READY":
        return None
    source = _json_dict(row.get("Sequential_PA_Uncertainty_Components")) or {}
    def risk(name: str, default: float = 0.0) -> float:
        return max(0.0, min(1.0, to_float(source.get(name), default=default)))
    return {
        "model": 0.025 * max(risk("batter_sample"), risk("pitcher_sample")),
        "calibration": 0.030,  # v1 intentionally has no positive calibration authority yet.
        "market_disagreement": 0.020,
        "player_role": 0.005,
        "lineup": 0.0 if lineup_status == "confirmed" else 0.050,
        "opportunity": 0.020 * risk("expected_pa", 0.25),
        "data_support": 0.025 * max(risk("contact_quality_missing"), risk("advanced_pitching_missing")),
        "distribution_shift": 0.020 * risk("data_freshness"),
    }


def valid_american_price(value: object) -> float | None:
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_EXPORT_JSON_HELPER")

    old = '''                "matchup_network_adjustment": to_float(row.get("Matchup_Network_Adjustment")),
                "player_mlbam_id": resolved_player_id,
'''
    new = '''                "matchup_network_adjustment": to_float(row.get("Matchup_Network_Adjustment")),
                # SEQUENTIAL_PA_EXPORT_FIELDS: preserve probability stages and
                # process diagnostics for the current unified/V2.1 adapters.
                "model_version": row.get("Probability_Model_Version") or row.get("Sequential_PA_Model_Version") or row.get("Matchup_Network_Version", ""),
                "probability_authority": row.get("Probability_Authority", ""),
                "raw_structural_probability": to_float(row.get("Sequential_PA_Raw_Probability"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "calibrated_probability": to_float(row.get("Sequential_PA_Calibrated_Probability"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "usable_probability": to_float(row.get("Sequential_PA_Usable_Probability"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "probability_lcb": to_float(row.get("Sequential_PA_Probability_LCB"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "uncertainty": to_float(row.get("Sequential_PA_Uncertainty"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "uncertainty_components": _sequential_v21_uncertainty(row, lineup_status),
                "support_status": row.get("Sequential_PA_Support_Status", ""),
                "ood_status": "IN_SUPPORT" if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" and str(row.get("Sequential_PA_Support_Status", "")).strip().upper() == "SUPPORTED" else "UNMEASURED",
                "expected_plate_appearances": to_float(row.get("Sequential_PA_Expected_PA"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "expected_at_bats": to_float(row.get("Sequential_PA_Expected_AB"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "expected_hits_sequential": to_float(row.get("Sequential_PA_Expected_H"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "expected_tb_sequential": to_float(row.get("Sequential_PA_Expected_TB"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "p_h_0": to_float(row.get("Sequential_PA_P_H_0"), default=float("nan")) if str(row.get("Sequential_PA_Status", "")).strip().upper() == "READY" else None,
                "sequential_pa_status": row.get("Sequential_PA_Status", ""),
                "sequential_pa_calibration_status": row.get("Sequential_PA_Calibration_Status", ""),
                "selected_side_price_time": row.get("Market_Over_Price_Time" if direction == "OVER" else "Market_Under_Price_Time", ""),
                "player_status": "ACTIVE" if lineup_status == "confirmed" else "UNKNOWN",
                "player_mlbam_id": resolved_player_id,
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_EXPORT_FIELDS")


def patch_unified_adapter() -> None:
    path = REPO / "sports/mlb/unified/adapters.py"
    old = '''    final_probability = play.get("final_hit_probability")
    price = play.get("selected_side_price", play.get("american_price"))
'''
    new = '''    final_probability = play.get("final_hit_probability")
    # SEQUENTIAL_PA_CANONICAL_ADAPTER: exact probability stages are preserved
    # when the current H/TB model produced them. Legacy rows remain unchanged.
    structural_probability = play.get("raw_structural_probability")
    if structural_probability is None:
        structural_probability = play.get("model_hit_probability")
    calibrated_probability = play.get("calibrated_probability")
    if calibrated_probability is None:
        calibrated_probability = final_probability
    usable_probability = play.get("usable_probability")
    if usable_probability is None and str(play.get("sequential_pa_status") or "").upper() != "READY":
        usable_probability = None
    uncertainty_components = play.get("uncertainty_components")
    uncertainty = play.get("uncertainty")
    if uncertainty is None and final_probability is not None and not uncertainty_components:
        uncertainty = 0.0
    price = play.get("selected_side_price", play.get("american_price"))
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CANONICAL_ADAPTER")

    old = '''        structural_probability=play.get("model_hit_probability"), market_conditioned_probability=None,
        raw_probability=play.get("estimated_hit_probability"), calibrated_probability=final_probability,
        uncertainty=uncertainty, usable_probability=None, support_status=support,
'''
    new = '''        structural_probability=structural_probability,
        market_conditioned_probability=play.get("market_conditioned_probability"),
        raw_probability=play.get("estimated_hit_probability"), calibrated_probability=calibrated_probability,
        uncertainty=uncertainty, usable_probability=usable_probability, support_status=(
            str(play.get("support_status") or "").upper() or support
        ),
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CANONICAL_PROBABILITIES")

    old = '''            "uncertainty_components": play.get("uncertainty_components"),
            "model_version": play.get("model_version") or play.get("matchup_network_version"),
'''
    new = '''            "uncertainty_components": uncertainty_components,
            "expected_plate_appearances": play.get("expected_plate_appearances"),
            "pa_probability_3_plus": play.get("pa_probability_3_plus"),
            "settlement_identity": f"{play.get('game_id')}:{play.get('player_mlbam_id') or play.get('player_id') or play.get('player')}:game",
            "model_version": play.get("model_version") or play.get("matchup_network_version"),
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CANONICAL_SOURCE_PAYLOAD")


def patch_pinned_runner() -> None:
    path = REPO / "sports/mlb/scripts/run_pinned_v2_1_daily_shadow.py"
    old = '''        _run(["git", "worktree", "add", "--detach", str(worktree), V21_COMMIT])
        EVIDENCE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        _run(
'''
    new = '''        _run(["git", "worktree", "add", "--detach", str(worktree), V21_COMMIT])
        # SEQUENTIAL_PA_V21_PLUMBING_OVERLAY: keep the policy/scoring code at
        # its frozen implementation commit, but overlay the current adapter and
        # canonical-contract modules so newly preserved lineup/quote/usable-
        # probability/uncertainty facts actually reach that frozen policy.
        # This is a data-contract repair, not policy tuning.
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
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_V21_PLUMBING_OVERLAY")


def patch_main_workflow() -> None:
    path = REPO / ".github/workflows/mlb-predictions.yml"
    old = '''      - 'sports/mlb/unified/**'
      - 'sports/mlb/scripts/run_unified_mlb_shadow.py'
'''
    new = '''      - 'sports/mlb/unified/**'
      # SEQUENTIAL_PA_WORKFLOW_PATHS
      - 'sports/mlb/advanced/**'
      - 'sports/mlb/scripts/run_sequential_pa_hitter_model.py'
      - 'sports/mlb/tests/test_sequential_pa_model.py'
      - 'sports/mlb/tests/test_advanced_mlb_profiles.py'
      - 'sports/mlb/requirements-advanced.txt'
      - 'sports/mlb/scripts/run_unified_mlb_shadow.py'
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_WORKFLOW_PATHS")

    old = '''            requirements.txt
            sports/mlb/requirements-same-game.txt
'''
    new = '''            requirements.txt
            sports/mlb/requirements-same-game.txt
            sports/mlb/requirements-advanced.txt # SEQUENTIAL_PA_WORKFLOW_DEPS
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_WORKFLOW_DEPS")

    old = '''          python -m pip install -r requirements.txt
          python -m pip install -r sports/mlb/requirements-same-game.txt
'''
    new = '''          python -m pip install -r requirements.txt
          python -m pip install -r sports/mlb/requirements-same-game.txt
          python -m pip install -r sports/mlb/requirements-advanced.txt  # SEQUENTIAL_PA_WORKFLOW_INSTALL
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_WORKFLOW_INSTALL")

    old = '''            sports/mlb/data/predictions/calibration/*.json
          key: mlb-daily-v4-${{ runner.os }}-${{ hashFiles('Player-Predictor/scripts/update_mlb_processed_data.py', 'sports/mlb/decision_engine/matchup_network.py', 'sports/mlb/scripts/backtest_latent_daily_pools.py') }}-${{ steps.run-config.outputs.run_stamp }}-${{ github.run_id }}
'''
    new = '''            sports/mlb/data/predictions/calibration/*.json
            sports/mlb/data/advanced
          # SEQUENTIAL_PA_WORKFLOW_CACHE
          key: mlb-daily-v4-${{ runner.os }}-${{ hashFiles('Player-Predictor/scripts/update_mlb_processed_data.py', 'sports/mlb/decision_engine/matchup_network.py', 'sports/mlb/advanced/**', 'sports/mlb/requirements-advanced.txt', 'sports/mlb/scripts/backtest_latent_daily_pools.py') }}-${{ steps.run-config.outputs.run_stamp }}-${{ github.run_id }}
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_WORKFLOW_CACHE")

    old = '''            sports/mlb/tests/test_validate_v4_live_players.py \\
            tests/test_run_daily_predictions.py \\
            -q
'''
    new = '''            sports/mlb/tests/test_validate_v4_live_players.py \\
            sports/mlb/tests/test_sequential_pa_model.py \\
            sports/mlb/tests/test_advanced_mlb_profiles.py \\
            tests/test_run_daily_predictions.py \\
            -q  # SEQUENTIAL_PA_WORKFLOW_TESTS
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_WORKFLOW_TESTS")

    old = '''            artifacts/mlb_slate_integrity_audit.json \\
            docs/mlb_v2_daily_evidence.md \\
'''
    new = '''            artifacts/mlb_slate_integrity_audit.json \\
            artifacts/mlb_sequential_pa_model_validation.json \\
            artifacts/mlb_sequential_pa_model_validation.md \\
            sports/mlb/web/data/sequential_pa_hitter_predictions.json \\
            docs/mlb_v2_daily_evidence.md \\
            # SEQUENTIAL_PA_WORKFLOW_STAGE
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_WORKFLOW_STAGE")

    old = '''            sports/mlb/web/data/mlb_engine_manifest.json
            sports/mlb/data/predictions/backtests/latent_daily_pool_replay_2026.json
'''
    new = '''            sports/mlb/web/data/mlb_engine_manifest.json
            sports/mlb/web/data/sequential_pa_hitter_predictions.json
            artifacts/mlb_sequential_pa_model_validation.json
            artifacts/mlb_sequential_pa_model_validation.md
            sports/mlb/data/predictions/backtests/latent_daily_pool_replay_2026.json
            # SEQUENTIAL_PA_WORKFLOW_ARTIFACT
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_WORKFLOW_ARTIFACT")


def patch_consistency_workflow() -> None:
    path = REPO / ".github/workflows/mlb-frontend-consistency.yml"
    old = '''            requirements.txt
            sports/mlb/requirements-same-game.txt
'''
    new = '''            requirements.txt
            sports/mlb/requirements-same-game.txt
            sports/mlb/requirements-advanced.txt # SEQUENTIAL_PA_CONSISTENCY_DEPS
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONSISTENCY_DEPS")

    old = '''          python -m pip install -r requirements.txt
          python -m pip install -r sports/mlb/requirements-same-game.txt
'''
    new = '''          python -m pip install -r requirements.txt
          python -m pip install -r sports/mlb/requirements-same-game.txt
          python -m pip install -r sports/mlb/requirements-advanced.txt  # SEQUENTIAL_PA_CONSISTENCY_INSTALL
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONSISTENCY_INSTALL")

    old = '''            sports/mlb/data/predictions/calibration/*.json
          key: mlb-frontend-sync-${{ runner.os }}-${{ hashFiles('Player-Predictor/scripts/update_mlb_processed_data.py', 'sports/mlb/decision_engine/matchup_network.py', 'sports/mlb/scripts/backtest_latent_daily_pools.py') }}-${{ steps.run-config.outputs.run_stamp }}-${{ github.run_id }}
'''
    new = '''            sports/mlb/data/predictions/calibration/*.json
            sports/mlb/data/advanced
          # SEQUENTIAL_PA_CONSISTENCY_CACHE
          key: mlb-frontend-sync-${{ runner.os }}-${{ hashFiles('Player-Predictor/scripts/update_mlb_processed_data.py', 'sports/mlb/decision_engine/matchup_network.py', 'sports/mlb/advanced/**', 'sports/mlb/requirements-advanced.txt', 'sports/mlb/scripts/backtest_latent_daily_pools.py') }}-${{ steps.run-config.outputs.run_stamp }}-${{ github.run_id }}
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONSISTENCY_CACHE")

    old = '''              "high_hit_parlay_predictions.json",
          )
'''
    new = '''              "high_hit_parlay_predictions.json",
              "sequential_pa_hitter_predictions.json",  # SEQUENTIAL_PA_CONSISTENCY_PRODUCT
          )
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONSISTENCY_PRODUCT")

    old = '''            sports/mlb/web/data/high_hit_parlay_predictions.json \\
            dist/mlb \\
'''
    new = '''            sports/mlb/web/data/high_hit_parlay_predictions.json \\
            sports/mlb/web/data/sequential_pa_hitter_predictions.json \\
            dist/mlb \\
            # SEQUENTIAL_PA_CONSISTENCY_STAGE
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONSISTENCY_STAGE")

    old = '''            dist/mlb/data/high_hit_parlay_predictions.json
            dist/mlb/predictions.js
'''
    new = '''            dist/mlb/data/high_hit_parlay_predictions.json
            dist/mlb/data/sequential_pa_hitter_predictions.json
            dist/mlb/predictions.js
            # SEQUENTIAL_PA_CONSISTENCY_ARTIFACT
'''
    replace_once(path, old, new, marker="SEQUENTIAL_PA_CONSISTENCY_ARTIFACT")


def main() -> int:
    patch_selector()
    patch_orchestrator()
    patch_exporter()
    patch_unified_adapter()
    patch_pinned_runner()
    patch_main_workflow()
    patch_consistency_workflow()
    print("sequential PA production wiring complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
