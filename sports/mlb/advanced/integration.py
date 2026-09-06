from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .data_layer import DEFAULT_ADVANCED_ROOT, load_profile_partition, refresh_advanced_profiles
from .game_conditioned_moe import (
    MODEL_VERSION as GAME_CONDITIONED_MODEL_VERSION,
    TARGETS as GAME_CONDITIONED_TARGETS,
    build_expert_state,
    choose_prior_probability,
    condition_probability,
    load_model_artifact,
)
from .schema import AdvancedCandidateContext, BatterProcessProfile, DirectMatchupProcess, PitcherProcessProfile
from .sequential_pa_model import MODEL_VERSION as STRUCTURAL_MODEL_VERSION, simulate_hitter_market

MLB_LIVE_FEED_ROOT = "https://statsapi.mlb.com/api/v1.1/game"
MLB_API_TIMEOUT_SECONDS = 12
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAME_SIM_PATH = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "game_simulation_predictions.json"


def _norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return " ".join(text.split())


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    number = _float(value)
    return int(number) if number is not None else default


def _fetch_live_feed(game_id: str) -> dict[str, Any]:
    response = requests.get(f"{MLB_LIVE_FEED_ROOT}/{game_id}/feed/live", timeout=MLB_API_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _game_identity_index(payload: dict[str, Any]) -> dict[str, Any]:
    game_data = payload.get("gameData") or {}
    players = game_data.get("players") or {}
    people: dict[str, int] = {}
    for item in players.values() if isinstance(players, dict) else []:
        if not isinstance(item, dict):
            continue
        pid = _int(item.get("id"))
        name = _norm(item.get("fullName"))
        if pid > 0 and name:
            people[name] = pid

    probable = game_data.get("probablePitchers") or {}
    probable_ids = {
        "home": _int((probable.get("home") or {}).get("id")) if isinstance(probable, dict) else 0,
        "away": _int((probable.get("away") or {}).get("id")) if isinstance(probable, dict) else 0,
    }
    teams = game_data.get("teams") or {}
    team_abbrev = {
        "home": str((teams.get("home") or {}).get("abbreviation") or "").upper(),
        "away": str((teams.get("away") or {}).get("abbreviation") or "").upper(),
    }

    batting_order: dict[int, int] = {}
    boxscore = (payload.get("liveData") or {}).get("boxscore") or {}
    box_teams = boxscore.get("teams") or {}
    for side in ("home", "away"):
        side_payload = box_teams.get(side) or {}
        order = side_payload.get("battingOrder") or []
        for index, pid in enumerate(order, start=1):
            resolved = _int(pid)
            if resolved > 0:
                batting_order[resolved] = index

    return {"people": people, "probable_ids": probable_ids, "team_abbrev": team_abbrev, "batting_order": batting_order}


def _nullable_int_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a pandas nullable-integer column suitable for MLBAM IDs."""
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Int64")
    return pd.to_numeric(frame[column], errors="coerce").astype("Int64")


def hydrate_pool_identities(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    for column in ("Player_MLBAM_ID", "Sequential_Batting_Order", "Opposing_Pitcher_ID"):
        result[column] = _nullable_int_column(result, column)

    diagnostics = {"games_requested": 0, "games_resolved": 0, "identity_failures": []}
    game_cache: dict[str, dict[str, Any]] = {}
    for game_id in sorted({str(value) for value in result.get("Game_ID", pd.Series(dtype=str)).dropna() if str(value)}):
        diagnostics["games_requested"] += 1
        try:
            game_cache[game_id] = _game_identity_index(_fetch_live_feed(game_id))
            diagnostics["games_resolved"] += 1
        except Exception as exc:
            diagnostics["identity_failures"].append({"game_id": game_id, "error": str(exc)})

    for index, row in result.iterrows():
        game_id = str(row.get("Game_ID") or "")
        game = game_cache.get(game_id)
        if not game:
            continue
        player_name = _norm(row.get("Player"))
        player_id = _int(row.get("Player_MLBAM_ID"))
        if player_id <= 0:
            player_id = int(game["people"].get(player_name) or 0)
        if player_id > 0:
            result.at[index, "Player_MLBAM_ID"] = player_id
            order = game["batting_order"].get(player_id)
            if order:
                result.at[index, "Sequential_Batting_Order"] = order

        pitcher_id = _int(row.get("Opposing_Pitcher_ID"))
        if pitcher_id <= 0:
            team = str(row.get("Team") or "").upper()
            if team and team == game["team_abbrev"].get("home"):
                pitcher_id = int(game["probable_ids"].get("away") or 0)
            elif team and team == game["team_abbrev"].get("away"):
                pitcher_id = int(game["probable_ids"].get("home") or 0)
        if pitcher_id > 0:
            result.at[index, "Opposing_Pitcher_ID"] = pitcher_id
    return result, diagnostics


def _batter_from_dict(row: dict[str, Any]) -> BatterProcessProfile:
    fields = BatterProcessProfile.__dataclass_fields__
    return BatterProcessProfile(**{key: value for key, value in row.items() if key in fields})


def _pitcher_from_dict(row: dict[str, Any]) -> PitcherProcessProfile:
    fields = PitcherProcessProfile.__dataclass_fields__
    return PitcherProcessProfile(**{key: value for key, value in row.items() if key in fields})


def _direct_from_dict(row: dict[str, Any]) -> DirectMatchupProcess:
    fields = DirectMatchupProcess.__dataclass_fields__
    return DirectMatchupProcess(**{key: value for key, value in row.items() if key in fields})


def _profile_freshness(manifest: dict[str, Any], run_date: str) -> str:
    if not manifest or str(manifest.get("run_date") or "") != run_date:
        return "STALE_OR_MISSING"
    failures = manifest.get("failures") or []
    total = int(manifest.get("candidate_identities") or 0)
    if total and len(failures) / total > 0.25:
        return "DEGRADED"
    return "FRESH"


def _team_expected_runs_direct(row: pd.Series) -> float | None:
    for key in ("Team_Expected_Runs", "Expected_Team_Runs", "Team_Run_Projection"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _team_expected_runs_map(frame: pd.DataFrame, run_date: str) -> dict[tuple[str, str], float]:
    """Resolve opportunity-only team scoring state without leaking into PA quality."""
    output: dict[tuple[str, str], float] = {}
    try:
        payload = json.loads(DEFAULT_GAME_SIM_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if isinstance(payload, dict) and str(payload.get("run_date") or "") == run_date:
        games = payload.get("games") or payload.get("predictions") or []
        if isinstance(games, list):
            for game in games:
                if not isinstance(game, dict):
                    continue
                game_id = str(game.get("game_id") or game.get("Game_ID") or "")
                home = str(game.get("home_team") or game.get("home") or "").upper()
                away = str(game.get("away_team") or game.get("away") or "").upper()
                home_runs = _float(game.get("home_expected_runs") or game.get("expected_home_runs"))
                away_runs = _float(game.get("away_expected_runs") or game.get("expected_away_runs"))
                if game_id and home and home_runs is not None:
                    output[(game_id, home)] = home_runs
                if game_id and away and away_runs is not None:
                    output[(game_id, away)] = away_runs

    if {"Game_ID", "Team", "Target", "Prediction"}.issubset(frame.columns):
        run_rows = frame.loc[frame["Target"].astype(str).str.upper().eq("R")].copy()
        run_rows["_prediction"] = pd.to_numeric(run_rows["Prediction"], errors="coerce")
        for (game_id, team), part in run_rows.groupby(["Game_ID", "Team"]):
            key = (str(game_id), str(team).upper())
            if key in output:
                continue
            values = part["_prediction"].dropna()
            if not values.empty:
                estimate = float(values.sum())
                if 1.5 <= estimate <= 9.5:
                    output[key] = estimate
    return output


def _pitch_compatibility_score(batter: BatterProcessProfile, pitcher: PitcherProcessProfile) -> tuple[float, dict[str, Any]]:
    """Match hitter pitch-type xwOBA to the starter's actual arsenal usage."""
    batter_by_pitch = batter.pitch_type_xwoba or {}
    arsenal = pitcher.arsenal or {}
    weighted = 0.0
    weight_sum = 0.0
    detail: dict[str, Any] = {}
    for pitch_type, metrics in arsenal.items():
        if not isinstance(metrics, dict):
            continue
        usage = _float(metrics.get("usage"), 0.0) or 0.0
        batter_xwoba = _float(batter_by_pitch.get(pitch_type))
        pitcher_xwoba = _float(metrics.get("xwoba_allowed_contact"))
        if usage <= 0 or batter_xwoba is None:
            continue
        matchup_xwoba = batter_xwoba if pitcher_xwoba is None else 0.58 * batter_xwoba + 0.42 * pitcher_xwoba
        weighted += usage * matchup_xwoba
        weight_sum += usage
        detail[pitch_type] = {
            "usage": usage,
            "batter_xwoba": batter_xwoba,
            "pitcher_xwoba_allowed": pitcher_xwoba,
            "matchup_xwoba": matchup_xwoba,
        }
    if weight_sum <= 0:
        return 0.0, {"status": "UNAVAILABLE", "pitch_types": {}}
    xwoba = weighted / weight_sum
    score = max(-1.5, min(1.5, (xwoba - 0.320) / 0.12))
    return score, {"status": "AVAILABLE", "weighted_matchup_xwoba": xwoba, "coverage": min(1.0, weight_sum), "pitch_types": detail}


def _numeric_columns() -> tuple[str, ...]:
    return (
        "Sequential_PA_Raw_Probability", "Sequential_PA_Calibrated_Probability", "Sequential_PA_Usable_Probability",
        "Sequential_PA_Probability_LCB", "Sequential_PA_Probability_SE", "Sequential_PA_Uncertainty", "Sequential_PA_Support",
        "Sequential_PA_Expected_PA", "Sequential_PA_Expected_AB", "Sequential_PA_Expected_H", "Sequential_PA_Expected_TB",
        "Sequential_PA_P_H_0", "Sequential_PA_P_H_1", "Sequential_PA_P_H_GE_2", "Sequential_PA_P_TB_0", "Sequential_PA_P_TB_1",
        "Sequential_PA_P_TB_GE_2", "Sequential_PA_P_HR_GE_1", "Game_Conditioned_Prior_Probability",
        "Game_Conditioned_Candidate_Probability", "Game_Conditioned_Production_Probability", "Game_Conditioned_Probability_LCB",
        "Game_Conditioned_Residual_Logit", "Game_Conditioned_Evidence_Strength", "Game_Conditioned_Pitch_Compatibility",
    )


def enrich_pool_with_sequential_pa(
    *,
    pool_csv: Path,
    run_date: str,
    advanced_root: Path = DEFAULT_ADVANCED_ROOT,
    refresh_data: bool = True,
    trials: int = 20000,
) -> dict[str, Any]:
    frame = pd.read_csv(pool_csv)
    hydrated, identity_diagnostics = hydrate_pool_identities(frame)
    hydrated.to_csv(pool_csv, index=False)

    refresh_manifest: dict[str, Any] = {}
    if refresh_data:
        try:
            refresh_manifest = refresh_advanced_profiles(pool_csv=pool_csv, run_date=run_date, advanced_root=advanced_root)
        except Exception as exc:
            refresh_manifest = {
                "run_date": run_date,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "ADVANCED_REFRESH_FAILED",
                "failures": [{"entity": "refresh", "error": str(exc)}],
            }

    batter_payload, pitcher_payload, matchup_payload, manifest = load_profile_partition(advanced_root, run_date)
    if not manifest:
        manifest = refresh_manifest
    freshness = _profile_freshness(manifest, run_date)
    batters = batter_payload.get("profiles") if isinstance(batter_payload, dict) else {}
    pitchers = pitcher_payload.get("profiles") if isinstance(pitcher_payload, dict) else {}
    matchups = matchup_payload.get("matchups") if isinstance(matchup_payload, dict) else {}
    batters = batters if isinstance(batters, dict) else {}
    pitchers = pitchers if isinstance(pitchers, dict) else {}
    matchups = matchups if isinstance(matchups, dict) else {}

    model_artifact = load_model_artifact()
    team_runs = _team_expected_runs_map(hydrated, run_date)
    output = hydrated.copy()
    columns: dict[str, Any] = {
        "Sequential_PA_Model_Version": "",
        "Sequential_PA_Structural_Model_Version": "",
        "Sequential_PA_Status": "NOT_APPLICABLE",
        "Sequential_PA_Support_Status": "",
        "Sequential_PA_Calibration_Status": "",
        "Sequential_PA_Uncertainty_Components": "",
        "Sequential_PA_Diagnostics": "",
        "Game_Conditioned_Model_Version": "",
        "Game_Conditioned_Authority": "",
        "Game_Conditioned_Expert_Weights": "",
        "Game_Conditioned_Expert_Signals": "",
        "Game_Conditioned_Expert_Activations": "",
        "Game_Conditioned_Expert_Contributions": "",
        "Game_Conditioned_Prior_Diagnostics": "",
        "Game_Conditioned_Pitch_Compatibility_Diagnostics": "",
    }
    for column in _numeric_columns():
        columns[column] = float("nan")
    for column, default in columns.items():
        if column not in output.columns:
            output[column] = default
    for column in _numeric_columns():
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(float)

    evaluated = 0
    ready = 0
    blocked = 0
    rows_report: list[dict[str, Any]] = []
    supported_targets = set(GAME_CONDITIONED_TARGETS)
    for index, row in output.iterrows():
        target = str(row.get("Target") or "").upper()
        if target not in supported_targets or str(row.get("Player_Type") or "").lower() != "hitter":
            continue
        evaluated += 1
        output.at[index, "Sequential_PA_Model_Version"] = GAME_CONDITIONED_MODEL_VERSION
        output.at[index, "Sequential_PA_Structural_Model_Version"] = STRUCTURAL_MODEL_VERSION
        output.at[index, "Game_Conditioned_Model_Version"] = GAME_CONDITIONED_MODEL_VERSION
        batter_id = _int(row.get("Player_MLBAM_ID"))
        pitcher_id = _int(row.get("Opposing_Pitcher_ID"))
        batter_data = batters.get(str(batter_id)) if batter_id > 0 else None
        pitcher_data = pitchers.get(str(pitcher_id)) if pitcher_id > 0 else None
        missing: list[str] = []
        if batter_id <= 0:
            missing.append("BATTER_MLBAM_ID")
        if pitcher_id <= 0:
            missing.append("OPPOSING_PITCHER_MLBAM_ID")
        if not batter_data:
            missing.append("BATTER_STATCAST_PROFILE")
        if not pitcher_data:
            missing.append("PITCHER_STATCAST_PROFILE")
        if freshness == "STALE_OR_MISSING":
            missing.append("ADVANCED_DATA_FRESHNESS")
        if missing:
            blocked += 1
            output.at[index, "Sequential_PA_Status"] = "BLOCKED_DATA"
            output.at[index, "Sequential_PA_Diagnostics"] = json.dumps({"missing_components": missing}, sort_keys=True)
            rows_report.append({"row": int(index), "player": str(row.get("Player") or ""), "target": target, "status": "BLOCKED_DATA", "missing": missing})
            continue

        batter = _batter_from_dict(batter_data)
        pitcher = _pitcher_from_dict(pitcher_data)
        pitcher = replace(
            pitcher,
            projected_ip=_float(row.get("Projected_IP"), pitcher.projected_ip),
            projected_pitches=_float(row.get("Projected_Pitches"), pitcher.projected_pitches),
        )
        direct_data = matchups.get(f"{batter_id}:{pitcher_id}")
        direct = _direct_from_dict(direct_data) if isinstance(direct_data, dict) else None
        batting_order = _int(row.get("Sequential_Batting_Order"), 0) or _int(row.get("Batting_Order"), 0) or None
        is_home = str(row.get("Is_Home") or "0").strip().lower() in {"1", "true", "yes"}
        park_factor = _float(row.get("Park_Factor"), 1.0) or 1.0
        game_id = str(row.get("Game_ID") or "")
        team = str(row.get("Team") or "").upper()
        expected_team_runs = _team_expected_runs_direct(row)
        if expected_team_runs is None:
            expected_team_runs = team_runs.get((game_id, team))
        temperature_f = _float(row.get("Temperature_F"), _float(row.get("Temperature")))

        context_missing: list[str] = []
        if batter.xba is None or batter.xslg is None:
            context_missing.append("BATTER_EXPECTED_CONTACT_METRICS")
        if pitcher.xba_allowed is None or pitcher.xslg_allowed is None:
            context_missing.append("PITCHER_EXPECTED_CONTACT_METRICS")
        if pitcher.xfip is None:
            context_missing.append("PITCHER_XFIP")
        if pitcher.siera is None:
            context_missing.append("PITCHER_SIERA")
        if expected_team_runs is None:
            context_missing.append("TEAM_RUN_ENVIRONMENT")
        if temperature_f is None:
            context_missing.append("WEATHER_TEMPERATURE")

        context = AdvancedCandidateContext(
            game_id=game_id, run_date=run_date, batter=batter, pitcher=pitcher,
            direct_matchup=direct, batting_order=batting_order, is_home=is_home,
            team_expected_runs=expected_team_runs, park_factor=float(park_factor),
            defense_residual=0.0,
            defense_status="AVERAGE_CONTEXT_RESIDUAL_ONLY_UNTIL_SPECIFIC_OAA_IS_AVAILABLE",
            data_freshness_status=freshness, missing_components=tuple(context_missing),
            temperature_f=temperature_f,
        )
        default_line = 1.5 if target == "TB" else 0.5
        market_line = float(_float(row.get("Market_Line"), default_line) or default_line)
        structural = simulate_hitter_market(context, target=target, market_line=market_line, side="OVER", trials=trials)
        pitch_compatibility, pitch_detail = _pitch_compatibility_score(batter, pitcher)
        expert_state = build_expert_state(
            context,
            structural,
            target=target,
            pitch_compatibility_score=pitch_compatibility,
        )

        target_model = ((model_artifact.get("targets") or {}).get(target) or {}) if isinstance(model_artifact, dict) else {}
        legacy_weight = _float(target_model.get("prior_legacy_weight"), 0.72) or 0.72
        prior_probability, prior_diagnostics = choose_prior_probability(
            legacy_projection=max(0.0, _float(row.get("Prediction"), 0.0) or 0.0),
            market_line=market_line,
            over_price=row.get("Market_Over_Price"),
            under_price=row.get("Market_Under_Price"),
            legacy_weight=legacy_weight,
        )
        conditioned = condition_probability(
            prior_probability,
            target=target,
            state=expert_state,
            artifact=model_artifact,
            sequential_uncertainty=structural.uncertainty,
        )

        ready += 1
        status = "READY" if structural.support_status == "SUPPORTED" and freshness == "FRESH" else "WEAK_SUPPORT"
        output.at[index, "Sequential_PA_Status"] = status
        output.at[index, "Sequential_PA_Raw_Probability"] = structural.raw_structural_probability
        output.at[index, "Sequential_PA_Calibrated_Probability"] = conditioned.candidate_probability
        output.at[index, "Sequential_PA_Usable_Probability"] = conditioned.production_probability
        output.at[index, "Sequential_PA_Probability_LCB"] = min(conditioned.production_probability, conditioned.lower_bound_probability)
        output.at[index, "Sequential_PA_Probability_SE"] = structural.probability_standard_error
        output.at[index, "Sequential_PA_Uncertainty"] = structural.uncertainty
        output.at[index, "Sequential_PA_Support"] = structural.support
        output.at[index, "Sequential_PA_Support_Status"] = structural.support_status
        output.at[index, "Sequential_PA_Calibration_Status"] = conditioned.authority_status
        output.at[index, "Sequential_PA_Expected_PA"] = structural.expected_pa
        output.at[index, "Sequential_PA_Expected_AB"] = structural.expected_ab
        output.at[index, "Sequential_PA_Expected_H"] = structural.expected_hits
        output.at[index, "Sequential_PA_Expected_TB"] = structural.expected_tb
        output.at[index, "Sequential_PA_P_H_0"] = structural.p_h_0
        output.at[index, "Sequential_PA_P_H_1"] = structural.p_h_1
        output.at[index, "Sequential_PA_P_H_GE_2"] = structural.p_h_ge_2
        output.at[index, "Sequential_PA_P_TB_0"] = structural.p_tb_0
        output.at[index, "Sequential_PA_P_TB_1"] = structural.p_tb_1
        output.at[index, "Sequential_PA_P_TB_GE_2"] = structural.p_tb_ge_2
        output.at[index, "Sequential_PA_P_HR_GE_1"] = structural.p_hr_ge_1
        output.at[index, "Sequential_PA_Uncertainty_Components"] = json.dumps(structural.uncertainty_components, sort_keys=True)

        output.at[index, "Game_Conditioned_Prior_Probability"] = conditioned.prior_probability
        output.at[index, "Game_Conditioned_Candidate_Probability"] = conditioned.candidate_probability
        output.at[index, "Game_Conditioned_Production_Probability"] = conditioned.production_probability
        output.at[index, "Game_Conditioned_Probability_LCB"] = conditioned.lower_bound_probability
        output.at[index, "Game_Conditioned_Residual_Logit"] = conditioned.residual_logit
        output.at[index, "Game_Conditioned_Evidence_Strength"] = conditioned.evidence_strength
        output.at[index, "Game_Conditioned_Pitch_Compatibility"] = pitch_compatibility
        output.at[index, "Game_Conditioned_Authority"] = conditioned.authority_status
        output.at[index, "Game_Conditioned_Expert_Weights"] = json.dumps(conditioned.expert_weights, sort_keys=True)
        output.at[index, "Game_Conditioned_Expert_Signals"] = json.dumps(conditioned.expert_signals, sort_keys=True)
        output.at[index, "Game_Conditioned_Expert_Activations"] = json.dumps(conditioned.expert_activations, sort_keys=True)
        output.at[index, "Game_Conditioned_Expert_Contributions"] = json.dumps(conditioned.expert_contributions, sort_keys=True)
        output.at[index, "Game_Conditioned_Prior_Diagnostics"] = json.dumps(prior_diagnostics, sort_keys=True)
        output.at[index, "Game_Conditioned_Pitch_Compatibility_Diagnostics"] = json.dumps(pitch_detail, sort_keys=True)
        output.at[index, "Sequential_PA_Diagnostics"] = json.dumps({
            "structural": structural.diagnostics,
            "game_conditioned": {
                "prior_probability": conditioned.prior_probability,
                "candidate_probability": conditioned.candidate_probability,
                "production_probability": conditioned.production_probability,
                "residual_logit": conditioned.residual_logit,
                "expert_weights": conditioned.expert_weights,
                "expert_contributions": conditioned.expert_contributions,
                "state": expert_state.diagnostics,
                "pitch_compatibility": pitch_detail,
            },
        }, sort_keys=True)

        rows_report.append({
            "row": int(index), "player": str(row.get("Player") or ""), "target": target, "status": status,
            "raw_probability": structural.raw_structural_probability,
            "prior_probability": conditioned.prior_probability,
            "game_conditioned_probability": conditioned.candidate_probability,
            "usable_probability": conditioned.production_probability,
            "lcb": min(conditioned.production_probability, conditioned.lower_bound_probability),
            "p_h_0": structural.p_h_0, "p_hr_ge_1": structural.p_hr_ge_1,
            "expected_pa": structural.expected_pa,
            "expected_h": structural.expected_hits, "expected_tb": structural.expected_tb,
            "support": structural.support, "uncertainty": structural.uncertainty,
            "authority": conditioned.authority_status,
            "expert_weights": conditioned.expert_weights,
            "pitch_compatibility": pitch_compatibility,
        })

    output.to_csv(pool_csv, index=False)
    target_validation = {}
    for target in GAME_CONDITIONED_TARGETS:
        model = ((model_artifact.get("targets") or {}).get(target) or {}) if isinstance(model_artifact, dict) else {}
        validation = model.get("validation") or {}
        target_validation[target] = {
            "positive_authority": bool(model.get("positive_authority", False)),
            "diagnostic_gate_passed": bool(validation.get("statistical_gate_passed", False)),
            "validation_status": validation.get("status"),
        }

    report = {
        "schema_version": "mlb_game_conditioned_hitter_daily_enrichment_v2",
        "model_version": GAME_CONDITIONED_MODEL_VERSION,
        "structural_model_version": STRUCTURAL_MODEL_VERSION,
        "run_date": run_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pool_csv": str(pool_csv),
        "evaluated_h_tb_hr_rows": evaluated,
        "modeled_rows": ready,
        "blocked_rows": blocked,
        "data_freshness_status": freshness,
        "identity": identity_diagnostics,
        "advanced_manifest": manifest,
        "model_artifact": {
            "training_status": model_artifact.get("training_status") if isinstance(model_artifact, dict) else None,
            "evidence_class": model_artifact.get("evidence_class") if isinstance(model_artifact, dict) else None,
            "target_authority": target_validation,
        },
        "authority": {
            "raw_probability_source": "sequential_plate_appearance_simulation",
            "calibrated_probability_source": "legacy_market_logit_prior_plus_game_conditioned_residual_moe",
            "per_game_weighting": "global_coefficients_x_game_specific_expert_activation",
            "promotion_status": "TARGET_SPECIFIC_EVIDENCE_GATED",
            "positive_adjustment_requires_exact_point_in_time_evidence": True,
            "negative_adjustment_requires_expanding_window_diagnostic_gate": True,
        },
        "rows": rows_report,
    }
    return report
