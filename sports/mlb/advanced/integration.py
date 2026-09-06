from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .data_layer import DEFAULT_ADVANCED_ROOT, load_profile_partition, refresh_advanced_profiles
from .schema import AdvancedCandidateContext, BatterProcessProfile, DirectMatchupProcess, PitcherProcessProfile
from .sequential_pa_model import MODEL_VERSION, simulate_hitter_market

MLB_LIVE_FEED_ROOT = "https://statsapi.mlb.com/api/v1.1/game"
MLB_API_TIMEOUT_SECONDS = 12


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


def hydrate_pool_identities(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    for column in ("Player_MLBAM_ID", "Sequential_Batting_Order"):
        if column not in result.columns:
            result[column] = ""
    if "Opposing_Pitcher_ID" not in result.columns:
        result["Opposing_Pitcher_ID"] = ""

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


def _team_expected_runs(row: pd.Series) -> float | None:
    for key in ("Team_Expected_Runs", "Expected_Team_Runs", "Team_Run_Projection"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


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

    output = hydrated.copy()
    columns = {
        "Sequential_PA_Model_Version": "",
        "Sequential_PA_Status": "NOT_APPLICABLE",
        "Sequential_PA_Raw_Probability": "",
        "Sequential_PA_Calibrated_Probability": "",
        "Sequential_PA_Usable_Probability": "",
        "Sequential_PA_Probability_LCB": "",
        "Sequential_PA_Probability_SE": "",
        "Sequential_PA_Uncertainty": "",
        "Sequential_PA_Support": "",
        "Sequential_PA_Support_Status": "",
        "Sequential_PA_Calibration_Status": "",
        "Sequential_PA_Expected_PA": "",
        "Sequential_PA_Expected_AB": "",
        "Sequential_PA_Expected_H": "",
        "Sequential_PA_Expected_TB": "",
        "Sequential_PA_P_H_0": "",
        "Sequential_PA_P_H_1": "",
        "Sequential_PA_P_H_GE_2": "",
        "Sequential_PA_P_TB_0": "",
        "Sequential_PA_P_TB_1": "",
        "Sequential_PA_P_TB_GE_2": "",
        "Sequential_PA_P_HR_GE_1": "",
        "Sequential_PA_Uncertainty_Components": "",
        "Sequential_PA_Diagnostics": "",
    }
    for column, default in columns.items():
        if column not in output.columns:
            output[column] = default

    evaluated = 0
    ready = 0
    blocked = 0
    rows_report: list[dict[str, Any]] = []
    for index, row in output.iterrows():
        target = str(row.get("Target") or "").upper()
        if target not in {"H", "TB"} or str(row.get("Player_Type") or "").lower() != "hitter":
            continue
        evaluated += 1
        output.at[index, "Sequential_PA_Model_Version"] = MODEL_VERSION
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
        context_missing: list[str] = []
        if batter.xba is None or batter.xslg is None:
            context_missing.append("BATTER_EXPECTED_CONTACT_METRICS")
        if pitcher.xba_allowed is None or pitcher.xslg_allowed is None:
            context_missing.append("PITCHER_EXPECTED_CONTACT_METRICS")
        if pitcher.xfip is None:
            context_missing.append("PITCHER_XFIP")
        if pitcher.siera is None:
            context_missing.append("PITCHER_SIERA")
        context = AdvancedCandidateContext(
            game_id=str(row.get("Game_ID") or ""), run_date=run_date, batter=batter, pitcher=pitcher,
            direct_matchup=direct, batting_order=batting_order, is_home=is_home,
            team_expected_runs=_team_expected_runs(row), park_factor=float(park_factor),
            defense_residual=0.0,
            defense_status="AVERAGE_CONTEXT_RESIDUAL_ONLY_UNTIL_SPECIFIC_OAA_IS_AVAILABLE",
            data_freshness_status=freshness, missing_components=tuple(context_missing),
        )
        market_line = float(_float(row.get("Market_Line"), 0.5) or 0.5)
        result = simulate_hitter_market(context, target=target, market_line=market_line, side="OVER", trials=trials)
        ready += 1
        status = "READY" if result.support_status == "SUPPORTED" and freshness == "FRESH" else "WEAK_SUPPORT"
        output.at[index, "Sequential_PA_Status"] = status
        output.at[index, "Sequential_PA_Raw_Probability"] = result.raw_structural_probability
        output.at[index, "Sequential_PA_Calibrated_Probability"] = result.calibrated_probability
        output.at[index, "Sequential_PA_Usable_Probability"] = result.usable_probability
        output.at[index, "Sequential_PA_Probability_LCB"] = result.probability_lcb
        output.at[index, "Sequential_PA_Probability_SE"] = result.probability_standard_error
        output.at[index, "Sequential_PA_Uncertainty"] = result.uncertainty
        output.at[index, "Sequential_PA_Support"] = result.support
        output.at[index, "Sequential_PA_Support_Status"] = result.support_status
        output.at[index, "Sequential_PA_Calibration_Status"] = result.calibration_status
        output.at[index, "Sequential_PA_Expected_PA"] = result.expected_pa
        output.at[index, "Sequential_PA_Expected_AB"] = result.expected_ab
        output.at[index, "Sequential_PA_Expected_H"] = result.expected_hits
        output.at[index, "Sequential_PA_Expected_TB"] = result.expected_tb
        output.at[index, "Sequential_PA_P_H_0"] = result.p_h_0
        output.at[index, "Sequential_PA_P_H_1"] = result.p_h_1
        output.at[index, "Sequential_PA_P_H_GE_2"] = result.p_h_ge_2
        output.at[index, "Sequential_PA_P_TB_0"] = result.p_tb_0
        output.at[index, "Sequential_PA_P_TB_1"] = result.p_tb_1
        output.at[index, "Sequential_PA_P_TB_GE_2"] = result.p_tb_ge_2
        output.at[index, "Sequential_PA_P_HR_GE_1"] = result.p_hr_ge_1
        output.at[index, "Sequential_PA_Uncertainty_Components"] = json.dumps(result.uncertainty_components, sort_keys=True)
        output.at[index, "Sequential_PA_Diagnostics"] = json.dumps(result.diagnostics, sort_keys=True)
        rows_report.append({
            "row": int(index), "player": str(row.get("Player") or ""), "target": target, "status": status,
            "raw_probability": result.raw_structural_probability, "usable_probability": result.usable_probability,
            "lcb": result.probability_lcb, "p_h_0": result.p_h_0, "expected_pa": result.expected_pa,
            "expected_h": result.expected_hits, "expected_tb": result.expected_tb, "support": result.support,
            "uncertainty": result.uncertainty,
        })

    output.to_csv(pool_csv, index=False)
    report = {
        "schema_version": "mlb_sequential_pa_daily_enrichment_v1",
        "model_version": MODEL_VERSION,
        "run_date": run_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pool_csv": str(pool_csv),
        "evaluated_h_tb_rows": evaluated,
        "modeled_rows": ready,
        "blocked_rows": blocked,
        "data_freshness_status": freshness,
        "identity": identity_diagnostics,
        "advanced_manifest": manifest,
        "authority": {
            "raw_probability_source": "sequential_plate_appearance_simulation",
            "usable_probability": "raw_minus_negative_only_uncertainty_haircut",
            "promotion_status": "NEGATIVE_AUTHORITY_UNTIL_INDEPENDENT_ADVANCED_MODEL_CALIBRATION",
            "can_raise_legacy_probability": False,
            "can_veto_overconfident_h_tb": True,
        },
        "rows": rows_report,
    }
    return report
