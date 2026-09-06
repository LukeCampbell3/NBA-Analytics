from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import data_layer as base


def _read_existing(advanced_root: Path, run_date: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return base.load_profile_partition(advanced_root, run_date)


def _valid_same_day_partition(manifest: dict[str, Any], run_day: date) -> bool:
    return bool(
        manifest
        and manifest.get("run_date") == run_day.isoformat()
        and manifest.get("effective_as_of_date") == (run_day - timedelta(days=1)).isoformat()
        and manifest.get("schema_version") == base.ADVANCED_SCHEMA_VERSION
    )


def _fangraphs_map_with_status(fn: Any, season: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        frame = fn(season, season, qual=0)
    except Exception as exc:  # network/upstream path
        return {}, {
            "status": "UNAVAILABLE",
            "rows": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "required_for_base_statcast_model": False,
        }
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Name" not in frame.columns:
        return {}, {
            "status": "UNAVAILABLE",
            "rows": 0,
            "error": "FanGraphs-compatible pitching_stats returned no named rows",
            "required_for_base_statcast_model": False,
        }
    mapping = {base._normalize_name(row.get("Name")): row.to_dict() for _, row in frame.iterrows()}
    available_fields = [field for field in ("ERA", "FIP", "xFIP", "SIERA", "xERA") if field in frame.columns]
    return mapping, {
        "status": "SUCCESS",
        "rows": int(len(frame)),
        "available_fields": available_fields,
        "required_for_base_statcast_model": False,
    }


def refresh_advanced_profiles_incremental(
    *,
    pool_csv: Path,
    run_date: str,
    advanced_root: Path = base.DEFAULT_ADVANCED_ROOT,
    lookback_days: int = base.PROFILE_LOOKBACK_DAYS,
    max_candidates: int = 80,
) -> dict[str, Any]:
    """Refresh only missing active-slate advanced profiles for one as-of date.

    Same-date partitions are safe to reuse because their effective cutoff is
    fixed at D-1. A new run date always gets a new partition/cutoff. Raw
    Statcast frames remain in pybaseball's cache rather than Git.
    """
    run_day = date.fromisoformat(run_date)
    effective_day = run_day - timedelta(days=1)
    start_day = max(date(run_day.year, 3, 1), run_day - timedelta(days=max(30, int(lookback_days))))
    partition = advanced_root / run_date.replace("-", "")
    partition.mkdir(parents=True, exist_ok=True)
    batter_path = partition / "batter_profiles.json"
    pitcher_path = partition / "pitcher_profiles.json"
    matchup_path = partition / "bvp_process.json"
    manifest_path = partition / "manifest.json"

    identities = base.read_pool_candidate_identities(pool_csv)[: max(1, int(max_candidates))]
    old_batter, old_pitcher, old_matchup, old_manifest = _read_existing(advanced_root, run_date)
    reuse_allowed = _valid_same_day_partition(old_manifest, run_day)

    batters: dict[str, dict[str, Any]] = dict((old_batter.get("profiles") or {}) if reuse_allowed else {})
    pitchers: dict[str, dict[str, Any]] = dict((old_pitcher.get("profiles") or {}) if reuse_allowed else {})
    matchups: dict[str, dict[str, Any]] = dict((old_matchup.get("matchups") or {}) if reuse_allowed else {})
    reused_batters = len(batters)
    reused_pitchers = len(pitchers)
    reused_matchups = len(matchups)

    needed_batters = {int(row["batter_id"] or 0) for row in identities if int(row["batter_id"] or 0) > 0}
    needed_pitchers = {int(row["pitcher_id"] or 0) for row in identities if int(row["pitcher_id"] or 0) > 0}
    missing_batters = sorted(pid for pid in needed_batters if str(pid) not in batters)
    missing_pitchers = sorted(pid for pid in needed_pitchers if str(pid) not in pitchers)
    missing_matchup_keys = {
        f"{int(row['batter_id'])}:{int(row['pitcher_id'])}"
        for row in identities
        if int(row["batter_id"] or 0) > 0 and int(row["pitcher_id"] or 0) > 0
        and f"{int(row['batter_id'])}:{int(row['pitcher_id'])}" not in matchups
    }

    tools = base._load_pybaseball()
    fg_map: dict[str, dict[str, Any]] = {}
    old_fg_status = ((old_manifest.get("source_status") or {}).get("fangraphs") or {}) if reuse_allowed else {}
    if missing_pitchers:
        fg_map, fangraphs_status = _fangraphs_map_with_status(tools["pitching_stats"], run_day.year)
    elif old_fg_status:
        fangraphs_status = dict(old_fg_status)
        fangraphs_status["status"] = "REUSED_SAME_ASOF" if old_fg_status.get("status") == "SUCCESS" else old_fg_status.get("status", "UNAVAILABLE")
    else:
        # No new pitcher requires FanGraphs this refresh; do not make a needless live call.
        fangraphs_status = {"status": "NOT_NEEDED_NO_NEW_PITCHERS", "rows": 0, "required_for_base_statcast_model": False}

    failures: list[dict[str, Any]] = []
    batter_frames: dict[int, pd.DataFrame] = {}
    new_batter_fetches = 0
    new_pitcher_fetches = 0
    new_matchups = 0

    by_batter: dict[int, dict[str, Any]] = {}
    by_pitcher: dict[int, dict[str, Any]] = {}
    for identity in identities:
        batter_id = int(identity["batter_id"] or 0)
        pitcher_id = int(identity["pitcher_id"] or 0)
        if batter_id > 0:
            by_batter.setdefault(batter_id, identity)
        if pitcher_id > 0:
            by_pitcher.setdefault(pitcher_id, identity)

    # Fetch each missing batter once. Also fetch a reused batter only when a new
    # direct BvP process pair requires the underlying pitch-level history.
    bvp_batters = {int(key.split(":", 1)[0]) for key in missing_matchup_keys}
    batter_fetch_ids = sorted(set(missing_batters) | bvp_batters)
    for batter_id in batter_fetch_ids:
        identity = by_batter.get(batter_id, {"batter_name": str(batter_id)})
        try:
            frame = base._safe_statcast_fetch(
                tools["statcast_batter"], start_day.isoformat(), effective_day.isoformat(), batter_id
            )
            batter_frames[batter_id] = frame
            if str(batter_id) not in batters:
                profile = base.build_batter_profile(
                    frame,
                    player_id=batter_id,
                    player_name=str(identity.get("batter_name") or batter_id),
                    as_of_date=run_date,
                )
                batters[str(batter_id)] = profile.to_dict()
                new_batter_fetches += 1
        except Exception as exc:
            failures.append({"source": "baseball_savant_statcast", "entity": "batter", "player_id": batter_id, "error": f"{type(exc).__name__}: {exc}"})

    for pitcher_id in missing_pitchers:
        identity = by_pitcher.get(pitcher_id, {"pitcher_name": str(pitcher_id)})
        try:
            frame = base._safe_statcast_fetch(
                tools["statcast_pitcher"], start_day.isoformat(), effective_day.isoformat(), pitcher_id
            )
            profile = base.build_pitcher_profile(
                frame,
                player_id=pitcher_id,
                player_name=str(identity.get("pitcher_name") or pitcher_id),
                as_of_date=run_date,
            )
            profile = base._attach_fangraphs_pitching(
                profile, fg_map.get(base._normalize_name(identity.get("pitcher_name")))
            )
            pitchers[str(pitcher_id)] = profile.to_dict()
            new_pitcher_fetches += 1
        except Exception as exc:
            failures.append({"source": "baseball_savant_statcast", "entity": "pitcher", "player_id": pitcher_id, "error": f"{type(exc).__name__}: {exc}"})

    for identity in identities:
        batter_id = int(identity["batter_id"] or 0)
        pitcher_id = int(identity["pitcher_id"] or 0)
        key = f"{batter_id}:{pitcher_id}"
        if batter_id <= 0 or pitcher_id <= 0 or key in matchups:
            continue
        frame = batter_frames.get(batter_id)
        if frame is None:
            continue
        direct = base.build_direct_matchup(frame, batter_id=batter_id, pitcher_id=pitcher_id)
        if direct is not None:
            matchups[key] = direct.to_dict()
            new_matchups += 1

    fetched_at = datetime.now(timezone.utc).isoformat()
    effective_as_of = effective_day.isoformat()
    statcast_failures = [row for row in failures if row.get("source") == "baseball_savant_statcast"]
    required_entities = len(needed_batters) + len(needed_pitchers)
    covered_entities = sum(1 for pid in needed_batters if str(pid) in batters) + sum(1 for pid in needed_pitchers if str(pid) in pitchers)
    if required_entities == 0:
        statcast_status = "NO_REAL_HTB_CANDIDATES"
    elif covered_entities == required_entities and not statcast_failures:
        statcast_status = "SUCCESS"
    elif covered_entities > 0:
        statcast_status = "DEGRADED"
    else:
        statcast_status = "FAILED"

    source_status = {
        "baseball_savant_statcast": {
            "status": statcast_status,
            "required_entities": required_entities,
            "covered_entities": covered_entities,
            "failures": len(statcast_failures),
            "effective_as_of_date": effective_as_of,
        },
        "fangraphs": {**fangraphs_status, "effective_as_of_date": effective_as_of},
        "mlb_stats_api_identity": {"status": "RESOLVED_BEFORE_ADVANCED_REFRESH", "effective_as_of_date": run_date},
    }

    common = {
        "schema_version": base.ADVANCED_SCHEMA_VERSION,
        "run_date": run_date,
        "effective_as_of_date": effective_as_of,
        "fetched_at_utc": fetched_at,
    }
    batter_payload = {**common, "source": base.SOURCE_STATCAST, "profiles": batters}
    pitcher_payload = {
        **common,
        "source": [base.SOURCE_STATCAST, base.SOURCE_FANGRAPHS],
        "source_status": source_status,
        "profiles": pitchers,
    }
    matchup_payload = {**common, "source": base.SOURCE_STATCAST, "matchups": matchups}
    batter_path.write_text(json.dumps(batter_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pitcher_path.write_text(json.dumps(pitcher_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matchup_path.write_text(json.dumps(matchup_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        **common,
        "lookback_start_date": start_day.isoformat(),
        "sources": [base.SOURCE_STATCAST, base.SOURCE_FANGRAPHS],
        "source_status": source_status,
        "candidate_identities": len(identities),
        "batter_profiles": len(batters),
        "pitcher_profiles": len(pitchers),
        "direct_matchups": len(matchups),
        "failures": failures,
        "cache": {
            "same_asof_partition_reused": reuse_allowed,
            "reused_batter_profiles": reused_batters,
            "reused_pitcher_profiles": reused_pitchers,
            "reused_direct_matchups": reused_matchups,
            "new_batter_statcast_fetches": new_batter_fetches,
            "new_pitcher_statcast_fetches": new_pitcher_fetches,
            "new_direct_matchups": new_matchups,
            "new_date_requires_new_partition": True,
        },
        "paths": {
            "batter_profiles": str(batter_path.relative_to(base.REPO_ROOT)),
            "pitcher_profiles": str(pitcher_path.relative_to(base.REPO_ROOT)),
            "bvp_process": str(matchup_path.relative_to(base.REPO_ROOT)),
        },
        "freshness_policy": {
            "max_profile_age_days": base.MAX_PROFILE_AGE_DAYS,
            "minimum_fresh_profile_pa": base.MIN_FRESH_PROFILE_PA,
            "stale_profiles_may_not_silently_authorize": True,
            "historical_cutoff": "strictly_before_run_date",
        },
        "defense_layer": {
            "status": "AVERAGE_CONTEXT_RESIDUAL_ONLY_UNTIL_SPECIFIC_OAA_IS_AVAILABLE",
            "default_residual": 0.0,
            "fabricated_oaa_allowed": False,
        },
        "sprint_speed": {
            "status": "NOT_APPLIED_AS_SEPARATE_RESIDUAL_WHEN_STATCAST_EXPECTED_METRICS_ARE_USED",
            "double_count_prevention": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
