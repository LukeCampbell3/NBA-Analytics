#!/usr/bin/env python3
"""
MLB Settlement One-Shot

Settles only real live entry + real live close evidence. Outcomes are resolved
from MLB StatsAPI when games are final, with Data-Proc-MLB as a stat fallback
only after a final MLB game is identified.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"))

MLB_SHADOW_DIR = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"
LEDGER_PATH = MLB_SHADOW_DIR / "mlb_evidence_ledger.csv"
SETTLEMENT_REPORT_PATH = MLB_SHADOW_DIR / "settlement_report.json"
SETTLEMENT_BACKUP_DIR = MLB_SHADOW_DIR / "backups" / "settlement"
OUTPUTS_DIR = MLB_SHADOW_DIR / "outputs"
SETTLEMENT_READINESS_REPORT_PATH = OUTPUTS_DIR / "mlb_settlement_readiness_report.json"
DATA_PROC_MLB_DIR = WORKSPACE / "Player-Predictor" / "Data-Proc-MLB"

STAT_MAP = {
    "pitcher_strikeouts": ("pitching", "strikeOuts", "SO"),
    "pitcher_hits_allowed": ("pitching", "hits", "H"),
    "pitcher_earned_runs": ("pitching", "earnedRuns", "ER"),
    "batter_hits": ("batting", "hits", "H"),
    "batter_total_bases": ("batting", "totalBases", "TB"),
    "batter_rbis": ("batting", "rbi", "RBI"),
    "batter_runs": ("batting", "runs", "R"),
    "batter_strikeouts": ("batting", "strikeOuts", "SO"),
}

TEAM_ALIASES = {
    "ARI": {"ARI", "AZ", "ARIZONA", "DIAMONDBACKS"},
    "ATL": {"ATL", "ATLANTA", "BRAVES"},
    "BAL": {"BAL", "BALTIMORE", "ORIOLES"},
    "BOS": {"BOS", "BOSTON", "RED SOX", "REDSOX"},
    "CHC": {"CHC", "CUBS", "CHICAGO CUBS"},
    "CWS": {"CWS", "CHW", "WHITE SOX", "WHITESOX", "CHICAGO WHITE SOX"},
    "CIN": {"CIN", "CINCINNATI", "REDS"},
    "CLE": {"CLE", "CLEVELAND", "GUARDIANS"},
    "COL": {"COL", "COLORADO", "ROCKIES"},
    "DET": {"DET", "DETROIT", "TIGERS"},
    "HOU": {"HOU", "HOUSTON", "ASTROS"},
    "KC": {"KC", "KCR", "KANSAS CITY", "ROYALS"},
    "LAA": {"LAA", "ANGELS", "LOS ANGELES ANGELS"},
    "LAD": {"LAD", "DODGERS", "LOS ANGELES DODGERS"},
    "MIA": {"MIA", "MIAMI", "MARLINS"},
    "MIL": {"MIL", "MILWAUKEE", "BREWERS"},
    "MIN": {"MIN", "MINNESOTA", "TWINS"},
    "NYM": {"NYM", "METS", "NEW YORK METS"},
    "NYY": {"NYY", "YANKEES", "NEW YORK YANKEES"},
    "ATH": {"ATH", "OAK", "ATHLETICS"},
    "PHI": {"PHI", "PHILADELPHIA", "PHILLIES"},
    "PIT": {"PIT", "PITTSBURGH", "PIRATES"},
    "SD": {"SD", "SDP", "SAN DIEGO", "PADRES"},
    "SEA": {"SEA", "SEATTLE", "MARINERS"},
    "SF": {"SF", "SFG", "SAN FRANCISCO", "GIANTS"},
    "STL": {"STL", "ST. LOUIS", "SAINT LOUIS", "CARDINALS"},
    "TB": {"TB", "TBR", "TAMPA BAY", "RAYS"},
    "TEX": {"TEX", "TEXAS", "RANGERS"},
    "TOR": {"TOR", "TORONTO", "BLUE JAYS", "BLUEJAYS"},
    "WSH": {"WSH", "WAS", "WSN", "WASHINGTON", "NATIONALS"},
}


def _json_default(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def _norm(value: Any) -> str:
    value = str(value or "").upper()
    value = re.sub(r"[^A-Z0-9 ]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _norm_name(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9 ]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _team_tokens(value: Any) -> set[str]:
    raw = _norm(value)
    tokens = {raw, raw.replace(" ", "")}
    for canonical, aliases in TEAM_ALIASES.items():
        normalized_aliases = {_norm(alias) for alias in aliases} | {_norm(alias).replace(" ", "") for alias in aliases}
        if raw in normalized_aliases or raw.replace(" ", "") in normalized_aliases:
            tokens.add(canonical)
            tokens.update(normalized_aliases)
    return {token for token in tokens if token}


def _team_matches(row_team: Any, api_team: Dict[str, Any]) -> bool:
    wanted = _team_tokens(row_team)
    names = {
        api_team.get("abbreviation"),
        api_team.get("teamName"),
        api_team.get("name"),
        api_team.get("shortName"),
        api_team.get("clubName"),
    }
    api_tokens = set()
    for name in names:
        api_tokens.update(_team_tokens(name))
    return bool(wanted & api_tokens)


def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _schedule_for_date(event_date: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if event_date not in cache:
        cache[event_date] = _fetch_json(
            "https://statsapi.mlb.com/api/v1/schedule",
            {"sportId": 1, "date": event_date, "hydrate": "team,linescore"},
        )
    return cache[event_date]


def _boxscore(game_pk: int, cache: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if game_pk not in cache:
        cache[game_pk] = _fetch_json(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore")
    return cache[game_pk]


def _is_final_game(game: Dict[str, Any]) -> bool:
    status = game.get("status", {})
    abstract = str(status.get("abstractGameState", "")).lower()
    detailed = str(status.get("detailedState", "")).lower()
    return abstract == "final" or detailed in {"final", "game over", "completed early"}


def _find_final_game(row: pd.Series, schedule_cache: Dict[str, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        from audit_mlb_settlement_matching import find_game_match
        match = find_game_match(row, schedule_cache)
    except Exception as exc:
        return None, f"statsapi_schedule_error:{type(exc).__name__}"

    if not match.get("selected"):
        reason = str(match.get("reason", "game_not_found"))
        return None, "ambiguous_game_match" if reason == "ambiguous_game_match" else reason
    if not match.get("is_final", False):
        return None, "game_not_final"
    return match.get("raw_game"), ""


def _extract_stat_from_boxscore(boxscore: Dict[str, Any], row: pd.Series) -> Tuple[Optional[float], str]:
    market_type = str(row.get("market_type", ""))
    stat_spec = STAT_MAP.get(market_type)
    if not stat_spec:
        return None, "unsupported_market_for_settlement"
    stat_group, statsapi_key, _ = stat_spec
    wanted_name = _norm_name(row.get("player_name", ""))

    for side in ("home", "away"):
        players = boxscore.get("teams", {}).get(side, {}).get("players", {})
        for player in players.values():
            person = player.get("person", {})
            if _norm_name(person.get("fullName", "")) != wanted_name:
                continue
            stats = player.get("stats", {}).get(stat_group, {})
            if not stats:
                return None, "lineup_role_mismatch"
            value = stats.get(statsapi_key)
            if value is None:
                return None, "stat_not_available"
            try:
                return float(value), ""
            except (TypeError, ValueError):
                return None, "stat_not_numeric"
    return None, "player_not_found"


def _lookup_data_proc_stat(row: pd.Series) -> Optional[float]:
    market_type = str(row.get("market_type", ""))
    stat_spec = STAT_MAP.get(market_type)
    if not stat_spec or not DATA_PROC_MLB_DIR.exists():
        return None
    _, _, data_proc_col = stat_spec
    player_dir = DATA_PROC_MLB_DIR / str(row.get("player_name", "")).replace(" ", "_")
    if not player_dir.exists():
        return None
    event_date = str(row.get("event_date", ""))[:10]
    for path in sorted(player_dir.glob("*processed*.csv"), reverse=True):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Date" not in df.columns or data_proc_col not in df.columns:
            continue
        dates = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        match = df[dates == event_date]
        if match.empty:
            continue
        value = pd.to_numeric(match.iloc[0][data_proc_col], errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


def _resolve_outcomes(awaiting: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    schedule_cache: Dict[str, Dict[str, Any]] = {}
    boxscore_cache: Dict[int, Dict[str, Any]] = {}
    final_game_pks: set[int] = set()
    failures: Counter[str] = Counter()

    for _, row in awaiting.iterrows():
        game, reason = _find_final_game(row, schedule_cache)
        base = {
            "game_id": str(row.get("game_id", "")),
            "player_id": str(row.get("player_id", "")),
            "player_name": str(row.get("player_name", "")),
            "market_type": str(row.get("market_type", "")),
            "settlement_source": "mlb_statsapi",
        }
        if game is None:
            failures[reason] += 1
            outcomes.append({**base, "actual_value": None, "is_final": False, "failure_reason": reason})
            continue

        game_pk = int(game["gamePk"])
        final_game_pks.add(game_pk)
        try:
            actual, stat_reason = _extract_stat_from_boxscore(_boxscore(game_pk, boxscore_cache), row)
        except Exception as exc:
            actual, stat_reason = None, f"statsapi_boxscore_error:{type(exc).__name__}"

        if actual is None:
            data_proc_actual = _lookup_data_proc_stat(row)
            if data_proc_actual is not None:
                outcomes.append({**base, "actual_value": data_proc_actual, "is_final": True, "settlement_source": "data_proc_mlb_final_game_verified"})
                continue
            failures[stat_reason or "stat_not_resolved"] += 1
            outcomes.append({**base, "actual_value": None, "is_final": True, "failure_reason": stat_reason or "stat_not_resolved"})
            continue

        outcomes.append({**base, "actual_value": actual, "is_final": True})

    resolver_report = {
        "final_games_available": len(final_game_pks),
        "settlement_failures_by_reason": dict(failures),
    }
    return outcomes, resolver_report


def _breakdown(df: pd.DataFrame, column: str) -> Dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].fillna("").value_counts().to_dict().items()}


def _find_game_for_readiness(row: pd.Series, schedule_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from audit_mlb_settlement_matching import find_game_match
        return find_game_match(row, schedule_cache)
    except Exception as exc:
        return {
            "selected": False,
            "reason": f"statsapi_schedule_error:{type(exc).__name__}",
            "statsapi_game_pk": "",
            "status": "",
            "statsapi_game_date": "",
            "is_final": False,
        }


def build_settlement_readiness_report(awaiting: pd.DataFrame, now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    schedule_cache: Dict[str, Dict[str, Any]] = {}
    game_rows: Dict[str, Dict[str, Any]] = {}
    statsapi_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for _, row in awaiting.iterrows():
        provider_game_id = str(row.get("game_id", ""))
        match = _find_game_for_readiness(row, schedule_cache)
        statsapi_game_pk = str(match.get("statsapi_game_pk") or "")
        status = str(match.get("status") or match.get("reason") or "unknown")
        scheduled_start = str(match.get("statsapi_game_date") or row.get("event_start_time", "") or row.get("entry_time", ""))
        reason = "ready_final_game" if match.get("selected") and match.get("is_final") else str(match.get("reason") or "game_not_final")
        key = f"{provider_game_id}|{statsapi_game_pk or 'unmatched'}"

        if key not in game_rows:
            game_rows[key] = {
                "provider_game_id": provider_game_id,
                "statsapi_game_pk": statsapi_game_pk,
                "game_status": status,
                "scheduled_start_time": scheduled_start,
                "last_checked_time": now.isoformat(),
                "rows_blocked": 0,
                "match_selected": bool(match.get("selected")),
                "is_final": bool(match.get("is_final")),
                "blocking_reason": reason,
            }
        game_rows[key]["rows_blocked"] += 1
        statsapi_counts[statsapi_game_pk or "unmatched"] += 1
        status_counts[status] += 1
        reason_counts[reason] += 1

    provider_counts = awaiting.get("game_id", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict() if not awaiting.empty else {}
    next_check = None if awaiting.empty else (now + timedelta(minutes=60)).isoformat()
    report = {
        "computed_at": now.isoformat(),
        "status_only_available": True,
        "awaiting_settlement_rows": int(len(awaiting)),
        "unresolved_rows_by_game_id": {str(k): int(v) for k, v in provider_counts.items()},
        "unresolved_rows_by_statsapi_gamePk": dict(statsapi_counts),
        "game_status_breakdown": dict(status_counts),
        "blocking_reason_breakdown": dict(reason_counts),
        "games": sorted(game_rows.values(), key=lambda row: (str(row["scheduled_start_time"]), str(row["provider_game_id"]))),
        "rows_blocked_by_each_game": {
            key: int(value["rows_blocked"])
            for key, value in sorted(game_rows.items(), key=lambda item: item[0])
        },
        "next_recommended_settlement_check_time": next_check,
        "staking_enabled": False,
        "promotion_allowed": False,
    }
    write_settlement_readiness_report(report)
    return report


def write_settlement_readiness_report(report: Dict[str, Any]) -> None:
    SETTLEMENT_READINESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTLEMENT_READINESS_REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")


def _outcomes_would_settle(outcomes: List[Dict[str, Any]]) -> bool:
    for outcome in outcomes:
        if not bool(outcome.get("is_final", False)):
            continue
        if outcome.get("void") or outcome.get("did_not_participate"):
            return True
        actual = outcome.get("actual_value")
        if actual is not None and not pd.isna(actual):
            return True
    return False


def _total_settled_gold_rows(ledger: pd.DataFrame) -> int:
    if ledger.empty:
        return 0
    status = ledger.get("status", pd.Series(dtype=str)).fillna("").astype(str)
    tier = ledger.get("evidence_tier", pd.Series(dtype=str)).fillna("").astype(str)
    return int(((status == "settled_gold") | (tier == "settled_gold")).sum())


def _unresolved_rows_count(ledger: pd.DataFrame) -> int:
    if ledger.empty:
        return 0
    status = ledger.get("status", pd.Series(dtype=str)).fillna("").astype(str)
    result = ledger.get("result", pd.Series(dtype=str)).fillna("").astype(str)
    return int(((status == "awaiting_settlement") | (result == "unresolved")).sum())


def _current_terminal_state(ledger: pd.DataFrame) -> str:
    try:
        from production_status_reporter import determine_terminal_state
        return determine_terminal_state(ledger, {})
    except Exception:
        total_settled = _total_settled_gold_rows(ledger)
        if total_settled > 0:
            return "MLB_ACCUMULATING_GOLD_EVIDENCE" if total_settled < 500 else "MLB_READY_FOR_REPLAY_VALIDATION"
        close_rows = int(ledger.get("evidence_tier", pd.Series(dtype=str)).fillna("").astype(str).isin(["live_close"]).sum()) if not ledger.empty else 0
        if close_rows > 0:
            return "MLB_WAITING_FOR_SETTLEMENT"
        live_rows = int(ledger.get("evidence_tier", pd.Series(dtype=str)).fillna("").astype(str).isin(["live_entry"]).sum()) if not ledger.empty else 0
        return "MLB_WAITING_FOR_CLOSE_LINES" if live_rows else "MLB_WAITING_FOR_FRESH_PROPS"


def settle_evidence_once(status_only: bool = False) -> Dict[str, Any]:
    from evidence_lifecycle import settle_evidence_rows
    from snapshot_identity import is_settlement_gold_eligible

    now = datetime.now(timezone.utc)
    result: Dict[str, Any] = {
        "sport": "MLB",
        "settled_at": now.isoformat(),
        "awaiting_settlement_rows": 0,
        "final_games_available": 0,
        "settled_rows": 0,
        "win_count": 0,
        "loss_count": 0,
        "push_count": 0,
        "void_count": 0,
        "unresolved_count": 0,
        "settlement_failures_by_reason": {},
        "settled_gold_rows_written": 0,
        "new_settled_rows_this_run": 0,
        "total_settled_gold_rows": 0,
        "unresolved_rows": 0,
        "market_type_breakdown": {},
        "book_breakdown": {},
        "terminal_state": "MLB_WAITING_FOR_SETTLEMENT",
        "failure_reason": "",
        "ledger_backup_path": None,
        "status_only": bool(status_only),
        "settlement_readiness_report_path": str(SETTLEMENT_READINESS_REPORT_PATH),
    }

    if not LEDGER_PATH.exists():
        result["failure_reason"] = "no_ledger"
        _write_report(result)
        return result

    ledger_before = pd.read_csv(LEDGER_PATH)
    result["total_settled_gold_rows"] = _total_settled_gold_rows(ledger_before)
    result["unresolved_rows"] = _unresolved_rows_count(ledger_before)
    result["terminal_state"] = _current_terminal_state(ledger_before)
    awaiting = ledger_before[ledger_before.get("status", pd.Series(dtype=str)) == "awaiting_settlement"].copy()
    result["awaiting_settlement_rows"] = int(len(awaiting))
    readiness_report = build_settlement_readiness_report(awaiting, now)
    result["settlement_readiness_summary"] = {
        "awaiting_settlement_rows": readiness_report["awaiting_settlement_rows"],
        "game_status_breakdown": readiness_report["game_status_breakdown"],
        "blocking_reason_breakdown": readiness_report["blocking_reason_breakdown"],
        "next_recommended_settlement_check_time": readiness_report["next_recommended_settlement_check_time"],
    }
    if status_only:
        result["failure_reason"] = "status_only"
        _write_report(result)
        return result
    if awaiting.empty:
        result["failure_reason"] = "no_rows_awaiting_settlement"
        _write_report(result)
        return result

    eligible = awaiting[awaiting.apply(is_settlement_gold_eligible, axis=1)].copy()
    if eligible.empty:
        result["failure_reason"] = "no_settlement_eligible_rows"
        _write_report(result)
        return result

    outcomes, resolver_report = _resolve_outcomes(eligible)
    result.update(resolver_report)
    if not outcomes:
        result["failure_reason"] = "no_outcomes_available"
        _write_report(result)
        return result

    if not _outcomes_would_settle(outcomes):
        result["failure_reason"] = "no_new_final_resolved_outcomes" if result["total_settled_gold_rows"] > 0 else "no_final_resolved_outcomes"
        result["unresolved_count"] = int(len(eligible))
        result["unresolved_rows"] = _unresolved_rows_count(ledger_before)
        result["terminal_state"] = _current_terminal_state(ledger_before)
        _write_report(result)
        return result

    result["ledger_backup_path"] = _backup_ledger(now)
    settle_evidence_rows(outcomes)
    ledger_after = pd.read_csv(LEDGER_PATH)

    before_settled_ids = set(
        ledger_before.loc[
            ledger_before.get("status", pd.Series(dtype=str)) == "settled_gold",
            "entry_row_id",
        ].astype(str)
    ) if "entry_row_id" in ledger_before.columns else set()
    settled_now = ledger_after[
        (ledger_after.get("status", pd.Series(dtype=str)) == "settled_gold")
        & ~ledger_after.get("entry_row_id", pd.Series(dtype=str)).astype(str).isin(before_settled_ids)
    ].copy()
    unresolved_now = ledger_after[
        (ledger_after.get("status", pd.Series(dtype=str)) == "awaiting_settlement")
        & (ledger_after.get("result", pd.Series(dtype=str)) == "unresolved")
    ].copy()

    results = settled_now.get("result", pd.Series(dtype=str)).astype(str)
    result["settled_rows"] = int(len(settled_now))
    result["win_count"] = int((results == "win").sum())
    result["loss_count"] = int((results == "loss").sum())
    result["push_count"] = int((results == "push").sum())
    result["void_count"] = int((results == "void").sum())
    result["unresolved_count"] = int(len(unresolved_now))
    result["unresolved_rows"] = _unresolved_rows_count(ledger_after)
    result["settled_gold_rows_written"] = int(len(settled_now))
    result["new_settled_rows_this_run"] = int(len(settled_now))
    result["total_settled_gold_rows"] = _total_settled_gold_rows(ledger_after)
    result["market_type_breakdown"] = _breakdown(settled_now, "market_type")
    result["book_breakdown"] = _breakdown(settled_now, "book")
    if result["settled_rows"] == 0 and result["unresolved_count"] > 0:
        result["failure_reason"] = "no_new_final_resolved_outcomes" if result["total_settled_gold_rows"] > 0 else "no_final_resolved_outcomes"

    try:
        from production_status_reporter import build_production_status
        status = build_production_status()
        result["terminal_state"] = status.get("terminal_state", result["terminal_state"])
    except Exception:
        result["terminal_state"] = _current_terminal_state(ledger_after)

    _write_report(result)
    return result


def _write_report(report: Dict[str, Any]) -> None:
    SETTLEMENT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTLEMENT_REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")


def _backup_ledger(now: datetime) -> str:
    SETTLEMENT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = SETTLEMENT_BACKUP_DIR / f"mlb_evidence_ledger_pre_settlement_{now.strftime('%Y%m%dT%H%M%SZ')}.csv"
    shutil.copy2(LEDGER_PATH, backup_path)
    return str(backup_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Settle MLB evidence once, or only report settlement readiness.")
    parser.add_argument("--status-only", action="store_true", help="Write settlement readiness/status reports without backing up or mutating the ledger.")
    args = parser.parse_args()
    result = settle_evidence_once(status_only=args.status_only)
    print(json.dumps(result, indent=2, default=_json_default))
    sys.exit(0)


if __name__ == "__main__":
    main()
