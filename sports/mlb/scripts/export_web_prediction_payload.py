#!/usr/bin/env python3
"""
Convert the MLB high-precision selection artifacts into the web payload consumed by
the MLB predictions pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
DEFAULT_OUT = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"
DEFAULT_OUT_DIST = REPO_ROOT / "dist" / "mlb" / "data" / "daily_predictions.json"
MLB_MANIFEST_PATH = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB" / "update_manifest_2026.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.parlay_analysis import annotate_parlay_board, evaluate_historical_parlays


MLB_STATS_API_ROOT = "https://statsapi.mlb.com/api/v1"
MLB_LIVE_FEED_ROOT = "https://statsapi.mlb.com/api/v1.1/game"
MLB_HEADSHOT_BASE_URL = "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{person_id}/headshot/67/current"
MLB_HEADSHOT_FALLBACK_URL = "https://midfield.mlbstatic.com/v1/people/{person_id}/headshot/67/current"
MLB_API_TIMEOUT_SECONDS = 5
ENABLE_PLAYER_SEARCH_FALLBACK = False
MLB_PARLAY_VALIDATION_CACHE = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "mlb_parlay_validation.json"
ENABLE_PARLAY_VALIDATION_REBUILD = False
STALE_DATA_REVIEW_DAYS = 14
STALE_DATA_WITHHOLD_DAYS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the MLB web prediction payload.")
    parser.add_argument(
        "--daily-runs-root",
        type=Path,
        default=DEFAULT_DAILY_RUNS_ROOT,
        help="Root directory containing MLB daily prediction run folders.",
    )
    parser.add_argument("--input-csv", type=Path, default=None, help="High-precision selection CSV.")
    parser.add_argument("--summary-json", type=Path, default=None, help="High-precision selection summary JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Destination web payload JSON.")
    parser.add_argument(
        "--output-dist",
        type=Path,
        default=DEFAULT_OUT_DIST,
        help="Optional destination for the published dist payload JSON.",
    )
    parser.add_argument(
        "--allow-date-regression",
        action="store_true",
        help="Allow an older run_date to overwrite a newer published payload.",
    )
    return parser.parse_args()


def find_latest_selected_csv(daily_runs_root: Path) -> Path:
    def sort_key(path: Path) -> tuple[int, float]:
        for source in (path.stem, path.parent.name):
            digits = "".join(char for char in source if char.isdigit())
            if len(digits) >= 8:
                return int(digits[:8]), path.stat().st_mtime
        return 0, path.stat().st_mtime

    candidates = sorted(
        daily_runs_root.glob("**/daily_prediction_pool_*_high_precision_predictions.csv"),
        key=sort_key,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No MLB high-precision selection CSV was found under {daily_runs_root}"
        )
    return candidates[0]


def infer_summary_path(selected_csv: Path) -> Path:
    return selected_csv.with_name(f"{selected_csv.stem}_summary.json")


def infer_run_date(selected_csv: Path, summary: dict[str, object], rows: list[dict[str, str]]) -> str:
    if rows:
        return str(rows[0].get("Prediction_Run_Date", "")).strip()

    selection = summary.get("selection", {})
    if isinstance(selection, dict):
        history_season = str(selection.get("history_season", "")).strip()
    else:
        history_season = ""

    pool_csv = str(summary.get("pool_csv", "")).strip()
    for source in [pool_csv, selected_csv.name]:
        digits = "".join(char for char in source if char.isdigit())
        if len(digits) >= 8:
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"

    return history_season


def infer_through_date(summary: dict[str, object], rows: list[dict[str, str]]) -> str:
    through_date = max((str(row.get("Last_History_Date", "")).strip() for row in rows), default="")
    if through_date:
        return through_date

    pool_csv = Path(str(summary.get("pool_csv", "")).strip())
    if not pool_csv.exists():
        return ""
    try:
        pool_rows = load_rows(pool_csv)
    except (OSError, csv.Error):
        return ""
    return max((str(row.get("Last_History_Date", "")).strip() for row in pool_rows), default="")


def read_payload_run_date(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("run_date", "")).strip()


def assert_no_date_regression(run_date: str, output_paths: list[Path], allow_regression: bool) -> None:
    if allow_regression or not run_date:
        return
    incoming = parse_date(run_date)
    if not incoming:
        return
    for path in output_paths:
        existing_text = read_payload_run_date(path)
        existing = parse_date(existing_text)
        if existing and existing.date() > incoming.date():
            raise RuntimeError(
                f"Refusing to overwrite newer payload {path} ({existing_text}) "
                f"with older run {run_date}. Pass --allow-date-regression to override."
            )


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_date(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10])
    except ValueError:
        return None


def days_between(start_value: object, end_value: object) -> int | None:
    start = parse_date(start_value)
    end = parse_date(end_value)
    if not start or not end:
        return None
    return max(0, (end.date() - start.date()).days)


def row_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    return to_float(row.get(key), default=default)


def prop_identity_key(row: dict[str, str]) -> tuple[str, ...]:
    """User-facing betting identity; intentionally excludes Game_ID for suspended-game slates."""
    return (
        normalize_player_name(str(row.get("Player", ""))),
        str(row.get("Team", "")).strip().upper(),
        str(row.get("Opponent", "")).strip().upper(),
        str(row.get("Game_Date", "")).strip(),
        str(row.get("Target", "")).strip().upper(),
        str(row.get("Direction", "")).strip().upper(),
        f"{row_float(row, 'Market_Line'):.3f}",
    )


def row_context_score(
    row: dict[str, str],
    game_context_lookup: dict[str, dict[str, object]] | None = None,
) -> tuple[float, float]:
    if not game_context_lookup:
        return 0.0, 0.0
    context = game_context_lookup.get(str(row.get("Game_ID", "")).strip(), {}) or {}
    official_date = str(context.get("official_date", "")).strip()
    market_date = str(row.get("Game_Date", "")).strip()
    date_match = 1.0 if official_date and official_date == market_date else 0.0
    players = context.get("players", {}) if isinstance(context.get("players"), dict) else {}
    participant = players.get(normalize_player_name(str(row.get("Player", ""))), {}) or {}
    team_match = 1.0 if participant and str(participant.get("team", "")).strip().upper() == str(row.get("Team", "")).strip().upper() else 0.0
    return date_match, team_match


def row_selection_score(
    row: dict[str, str],
    game_context_lookup: dict[str, dict[str, object]] | None = None,
) -> tuple[float, float, float, float, float, float]:
    date_match, team_match = row_context_score(row, game_context_lookup)
    return (
        date_match,
        team_match,
        row_float(row, "Selection_Score"),
        row_float(row, "Expected_Value_Per_Unit", default=-999.0),
        row_float(row, "Precision_Score"),
        row_float(row, "Abs_Edge"),
    )


def suppress_duplicate_props(
    rows: list[dict[str, str]],
    game_context_lookup: dict[str, dict[str, object]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    kept: dict[tuple[str, ...], dict[str, str]] = {}
    suppressed: list[dict[str, object]] = []
    for row in rows:
        key = prop_identity_key(row)
        existing = kept.get(key)
        if existing is None:
            kept[key] = row
            continue
        incumbent_score = row_selection_score(existing, game_context_lookup)
        challenger_score = row_selection_score(row, game_context_lookup)
        keep_new = challenger_score > incumbent_score
        removed = existing if keep_new else row
        if keep_new:
            kept[key] = row
        suppressed.append(
            {
                "player": str(removed.get("Player", "")).strip(),
                "team": str(removed.get("Team", "")).strip(),
                "opponent": str(removed.get("Opponent", "")).strip(),
                "market_date": str(removed.get("Game_Date", "")).strip(),
                "target": str(removed.get("Target", "")).strip(),
                "direction": str(removed.get("Direction", "")).strip(),
                "market_line": row_float(removed, "Market_Line"),
                "game_id": str(removed.get("Game_ID", "")).strip(),
                "reason": "duplicate prop identity on same slate",
            }
        )
    deduped = list(kept.values())
    deduped.sort(key=lambda row: to_int(row.get("Rank"), default=999999))
    return deduped, suppressed


def suppress_closed_games(
    rows: list[dict[str, str]],
    game_context_lookup: dict[str, dict[str, object]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    if not game_context_lookup:
        return rows, []

    kept: list[dict[str, str]] = []
    suppressed: list[dict[str, object]] = []
    for row in rows:
        game_id = str(row.get("Game_ID", "")).strip()
        context = game_context_lookup.get(game_id, {}) or {}
        abstract_state = str(context.get("abstract_state", "")).strip().lower()
        detailed_status = str(context.get("status", "")).strip()
        if abstract_state not in {"live", "final"}:
            kept.append(row)
            continue
        suppressed.append(
            {
                "player": str(row.get("Player", "")).strip(),
                "team": str(row.get("Team", "")).strip(),
                "opponent": str(row.get("Opponent", "")).strip(),
                "market_date": str(row.get("Game_Date", "")).strip(),
                "target": str(row.get("Target", "")).strip(),
                "direction": str(row.get("Direction", "")).strip(),
                "market_line": row_float(row, "Market_Line"),
                "game_id": game_id,
                "official_game_status": detailed_status,
                "reason": "game is no longer open for pregame predictions",
            }
        )
    return kept, suppressed


def build_data_quality(
    run_date: str,
    through_date: str,
    duplicate_count: int,
    play_count: int | None = None,
) -> dict[str, object]:
    lag_days = days_between(through_date, run_date)
    status = "ready"
    reasons: list[str] = []
    if lag_days is not None and lag_days > STALE_DATA_WITHHOLD_DAYS:
        status = "withheld"
        reasons.append(f"data history is {lag_days} days behind run date")
    elif lag_days is not None and lag_days > STALE_DATA_REVIEW_DAYS:
        status = "review"
        reasons.append(f"data history is {lag_days} days behind run date")
    if duplicate_count:
        if status == "ready":
            status = "review"
        reasons.append(f"{duplicate_count} duplicate prop card{'s' if duplicate_count != 1 else ''} suppressed")
    if play_count == 0:
        status = "withheld"
        reasons.append("no plays passed publication filters")
    return {
        "status": status,
        "lag_days": lag_days,
        "review_threshold_days": STALE_DATA_REVIEW_DAYS,
        "withhold_threshold_days": STALE_DATA_WITHHOLD_DAYS,
        "reasons": reasons,
    }


def is_whole_number_line(value: float) -> bool:
    rounded = round(float(value))
    return abs(float(value) - rounded) < 1e-9


def build_splits(source: dict[str, int], total: int) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key, count in source.items():
        count_int = int(count)
        out[str(key)] = {
            "count": count_int,
            "share": (count_int / total) if total else 0.0,
        }
    return out


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_mlb_parlay_leg_quality(
    *,
    graded_hit_rate: float,
    precision_score: float,
    historical_bucket_support: float,
    historical_bucket_win_rate: float,
    expected_value_per_unit: float | None = None,
) -> float:
    support_score = 0.0
    if historical_bucket_support > 0:
        support_score = math.log1p(max(0.0, historical_bucket_support)) / math.log1p(3000.0)
    ev_score = 0.0
    if expected_value_per_unit is not None:
        ev_score = clamp01((float(expected_value_per_unit) + 0.05) / 0.20)
    return clamp01(
        (0.50 * clamp01(graded_hit_rate))
        + (0.20 * clamp01(precision_score / 1.15))
        + (0.15 * clamp01(support_score))
        + (0.10 * clamp01(historical_bucket_win_rate))
        + (0.05 * ev_score)
    )


def is_mlb_parlay_leg_eligible(
    *,
    graded_hit_rate: float,
    leg_quality: float,
    historical_bucket_support: float,
    expected_value_per_unit: float | None = None,
) -> bool:
    if graded_hit_rate < 0.68 or leg_quality < 0.82:
        return False
    if historical_bucket_support > 0 and historical_bucket_support < 250:
        return False
    if expected_value_per_unit is not None and expected_value_per_unit < -0.02:
        return False
    return True


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    lam = max(0.0, float(lam))
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    log_p = (-lam) + (k * math.log(lam)) - math.lgamma(k + 1)
    return math.exp(log_p)


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    return min(1.0, sum(poisson_pmf(i, lam) for i in range(k + 1)))


def estimate_count_hit_probabilities(prediction: float, market_line: float, direction: str) -> tuple[float, float, float]:
    lam = max(0.0, float(prediction))
    rounded = round(float(market_line))
    is_integer_line = abs(float(market_line) - rounded) < 1e-9

    if is_integer_line:
        push_probability = poisson_pmf(int(rounded), lam)
        if direction == "OVER":
            hit_probability = 1.0 - poisson_cdf(int(rounded), lam)
        else:
            hit_probability = poisson_cdf(int(rounded) - 1, lam)
    else:
        floor_line = math.floor(float(market_line))
        push_probability = 0.0
        if direction == "OVER":
            hit_probability = 1.0 - poisson_cdf(int(floor_line), lam)
        else:
            hit_probability = poisson_cdf(int(floor_line), lam)

    settle_probability = max(1e-9, 1.0 - push_probability)
    graded_hit_rate = hit_probability / settle_probability
    return (
        max(0.0, min(1.0, hit_probability)),
        max(0.0, min(1.0, push_probability)),
        max(0.0, min(1.0, graded_hit_rate)),
    )


def build_mlb_parlay_validation(manifest_path: Path) -> dict:
    if MLB_PARLAY_VALIDATION_CACHE.exists():
        try:
            return json.loads(MLB_PARLAY_VALIDATION_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not ENABLE_PARLAY_VALIDATION_REBUILD:
        return {"available": False, "reason": "parlay validation rebuild disabled for fast web export"}

    if not manifest_path.exists():
        return {"available": False, "reason": f"manifest not found: {manifest_path}"}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "reason": f"failed reading manifest: {exc}"}

    written = manifest.get("written", {})
    if not isinstance(written, dict) or not written:
        return {"available": False, "reason": "manifest does not contain processed MLB player files"}

    target_map = {
        "H": ("H", "Market_H", "H_market_gap"),
        "TB": ("TB", "Market_TB", "TB_market_gap"),
        "R": ("R", "Market_R", "R_market_gap"),
        "HR": ("HR", "Market_HR", "HR_market_gap"),
        "RBI": ("RBI", "Market_RBI", "RBI_market_gap"),
        "K": ("K", "Market_K", "K_market_gap"),
    }
    rows: list[dict] = []

    for player_name, item in written.items():
        raw_path = item.get("path")
        if not raw_path:
            continue
        source_path = Path(raw_path)
        if not source_path.exists():
            fallback = manifest_path.parent / player_name / "2026_processed_processed.csv"
            source_path = fallback if fallback.exists() else source_path
        if not source_path.exists():
            continue

        try:
            frame = pd.read_csv(source_path)
        except Exception:
            continue
        if frame.empty:
            continue

        for _, row in frame.iterrows():
            market_date = str(row.get("Date", "")).strip()
            player = str(row.get("Player", "") or player_name).strip()
            team = str(row.get("Team", "")).strip()
            opponent = str(row.get("Opponent", "")).strip()
            game_id = str(row.get("Game_ID", "")).strip()
            for target, (actual_col, market_col, gap_col) in target_map.items():
                try:
                    market_line = float(row.get(market_col))
                    gap = float(row.get(gap_col))
                    actual = float(row.get(actual_col))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(market_line) or not math.isfinite(gap) or not math.isfinite(actual) or abs(gap) < 1e-9:
                    continue
                prediction = market_line + gap
                direction = "OVER" if gap > 0 else "UNDER"
                _, _, graded_hit_rate = estimate_count_hit_probabilities(prediction, market_line, direction)
                if direction == "OVER":
                    result = "win" if actual > market_line else "push" if actual == market_line else "loss"
                else:
                    result = "win" if actual < market_line else "push" if actual == market_line else "loss"
                rows.append(
                    {
                        "market_date": market_date,
                        "player": player,
                        "player_display_name": player,
                        "team": team,
                        "opponent": opponent,
                        "game_id": game_id,
                        "target": target,
                        "direction": direction,
                        "estimated_graded_hit_rate": graded_hit_rate,
                        "result": result,
                    }
                )

    history = pd.DataFrame(rows)
    if history.empty:
        return {"available": False, "reason": "processed MLB history did not yield usable pair rows"}

    summary = evaluate_historical_parlays(
        history,
        sport="mlb",
        date_col="market_date",
        probability_col="estimated_graded_hit_rate",
        result_col="result",
        max_pairs_per_day=1,
    )
    summary["source_manifest"] = str(manifest_path)
    summary["history_row_count"] = int(len(history))
    MLB_PARLAY_VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    MLB_PARLAY_VALIDATION_CACHE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def normalize_player_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    cleaned = []
    for char in lowered:
        cleaned.append(char if char.isalnum() else " ")
    normalized = " ".join("".join(cleaned).split())
    normalized = normalized.replace(" jr", "").replace(" sr", "")
    normalized = normalized.replace(" ii", "").replace(" iii", "").replace(" iv", "")
    return " ".join(normalized.split())


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=MLB_API_TIMEOUT_SECONDS) as response:
        return json.load(response)


def build_game_context_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    game_ids = sorted({
        str(row.get("Game_ID", "")).strip()
        for row in rows
        if str(row.get("Game_ID", "")).strip()
    })
    for game_id in game_ids:
        try:
            payload = fetch_json(f"{MLB_LIVE_FEED_ROOT}/{game_id}/feed/live")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue

        game_data = payload.get("gameData") or {}
        live_data = payload.get("liveData") or {}
        boxscore = live_data.get("boxscore") or {}
        boxscore_teams = boxscore.get("teams") or {}
        game_teams = game_data.get("teams") or {}
        players: dict[str, dict[str, object]] = {}
        team_abbreviations: dict[str, str] = {}

        for side in ("away", "home"):
            team = game_teams.get(side) or {}
            team_abbr = str(team.get("abbreviation", "")).strip().upper()
            team_abbreviations[side] = team_abbr
            team_boxscore = boxscore_teams.get(side) or {}
            for player in (team_boxscore.get("players") or {}).values():
                batting_order = str(player.get("battingOrder", "")).strip()
                if not batting_order:
                    continue
                person = player.get("person") or {}
                full_name = str(person.get("fullName", "")).strip()
                if not full_name:
                    continue
                try:
                    person_id = int(person.get("id"))
                except (TypeError, ValueError):
                    person_id = None
                players[normalize_player_name(full_name)] = {
                    "player": full_name,
                    "player_mlbam_id": person_id,
                    "team": team_abbr,
                    "batting_order": batting_order,
                }

        game_datetime = game_data.get("datetime") or {}
        game_status = game_data.get("status") or {}
        lookup[game_id] = {
            "official_date": str(game_datetime.get("officialDate", "")).strip(),
            "status": str(game_status.get("detailedState", "")).strip(),
            "abstract_state": str(game_status.get("abstractGameState", "")).strip(),
            "teams": team_abbreviations,
            "players": players,
            "lineup_available": bool(players),
        }
    return lookup


def fetch_team_id_lookup(season: int) -> dict[str, int]:
    url = f"{MLB_STATS_API_ROOT}/teams?{urlencode({'sportId': 1, 'season': season})}"
    payload = fetch_json(url)
    lookup: dict[str, int] = {}
    for team in payload.get("teams", []):
        try:
            team_id = int(team.get("id"))
        except (TypeError, ValueError):
            continue
        abbr = str(team.get("abbreviation", "")).strip().upper()
        if abbr:
            lookup[abbr] = team_id
    return lookup


def fetch_team_roster_lookup(team_id: int, run_date: str) -> dict[str, int]:
    url = f"{MLB_STATS_API_ROOT}/teams/{int(team_id)}/roster?{urlencode({'rosterType': 'active', 'date': run_date, 'hydrate': 'person'})}"
    payload = fetch_json(url)
    lookup: dict[str, int] = {}
    for entry in payload.get("roster", []):
        person = entry.get("person") or {}
        try:
            person_id = int(person.get("id"))
        except (TypeError, ValueError):
            continue
        full_name = str(person.get("fullName", "")).strip()
        if not full_name:
            continue
        lookup[normalize_player_name(full_name)] = person_id
    return lookup


def search_person_id_by_name(player_name: str) -> int | None:
    query = str(player_name or "").strip()
    if not query:
        return None
    url = f"{MLB_STATS_API_ROOT}/people/search?{urlencode({'names': query})}"
    payload = fetch_json(url)
    normalized_query = normalize_player_name(query)
    for person in payload.get("people", []):
        full_name = str(person.get("fullName", "")).strip()
        if normalize_player_name(full_name) != normalized_query:
            continue
        try:
            return int(person.get("id"))
        except (TypeError, ValueError):
            continue
    return None


def build_headshot_url(person_id: int | None) -> str | None:
    if not person_id:
        return None
    return MLB_HEADSHOT_BASE_URL.format(person_id=int(person_id))


def build_headshot_fallback_url(person_id: int | None) -> str | None:
    if not person_id:
        return None
    return MLB_HEADSHOT_FALLBACK_URL.format(person_id=int(person_id))


def build_player_headshot_lookup(rows: list[dict[str, str]], run_date: str) -> dict[tuple[str, str], dict[str, int | str | None]]:
    if not rows or not run_date:
        return {}

    try:
        season = int(str(run_date).split("-", 1)[0])
    except (TypeError, ValueError):
        return {}

    try:
        team_id_lookup = fetch_team_id_lookup(season)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return {}

    teams_needed = {
        str(row.get("Team", "")).strip().upper()
        for row in rows
        if str(row.get("Team", "")).strip()
    }
    roster_by_team: dict[str, dict[str, int]] = {}
    for team_abbr in sorted(teams_needed):
        team_id = team_id_lookup.get(team_abbr)
        if not team_id:
            continue
        try:
            roster_by_team[team_abbr] = fetch_team_roster_lookup(team_id, run_date)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue

    lookup: dict[tuple[str, str], dict[str, int | str | None]] = {}
    for row in rows:
        team_abbr = str(row.get("Team", "")).strip().upper()
        player_name = str(row.get("Player", "")).strip()
        if not team_abbr or not player_name:
            continue
        roster_lookup = roster_by_team.get(team_abbr, {})
        person_id = roster_lookup.get(normalize_player_name(player_name))
        if not person_id and ENABLE_PLAYER_SEARCH_FALLBACK:
            try:
                person_id = search_person_id_by_name(player_name)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
                person_id = None
        lookup[(team_abbr, player_name)] = {
            "player_mlbam_id": person_id,
            "player_headshot_url": build_headshot_url(person_id),
            "player_headshot_fallback_url": build_headshot_fallback_url(person_id),
        }
    return lookup


def main() -> None:
    args = parse_args()
    selected_csv = args.input_csv.resolve() if args.input_csv else find_latest_selected_csv(args.daily_runs_root.resolve())
    summary_json = args.summary_json.resolve() if args.summary_json else infer_summary_path(selected_csv)
    rows = load_rows(selected_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8-sig"))
    through_date = infer_through_date(summary, rows)
    run_date = infer_run_date(selected_csv, summary, rows)
    output_paths = [args.output]
    if args.output_dist:
        output_paths.append(args.output_dist)
    assert_no_date_regression(run_date, output_paths, args.allow_date_regression)
    original_rows = rows
    game_context_lookup = build_game_context_lookup(original_rows)
    rows, suppressed_closed_games = suppress_closed_games(rows, game_context_lookup)
    rows, suppressed_duplicates = suppress_duplicate_props(rows, game_context_lookup)
    total = len(rows)
    data_quality = build_data_quality(run_date, through_date, len(suppressed_duplicates), play_count=total)
    publication_status = str(data_quality.get("status") or "ready")
    multi_game_slate_keys: set[tuple[str, str, str]] = set()
    game_ids_by_slate: dict[tuple[str, str, str], set[str]] = {}
    for row in original_rows:
        teams = sorted([
            str(row.get("Team", "")).strip().upper(),
            str(row.get("Opponent", "")).strip().upper(),
        ])
        if len(teams) != 2 or not teams[0] or not teams[1]:
            continue
        slate_key = (str(row.get("Game_Date", "")).strip(), teams[0], teams[1])
        game_ids_by_slate.setdefault(slate_key, set()).add(str(row.get("Game_ID", "")).strip())
    for slate_key, game_ids in game_ids_by_slate.items():
        if len({game_id for game_id in game_ids if game_id}) > 1:
            multi_game_slate_keys.add(slate_key)

    headshot_lookup = build_player_headshot_lookup(rows, run_date)
    plays = []
    for display_rank, row in enumerate(rows, start=1):
        is_home = to_int(row.get("Is_Home", "0"))
        team = str(row.get("Team", "")).strip()
        opponent = str(row.get("Opponent", "")).strip()
        player_name = str(row.get("Player", "")).strip()
        home_team = team if is_home else opponent
        away_team = opponent if is_home else team
        player_lookup = headshot_lookup.get((team.upper(), player_name), {}) or {}
        game_id = str(row.get("Game_ID", "")).strip()
        game_context = game_context_lookup.get(game_id, {}) or {}
        game_players = game_context.get("players", {}) if isinstance(game_context.get("players"), dict) else {}
        participant = game_players.get(normalize_player_name(player_name), {}) or {}
        participant_id = participant.get("player_mlbam_id")
        resolved_player_id = participant_id or player_lookup.get("player_mlbam_id")
        lineup_available = bool(game_context.get("lineup_available"))
        participant_team = str(participant.get("team", "")).strip().upper()
        official_game_date = str(game_context.get("official_date", "")).strip()
        market_date = str(row.get("Game_Date", "")).strip()
        lineup_status = (
            "confirmed"
            if participant
            else "not_in_posted_lineup"
            if lineup_available
            else "unconfirmed"
        )
        estimated_graded_hit_rate = to_float(row.get("Estimated_Graded_Hit_Rate"))
        precision_score = to_float(row.get("Precision_Score"))
        historical_bucket_support = to_float(row.get("Historical_Bucket_Support"))
        historical_bucket_win_rate = to_float(row.get("Historical_Bucket_Win_Rate"))
        expected_value_per_unit = to_float(row.get("Expected_Value_Per_Unit"), default=float("nan"))
        if not math.isfinite(expected_value_per_unit):
            expected_value_per_unit = None
        market_implied_probability = to_float(row.get("Market_Implied_Probability"), default=float("nan"))
        if not math.isfinite(market_implied_probability):
            market_implied_probability = None
        push_probability = to_float(row.get("Estimated_Push_Probability"), default=0.0)
        market_line = to_float(row.get("Market_Line"))
        slate_teams = sorted([team.upper(), opponent.upper()])
        multi_game_slate = (
            len(slate_teams) == 2
            and (str(row.get("Game_Date", "")).strip(), slate_teams[0], slate_teams[1]) in multi_game_slate_keys
        )
        risk_flags: list[str] = []
        if data_quality.get("lag_days") is not None and int(data_quality["lag_days"]) > STALE_DATA_REVIEW_DAYS:
            risk_flags.append("stale_history")
        if lineup_status != "confirmed":
            risk_flags.append("lineup_unconfirmed")
        if participant_team and participant_team != team.upper():
            risk_flags.append("team_mismatch")
        if official_game_date and market_date and official_game_date != market_date:
            risk_flags.append("game_date_mismatch")
        if not resolved_player_id:
            risk_flags.append("roster_unverified")
        if push_probability >= 0.05 or is_whole_number_line(market_line):
            risk_flags.append("push_exposure")
        if multi_game_slate:
            risk_flags.append("multi_game_slate_review")
        parlay_leg_quality = build_mlb_parlay_leg_quality(
            graded_hit_rate=estimated_graded_hit_rate,
            precision_score=precision_score,
            historical_bucket_support=historical_bucket_support,
            historical_bucket_win_rate=historical_bucket_win_rate,
            expected_value_per_unit=expected_value_per_unit,
        )
        parlay_eligible = (
            publication_status == "ready"
            and not risk_flags
            and is_mlb_parlay_leg_eligible(
                graded_hit_rate=estimated_graded_hit_rate,
                leg_quality=parlay_leg_quality,
                historical_bucket_support=historical_bucket_support,
                expected_value_per_unit=expected_value_per_unit,
            )
        )
        raw_confidence_tier = row.get("Confidence_Tier", "consider")
        plays.append(
            {
                "rank": display_rank,
                "source_rank": to_int(row.get("Rank")),
                "player": player_name,
                "player_display_name": player_name,
                "player_id": row.get("Player_ID", ""),
                "player_mlbam_id": resolved_player_id,
                "player_headshot_url": build_headshot_url(resolved_player_id),
                "player_headshot_fallback_url": build_headshot_fallback_url(resolved_player_id),
                "team": team,
                "opponent": opponent,
                "market_home_team": home_team,
                "market_away_team": away_team,
                "market_date": market_date,
                "official_game_date": official_game_date,
                "commence_time_utc": row.get("Commence_Time_UTC", ""),
                "game_id": game_id,
                "game_status_code": row.get("Game_Status_Code", ""),
                "official_game_status": game_context.get("status", ""),
                "direction": row.get("Direction", ""),
                "target": row.get("Target", ""),
                "prediction": to_float(row.get("Prediction")),
                "market_line": market_line,
                "market_source": row.get("Market_Source", "synthetic"),
                "edge": to_float(row.get("Edge")),
                "abs_edge": to_float(row.get("Abs_Edge")),
                "estimated_hit_probability": to_float(row.get("Estimated_Hit_Probability")),
                "estimated_push_probability": push_probability,
                "estimated_graded_hit_rate": estimated_graded_hit_rate,
                "model_estimate_status": "review" if risk_flags or publication_status != "ready" else "calibrated",
                "precision_score": precision_score,
                "value_score": precision_score * to_float(row.get("Abs_Edge")),
                "historical_bucket_key": row.get("Historical_Bucket_Key", ""),
                "historical_prior_source": row.get("Historical_Prior_Source", ""),
                "historical_bucket_win_rate": historical_bucket_win_rate,
                "historical_bucket_support": historical_bucket_support,
                "market_bucket": row.get("Market_Bucket", ""),
                "market_books": to_int(row.get("Market_Books", "0")),
                "market_line_std": to_float(row.get("Market_Line_Std")),
                "market_implied_probability": market_implied_probability,
                "expected_value_per_unit": expected_value_per_unit,
                "final_pool_quality_score": parlay_leg_quality,
                "parlay_precision_eligible": parlay_eligible,
                "model_confidence_tier": raw_confidence_tier,
                "confidence_tier": "review" if risk_flags or publication_status != "ready" else raw_confidence_tier,
                "publication_status": publication_status,
                "action_status": "review" if risk_flags or publication_status != "ready" else "ready",
                "lineup_status": lineup_status,
                "confirmed_team": participant_team,
                "batting_order": participant.get("batting_order", ""),
                "risk_flags": risk_flags,
            }
        )

    parlay_payload = annotate_parlay_board(
        plays,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
        eligibility_field="parlay_precision_eligible",
    )
    plays = parlay_payload["plays"]
    target_counts: dict[str, int] = {}
    direction_counts: dict[str, int] = {}
    for row in rows:
        target = str(row.get("Target", "")).strip()
        direction = str(row.get("Direction", "")).strip()
        target_counts[target] = target_counts.get(target, 0) + 1
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

    payload = {
        "sport": "MLB",
        "board_title": "MLB Prediction Bounties",
        "run_date": run_date,
        "through_date": through_date,
        "model_run_id": "mlb_high_precision_selector_v2",
        "policy_profile": str(summary.get("publication_strategy") or "core_market_props"),
        "publication_status": publication_status,
        "publication_message": "; ".join(data_quality.get("reasons", [])) or "Board passed publication checks.",
        "data_quality": data_quality,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "play_count": total,
            "source_play_count": len(original_rows),
            "closed_game_cards_suppressed": len(suppressed_closed_games),
            "duplicate_cards_suppressed": len(suppressed_duplicates),
            "supported_rows": int(summary.get("rows_supported", 0)),
            "rows_after_filters": int(summary.get("rows_after_filters", 0)),
            "rejected_rows": max(0, int(summary.get("rows_supported", 0)) - int(summary.get("rows_after_filters", 0))),
            "avg_expected_hit_rate": (sum(to_float(row.get("Estimated_Hit_Probability")) for row in rows) / total) if total else 0.0,
            "avg_graded_hit_rate": (sum(to_float(row.get("Estimated_Graded_Hit_Rate")) for row in rows) / total) if total else 0.0,
            "avg_edge": (sum(to_float(row.get("Edge")) for row in rows) / total) if total else 0.0,
            "avg_abs_edge": (sum(to_float(row.get("Abs_Edge")) for row in rows) / total) if total else 0.0,
            "avg_value_score": (sum(to_float(row.get("Precision_Score")) * to_float(row.get("Abs_Edge")) for row in rows) / total) if total else 0.0,
            "avg_precision_score": (sum(to_float(row.get("Precision_Score")) for row in rows) / total) if total else 0.0,
        },
        "selection": summary.get("selection", {}),
        "filter_rejections": summary.get("filter_rejections", {}),
        "by_target": build_splits(target_counts, total),
        "by_direction": build_splits(direction_counts, total),
        "parlay_summary": parlay_payload["summary"],
        "parlay_pairs": parlay_payload["pairs"],
        "parlay_validation": build_mlb_parlay_validation(MLB_MANIFEST_PATH),
        "suppressed_closed_games": suppressed_closed_games,
        "suppressed_duplicates": suppressed_duplicates,
        "plays": plays,
    }
    payload["summary"]["parlay_tagged_plays"] = int(payload["parlay_summary"].get("tagged_play_count", 0))
    payload["summary"]["parlay_pairs"] = int(payload["parlay_summary"].get("selected_pair_count", 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_dist:
        args.output_dist.parent.mkdir(parents=True, exist_ok=True)
        args.output_dist.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote MLB web payload -> {args.output}")
    if args.output_dist:
        print(f"Wrote MLB dist payload -> {args.output_dist}")


if __name__ == "__main__":
    main()
