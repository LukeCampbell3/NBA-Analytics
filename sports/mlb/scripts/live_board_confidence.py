#!/usr/bin/env python3
"""Build and apply leakage-safe confidence corrections from settled MLB boards."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DAILY_RUNS_ROOT = SPORT_ROOT / "data" / "predictions" / "daily_runs"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_OUTPUT_ROOT = SPORT_ROOT / "data" / "predictions" / "calibration"
DEFAULT_PUBLISHED_HISTORY_ROOT = SPORT_ROOT / "web" / "data" / "history"
DEFAULT_PUBLISHED_CURRENT_JSON = SPORT_ROOT / "web" / "data" / "daily_predictions.json"
TARGET_TO_ACTUAL_COL = {"H": "H", "TB": "TB", "R": "R", "HR": "HR", "RBI": "RBI", "K": "K", "ER": "ER"}
DEFAULT_PRIOR_STRENGTH = 20.0
# Was 0.05 -- far too small a safety cap given what the real settled
# history actually shows. validate_historical_final_pools.py's real
# walk-forward report (source_file_count=25, through 2026-08-11) found
# the live board's own avg_estimated_graded_hit_rate (~74%) sitting
# ~16 points above its real priced hit rate (~39%) overall, and the
# single largest real segment here (TB|OVER, 19 of 26 graded
# calibration rows) specifically off by ~17.5 points -- both well past
# the old 5-point ceiling, so the correction this function itself
# already computes (credibility-weighted toward each segment's own real
# win rate, shrunk by prior_strength for thin segments) was never
# allowed to apply more than a third of the way. Raised to let a real,
# well-evidenced segment correct most of the way to its measured rate;
# min_segment_rows + the credibility weighting below still do the real
# work of not over-reacting to a single thin/noisy segment.
DEFAULT_MAX_ABS_ADJUSTMENT = 0.20
DEFAULT_MIN_SEGMENT_ROWS = 3
CURRENT_PROFILE_REQUIRED_COLUMN = "Selection_Profile"
MAIN_BOARD_PATTERN = re.compile(r"^daily_prediction_pool_(\d{8})_high_precision_predictions\.csv$")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def normalize_player_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", text)


def to_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def is_valid_american_price(value: Any) -> bool:
    price = to_float(value)
    return bool(price is not None and abs(price) >= 100.0 and abs(price - round(price)) <= 1e-6)


def iter_main_board_paths(daily_runs_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in daily_runs_root.glob("*/daily_prediction_pool_*_high_precision_predictions.csv"):
        match = MAIN_BOARD_PATTERN.fullmatch(path.name)
        if match and path.parent.name == match.group(1):
            paths.append(path)
    return sorted(paths)


def published_play_key(row: dict[str, Any], run_date: str | None = None) -> tuple[str, str, str, str, str, float]:
    return (
        str(run_date or row.get("official_game_date") or row.get("market_date") or row.get("Game_Date") or "")[:10],
        re.sub(r"\.0$", "", str(row.get("game_id") or row.get("Game_ID") or "")),
        normalize_player_key(row.get("player_id") or row.get("Player_ID") or row.get("player") or row.get("Player")),
        str(row.get("target") or row.get("Target") or "").strip().upper(),
        str(row.get("direction") or row.get("Direction") or "").strip().upper(),
        round(float(to_float(row.get("market_line") if "market_line" in row else row.get("Market_Line")) or -999.0), 6),
    )


def load_published_play_index(
    history_root: Path,
    current_json: Path | None,
    *,
    before_date: date,
    policy_version: str | None,
) -> dict[tuple[str, str, str, str, str, float], dict[str, Any]]:
    payload_paths = sorted(history_root.glob("*.json"))
    if current_json is not None and current_json.exists():
        payload_paths.append(current_json)
    index: dict[tuple[str, str, str, str, str, float], dict[str, Any]] = {}
    for path in payload_paths:
        if path.name == "index.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_date = str(payload.get("run_date", ""))[:10]
        try:
            if date.fromisoformat(run_date) >= before_date:
                continue
        except ValueError:
            continue
        observed_policy = str(payload.get("policy_profile", "")).strip()
        if policy_version and observed_policy != policy_version:
            continue
        for play in payload.get("plays", []):
            if not isinstance(play, dict):
                continue
            key = published_play_key(play, run_date)
            index[key] = {**play, "policy_version": observed_policy, "run_date": run_date}
    return index


def build_actual_lookup(processed_root: Path, season: int) -> dict[tuple[str, str, str, str], float]:
    lookup: dict[tuple[str, str, str, str], float] = {}
    usecols = ["Date", "Player", "Game_ID", *TARGET_TO_ACTUAL_COL.values()]
    for path in processed_root.glob(f"*/{int(season)}_processed_processed.csv"):
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in usecols)
        except Exception:
            continue
        if frame.empty or not {"Date", "Player", "Game_ID"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["Player_Key"] = frame["Player"].map(normalize_player_key)
        frame["Game_ID"] = frame["Game_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
        for target, actual_col in TARGET_TO_ACTUAL_COL.items():
            if actual_col not in frame.columns:
                continue
            actual = pd.to_numeric(frame[actual_col], errors="coerce")
            mask = frame["Date"].notna() & frame["Player_Key"].ne("") & frame["Game_ID"].ne("") & actual.notna()
            for row, value in zip(frame.loc[mask, ["Date", "Player_Key", "Game_ID"]].itertuples(index=False), actual.loc[mask]):
                lookup[(str(row.Date), str(row.Player_Key), target, str(row.Game_ID))] = float(value)
    return lookup


def grade_result(actual: float, market_line: float, direction: str) -> str:
    if direction == "OVER":
        return "win" if actual > market_line else "push" if actual == market_line else "loss"
    return "win" if actual < market_line else "push" if actual == market_line else "loss"


def diagnose_loss(
    *,
    result: str,
    direction: str,
    actual: float,
    prediction: float | None,
    published_probability: float,
    model_probability: float | None,
    historical_probability: float | None,
) -> list[str]:
    if result != "loss":
        return []
    diagnostics: list[str] = []
    if published_probability >= 0.75:
        diagnostics.append("high_confidence_miss")
    if model_probability is not None and model_probability >= 0.75:
        diagnostics.append("model_overconfidence")
    if historical_probability is not None and historical_probability >= 0.75:
        diagnostics.append("historical_prior_overconfidence")
    if prediction is not None:
        if direction == "UNDER" and actual > prediction:
            diagnostics.append("actual_above_projection")
        elif direction == "OVER" and actual < prediction:
            diagnostics.append("actual_below_projection")
    return diagnostics


def _official_game_actuals(game_id: str) -> dict[tuple[str, str], float]:
    if not str(game_id).isdigit():
        return {}
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    actuals: dict[tuple[str, str], float] = {}
    teams = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
    for side in ("away", "home"):
        for player in teams.get(side, {}).get("players", {}).values():
            player_key = normalize_player_key(player.get("person", {}).get("fullName", ""))
            if not player_key:
                continue
            batting = player.get("stats", {}).get("batting", {})
            pitching = player.get("stats", {}).get("pitching", {})
            values = {
                "H": batting.get("hits"),
                "TB": batting.get("totalBases"),
                "R": batting.get("runs"),
                "HR": batting.get("homeRuns"),
                "RBI": batting.get("rbi"),
                "K": pitching.get("strikeOuts"),
                "ER": pitching.get("earnedRuns"),
            }
            for target, value in values.items():
                numeric = to_float(value)
                if numeric is not None:
                    actuals[(player_key, target)] = numeric
    return actuals


def _segment_payload(
    rows: list[dict[str, Any]],
    *,
    prior_strength: float,
    max_abs_adjustment: float,
    min_segment_rows: int,
) -> dict[str, Any]:
    graded_rows = len(rows)
    wins = sum(int(row["win"]) for row in rows)
    mean_probability = sum(float(row["probability"]) for row in rows) / graded_rows
    actual_rate = wins / graded_rows
    credibility = graded_rows / (graded_rows + max(1.0, float(prior_strength)))
    raw_adjustment = credibility * (actual_rate - mean_probability)
    adjustment = max(-max_abs_adjustment, min(max_abs_adjustment, raw_adjustment))
    return {
        "graded_rows": graded_rows,
        "wins": wins,
        "losses": graded_rows - wins,
        "mean_probability": mean_probability,
        "actual_win_rate": actual_rate,
        "credibility_weight": credibility,
        "adjustment": adjustment if graded_rows >= min_segment_rows else 0.0,
        "active": graded_rows >= min_segment_rows,
    }


def _walk_forward_brier(
    segments: dict[str, list[dict[str, Any]]],
    *,
    prior_strength: float,
    max_abs_adjustment: float,
    min_segment_rows: int,
) -> dict[str, float | int | None]:
    dated_rows = sorted(
        ((str(row["date"]), key, row) for key, rows in segments.items() for row in rows),
        key=lambda item: item[0],
    )
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    before_errors: list[float] = []
    after_errors: list[float] = []
    adjusted_rows = 0
    for evaluation_date in sorted({item[0] for item in dated_rows}):
        day_rows = [item for item in dated_rows if item[0] == evaluation_date]
        for _, key, row in day_rows:
            probability = float(row["probability"])
            actual = float(row["win"])
            segment = _segment_payload(
                history[key],
                prior_strength=prior_strength,
                max_abs_adjustment=max_abs_adjustment,
                min_segment_rows=min_segment_rows,
            ) if history[key] else {"active": False, "adjustment": 0.0}
            adjustment = float(segment.get("adjustment", 0.0) or 0.0)
            adjusted_probability = max(0.01, min(0.99, probability + adjustment))
            before_errors.append((probability - actual) ** 2)
            after_errors.append((adjusted_probability - actual) ** 2)
            adjusted_rows += int(bool(segment.get("active")))
        for _, key, row in day_rows:
            history[key].append(row)
    return {
        "rows": len(before_errors),
        "adjusted_rows": adjusted_rows,
        "brier_score_before": sum(before_errors) / len(before_errors) if before_errors else None,
        "brier_score_after": sum(after_errors) / len(after_errors) if after_errors else None,
    }


def build_live_board_calibration(
    *,
    daily_runs_root: Path,
    processed_root: Path,
    season: int,
    before_date: date,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    max_abs_adjustment: float = DEFAULT_MAX_ABS_ADJUSTMENT,
    min_segment_rows: int = DEFAULT_MIN_SEGMENT_ROWS,
    official_api_fallback: bool = False,
    published_history_root: Path | None = None,
    published_current_json: Path | None = None,
    policy_version: str | None = None,
) -> dict[str, Any]:
    actual_lookup = build_actual_lookup(processed_root, season)
    official_cache: dict[str, dict[tuple[str, str], float]] = {}
    segments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_paths: list[Path] = []
    skipped_legacy_files = 0
    published_index = None
    if published_history_root is not None:
        published_index = load_published_play_index(
            published_history_root,
            published_current_json,
            before_date=before_date,
            policy_version=policy_version,
        )
    matched_published_keys: set[tuple[str, str, str, str, str, float]] = set()
    settled_rows: list[dict[str, Any]] = []

    for path in iter_main_board_paths(daily_runs_root):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or CURRENT_PROFILE_REQUIRED_COLUMN not in frame.columns:
            skipped_legacy_files += 1
            continue
        source_paths.append(path)
        for row in frame.to_dict(orient="records"):
            run_date = str(row.get("Game_Date", ""))[:10]
            try:
                game_date = date.fromisoformat(run_date)
            except ValueError:
                continue
            if game_date >= before_date:
                continue
            target = str(row.get("Target", "")).strip().upper()
            direction = str(row.get("Direction", "")).strip().upper()
            game_id = re.sub(r"\.0$", "", str(row.get("Game_ID", "")))
            player_key = normalize_player_key(row.get("Player_ID") or row.get("Player"))
            line = to_float(row.get("Market_Line"))
            published_probability = to_float(row.get("Estimated_Graded_Hit_Rate"))
            probability = published_probability
            previous_adjustment = to_float(row.get("Live_Confidence_Calibration_Adjustment")) or 0.0
            if probability is not None:
                probability -= previous_adjustment
            price_confirmed = bool(int(to_float(row.get("Price_Confirmed")) or 0))
            side_price = to_float(row.get("Selected_Side_Price"))
            if (
                target not in TARGET_TO_ACTUAL_COL
                or direction not in {"OVER", "UNDER"}
                or line is None
                or probability is None
                or not price_confirmed
                or not is_valid_american_price(side_price)
            ):
                continue
            board_key = published_play_key(row, run_date)
            published_play = published_index.get(board_key) if published_index is not None else None
            if published_index is not None and published_play is None:
                continue
            if published_play is not None:
                matched_published_keys.add(board_key)
            actual = actual_lookup.get((run_date, player_key, target, game_id))
            if actual is None and official_api_fallback:
                if game_id not in official_cache:
                    try:
                        official_cache[game_id] = _official_game_actuals(game_id)
                    except Exception:
                        official_cache[game_id] = {}
                actual = official_cache[game_id].get((normalize_player_key(row.get("Player")), target))
            if actual is None:
                continue
            result = grade_result(float(actual), float(line), direction)
            if result == "push":
                continue
            probability = max(0.01, min(0.99, probability))
            segments[f"{target}|{direction}"].append(
                {"probability": probability, "win": result == "win", "date": run_date}
            )
            profit_if_win = float(side_price) / 100.0 if float(side_price) > 0 else 100.0 / abs(float(side_price))
            settled_rows.append(
                {
                    "run_date": run_date,
                    "policy_version": str((published_play or {}).get("policy_version", policy_version or "unscoped")),
                    "selection_profile": str(row.get("Selection_Profile", "")),
                    "game_id": game_id,
                    "player": str(row.get("Player", "")),
                    "target": target,
                    "direction": direction,
                    "market_line": float(line),
                    "selected_side_price": float(side_price),
                    "prediction": to_float(row.get("Prediction")),
                    "model_hit_probability": to_float(row.get("Model_Hit_Probability")),
                    "published_graded_hit_rate": float(published_probability),
                    "calibration_input_probability": probability,
                    "historical_bucket_win_rate": to_float(row.get("Historical_Bucket_Win_Rate")),
                    "historical_bucket_support": int(to_float(row.get("Historical_Bucket_Support")) or 0),
                    "market_books": int(to_float(row.get("Market_Books")) or 0),
                    "actual": float(actual),
                    "projection_residual": (
                        None if to_float(row.get("Prediction")) is None else float(actual) - float(to_float(row.get("Prediction")))
                    ),
                    "market_result_margin": (
                        float(actual) - float(line) if direction == "OVER" else float(line) - float(actual)
                    ),
                    "result": result,
                    "unit_return": profit_if_win if result == "win" else -1.0,
                    "probability_error": float(published_probability) - float(result == "win"),
                    "loss_diagnostics": diagnose_loss(
                        result=result,
                        direction=direction,
                        actual=float(actual),
                        prediction=to_float(row.get("Prediction")),
                        published_probability=float(published_probability),
                        model_probability=to_float(row.get("Model_Hit_Probability")),
                        historical_probability=to_float(row.get("Historical_Bucket_Win_Rate")),
                    ),
                }
            )

    if published_index is not None:
        for board_key in sorted(set(published_index) - matched_published_keys):
            play = published_index[board_key]
            run_date, game_id, player_key, target, direction, line = board_key
            published_probability = to_float(play.get("estimated_graded_hit_rate"))
            side_price = to_float(play.get("selected_side_price"))
            if (
                published_probability is None
                or side_price is None
                or not is_valid_american_price(side_price)
                or target not in TARGET_TO_ACTUAL_COL
                or direction not in {"OVER", "UNDER"}
            ):
                continue
            previous_adjustment = to_float(play.get("live_confidence_calibration_adjustment")) or 0.0
            probability = max(0.01, min(0.99, published_probability - previous_adjustment))
            actual = actual_lookup.get((run_date, player_key, target, game_id))
            if actual is None and official_api_fallback:
                if game_id not in official_cache:
                    try:
                        official_cache[game_id] = _official_game_actuals(game_id)
                    except Exception:
                        official_cache[game_id] = {}
                actual = official_cache[game_id].get((normalize_player_key(play.get("player")), target))
            if actual is None:
                continue
            result = grade_result(float(actual), float(line), direction)
            if result == "push":
                continue
            matched_published_keys.add(board_key)
            segments[f"{target}|{direction}"].append(
                {"probability": probability, "win": result == "win", "date": run_date}
            )
            profit_if_win = side_price / 100.0 if side_price > 0 else 100.0 / abs(side_price)
            settled_rows.append(
                {
                    "run_date": run_date,
                    "policy_version": str(play.get("policy_version", policy_version or "unscoped")),
                    "selection_profile": str(play.get("selection_profile", "")),
                    "game_id": game_id,
                    "player": str(play.get("player", "")),
                    "target": target,
                    "direction": direction,
                    "market_line": float(line),
                    "selected_side_price": side_price,
                    "prediction": to_float(play.get("prediction")),
                    "model_hit_probability": to_float(play.get("model_hit_probability")),
                    "published_graded_hit_rate": published_probability,
                    "calibration_input_probability": probability,
                    "historical_bucket_win_rate": to_float(play.get("historical_bucket_win_rate")),
                    "historical_bucket_support": int(to_float(play.get("historical_bucket_support")) or 0),
                    "market_books": int(to_float(play.get("market_books")) or 0),
                    "actual": float(actual),
                    "projection_residual": (
                        None if to_float(play.get("prediction")) is None else float(actual) - float(to_float(play.get("prediction")))
                    ),
                    "market_result_margin": (
                        float(actual) - float(line) if direction == "OVER" else float(line) - float(actual)
                    ),
                    "result": result,
                    "unit_return": profit_if_win if result == "win" else -1.0,
                    "probability_error": published_probability - float(result == "win"),
                    "loss_diagnostics": diagnose_loss(
                        result=result,
                        direction=direction,
                        actual=float(actual),
                        prediction=to_float(play.get("prediction")),
                        published_probability=published_probability,
                        model_probability=to_float(play.get("model_hit_probability")),
                        historical_probability=to_float(play.get("historical_bucket_win_rate")),
                    ),
                }
            )

    segment_payloads = {
        key: _segment_payload(
            rows,
            prior_strength=prior_strength,
            max_abs_adjustment=max_abs_adjustment,
            min_segment_rows=min_segment_rows,
        )
        for key, rows in sorted(segments.items())
    }
    all_rows = [row for rows in segments.values() for row in rows]
    before_brier = None
    after_brier = None
    if all_rows:
        before_brier = sum((float(row["probability"]) - float(row["win"])) ** 2 for row in all_rows) / len(all_rows)
        after_brier = sum(
            (
                max(0.01, min(0.99, float(row["probability"]) + float(segment_payloads[key]["adjustment"])))
                - float(row["win"])
            )
            ** 2
            for key, rows in segments.items()
            for row in rows
        ) / len(all_rows)
    walk_forward = _walk_forward_brier(
        segments,
        prior_strength=prior_strength,
        max_abs_adjustment=max_abs_adjustment,
        min_segment_rows=min_segment_rows,
    )
    loss_diagnostic_counts: dict[str, int] = defaultdict(int)
    for row in settled_rows:
        for diagnostic in row.get("loss_diagnostics", []):
            loss_diagnostic_counts[str(diagnostic)] += 1

    return {
        "schema_version": 2,
        "season": int(season),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_before_date": before_date.isoformat(),
        "method": "empirical_bayes_target_direction_residual",
        "profile_compatibility": "exact_published_policy_only" if policy_version else "requires_selection_profile_column",
        "policy_version": policy_version or "unscoped",
        "candidate_universe": "published_frontend_plays_only" if published_index is not None else "selected_board_rows",
        "priced_rows_only": True,
        "prior_strength": float(prior_strength),
        "max_abs_adjustment": float(max_abs_adjustment),
        "min_segment_rows": int(min_segment_rows),
        "source_file_count": len(source_paths),
        "skipped_legacy_file_count": skipped_legacy_files,
        "graded_rows": len(all_rows),
        "published_play_count": len(published_index) if published_index is not None else None,
        "matched_published_play_count": len(matched_published_keys) if published_index is not None else None,
        "unmatched_published_play_count": (
            len(set(published_index) - matched_published_keys) if published_index is not None else None
        ),
        "brier_score_before": before_brier,
        "brier_score_after": after_brier,
        "walk_forward_validation": walk_forward,
        "segments": segment_payloads,
        "settled_rows": sorted(settled_rows, key=lambda row: (row["run_date"], row["game_id"], row["player"])),
        "loss_diagnostic_counts": dict(sorted(loss_diagnostic_counts.items())),
        "daily_runs_root": portable_path(daily_runs_root),
        "processed_root": portable_path(processed_root),
    }


def apply_live_board_calibration(
    probability: float,
    calibration: dict[str, Any] | None,
    *,
    target: str,
    direction: str,
) -> tuple[float, str, int, float]:
    bounded = max(0.0, min(1.0, float(probability)))
    if not isinstance(calibration, dict):
        return bounded, "disabled", 0, 0.0
    key = f"{str(target).strip().upper()}|{str(direction).strip().upper()}"
    segment = calibration.get("segments", {}).get(key, {})
    support = int(segment.get("graded_rows", 0) or 0)
    if not bool(segment.get("active")):
        return bounded, "insufficient_support", support, 0.0
    adjustment = float(segment.get("adjustment", 0.0) or 0.0)
    return max(0.0, min(1.0, bounded + adjustment)), key, support, adjustment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-runs-root", type=Path, default=DEFAULT_DAILY_RUNS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--before-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH)
    parser.add_argument("--max-abs-adjustment", type=float, default=DEFAULT_MAX_ABS_ADJUSTMENT)
    parser.add_argument("--min-segment-rows", type=int, default=DEFAULT_MIN_SEGMENT_ROWS)
    parser.add_argument("--official-api-fallback", action="store_true")
    parser.add_argument("--published-history-root", type=Path, default=DEFAULT_PUBLISHED_HISTORY_ROOT)
    parser.add_argument("--published-current-json", type=Path, default=DEFAULT_PUBLISHED_CURRENT_JSON)
    parser.add_argument("--policy-version", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json or DEFAULT_OUTPUT_ROOT / f"live_board_confidence_calibration_{args.season}.json"
    payload = build_live_board_calibration(
        daily_runs_root=args.daily_runs_root.resolve(),
        processed_root=args.processed_root.resolve(),
        season=args.season,
        before_date=args.before_date,
        prior_strength=args.prior_strength,
        max_abs_adjustment=args.max_abs_adjustment,
        min_segment_rows=args.min_segment_rows,
        official_api_fallback=bool(args.official_api_fallback),
        published_history_root=args.published_history_root.resolve(),
        published_current_json=args.published_current_json.resolve(),
        policy_version=str(args.policy_version).strip() if args.policy_version else None,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Confidence calibration: {output_json}")
    print(f"Graded current-profile rows: {payload['graded_rows']}")
    print(f"Brier score: {payload['brier_score_before']} -> {payload['brier_score_after']}")


if __name__ == "__main__":
    main()
