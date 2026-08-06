#!/usr/bin/env python3
"""Select one adaptive, OVER-only MLB parlay from the broader daily pool."""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.parlay_analysis import score_candidate_parlays

try:
    from . import select_high_precision_predictions as selector
except ImportError:
    import select_high_precision_predictions as selector


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
CALIBRATION_ROOT = SPORT_ROOT / "data" / "predictions" / "calibration"
POLICY_VERSION = "mlb_over_consistency_parlay_v1"
ALLOWED_TARGETS = ("H", "TB", "R", "RBI", "K")
MIN_LEGS = 2
MAX_LEGS = 4
MIN_LEG_PROBABILITY = 0.62
MIN_TICKET_PROBABILITY = 0.40
MIN_COMBINED_DECIMAL_PRICE = 2.0
MIN_EXPECTED_RETURN = 0.0
MAX_CANDIDATES_PER_BOOK = 12
MLB_LIVE_FEED_ROOT = "https://statsapi.mlb.com/api/v1.1/game"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an adaptive OVER-only MLB consistency parlay.")
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--history-dir", type=Path, default=selector.DEFAULT_HISTORY_DIR)
    parser.add_argument("--min-legs", type=int, default=MIN_LEGS)
    parser.add_argument("--max-legs", type=int, default=MAX_LEGS)
    parser.add_argument("--min-leg-probability", type=float, default=MIN_LEG_PROBABILITY)
    parser.add_argument("--min-ticket-probability", type=float, default=MIN_TICKET_PROBABILITY)
    parser.add_argument("--min-combined-decimal-price", type=float, default=MIN_COMBINED_DECIMAL_PRICE)
    parser.add_argument("--min-expected-return", type=float, default=MIN_EXPECTED_RETURN)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _wilson_interval(wins: int, rows: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if rows <= 0:
        return None, None
    probability = wins / rows
    denominator = 1.0 + (z * z / rows)
    center = (probability + z * z / (2.0 * rows)) / denominator
    margin = z * math.sqrt((probability * (1.0 - probability) / rows) + (z * z / (4.0 * rows * rows))) / denominator
    return center - margin, center + margin


def _candidate_probability(candidate: selector.Candidate) -> float:
    return float(candidate.calibrated_graded_hit_rate)


def _candidate_to_play(candidate: selector.Candidate) -> dict[str, Any]:
    probability = _candidate_probability(candidate)
    market_date = str(candidate.raw.get("Game_Date", candidate.run_date.isoformat()))
    return {
        "play_key": "|".join(
            (market_date, candidate.game_id, candidate.player, candidate.target, candidate.direction)
        ),
        "player": candidate.player,
        "player_display_name": candidate.player.replace("_", " "),
        "player_id": candidate.player_id,
        "player_type": str(candidate.raw.get("Player_Type", "")),
        "team": candidate.team,
        "opponent": str(candidate.raw.get("Opponent", "")),
        "game_id": candidate.game_id,
        "market_date": market_date,
        "commence_time_utc": str(candidate.raw.get("Commence_Time_UTC", "")),
        "game_status_code": candidate.game_status_code,
        "target": candidate.target,
        "direction": candidate.direction,
        "prediction": candidate.prediction,
        "market_line": candidate.market_line,
        "market_source": candidate.market_source,
        "market_bucket": candidate.market_bucket,
        "historical_bucket_key": candidate.historical_bucket_key,
        "estimated_graded_hit_rate": probability,
        "model_hit_probability": candidate.model_hit_probability,
        "final_pool_quality_score": candidate.selection_score,
        "expected_value_per_unit": candidate.expected_value_per_unit,
        "selected_side_price": candidate.selected_side_price,
        "selected_sportsbook_key": candidate.selected_sportsbook_key,
        "selected_sportsbook": candidate.selected_sportsbook,
        "price_confirmed": candidate.price_confirmed,
        "market_books": candidate.market_books,
        "market_common_books": candidate.market_common_books,
        "market_book_keys": [value for value in candidate.market_book_keys.split("|") if value],
        "market_common_book_keys": [value for value in candidate.market_common_book_keys.split("|") if value],
        "history_rows": candidate.history_rows,
        "days_since_history": candidate.days_since_history,
        "selection_score": candidate.selection_score,
        "parlay_precision_eligible": True,
    }


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def fetch_official_game_contexts(candidates: list[selector.Candidate]) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for game_id in sorted({candidate.game_id for candidate in candidates if candidate.game_id}):
        request = Request(f"{MLB_LIVE_FEED_ROOT}/{game_id}/feed/live", headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=5) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        game_data = payload.get("gameData") or {}
        status = game_data.get("status") or {}
        boxscore_teams = ((payload.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        game_teams = game_data.get("teams") or {}
        lineups: dict[str, set[str]] = {}
        for side in ("away", "home"):
            team = str((game_teams.get(side) or {}).get("abbreviation", "")).strip().upper()
            names = {
                _normalize_name((player.get("person") or {}).get("fullName", ""))
                for player in ((boxscore_teams.get(side) or {}).get("players") or {}).values()
                if str(player.get("battingOrder", "")).strip()
            }
            if team and names:
                lineups[team] = names
        contexts[game_id] = {
            "state": str(status.get("abstractGameState", "")).strip().lower(),
            "lineups": lineups,
        }
    return contexts


def filter_anchor_candidates(candidates: list[selector.Candidate], *, min_leg_probability: float) -> tuple[list[selector.Candidate], Counter[str]]:
    kept: list[selector.Candidate] = []
    rejected: Counter[str] = Counter()
    for candidate in candidates:
        if candidate.direction != "OVER":
            rejected["not_over"] += 1
            continue
        if candidate.target not in ALLOWED_TARGETS:
            rejected["unsupported_anchor_target"] += 1
            continue
        if str(candidate.model_selected).strip().lower() == "baseline":
            rejected["baseline_model"] += 1
            continue
        if candidate.market_source != "real" or not candidate.price_confirmed:
            rejected["unconfirmed_market_price"] += 1
            continue
        if candidate.market_books < 5 or candidate.market_common_books < 2:
            rejected["insufficient_book_coverage"] += 1
            continue
        if not selector.is_standard_bettable_line(candidate.target, candidate.market_line):
            rejected["nonstandard_line"] += 1
            continue
        if candidate.history_rows < 35:
            rejected["history_too_short"] += 1
            continue
        if candidate.days_since_history is None or candidate.days_since_history > 4:
            rejected["history_too_stale"] += 1
            continue
        if not selector.is_upcoming_status(candidate.game_status_code, str(candidate.raw.get("Game_Status_Detail", ""))):
            rejected["game_not_upcoming"] += 1
            continue
        if candidate.selected_side_price is None or not (-250.0 <= candidate.selected_side_price <= 125.0):
            rejected["price_outside_playable_range"] += 1
            continue
        if _candidate_probability(candidate) < float(min_leg_probability):
            rejected["leg_probability_too_low"] += 1
            continue
        if candidate.expected_value_per_unit is None or candidate.expected_value_per_unit < -0.03:
            rejected["leg_value_too_low"] += 1
            continue
        kept.append(candidate)
    return kept, rejected


def select_ticket(
    candidates: list[selector.Candidate],
    *,
    min_legs: int,
    max_legs: int,
    min_leg_probability: float,
    min_ticket_probability: float,
    min_combined_decimal_price: float,
    min_expected_return: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    plays = [_candidate_to_play(candidate) for candidate in candidates]
    limited: list[dict[str, Any]] = []
    by_book: dict[str, list[dict[str, Any]]] = {}
    for play in plays:
        by_book.setdefault(str(play["selected_sportsbook_key"]), []).append(play)
    for book_plays in by_book.values():
        book_plays.sort(
            key=lambda row: (
                float(row["estimated_graded_hit_rate"]),
                float(row.get("expected_value_per_unit") or -999.0),
                float(row.get("selection_score") or 0.0),
            ),
            reverse=True,
        )
        limited.extend(book_plays[:MAX_CANDIDATES_PER_BOOK])

    tickets = score_candidate_parlays(
        limited,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
        eligibility_field="parlay_precision_eligible",
        min_leg_probability=min_leg_probability,
        min_pair_probability=min_ticket_probability,
        min_legs_per_parlay=min_legs,
        max_legs_per_parlay=max_legs,
        forbid_same_market_bucket_parlay=False,
        min_expected_return_per_unit=min_expected_return,
    )
    executable = [
        ticket
        for ticket in tickets
        if float(ticket.get("combined_decimal_price") or 0.0) >= float(min_combined_decimal_price)
        and float(ticket.get("expected_return_per_unit") or -999.0) >= float(min_expected_return)
        and not bool(ticket.get("same_game"))
        and not bool(ticket.get("same_player"))
    ]
    executable.sort(
        key=lambda row: (
            float(row.get("projected_probability") or 0.0),
            float(row.get("expected_return_per_unit") or -999.0),
            float(row.get("avg_leg_quality") or 0.0),
            -int(row.get("leg_count") or 0),
        ),
        reverse=True,
    )
    if not executable:
        return None, limited
    ticket = dict(executable[0])
    ticket["legs"] = [limited[int(index)] for index in ticket["leg_indices"]]
    return ticket, limited


def _historical_over_events(history_dir: Path, season: int, before_date: date, min_leg_probability: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(history_dir.glob(f"*/{season}_processed_processed.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Date" not in frame:
            continue
        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.loc[frame["Date"].dt.date < before_date].sort_values("Date")
        for history_rows, (_, row) in enumerate(frame.iterrows(), start=1):
            if history_rows < 35:
                continue
            for target in ALLOWED_TARGETS:
                actual = selector.to_float(row.get(target), default=float("nan"))
                line = selector.to_float(row.get(f"Market_{target}"), default=float("nan"))
                gap = selector.to_float(row.get(f"{target}_market_gap"), default=float("nan"))
                if not all(math.isfinite(value) for value in (actual, line, gap)) or gap <= 0.0:
                    continue
                if abs(actual - line) <= 1e-9:
                    continue
                if not selector.is_standard_bettable_line(target, line):
                    continue
                probability = selector.estimate_count_hit_probabilities(max(0.0, line + gap), line, "OVER")[2]
                if probability < min_leg_probability:
                    continue
                records.append(
                    {
                        "date": row["Date"].date().isoformat(),
                        "player": str(row.get("Player", path.parent.name)),
                        "game_id": str(row.get("Game_ID", "")).removesuffix(".0"),
                        "target": target,
                        "probability": probability,
                        "win": int(actual > line),
                    }
                )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.drop_duplicates(["date", "player", "game_id", "target"], keep="last")


def _grade_leg_count(events: pd.DataFrame, leg_count: int) -> dict[str, Any]:
    tickets: list[dict[str, Any]] = []
    for slate_date, frame in events.groupby("date", sort=True):
        frame = frame.sort_values("probability", ascending=False).drop_duplicates("player", keep="first")
        selected: list[pd.Series] = []
        games: set[str] = set()
        for _, row in frame.iterrows():
            game_id = str(row["game_id"])
            if not game_id or game_id in games:
                continue
            selected.append(row)
            games.add(game_id)
            if len(selected) == leg_count:
                break
        if len(selected) != leg_count:
            continue
        projected = math.prod(float(row["probability"]) for row in selected)
        tickets.append(
            {
                "date": slate_date,
                "projected_probability": projected,
                "hit": int(all(int(row["win"]) == 1 for row in selected)),
            }
        )
    if not tickets:
        return {"leg_count": leg_count, "tickets": 0, "dates": 0, "hits": 0, "hit_rate": None}
    frame = pd.DataFrame(tickets)
    holdout_size = min(20, max(1, len(frame) // 5))
    development = frame.iloc[:-holdout_size]
    holdout = frame.iloc[-holdout_size:]

    def metrics(part: pd.DataFrame) -> dict[str, Any]:
        if part.empty:
            return {"tickets": 0, "hits": 0, "hit_rate": None}
        wins = int(part["hit"].sum())
        low, high = _wilson_interval(wins, len(part))
        return {
            "tickets": int(len(part)),
            "dates": int(part["date"].nunique()),
            "hits": wins,
            "hit_rate": float(part["hit"].mean()),
            "win_rate_wilson_95_low": low,
            "win_rate_wilson_95_high": high,
            "mean_projected_probability": float(part["projected_probability"].mean()),
            "brier_score": float(((part["hit"] - part["projected_probability"]) ** 2).mean()),
        }

    return {
        "leg_count": leg_count,
        "all_dates": metrics(frame),
        "development": metrics(development),
        "fixed_recent_holdout": metrics(holdout),
    }


def build_validation(history_dir: Path, season: int, before_date: date, min_leg_probability: float) -> dict[str, Any]:
    events = _historical_over_events(history_dir, season, before_date, min_leg_probability)
    return {
        "method": "stored_pre_event_projection_synthetic_line_grading",
        "price_validation": "unavailable_historically; no ROI claim",
        "history_before_date": before_date.isoformat(),
        "event_rows": int(len(events)),
        "dates": int(events["date"].nunique()) if not events.empty else 0,
        "by_leg_count": [_grade_leg_count(events, leg_count) for leg_count in range(MIN_LEGS, MAX_LEGS + 1)],
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    pool_csv = args.pool_csv.resolve()
    run_date = selector.infer_pool_run_date(pool_csv)
    if run_date is None:
        raise ValueError(f"Could not infer run date from {pool_csv}")
    season = selector.infer_history_season(pool_csv, None)
    calibration = _load_json(selector.default_history_cache_path(season))
    bet_profiles = _load_json(selector.default_bet_profile_cache_path(season))
    live_calibration = _load_json(selector.default_live_confidence_cache_path(season))
    survival_model = _load_json(selector.default_pick_survival_cache_path(season))
    candidates = selector.load_candidates(
        pool_csv,
        calibration=calibration,
        bet_profile_priors=bet_profiles,
        live_confidence_calibration=live_calibration,
        pick_survival_model=survival_model,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
        min_bet_profile_rows=12,
        max_bet_profile_prior_weight=0.25,
        bet_profile_prior_strength=80.0,
        min_market_availability_rows=12,
    )
    anchors, rejected = filter_anchor_candidates(candidates, min_leg_probability=args.min_leg_probability)
    official_contexts = fetch_official_game_contexts(anchors)
    open_anchors: list[selector.Candidate] = []
    for candidate in anchors:
        context = official_contexts.get(candidate.game_id, {})
        if context.get("state") in {"live", "final"}:
            rejected["official_game_closed"] += 1
            continue
        team_lineup = (context.get("lineups") or {}).get(candidate.team.upper())
        if team_lineup and _normalize_name(candidate.player) not in team_lineup:
            rejected["not_in_posted_lineup"] += 1
            continue
        open_anchors.append(candidate)
    anchors = open_anchors
    ticket, considered = select_ticket(
        anchors,
        min_legs=max(MIN_LEGS, int(args.min_legs)),
        max_legs=min(MAX_LEGS, max(int(args.min_legs), int(args.max_legs))),
        min_leg_probability=float(args.min_leg_probability),
        min_ticket_probability=float(args.min_ticket_probability),
        min_combined_decimal_price=float(args.min_combined_decimal_price),
        min_expected_return=float(args.min_expected_return),
    )
    validation = build_validation(args.history_dir.resolve(), season, run_date, float(args.min_leg_probability))
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date.isoformat(),
        "status": "ready" if ticket is not None else "withheld",
        "objective": "ticket_hit_probability_then_expected_return",
        "direction_policy": "OVER_ONLY",
        "allowed_leg_counts": list(range(max(MIN_LEGS, int(args.min_legs)), min(MAX_LEGS, int(args.max_legs)) + 1)),
        "gates": {
            "min_leg_probability": float(args.min_leg_probability),
            "min_ticket_probability": float(args.min_ticket_probability),
            "min_combined_decimal_price": float(args.min_combined_decimal_price),
            "min_expected_return_per_unit": float(args.min_expected_return),
            "min_market_books": 5,
            "min_common_market_books": 2,
            "same_sportsbook_required": True,
            "distinct_games_required": True,
        },
        "pool_candidate_count": int(len(candidates)),
        "eligible_anchor_count": int(len(anchors)),
        "considered_anchor_count": int(len(considered)),
        "filter_rejections": dict(sorted(rejected.items())),
        "selected_ticket": ticket,
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    output = args.out_json or args.pool_csv.with_name(f"{args.pool_csv.stem}_daily_parlay.json")
    payload = build_payload(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ticket = payload.get("selected_ticket") or {}
    print(f"Daily MLB parlay: {output}")
    print(f"Status: {payload['status']}; anchors={payload['eligible_anchor_count']}")
    if ticket:
        names = ", ".join(str(value) for value in ticket.get("leg_names", []))
        print(
            f"Selected {ticket.get('leg_count')} legs at {ticket.get('sportsbook')}: "
            f"p={float(ticket.get('projected_probability') or 0.0):.3f}, "
            f"EV={float(ticket.get('expected_return_per_unit') or 0.0):+.3f} ({names})"
        )


if __name__ == "__main__":
    main()
