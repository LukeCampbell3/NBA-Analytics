from __future__ import annotations

import math
from itertools import combinations
from collections import Counter
from typing import Any

import pandas as pd


SPORT_CONFIG: dict[str, dict[str, float | int]] = {
    "nba": {
        "min_leg_probability": 0.60,
        "min_pair_probability": 0.36,
        "max_pairs": 1,
        "min_legs_per_parlay": 2,
        "max_legs_per_parlay": 2,
        "fallback_min_leg_probability": 0.50,
        "fallback_min_pair_probability": 0.24,
        "fallback_max_pairs": 1,
        "fallback_min_legs_per_parlay": 2,
        "fallback_max_legs_per_parlay": 2,
        "same_player_factor": 0.52,
        "same_game_factor": 0.84,
        "same_team_factor": 0.94,
        "same_target_factor": 0.98,
        "same_direction_factor": 1.01,
        "mixed_direction_factor": 0.99,
        "different_game_bonus": 1.05,
        "different_team_bonus": 1.02,
        "same_script_cluster_factor": 0.88,
        "same_market_bucket_factor": 0.90,
        "forbid_same_market_bucket_parlay": 1,
        "avoid_reused_market_buckets_across_tickets": 1,
        "cap_projected_probability_to_independent": 1,
    },
    "mlb": {
        "min_leg_probability": 0.60,
        "min_pair_probability": 0.38,
        "max_pairs": 2,
        "min_legs_per_parlay": 2,
        "max_legs_per_parlay": 2,
        "fallback_min_leg_probability": 0.58,
        "fallback_min_pair_probability": 0.34,
        "fallback_max_pairs": 1,
        "fallback_min_legs_per_parlay": 2,
        "fallback_max_legs_per_parlay": 2,
        "same_player_factor": 0.72,
        "same_game_factor": 0.95,
        "same_team_factor": 0.97,
        "same_target_factor": 0.99,
        "same_direction_factor": 1.03,
        "mixed_direction_factor": 0.98,
        "different_game_bonus": 1.06,
        "different_team_bonus": 1.03,
        "same_script_cluster_factor": 0.96,
        "same_market_bucket_factor": 0.90,
        "forbid_same_market_bucket_parlay": 1,
        "avoid_reused_market_buckets_across_tickets": 1,
        "cap_projected_probability_to_independent": 1,
    },
}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_text(value: Any) -> str:
    return _clean_text(value).lower()


def _normalized_script_cluster(value: Any) -> str:
    token = _normalized_text(value)
    if token in {"", "nan", "none", "null", "unknown", "script=unknown", "uninferred", "script=uninferred"}:
        return ""
    return token


def _normalized_market_bucket(value: Any) -> str:
    token = _normalized_text(value)
    if token in {"", "nan", "none", "null"}:
        return ""
    return token


def _leg_quality(play: dict[str, Any], probability_field: str) -> float:
    for key in ("parlay_leg_quality_score", "final_pool_quality_score"):
        quality = _safe_float(play.get(key))
        if quality is not None:
            return max(0.0, min(1.0, quality))
    probability = _safe_float(play.get(probability_field))
    confidence = _safe_float(play.get("final_confidence")) or 0.0
    ev = _safe_float(play.get("ev")) or 0.0
    derived = 0.75 * (probability if probability is not None else 0.5) + 0.20 * max(0.0, min(1.0, confidence)) + 0.05 * max(0.0, min(1.0, 0.5 + (4.0 * ev)))
    return max(0.0, min(1.0, derived))


def _play_key(play: dict[str, Any], fallback_index: int) -> str:
    player = _normalized_text(play.get("player_display_name") or play.get("player"))
    target = _normalized_text(play.get("target"))
    direction = _normalized_text(play.get("direction"))
    market_date = _clean_text(play.get("market_date"))
    game_key = _clean_text(play.get("game_key") or play.get("game_id"))
    parts = [part for part in [market_date, game_key, player, target, direction] if part]
    return "|".join(parts) if parts else f"play-{fallback_index + 1}"


def _pair_outcome(left_result: str, right_result: str) -> str:
    results = {_normalized_text(left_result), _normalized_text(right_result)}
    if "unresolved" in results or "" in results:
        return "unresolved"
    if "loss" in results:
        return "miss"
    if results == {"win"}:
        return "hit"
    if "push" in results:
        return "push"
    return "unresolved"


def _parlay_outcome(results: list[str]) -> str:
    normalized = {_normalized_text(result) for result in results}
    if not normalized or "unresolved" in normalized or "" in normalized:
        return "unresolved"
    if "loss" in normalized:
        return "miss"
    if normalized == {"win"}:
        return "hit"
    if "push" in normalized:
        return "push"
    return "unresolved"


def _resolve_sport_config(
    sport: str,
    *,
    min_leg_probability: float | None = None,
    min_pair_probability: float | None = None,
    max_pairs: int | None = None,
    min_legs_per_parlay: int | None = None,
    max_legs_per_parlay: int | None = None,
) -> dict[str, float | int]:
    config = dict(SPORT_CONFIG.get(str(sport or "").strip().lower(), SPORT_CONFIG["nba"]))
    if min_leg_probability is not None:
        config["min_leg_probability"] = float(min_leg_probability)
    if min_pair_probability is not None:
        config["min_pair_probability"] = float(min_pair_probability)
    if max_pairs is not None:
        config["max_pairs"] = int(max_pairs)
    if min_legs_per_parlay is not None:
        config["min_legs_per_parlay"] = int(min_legs_per_parlay)
    if max_legs_per_parlay is not None:
        config["max_legs_per_parlay"] = int(max_legs_per_parlay)
    return config


def _pair_adjustment_factor(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    config: dict[str, float | int],
) -> tuple[float, dict[str, bool]]:
    left_player = _normalized_text(left.get("player_display_name") or left.get("player"))
    left_team = _normalized_text(left.get("team"))
    left_target = _normalized_text(left.get("target"))
    left_direction = _normalized_text(left.get("direction"))
    left_game = _normalized_text(left.get("game_id") or left.get("game_key"))
    left_script_cluster = _normalized_script_cluster(left.get("script_cluster_id"))
    left_market_bucket = _normalized_market_bucket(left.get("market_bucket") or left.get("historical_bucket_key"))

    right_player = _normalized_text(right.get("player_display_name") or right.get("player"))
    right_team = _normalized_text(right.get("team"))
    right_target = _normalized_text(right.get("target"))
    right_direction = _normalized_text(right.get("direction"))
    right_game = _normalized_text(right.get("game_id") or right.get("game_key"))
    right_script_cluster = _normalized_script_cluster(right.get("script_cluster_id"))
    right_market_bucket = _normalized_market_bucket(right.get("market_bucket") or right.get("historical_bucket_key"))

    same_player = bool(left_player and left_player == right_player)
    same_game = bool(left_game and left_game == right_game)
    same_team = bool(left_team and left_team == right_team)
    same_target = bool(left_target and left_target == right_target)
    same_direction = bool(left_direction and left_direction == right_direction)
    same_script_cluster = bool(left_script_cluster and left_script_cluster == right_script_cluster)
    same_market_bucket = bool(left_market_bucket and left_market_bucket == right_market_bucket)

    factor = 1.0
    if same_player:
        factor *= float(config["same_player_factor"])
    if same_game:
        factor *= float(config["same_game_factor"])
    else:
        factor *= float(config["different_game_bonus"])
    if same_team:
        factor *= float(config["same_team_factor"])
    else:
        factor *= float(config["different_team_bonus"])
    if same_target:
        factor *= float(config["same_target_factor"])
    factor *= float(config["same_direction_factor"] if same_direction else config["mixed_direction_factor"])
    if same_script_cluster:
        factor *= float(config["same_script_cluster_factor"])
    if same_market_bucket:
        factor *= float(config.get("same_market_bucket_factor", 1.0))

    return factor, {
        "same_player": same_player,
        "same_game": same_game,
        "same_team": same_team,
        "same_target": same_target,
        "same_direction": same_direction,
        "same_script_cluster": same_script_cluster,
        "same_market_bucket": same_market_bucket,
    }


def score_candidate_parlays(
    plays: list[dict[str, Any]],
    *,
    sport: str,
    probability_field: str,
    eligibility_field: str | None = None,
    min_leg_probability: float | None = None,
    min_pair_probability: float | None = None,
    min_legs_per_parlay: int | None = None,
    max_legs_per_parlay: int | None = None,
) -> list[dict[str, Any]]:
    config = _resolve_sport_config(
        sport,
        min_leg_probability=min_leg_probability,
        min_pair_probability=min_pair_probability,
        min_legs_per_parlay=min_legs_per_parlay,
        max_legs_per_parlay=max_legs_per_parlay,
    )
    min_leg = float(config["min_leg_probability"])
    min_ticket = float(config["min_pair_probability"])
    min_legs = max(2, int(config.get("min_legs_per_parlay", 2)))
    max_legs = max(min_legs, int(config.get("max_legs_per_parlay", min_legs)))

    eligible_rows: list[tuple[int, dict[str, Any], float, float]] = []
    for index, play in enumerate(plays):
        if eligibility_field is not None and not bool(play.get(eligibility_field)):
            continue
        probability = _safe_float(play.get(probability_field))
        if probability is None or probability < min_leg:
            continue
        eligible_rows.append((index, play, probability, _leg_quality(play, probability_field)))

    tickets: list[dict[str, Any]] = []
    if len(eligible_rows) < min_legs:
        return tickets

    for leg_count in range(min_legs, min(max_legs, len(eligible_rows)) + 1):
        for combo in combinations(eligible_rows, leg_count):
            indices = [int(item[0]) for item in combo]
            combo_plays = [item[1] for item in combo]
            probabilities = [float(item[2]) for item in combo]
            qualities = [float(item[3]) for item in combo]

            independent_probability = 1.0
            for probability in probabilities:
                independent_probability *= probability

            pair_factors: list[float] = []
            same_player = False
            same_game = False
            same_team = False
            same_target = False
            same_script_cluster = False
            same_market_bucket = False
            direction_tokens: set[str] = set()
            for left_pos, right_pos in combinations(range(len(combo_plays)), 2):
                factor, flags = _pair_adjustment_factor(combo_plays[left_pos], combo_plays[right_pos], config=config)
                pair_factors.append(float(factor))
                same_player = same_player or bool(flags["same_player"])
                same_game = same_game or bool(flags["same_game"])
                same_team = same_team or bool(flags["same_team"])
                same_target = same_target or bool(flags["same_target"])
                same_script_cluster = same_script_cluster or bool(flags["same_script_cluster"])
                same_market_bucket = same_market_bucket or bool(flags["same_market_bucket"])
            for play in combo_plays:
                direction = _normalized_text(play.get("direction"))
                if direction:
                    direction_tokens.add(direction)

            if same_market_bucket and int(config.get("forbid_same_market_bucket_parlay", 0)):
                continue

            factor = math.prod(pair_factors) ** (1.0 / len(pair_factors)) if pair_factors else 1.0
            projected_probability = max(0.0, min(1.0, independent_probability * factor))
            if int(config.get("cap_projected_probability_to_independent", 0)):
                projected_probability = min(projected_probability, independent_probability)
            projected_probability = min(projected_probability, min(probabilities))
            if projected_probability < min_ticket:
                continue

            distinct_games = len({_normalized_text(play.get("game_id") or play.get("game_key")) for play in combo_plays if _normalized_text(play.get("game_id") or play.get("game_key"))})
            distinct_teams = len({_normalized_text(play.get("team")) for play in combo_plays if _normalized_text(play.get("team"))})
            diversity_bonus = 1.0 + (0.03 * max(0, distinct_games - 1)) + (0.02 * max(0, distinct_teams - 1))
            if same_player:
                diversity_bonus -= 0.10
            if same_game:
                diversity_bonus -= 0.04
            if same_script_cluster:
                diversity_bonus -= 0.03
            if same_market_bucket:
                diversity_bonus -= 0.06

            avg_leg_quality = sum(qualities) / len(qualities)
            quality_factor = 0.85 + (0.30 * avg_leg_quality)
            parlay_score = projected_probability * max(0.75, diversity_bonus) * quality_factor
            tickets.append(
                {
                    "leg_indices": indices,
                    "leg_keys": [play.get("play_key") for play in combo_plays],
                    "leg_names": [play.get("player_display_name") or play.get("player") for play in combo_plays],
                    "leg_targets": [play.get("target") for play in combo_plays],
                    "leg_directions": [play.get("direction") for play in combo_plays],
                    "leg_count": int(len(combo_plays)),
                    "projected_probability": projected_probability,
                    "independent_probability": independent_probability,
                    "pairwise_adjustment_factor": factor,
                    "parlay_score": parlay_score,
                    "avg_leg_quality": avg_leg_quality,
                    "same_player": same_player,
                    "same_game": same_game,
                    "same_team": same_team,
                    "same_target": same_target,
                    "same_market_bucket": same_market_bucket,
                    "mixed_direction": len(direction_tokens) > 1,
                }
            )

    tickets.sort(
        key=lambda row: (
            float(row["parlay_score"]),
            float(row["projected_probability"]),
            -int(row["leg_count"]),
            float(row["independent_probability"]),
        ),
        reverse=True,
    )
    return tickets


def score_candidate_pairs(
    plays: list[dict[str, Any]],
    *,
    sport: str,
    probability_field: str,
    eligibility_field: str | None = None,
    min_leg_probability: float | None = None,
    min_pair_probability: float | None = None,
) -> list[dict[str, Any]]:
    parlays = score_candidate_parlays(
        plays,
        sport=sport,
        probability_field=probability_field,
        eligibility_field=eligibility_field,
        min_leg_probability=min_leg_probability,
        min_pair_probability=min_pair_probability,
        min_legs_per_parlay=2,
        max_legs_per_parlay=2,
    )
    pairs: list[dict[str, Any]] = []
    for parlay in parlays:
        left_index, right_index = [int(value) for value in parlay["leg_indices"]]
        pairs.append(
            {
                "left_index": left_index,
                "right_index": right_index,
                "left_key": parlay["leg_keys"][0],
                "right_key": parlay["leg_keys"][1],
                "left_name": parlay["leg_names"][0],
                "right_name": parlay["leg_names"][1],
                "left_target": parlay["leg_targets"][0],
                "right_target": parlay["leg_targets"][1],
                "left_direction": parlay["leg_directions"][0],
                "right_direction": parlay["leg_directions"][1],
                "projected_probability": parlay["projected_probability"],
                "independent_probability": parlay["independent_probability"],
                "pair_score": parlay["parlay_score"],
                "avg_leg_quality": parlay["avg_leg_quality"],
                "adjustment_factor": parlay["pairwise_adjustment_factor"],
                "same_player": parlay["same_player"],
                "same_game": parlay["same_game"],
                "same_team": parlay["same_team"],
                "same_target": parlay["same_target"],
                "same_market_bucket": parlay["same_market_bucket"],
                "same_direction": not parlay["mixed_direction"],
            }
        )
    return pairs


def annotate_parlay_board(
    plays: list[dict[str, Any]],
    *,
    sport: str,
    probability_field: str,
    eligibility_field: str | None = None,
    allow_fallback: bool = True,
    min_leg_probability: float | None = None,
    min_pair_probability: float | None = None,
    max_pairs: int | None = None,
    min_legs_per_parlay: int | None = None,
    max_legs_per_parlay: int | None = None,
) -> dict[str, Any]:
    config = _resolve_sport_config(
        sport,
        min_leg_probability=min_leg_probability,
        min_pair_probability=min_pair_probability,
        max_pairs=max_pairs,
        min_legs_per_parlay=min_legs_per_parlay,
        max_legs_per_parlay=max_legs_per_parlay,
    )
    prepared: list[dict[str, Any]] = []
    for index, play in enumerate(plays):
        item = dict(play)
        item["play_key"] = _play_key(item, index)
        item["parlay_tag"] = ""
        item["parlay_candidate"] = False
        item["parlay_ticket_rank"] = None
        item["parlay_pair_rank"] = None
        item["parlay_score"] = None
        item["parlay_projected_hit_rate"] = None
        item["parlay_leg_count"] = None
        item["parlay_partner_key"] = None
        item["parlay_partner_name"] = None
        item["parlay_partner_keys"] = []
        item["parlay_partner_names"] = []
        prepared.append(item)

    candidate_parlays = score_candidate_parlays(
        prepared,
        sport=sport,
        probability_field=probability_field,
        eligibility_field=eligibility_field,
        min_leg_probability=float(config["min_leg_probability"]),
        min_pair_probability=float(config["min_pair_probability"]),
        min_legs_per_parlay=int(config.get("min_legs_per_parlay", 2)),
        max_legs_per_parlay=int(config.get("max_legs_per_parlay", 2)),
    )

    selection_mode = "strict"
    if allow_fallback and not candidate_parlays:
        fallback_parlays = score_candidate_parlays(
            prepared,
            sport=sport,
            probability_field=probability_field,
            eligibility_field=eligibility_field,
            min_leg_probability=float(config.get("fallback_min_leg_probability", config["min_leg_probability"])),
            min_pair_probability=float(config.get("fallback_min_pair_probability", config["min_pair_probability"])),
            min_legs_per_parlay=int(config.get("fallback_min_legs_per_parlay", config.get("min_legs_per_parlay", 2))),
            max_legs_per_parlay=int(config.get("fallback_max_legs_per_parlay", config.get("max_legs_per_parlay", 2))),
        )
        if fallback_parlays:
            candidate_parlays = fallback_parlays
            selection_mode = "fallback"
            non_negative_ev_parlays = [
                parlay
                for parlay in candidate_parlays
                if all((_safe_float(prepared[int(index)].get("ev")) or 0.0) >= 0.0 for index in parlay["leg_indices"])
            ]
            if non_negative_ev_parlays:
                candidate_parlays = non_negative_ev_parlays

    selected_parlays: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    used_market_buckets: set[str] = set()
    max_parlays_to_select = int(
        config["max_pairs"]
        if selection_mode == "strict"
        else config.get("fallback_max_pairs", config["max_pairs"])
    )
    for parlay in candidate_parlays:
        leg_indices = [int(index) for index in parlay["leg_indices"]]
        if any(index in used_indices for index in leg_indices):
            continue
        parlay_market_buckets = {
            _normalized_market_bucket(prepared[index].get("market_bucket") or prepared[index].get("historical_bucket_key"))
            for index in leg_indices
        }
        parlay_market_buckets.discard("")
        if int(config.get("avoid_reused_market_buckets_across_tickets", 0)) and parlay_market_buckets & used_market_buckets:
            continue
        selected_parlays.append(dict(parlay))
        used_indices.update(leg_indices)
        used_market_buckets.update(parlay_market_buckets)
        if len(selected_parlays) >= max_parlays_to_select:
            break

    for parlay_rank, parlay in enumerate(selected_parlays, start=1):
        leg_indices = [int(index) for index in parlay["leg_indices"]]
        leg_rows = [prepared[index] for index in leg_indices]
        parlay["pair_rank"] = parlay_rank
        parlay["ticket_rank"] = parlay_rank
        parlay["pair_score"] = parlay["parlay_score"]
        parlay["legs"] = [
            {
                "play_key": leg["play_key"],
                "player": leg.get("player_display_name") or leg.get("player"),
                "target": leg.get("target"),
                "direction": leg.get("direction"),
            }
            for leg in leg_rows
        ]
        for current in leg_rows:
            partner_keys = [leg["play_key"] for leg in leg_rows if leg["play_key"] != current["play_key"]]
            partner_names = [
                leg.get("player_display_name") or leg.get("player")
                for leg in leg_rows
                if leg["play_key"] != current["play_key"]
            ]
            current["parlay_tag"] = "parlay"
            current["parlay_candidate"] = True
            current["parlay_ticket_rank"] = parlay_rank
            current["parlay_pair_rank"] = parlay_rank
            current["parlay_score"] = parlay["parlay_score"]
            current["parlay_projected_hit_rate"] = parlay["projected_probability"]
            current["parlay_leg_count"] = int(parlay["leg_count"])
            current["parlay_partner_keys"] = partner_keys
            current["parlay_partner_names"] = partner_names
            if len(partner_keys) == 1:
                current["parlay_partner_key"] = partner_keys[0]
                current["parlay_partner_name"] = partner_names[0]
            else:
                current["parlay_partner_key"] = None
                current["parlay_partner_name"] = None

    tagged_probability = [
        float(parlay["projected_probability"])
        for parlay in selected_parlays
        if _safe_float(parlay.get("projected_probability")) is not None
    ]

    summary = {
        "selection_mode": selection_mode,
        "candidate_pair_count": int(len(candidate_parlays)),
        "selected_pair_count": int(len(selected_parlays)),
        "candidate_parlay_count": int(len(candidate_parlays)),
        "selected_parlay_count": int(len(selected_parlays)),
        "tagged_play_count": int(sum(1 for play in prepared if play["parlay_candidate"])),
        "avg_projected_pair_hit_rate": _safe_mean(tagged_probability),
        "best_projected_pair_hit_rate": max(tagged_probability) if tagged_probability else None,
        "avg_projected_parlay_hit_rate": _safe_mean(tagged_probability),
        "best_projected_parlay_hit_rate": max(tagged_probability) if tagged_probability else None,
        "min_leg_probability": float(config["min_leg_probability"]),
        "min_pair_probability": float(config["min_pair_probability"]),
        "min_legs_per_parlay": int(config.get("min_legs_per_parlay", 2)),
        "max_legs_per_parlay": int(config.get("max_legs_per_parlay", 2)),
        "fallback_min_leg_probability": float(config.get("fallback_min_leg_probability", config["min_leg_probability"])),
        "fallback_min_pair_probability": float(config.get("fallback_min_pair_probability", config["min_pair_probability"])),
        "fallback_min_legs_per_parlay": int(config.get("fallback_min_legs_per_parlay", config.get("min_legs_per_parlay", 2))),
        "fallback_max_legs_per_parlay": int(config.get("fallback_max_legs_per_parlay", config.get("max_legs_per_parlay", 2))),
    }
    return {
        "plays": prepared,
        "pairs": selected_parlays,
        "summary": summary,
    }


def evaluate_historical_parlays(
    history_rows: pd.DataFrame,
    *,
    sport: str,
    date_col: str,
    probability_col: str,
    result_col: str = "result",
    min_leg_probability: float | None = None,
    min_pair_probability: float | None = None,
    max_pairs_per_day: int = 1,
    min_legs_per_parlay: int | None = None,
    max_legs_per_parlay: int | None = None,
) -> dict[str, Any]:
    if history_rows.empty:
        return {"available": False, "reason": "history rows are empty"}
    if date_col not in history_rows.columns:
        return {"available": False, "reason": f"missing date column: {date_col}"}
    if result_col not in history_rows.columns:
        return {"available": False, "reason": f"missing result column: {result_col}"}
    if probability_col not in history_rows.columns:
        return {"available": False, "reason": f"missing probability column: {probability_col}"}

    working = history_rows.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    working[result_col] = working[result_col].astype(str).str.lower().str.strip()
    working = working.loc[working[date_col].notna()].copy()
    if working.empty:
        return {"available": False, "reason": "no dated rows available for parlay validation"}

    selected_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    dates_with_candidates = 0

    for market_date, part in working.groupby(date_col, dropna=False):
        rows = part.to_dict(orient="records")
        for index, row in enumerate(rows):
            row["play_key"] = _play_key(row, index)
        candidate_parlays = score_candidate_parlays(
            rows,
            sport=sport,
            probability_field=probability_col,
            min_leg_probability=min_leg_probability,
            min_pair_probability=min_pair_probability,
            min_legs_per_parlay=min_legs_per_parlay,
            max_legs_per_parlay=max_legs_per_parlay,
        )
        if not candidate_parlays:
            continue

        dates_with_candidates += 1
        for parlay in candidate_parlays:
            leg_results = [str(rows[int(index)].get(result_col, "unresolved")) for index in parlay["leg_indices"]]
            baseline_records.append(
                {
                    "market_date": market_date,
                    "pair_outcome": _parlay_outcome(leg_results),
                    "projected_probability": parlay["projected_probability"],
                    "leg_count": int(parlay["leg_count"]),
                }
            )

        chosen = 0
        used_indices: set[int] = set()
        for parlay in candidate_parlays:
            leg_indices = [int(index) for index in parlay["leg_indices"]]
            if any(index in used_indices for index in leg_indices):
                continue
            leg_results = [str(rows[index].get(result_col, "unresolved")) for index in leg_indices]
            selected_records.append(
                {
                    "market_date": market_date,
                    "pair_outcome": _parlay_outcome(leg_results),
                    "projected_probability": parlay["projected_probability"],
                    "leg_count": int(parlay["leg_count"]),
                }
            )
            used_indices.update(leg_indices)
            chosen += 1
            if chosen >= int(max_pairs_per_day):
                break

    if not selected_records:
        return {"available": False, "reason": "no historical parlay candidates met thresholds"}

    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        outcome_counts = Counter(str(row.get("pair_outcome", "unresolved")) for row in records)
        graded = int(outcome_counts.get("hit", 0) + outcome_counts.get("miss", 0))
        projected = [
            float(row["projected_probability"])
            for row in records
            if row.get("pair_outcome") in {"hit", "miss"} and _safe_float(row.get("projected_probability")) is not None
        ]
        return {
            "pair_count": int(len(records)),
            "parlay_count": int(len(records)),
            "graded_pair_count": graded,
            "graded_parlay_count": graded,
            "hit_pair_count": int(outcome_counts.get("hit", 0)),
            "hit_parlay_count": int(outcome_counts.get("hit", 0)),
            "miss_pair_count": int(outcome_counts.get("miss", 0)),
            "miss_parlay_count": int(outcome_counts.get("miss", 0)),
            "push_pair_count": int(outcome_counts.get("push", 0)),
            "push_parlay_count": int(outcome_counts.get("push", 0)),
            "unresolved_pair_count": int(outcome_counts.get("unresolved", 0)),
            "unresolved_parlay_count": int(outcome_counts.get("unresolved", 0)),
            "pair_hit_rate": (_safe_mean([1.0] * int(outcome_counts.get("hit", 0)) + [0.0] * int(outcome_counts.get("miss", 0))) if graded else None),
            "parlay_hit_rate": (_safe_mean([1.0] * int(outcome_counts.get("hit", 0)) + [0.0] * int(outcome_counts.get("miss", 0))) if graded else None),
            "avg_projected_pair_hit_rate": _safe_mean(projected),
            "avg_projected_parlay_hit_rate": _safe_mean(projected),
        }

    selected_summary = summarize(selected_records)
    baseline_summary = summarize(baseline_records)

    selected_hit_rate = _safe_float(selected_summary.get("pair_hit_rate"))
    baseline_hit_rate = _safe_float(baseline_summary.get("pair_hit_rate"))
    hit_rate_lift = None
    if selected_hit_rate is not None and baseline_hit_rate is not None:
        hit_rate_lift = selected_hit_rate - baseline_hit_rate

    return {
        "available": True,
        "sample_dates": int(dates_with_candidates),
        "max_pairs_per_day": int(max_pairs_per_day),
        "max_parlays_per_day": int(max_pairs_per_day),
        "selected": selected_summary,
        "baseline_all_pairs": baseline_summary,
        "baseline_all_parlays": baseline_summary,
        "hit_rate_lift_vs_all_pairs": hit_rate_lift,
        "hit_rate_lift_vs_all_parlays": hit_rate_lift,
    }
