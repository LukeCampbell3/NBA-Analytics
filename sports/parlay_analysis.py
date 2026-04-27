from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd


SPORT_CONFIG: dict[str, dict[str, float | int]] = {
    "nba": {
        "min_leg_probability": 0.54,
        "min_pair_probability": 0.29,
        "max_pairs": 3,
        "fallback_min_leg_probability": 0.50,
        "fallback_min_pair_probability": 0.24,
        "fallback_max_pairs": 1,
        "same_player_factor": 0.52,
        "same_game_factor": 0.90,
        "same_team_factor": 0.96,
        "same_target_factor": 0.98,
        "same_direction_factor": 1.01,
        "mixed_direction_factor": 0.99,
        "different_game_bonus": 1.05,
        "different_team_bonus": 1.02,
        "same_script_cluster_factor": 0.92,
    },
    "mlb": {
        "min_leg_probability": 0.60,
        "min_pair_probability": 0.38,
        "max_pairs": 3,
        "fallback_min_leg_probability": 0.58,
        "fallback_min_pair_probability": 0.34,
        "fallback_max_pairs": 2,
        "same_player_factor": 0.72,
        "same_game_factor": 0.95,
        "same_team_factor": 0.97,
        "same_target_factor": 0.99,
        "same_direction_factor": 1.03,
        "mixed_direction_factor": 0.98,
        "different_game_bonus": 1.06,
        "different_team_bonus": 1.03,
        "same_script_cluster_factor": 0.96,
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


def _resolve_sport_config(
    sport: str,
    *,
    min_leg_probability: float | None = None,
    min_pair_probability: float | None = None,
    max_pairs: int | None = None,
) -> dict[str, float | int]:
    config = dict(SPORT_CONFIG.get(str(sport or "").strip().lower(), SPORT_CONFIG["nba"]))
    if min_leg_probability is not None:
        config["min_leg_probability"] = float(min_leg_probability)
    if min_pair_probability is not None:
        config["min_pair_probability"] = float(min_pair_probability)
    if max_pairs is not None:
        config["max_pairs"] = int(max_pairs)
    return config


def score_candidate_pairs(
    plays: list[dict[str, Any]],
    *,
    sport: str,
    probability_field: str,
    min_leg_probability: float | None = None,
    min_pair_probability: float | None = None,
) -> list[dict[str, Any]]:
    config = _resolve_sport_config(
        sport,
        min_leg_probability=min_leg_probability,
        min_pair_probability=min_pair_probability,
    )
    min_leg = float(config["min_leg_probability"])
    min_pair = float(config["min_pair_probability"])

    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(plays):
        left_probability = _safe_float(left.get(probability_field))
        if left_probability is None or left_probability < min_leg:
            continue
        left_quality = _leg_quality(left, probability_field)

        left_player = _normalized_text(left.get("player_display_name") or left.get("player"))
        left_team = _normalized_text(left.get("team"))
        left_target = _normalized_text(left.get("target"))
        left_direction = _normalized_text(left.get("direction"))
        left_game = _normalized_text(left.get("game_id") or left.get("game_key"))
        left_script_cluster = _normalized_text(left.get("script_cluster_id"))

        for right_index in range(left_index + 1, len(plays)):
            right = plays[right_index]
            right_probability = _safe_float(right.get(probability_field))
            if right_probability is None or right_probability < min_leg:
                continue
            right_quality = _leg_quality(right, probability_field)

            right_player = _normalized_text(right.get("player_display_name") or right.get("player"))
            right_team = _normalized_text(right.get("team"))
            right_target = _normalized_text(right.get("target"))
            right_direction = _normalized_text(right.get("direction"))
            right_game = _normalized_text(right.get("game_id") or right.get("game_key"))
            right_script_cluster = _normalized_text(right.get("script_cluster_id"))

            same_player = bool(left_player and left_player == right_player)
            same_game = bool(left_game and left_game == right_game)
            same_team = bool(left_team and left_team == right_team)
            same_target = bool(left_target and left_target == right_target)
            same_direction = bool(left_direction and left_direction == right_direction)
            same_script_cluster = bool(left_script_cluster and left_script_cluster == right_script_cluster)

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

            independent_probability = left_probability * right_probability
            projected_probability = max(0.0, min(1.0, independent_probability * factor))
            if projected_probability < min_pair:
                continue

            diversity_bonus = 1.0
            if not same_game:
                diversity_bonus += 0.05
            if not same_team:
                diversity_bonus += 0.03
            if same_player:
                diversity_bonus -= 0.10
            if same_game:
                diversity_bonus -= 0.04
            if same_script_cluster:
                diversity_bonus -= 0.03

            avg_leg_quality = (left_quality + right_quality) / 2.0
            quality_factor = 0.85 + (0.30 * avg_leg_quality)
            pair_score = projected_probability * max(0.75, diversity_bonus) * quality_factor
            pairs.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_key": left.get("play_key"),
                    "right_key": right.get("play_key"),
                    "left_name": left.get("player_display_name") or left.get("player"),
                    "right_name": right.get("player_display_name") or right.get("player"),
                    "left_target": left.get("target"),
                    "right_target": right.get("target"),
                    "left_direction": left.get("direction"),
                    "right_direction": right.get("direction"),
                    "projected_probability": projected_probability,
                    "independent_probability": independent_probability,
                    "pair_score": pair_score,
                    "avg_leg_quality": avg_leg_quality,
                    "adjustment_factor": factor,
                    "same_player": same_player,
                    "same_game": same_game,
                    "same_team": same_team,
                    "same_target": same_target,
                    "same_direction": same_direction,
                }
            )

    pairs.sort(
        key=lambda row: (
            float(row["pair_score"]),
            float(row["projected_probability"]),
            float(row["independent_probability"]),
        ),
        reverse=True,
    )
    return pairs


def annotate_parlay_board(
    plays: list[dict[str, Any]],
    *,
    sport: str,
    probability_field: str,
    min_leg_probability: float | None = None,
    min_pair_probability: float | None = None,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    config = _resolve_sport_config(
        sport,
        min_leg_probability=min_leg_probability,
        min_pair_probability=min_pair_probability,
        max_pairs=max_pairs,
    )
    prepared: list[dict[str, Any]] = []
    for index, play in enumerate(plays):
        item = dict(play)
        item["play_key"] = _play_key(item, index)
        item["parlay_tag"] = ""
        item["parlay_candidate"] = False
        item["parlay_pair_rank"] = None
        item["parlay_score"] = None
        item["parlay_projected_hit_rate"] = None
        item["parlay_partner_key"] = None
        item["parlay_partner_name"] = None
        prepared.append(item)

    candidate_pairs = score_candidate_pairs(
        prepared,
        sport=sport,
        probability_field=probability_field,
        min_leg_probability=float(config["min_leg_probability"]),
        min_pair_probability=float(config["min_pair_probability"]),
    )

    selection_mode = "strict"
    if not candidate_pairs:
        fallback_pairs = score_candidate_pairs(
            prepared,
            sport=sport,
            probability_field=probability_field,
            min_leg_probability=float(config.get("fallback_min_leg_probability", config["min_leg_probability"])),
            min_pair_probability=float(config.get("fallback_min_pair_probability", config["min_pair_probability"])),
        )
        if fallback_pairs:
            candidate_pairs = fallback_pairs
            selection_mode = "fallback"
            non_negative_ev_pairs = [
                pair
                for pair in candidate_pairs
                if (_safe_float(prepared[int(pair["left_index"])].get("ev")) or 0.0) >= 0.0
                and (_safe_float(prepared[int(pair["right_index"])].get("ev")) or 0.0) >= 0.0
            ]
            if non_negative_ev_pairs:
                candidate_pairs = non_negative_ev_pairs

    selected_pairs: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    max_pairs_to_select = int(
        config["max_pairs"]
        if selection_mode == "strict"
        else config.get("fallback_max_pairs", config["max_pairs"])
    )
    for pair in candidate_pairs:
        left_index = int(pair["left_index"])
        right_index = int(pair["right_index"])
        if left_index in used_indices or right_index in used_indices:
            continue
        selected_pairs.append(dict(pair))
        used_indices.add(left_index)
        used_indices.add(right_index)
        if len(selected_pairs) >= max_pairs_to_select:
            break

    for pair_rank, pair in enumerate(selected_pairs, start=1):
        left_index = int(pair["left_index"])
        right_index = int(pair["right_index"])
        left = prepared[left_index]
        right = prepared[right_index]
        pair["pair_rank"] = pair_rank
        pair["legs"] = [
            {
                "play_key": left["play_key"],
                "player": left.get("player_display_name") or left.get("player"),
                "target": left.get("target"),
                "direction": left.get("direction"),
            },
            {
                "play_key": right["play_key"],
                "player": right.get("player_display_name") or right.get("player"),
                "target": right.get("target"),
                "direction": right.get("direction"),
            },
        ]
        for current, partner in ((left, right), (right, left)):
            current["parlay_tag"] = "parlay"
            current["parlay_candidate"] = True
            current["parlay_pair_rank"] = pair_rank
            current["parlay_score"] = pair["pair_score"]
            current["parlay_projected_hit_rate"] = pair["projected_probability"]
            current["parlay_partner_key"] = partner["play_key"]
            current["parlay_partner_name"] = partner.get("player_display_name") or partner.get("player")

    tagged_probability = [
        float(pair["projected_probability"])
        for pair in selected_pairs
        if _safe_float(pair.get("projected_probability")) is not None
    ]

    summary = {
        "selection_mode": selection_mode,
        "candidate_pair_count": int(len(candidate_pairs)),
        "selected_pair_count": int(len(selected_pairs)),
        "tagged_play_count": int(sum(1 for play in prepared if play["parlay_candidate"])),
        "avg_projected_pair_hit_rate": _safe_mean(tagged_probability),
        "best_projected_pair_hit_rate": max(tagged_probability) if tagged_probability else None,
        "min_leg_probability": float(config["min_leg_probability"]),
        "min_pair_probability": float(config["min_pair_probability"]),
        "fallback_min_leg_probability": float(config.get("fallback_min_leg_probability", config["min_leg_probability"])),
        "fallback_min_pair_probability": float(config.get("fallback_min_pair_probability", config["min_pair_probability"])),
    }
    return {
        "plays": prepared,
        "pairs": selected_pairs,
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
        return {"available": False, "reason": "no dated rows available for pair validation"}

    selected_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    dates_with_candidates = 0

    for market_date, part in working.groupby(date_col, dropna=False):
        rows = part.to_dict(orient="records")
        for index, row in enumerate(rows):
            row["play_key"] = _play_key(row, index)
        candidate_pairs = score_candidate_pairs(
            rows,
            sport=sport,
            probability_field=probability_col,
            min_leg_probability=min_leg_probability,
            min_pair_probability=min_pair_probability,
        )
        if not candidate_pairs:
            continue

        dates_with_candidates += 1
        for pair in candidate_pairs:
            left_result = str(rows[int(pair["left_index"])].get(result_col, "unresolved"))
            right_result = str(rows[int(pair["right_index"])].get(result_col, "unresolved"))
            baseline_records.append(
                {
                    "market_date": market_date,
                    "pair_outcome": _pair_outcome(left_result, right_result),
                    "projected_probability": pair["projected_probability"],
                }
            )

        chosen = 0
        used_indices: set[int] = set()
        for pair in candidate_pairs:
            left_index = int(pair["left_index"])
            right_index = int(pair["right_index"])
            if left_index in used_indices or right_index in used_indices:
                continue
            left_result = str(rows[left_index].get(result_col, "unresolved"))
            right_result = str(rows[right_index].get(result_col, "unresolved"))
            selected_records.append(
                {
                    "market_date": market_date,
                    "pair_outcome": _pair_outcome(left_result, right_result),
                    "projected_probability": pair["projected_probability"],
                }
            )
            used_indices.add(left_index)
            used_indices.add(right_index)
            chosen += 1
            if chosen >= int(max_pairs_per_day):
                break

    if not selected_records:
        return {"available": False, "reason": "no historical pair candidates met thresholds"}

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
            "graded_pair_count": graded,
            "hit_pair_count": int(outcome_counts.get("hit", 0)),
            "miss_pair_count": int(outcome_counts.get("miss", 0)),
            "push_pair_count": int(outcome_counts.get("push", 0)),
            "unresolved_pair_count": int(outcome_counts.get("unresolved", 0)),
            "pair_hit_rate": (_safe_mean([1.0] * int(outcome_counts.get("hit", 0)) + [0.0] * int(outcome_counts.get("miss", 0))) if graded else None),
            "avg_projected_pair_hit_rate": _safe_mean(projected),
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
        "selected": selected_summary,
        "baseline_all_pairs": baseline_summary,
        "hit_rate_lift_vs_all_pairs": hit_rate_lift,
    }
