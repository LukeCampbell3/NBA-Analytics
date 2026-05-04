from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from .gating import StrategyConfig


@dataclass
class BoardOptimizationResult:
    selected_board: pd.DataFrame
    candidate_pool: pd.DataFrame
    board_score: float
    beam_width: int


def numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def string_series(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in df.columns:
        return df[column].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def resolve_target_caps(config: StrategyConfig) -> dict[str, int]:
    if config.max_plays_per_target > 0:
        cap = int(config.max_plays_per_target)
        return {"PTS": cap, "TRB": cap, "AST": cap}
    return {"PTS": int(config.max_pts_plays), "TRB": int(config.max_trb_plays), "AST": int(config.max_ast_plays)}


def build_target_direction_key(target: pd.Series, direction: pd.Series) -> pd.Series:
    return target.fillna("").astype(str).str.upper().str.strip() + "|" + direction.fillna("").astype(str).str.upper().str.strip()


def _normalize_script_cluster(value: object) -> str:
    token = str(value or "").strip()
    if not token or token.lower() == "script=unknown":
        return ""
    return token


def _build_game_key(df: pd.DataFrame) -> pd.Series:
    # Keep portfolio grouping aligned with the script-layer optimizer:
    # use canonical `game_key`, never the legacy `game_id`.
    existing = string_series(df, "game_key").str.strip()
    event = string_series(df, "market_event_id").str.strip()
    home = string_series(df, "market_home_team").str.strip()
    away = string_series(df, "market_away_team").str.strip()
    teams = (home + "::" + away).str.strip(":")
    fallback = event.where(event != "", teams)
    return existing.where(existing != "", fallback)


def _normalize_belief_uncertainty(values: pd.Series, config: StrategyConfig) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").fillna(1.0)
    span = max(float(config.belief_uncertainty_upper) - float(config.belief_uncertainty_lower), 1e-9)
    return ((raw - float(config.belief_uncertainty_lower)) / span).clip(lower=0.0, upper=1.0)


def prepare_board_candidates(candidates: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    out = candidates.copy()
    out["player"] = string_series(out, "player").str.strip()
    out["target"] = string_series(out, "target").str.upper().str.strip()
    out["direction"] = string_series(out, "direction").str.upper().str.strip()
    missing_direction = out["direction"] == ""
    if missing_direction.any():
        edge = numeric_series(out, "edge", 0.0)
        out.loc[missing_direction, "direction"] = np.where(
            edge.loc[missing_direction] > 0.0,
            "OVER",
            np.where(edge.loc[missing_direction] < 0.0, "UNDER", "PUSH"),
        )
    out["game_key"] = _build_game_key(out).fillna("").astype(str).str.strip()
    out["script_cluster_id"] = string_series(out, "script_cluster_id").map(_normalize_script_cluster)
    out["belief_uncertainty"] = numeric_series(out, "belief_uncertainty", 1.0)
    out["belief_uncertainty_normalized"] = _normalize_belief_uncertainty(out["belief_uncertainty"], config)
    out["expected_win_rate"] = numeric_series(out, "expected_win_rate", 0.0)
    out["final_confidence"] = numeric_series(out, "final_confidence", 0.0)
    out["abs_edge"] = numeric_series(out, "abs_edge", 0.0)
    out["ev_adjusted"] = numeric_series(out, "ev_adjusted", np.nan).fillna(numeric_series(out, "ev", 0.0))
    out["board_play_win_prob"] = numeric_series(out, "board_play_win_prob", np.nan).fillna(out["expected_win_rate"])
    out["board_target_empirical_win_rate"] = numeric_series(out, "board_target_empirical_win_rate", np.nan)
    missing_target_empirical = out["board_target_empirical_win_rate"].isna()
    if missing_target_empirical.any():
        out.loc[missing_target_empirical, "board_target_empirical_win_rate"] = (
            out.loc[missing_target_empirical].groupby("target")["expected_win_rate"].transform("median").fillna(out["expected_win_rate"])
        )
    out["board_segment_empirical_win_rate"] = numeric_series(out, "board_segment_empirical_win_rate", np.nan)
    missing_segment_empirical = out["board_segment_empirical_win_rate"].isna()
    if missing_segment_empirical.any():
        segment_fallback = build_target_direction_key(out["target"], out["direction"])
        out.loc[missing_segment_empirical, "board_segment_empirical_win_rate"] = (
            out.loc[missing_segment_empirical]
            .assign(_segment_key=segment_fallback.loc[missing_segment_empirical])
            .groupby("_segment_key")["expected_win_rate"]
            .transform("median")
            .fillna(out.loc[missing_segment_empirical, "expected_win_rate"])
            .to_numpy(dtype="float64")
        )
    out["board_segment_empirical_rows"] = numeric_series(out, "board_segment_empirical_rows", 0.0)
    out["board_segment_recent_weakness"] = numeric_series(out, "board_segment_recent_weakness", 0.0).clip(lower=0.0, upper=1.0)
    out["board_target_direction_key"] = build_target_direction_key(out["target"], out["direction"])
    out["final_pool_quality_score"] = numeric_series(out, "final_pool_quality_score", 0.50).clip(lower=0.0, upper=1.0)
    out["board_objective_base_score"] = (
        1.00 * out["ev_adjusted"]
        + 0.35 * out["expected_win_rate"]
        + 0.20 * out["final_confidence"]
        + 0.10 * out["abs_edge"]
        + float(config.final_pool_quality_weight) * out["final_pool_quality_score"]
        - 0.20 * out["belief_uncertainty_normalized"]
    )
    return out


def _resolve_target_direction_constraints(
    candidates: pd.DataFrame,
    target_caps: dict[str, int],
    config: StrategyConfig,
    target_size: int,
) -> tuple[dict[str, int], dict[str, int], pd.DataFrame]:
    columns = [
        "board_target_direction_key",
        "board_target_direction_cap_effective",
        "board_target_direction_reserve_min",
        "board_target_direction_strength",
        "board_target_direction_penalty",
    ]
    annotated = candidates.copy()
    if annotated.empty:
        for column in columns:
            annotated[column] = pd.Series(dtype="float64" if column != "board_target_direction_key" else "object")
        return {}, {}, annotated
    if not bool(config.target_direction_allocation_enabled):
        annotated["board_target_direction_cap_effective"] = 0
        annotated["board_target_direction_reserve_min"] = 0
        annotated["board_target_direction_strength"] = 0.0
        annotated["board_target_direction_penalty"] = 0.0
        return {}, {}, annotated

    summary = (
        annotated.groupby("board_target_direction_key", dropna=False)
        .agg(
            target=("target", "first"),
            direction=("direction", "first"),
            candidate_count=("board_target_direction_key", "size"),
            empirical_rate=("board_segment_empirical_win_rate", "median"),
            empirical_rows=("board_segment_empirical_rows", "max"),
            selection_prob=("board_play_win_prob", "mean"),
            target_rate=("board_target_empirical_win_rate", "median"),
            recent_weakness=("board_segment_recent_weakness", "mean"),
        )
        .reset_index()
    )
    if summary.empty:
        annotated["board_target_direction_cap_effective"] = 0
        annotated["board_target_direction_reserve_min"] = 0
        annotated["board_target_direction_strength"] = 0.0
        annotated["board_target_direction_penalty"] = 0.0
        return {}, {}, annotated

    global_prob = float(pd.to_numeric(summary["selection_prob"], errors="coerce").fillna(0.5).mean())
    global_rate = float(
        np.average(
            pd.to_numeric(summary["empirical_rate"], errors="coerce").fillna(global_prob).to_numpy(dtype="float64"),
            weights=np.maximum(pd.to_numeric(summary["empirical_rows"], errors="coerce").fillna(1.0).to_numpy(dtype="float64"), 1.0),
        )
    )
    summary["target_rate"] = pd.to_numeric(summary["target_rate"], errors="coerce").fillna(pd.to_numeric(summary["empirical_rate"], errors="coerce").fillna(global_rate))
    summary["support_ok"] = pd.to_numeric(summary["empirical_rows"], errors="coerce").fillna(0.0) >= float(config.target_direction_min_rows)
    summary["strength"] = (
        0.70 * (pd.to_numeric(summary["empirical_rate"], errors="coerce").fillna(global_rate) - pd.to_numeric(summary["target_rate"], errors="coerce").fillna(global_rate))
        + 0.25 * (pd.to_numeric(summary["selection_prob"], errors="coerce").fillna(global_prob) - global_prob)
        - 0.20 * pd.to_numeric(summary["recent_weakness"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    )

    segment_caps: dict[str, int] = {}
    for _, row in summary.iterrows():
        segment_key = str(row["board_target_direction_key"])
        target = str(row["target"]).upper().strip()
        target_cap = int(target_caps.get(target, 0))
        if target_cap <= 0:
            continue
        candidate_count = int(pd.to_numeric(pd.Series([row["candidate_count"]]), errors="coerce").fillna(0).iloc[0])
        if candidate_count <= 0:
            continue
        support_ok = bool(row["support_ok"])
        empirical_rate = float(pd.to_numeric(pd.Series([row["empirical_rate"]]), errors="coerce").fillna(global_rate).iloc[0])
        strength = float(pd.to_numeric(pd.Series([row["strength"]]), errors="coerce").fillna(0.0).iloc[0])
        recent_weakness = float(pd.to_numeric(pd.Series([row["recent_weakness"]]), errors="coerce").fillna(0.0).iloc[0])
        effective_cap = target_cap
        if support_ok and (
            empirical_rate <= float(config.target_direction_very_weak_rate)
            or (recent_weakness >= 0.22 and strength <= -0.01)
        ):
            effective_cap = min(target_cap, int(config.target_direction_max_very_weak_cap))
        elif support_ok and (
            empirical_rate <= float(config.target_direction_weak_rate)
            or strength <= -0.02
            or recent_weakness >= 0.16
        ):
            effective_cap = min(target_cap, int(config.target_direction_max_weak_cap))
        segment_caps[segment_key] = max(0, min(candidate_count, int(effective_cap)))

    reserve_budget = max(0, min(int(config.target_direction_reserve_max_slots), int(target_size)))
    reserve_candidates = summary.loc[
        summary["support_ok"]
        & (pd.to_numeric(summary["empirical_rate"], errors="coerce").fillna(global_rate) >= float(config.target_direction_strong_rate))
        & (pd.to_numeric(summary["selection_prob"], errors="coerce").fillna(global_prob) >= float(config.target_direction_reserve_min_prob))
        & (pd.to_numeric(summary["strength"], errors="coerce").fillna(0.0) >= 0.0)
    ].sort_values(["strength", "empirical_rate", "empirical_rows"], ascending=[False, False, False])

    segment_reserves: dict[str, int] = {}
    reserved_targets: set[str] = set()
    for _, row in reserve_candidates.iterrows():
        if reserve_budget <= 0:
            break
        segment_key = str(row["board_target_direction_key"])
        target = str(row["target"]).upper().strip()
        effective_cap = int(segment_caps.get(segment_key, int(target_caps.get(target, 0))))
        if effective_cap <= 0 or target in reserved_targets:
            continue
        segment_reserves[segment_key] = 1
        reserved_targets.add(target)
        reserve_budget -= 1

    strength_map = summary.set_index("board_target_direction_key")["strength"].to_dict()
    rate_map = pd.to_numeric(summary.set_index("board_target_direction_key")["empirical_rate"], errors="coerce").fillna(global_rate).to_dict()
    weakness_map = pd.to_numeric(summary.set_index("board_target_direction_key")["recent_weakness"], errors="coerce").fillna(0.0).to_dict()
    rows_map = pd.to_numeric(summary.set_index("board_target_direction_key")["empirical_rows"], errors="coerce").fillna(0.0).to_dict()

    def _penalty_for(segment_key: str) -> float:
        empirical_rate = float(rate_map.get(segment_key, global_rate))
        recent_weakness = float(np.clip(weakness_map.get(segment_key, 0.0), 0.0, 1.0))
        rows = float(rows_map.get(segment_key, 0.0))
        strength = float(strength_map.get(segment_key, 0.0))
        support_scale = float(np.clip(rows / max(float(config.target_direction_min_rows), 1.0), 0.0, 1.0))
        weak_gap = float(np.clip(float(config.target_direction_weak_rate) - empirical_rate, 0.0, 0.20))
        penalty = float(config.target_direction_weak_penalty_weight) * support_scale * (weak_gap / 0.20)
        penalty += float(config.target_direction_recent_weakness_penalty_weight) * recent_weakness
        if strength <= -0.02:
            penalty += 0.5 * float(config.target_direction_weak_penalty_weight) * support_scale
        return float(max(0.0, penalty))

    penalty_map = {key: _penalty_for(str(key)) for key in summary["board_target_direction_key"].tolist()}
    annotated["board_target_direction_cap_effective"] = annotated["board_target_direction_key"].map(segment_caps).fillna(0).astype("int64")
    annotated["board_target_direction_reserve_min"] = annotated["board_target_direction_key"].map(segment_reserves).fillna(0).astype("int64")
    annotated["board_target_direction_strength"] = annotated["board_target_direction_key"].map(strength_map).fillna(0.0).astype("float64")
    annotated["board_target_direction_penalty"] = annotated["board_target_direction_key"].map(penalty_map).fillna(0.0).astype("float64")
    return segment_caps, segment_reserves, annotated


def _pairwise_dependency_penalty(board: pd.DataFrame, config: StrategyConfig) -> float:
    if len(board) <= 1:
        return 0.0

    players = board["player"].to_numpy(dtype=str)
    targets = board["target"].to_numpy(dtype=str)
    directions = board["direction"].to_numpy(dtype=str)
    games = board["game_key"].to_numpy(dtype=str)
    scripts = board["script_cluster_id"].to_numpy(dtype=str)

    penalty = 0.0
    for left in range(len(board)):
        for right in range(left + 1, len(board)):
            if players[left] and players[left] == players[right]:
                penalty += float(config.board_objective_corr_same_player)
            if games[left] and games[left] == games[right]:
                penalty += float(config.board_objective_corr_same_game)
            if targets[left] and targets[left] == targets[right]:
                penalty += float(config.board_objective_corr_same_target)
            if directions[left] and directions[left] == directions[right]:
                penalty += float(config.board_objective_corr_same_direction)
            if scripts[left] and scripts[left] == scripts[right]:
                penalty += float(config.board_objective_corr_same_script_cluster)
    return penalty / max(len(board), 1)


def _field_concentration_penalty(values: pd.Series, weight: float) -> float:
    filtered = values.loc[values.astype(str).str.strip() != ""].astype(str)
    if filtered.empty:
        return 0.0
    counts = filtered.value_counts()
    pair_count = float(((counts * (counts - 1)) / 2.0).sum())
    return float(weight) * pair_count / max(len(filtered), 1)


def board_penalty(board: pd.DataFrame, config: StrategyConfig) -> float:
    if board.empty:
        return 0.0

    concentration = 0.0
    concentration += _field_concentration_penalty(board["player"], weight=0.75)
    concentration += _field_concentration_penalty(board["game_key"], weight=0.35)
    concentration += _field_concentration_penalty(board["target"], weight=0.20)
    concentration += _field_concentration_penalty(board["direction"], weight=0.15)
    concentration += _field_concentration_penalty(board["script_cluster_id"], weight=0.20)

    uncertainty = float(pd.to_numeric(board["belief_uncertainty_normalized"], errors="coerce").fillna(0.0).mean())
    dependency = _pairwise_dependency_penalty(board, config)

    # --- Direction diversity bonus: reduce penalty when the board has a mix
    # of OVER and UNDER picks.  A board that is 100% one direction carries
    # more correlated risk than a balanced board.  The bonus is small enough
    # that it never overrides quality, but large enough to break ties in
    # favour of diversity when qualified OVERs exist.
    direction_series = board["direction"].astype(str).str.upper().str.strip()
    n_over = int((direction_series == "OVER").sum())
    n_under = int((direction_series == "UNDER").sum())
    n_total = max(1, n_over + n_under)
    direction_balance = float(min(n_over, n_under)) / float(n_total)  # 0 = mono, 0.5 = perfect balance
    direction_diversity_bonus = 0.025 * direction_balance  # max ~0.0125 bonus

    return (
        float(config.board_objective_lambda_corr) * dependency
        + float(config.board_objective_lambda_conc) * concentration
        + float(config.board_objective_lambda_unc) * uncertainty
        - direction_diversity_bonus
    )


def board_score(board: pd.DataFrame, config: StrategyConfig) -> float:
    if board.empty:
        return -np.inf
    base = float(pd.to_numeric(board["board_objective_base_score"], errors="coerce").fillna(0.0).sum())
    return base - board_penalty(board, config)


def _board_passes_caps(
    board: pd.DataFrame,
    config: StrategyConfig,
    target_size: int,
    target_direction_caps: dict[str, int] | None = None,
    target_direction_reserves: dict[str, int] | None = None,
) -> bool:
    if board.empty:
        return True
    if target_size > 0 and len(board) > int(target_size):
        return False

    if int(config.max_plays_per_player) > 0:
        player_counts = board["player"].astype(str).value_counts()
        if not player_counts.empty and int(player_counts.max()) > int(config.max_plays_per_player):
            return False

    caps = resolve_target_caps(config)
    target_counts = board["target"].astype(str).value_counts()
    for target, count in target_counts.items():
        target_cap = int(caps.get(str(target), 0))
        if target_cap > 0 and int(count) > target_cap:
            return False

    segment_caps = {str(key): int(value) for key, value in (target_direction_caps or {}).items()}
    if segment_caps:
        segment_counts = build_target_direction_key(board["target"], board["direction"]).astype(str).value_counts()
        for segment_key, count in segment_counts.items():
            if segment_key in segment_caps and int(count) > int(segment_caps[segment_key]):
                return False

    if int(config.max_plays_per_game) > 0:
        game_counts = board.loc[board["game_key"].astype(str).str.strip() != "", "game_key"].astype(str).value_counts()
        if not game_counts.empty and int(game_counts.max()) > int(config.max_plays_per_game):
            return False

    if int(config.max_plays_per_script_cluster) > 0:
        script_counts = board.loc[board["script_cluster_id"].astype(str).str.strip() != "", "script_cluster_id"].astype(str).value_counts()
        if not script_counts.empty and int(script_counts.max()) > int(config.max_plays_per_script_cluster):
            return False

    if target_direction_reserves:
        segment_counts = build_target_direction_key(board["target"], board["direction"]).astype(str).value_counts()
        for segment_key, minimum in target_direction_reserves.items():
            if int(segment_counts.get(str(segment_key), 0)) < int(minimum):
                return False

    return True


def _candidate_limit(candidate_count: int, target_size: int, config: StrategyConfig) -> int:
    if candidate_count <= 0:
        return 0
    if target_size <= 0:
        return candidate_count

    # Beam search can tolerate a wider universe than the script-layer exact solver,
    # but we still default to the same 30-36 candidate range unless the policy says otherwise.
    overfetch_count = int(ceil(max(float(config.board_objective_overfetch), 1.0) * int(target_size)))
    base_limit = max(int(target_size), overfetch_count)
    hard_cap = int(config.board_objective_candidate_limit)
    if hard_cap > 0:
        base_limit = min(base_limit, hard_cap)
    return min(candidate_count, max(int(target_size), base_limit))


def _beam_width(candidate_count: int, target_size: int, config: StrategyConfig) -> int:
    if candidate_count <= 0:
        return 0
    search_budget = max(int(config.board_objective_max_search_nodes), 1000)
    width = search_budget // max(candidate_count * max(target_size, 1), 1)
    return int(np.clip(width, 16, 128))


def optimize_board(candidates: pd.DataFrame, config: StrategyConfig) -> BoardOptimizationResult:
    prepared = prepare_board_candidates(candidates, config)
    if prepared.empty:
        empty = prepared.iloc[0:0].copy()
        return BoardOptimizationResult(selected_board=empty, candidate_pool=empty, board_score=-np.inf, beam_width=0)

    target_size = int(config.max_total_plays) if int(config.max_total_plays) > 0 else int(len(prepared))
    target_size = min(target_size, int(len(prepared)))
    if target_size <= 0:
        empty = prepared.iloc[0:0].copy()
        return BoardOptimizationResult(selected_board=empty, candidate_pool=empty, board_score=-np.inf, beam_width=0)

    target_caps = resolve_target_caps(config)
    target_direction_caps, target_direction_reserves, prepared = _resolve_target_direction_constraints(prepared, target_caps, config, target_size)
    prepared["board_objective_base_score"] = prepared["board_objective_base_score"] - pd.to_numeric(
        prepared.get("board_target_direction_penalty"),
        errors="coerce",
    ).fillna(0.0)

    prepared = prepared.sort_values(
        ["board_objective_base_score", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"],
        ascending=[False, False, False, False, False],
    ).copy()
    candidate_limit = _candidate_limit(len(prepared), target_size, config)
    candidate_pool = prepared.head(candidate_limit).copy()
    if candidate_pool.empty:
        empty = prepared.iloc[0:0].copy()
        return BoardOptimizationResult(selected_board=empty, candidate_pool=empty, board_score=-np.inf, beam_width=0)

    beam_width = _beam_width(len(candidate_pool), target_size, config)
    beams: list[tuple[tuple[int, ...], float]] = [(tuple(), 0.0)]

    for idx in candidate_pool.index.tolist():
        next_beams = list(beams)
        for selected_indices, _ in beams:
            if len(selected_indices) >= target_size:
                continue
            trial_indices = selected_indices + (int(idx),)
            trial_board = candidate_pool.loc[list(trial_indices)].copy()
            if not _board_passes_caps(
                trial_board,
                config,
                target_size,
                target_direction_caps=target_direction_caps,
                target_direction_reserves=None,
            ):
                continue
            next_beams.append((trial_indices, board_score(trial_board, config)))

        pruned: list[tuple[tuple[int, ...], float]] = []
        for size in sorted({len(indices) for indices, _ in next_beams}):
            bucket = [state for state in next_beams if len(state[0]) == size]
            bucket.sort(key=lambda item: item[1], reverse=True)
            pruned.extend(bucket[:beam_width])
        beams = pruned

    required_size = max(1, min(int(config.min_board_plays), target_size)) if int(config.min_board_plays) > 0 else target_size
    full_size_states = [state for state in beams if len(state[0]) == target_size]
    min_size_states = [state for state in beams if len(state[0]) >= required_size]
    non_empty_states = [state for state in beams if len(state[0]) > 0]
    reserve_qualified = [
        state
        for state in (full_size_states or min_size_states or non_empty_states or beams)
        if _board_passes_caps(
            candidate_pool.loc[list(state[0])].copy(),
            config,
            target_size,
            target_direction_caps=target_direction_caps,
            target_direction_reserves=target_direction_reserves,
        )
    ]
    selected_indices, score = max(reserve_qualified or full_size_states or min_size_states or non_empty_states or beams, key=lambda item: item[1])

    if not selected_indices:
        selected_board = candidate_pool.iloc[0:0].copy()
    else:
        selected_board = candidate_pool.loc[list(selected_indices)].copy().sort_values(
            ["board_objective_base_score", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"],
            ascending=[False, False, False, False, False],
        )
        selected_board["board_objective_score"] = float(score)
        selected_board["board_objective_beam_width"] = int(beam_width)
        selected_board["board_objective_candidate_count"] = int(len(candidate_pool))
        selected_board["board_objective_solver_mode"] = "beam_search_v1"

    candidate_pool["board_objective_beam_width"] = int(beam_width)
    candidate_pool["board_objective_candidate_count"] = int(len(candidate_pool))
    candidate_pool["board_objective_solver_mode"] = "beam_search_v1"
    return BoardOptimizationResult(
        selected_board=selected_board,
        candidate_pool=candidate_pool,
        board_score=float(score),
        beam_width=int(beam_width),
    )
