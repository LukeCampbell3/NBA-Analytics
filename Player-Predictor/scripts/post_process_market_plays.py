#!/usr/bin/env python3
"""
Post-process ranked market plays into a final actionable board.

This layer is intentionally separate from selection so we can:
- compute EV from expected win rate
- de-correlate by player
- filter to positive-EV / minimum-quality plays
- produce a tighter, final board for execution
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decision_engine.sizing import apply_tiered_bet_sizing
try:
    from decision_engine.uncertainty import (
        BELIEF_UNCERTAINTY_LOWER,
        BELIEF_UNCERTAINTY_UPPER,
        belief_confidence_factor,
        normalize_belief_uncertainty,
    )
except Exception:  # pragma: no cover - fallback for standalone execution
    BELIEF_UNCERTAINTY_LOWER = 0.75
    BELIEF_UNCERTAINTY_UPPER = 1.15

    def normalize_belief_uncertainty(value, default: float = 1.0, lower: float = BELIEF_UNCERTAINTY_LOWER, upper: float = BELIEF_UNCERTAINTY_UPPER):
        span = max(float(upper) - float(lower), 1e-9)
        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce").fillna(float(default))
            return ((numeric - float(lower)) / span).clip(lower=0.0, upper=1.0)
        try:
            numeric = float(value)
            if np.isnan(numeric):
                numeric = float(default)
        except Exception:
            numeric = float(default)
        return float(np.clip((numeric - float(lower)) / span, 0.0, 1.0))

    def belief_confidence_factor(value, default: float = 1.0, lower: float = BELIEF_UNCERTAINTY_LOWER, upper: float = BELIEF_UNCERTAINTY_UPPER):
        normalized = normalize_belief_uncertainty(value, default=default, lower=lower, upper=upper)
        if isinstance(normalized, pd.Series):
            return (1.0 - normalized).clip(lower=0.0, upper=1.0)
        return float(np.clip(1.0 - float(normalized), 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process selected market plays into a final board.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("model/analysis/upcoming_market_play_selector.csv"),
        help="Selector CSV from select_market_plays.py",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("model/analysis/final_market_plays.csv"),
        help="Output CSV path for the final board.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("model/analysis/final_market_plays.json"),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--american-odds",
        type=int,
        default=-110,
        help="Assumed book odds for EV calculation when actual odds are unavailable.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.0,
        help="Minimum expected value required to keep a play.",
    )
    parser.add_argument(
        "--min-final-confidence",
        type=float,
        default=0.03,
        help="Minimum final confidence required to keep a play.",
    )
    parser.add_argument(
        "--min-recommendation",
        type=str,
        default="consider",
        choices=["pass", "consider", "strong", "elite"],
        help="Lowest selector recommendation allowed into the final board.",
    )
    parser.add_argument(
        "--max-plays-per-player",
        type=int,
        default=1,
        help="Maximum number of plays to keep per player after ranking.",
    )
    parser.add_argument(
        "--max-plays-per-target",
        type=int,
        default=0,
        help="Maximum number of final plays to keep per target when target-specific caps are not supplied.",
    )
    parser.add_argument(
        "--max-pts-plays",
        type=int,
        default=6,
        help="Maximum final PTS plays to keep.",
    )
    parser.add_argument(
        "--max-trb-plays",
        type=int,
        default=4,
        help="Maximum final TRB plays to keep.",
    )
    parser.add_argument(
        "--max-ast-plays",
        type=int,
        default=2,
        help="Maximum final AST plays to keep.",
    )
    parser.add_argument(
        "--max-total-plays",
        type=int,
        default=12,
        help="Maximum number of final plays to keep overall.",
    )
    parser.add_argument(
        "--non-pts-min-gap-percentile",
        type=float,
        default=0.90,
        help="Minimum disagreement percentile required for TRB/AST plays.",
    )
    parser.add_argument(
        "--edge-adjust-k",
        type=float,
        default=0.30,
        help="Weight for edge-adjusted EV ranking.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="thompson_ev",
        choices=["ev_adjusted", "edge", "xgb_ltr", "robust_reranker", "thompson_ev", "set_theory", "edge_append_shadow"],
        help="Final ranking mode used before portfolio constraints are applied.",
    )
    parser.add_argument(
        "--max-plays-per-game",
        type=int,
        default=2,
        help="Maximum selected plays from the same game/event to limit correlation.",
    )
    parser.add_argument(
        "--max-plays-per-script-cluster",
        type=int,
        default=2,
        help="Maximum selected plays from the same inferred script cluster.",
    )
    parser.add_argument(
        "--thompson-temperature",
        type=float,
        default=1.0,
        help="Temperature on Thompson beta posterior sampling (>1 explores more).",
    )
    parser.add_argument(
        "--thompson-seed",
        type=int,
        default=17,
        help="Seed salt for deterministic Thompson sampling.",
    )
    parser.add_argument("--min-bet-win-rate", type=float, default=0.57, help="Minimum expected win rate required to place any bet.")
    parser.add_argument("--medium-bet-win-rate", type=float, default=0.60, help="Expected win rate for a medium-sized bet.")
    parser.add_argument("--full-bet-win-rate", type=float, default=0.65, help="Expected win rate for a full-sized bet.")
    parser.add_argument("--medium-tier-percentile", type=float, default=0.80, help="Minimum percentile for a medium-tier candidate.")
    parser.add_argument("--strong-tier-percentile", type=float, default=0.90, help="Minimum percentile for a strong-tier candidate.")
    parser.add_argument("--elite-tier-percentile", type=float, default=0.95, help="Minimum percentile for an elite-tier candidate.")
    parser.add_argument("--small-bet-fraction", type=float, default=0.005, help="Bankroll fraction for a small bet.")
    parser.add_argument("--medium-bet-fraction", type=float, default=0.010, help="Bankroll fraction for a medium bet.")
    parser.add_argument("--full-bet-fraction", type=float, default=0.015, help="Bankroll fraction for a full bet.")
    parser.add_argument("--max-bet-fraction", type=float, default=0.02, help="Maximum bankroll fraction per play.")
    parser.add_argument("--max-total-bet-fraction", type=float, default=0.05, help="Maximum total bankroll fraction across the board.")
    parser.add_argument(
        "--belief-uncertainty-lower",
        type=float,
        default=BELIEF_UNCERTAINTY_LOWER,
        help="Lower anchor used when converting latent belief uncertainty into a confidence penalty.",
    )
    parser.add_argument(
        "--belief-uncertainty-upper",
        type=float,
        default=BELIEF_UNCERTAINTY_UPPER,
        help="Upper anchor used when converting latent belief uncertainty into a confidence penalty.",
    )
    parser.add_argument(
        "--append-agreement-min",
        type=int,
        default=3,
        help="Minimum E/T/V agreement count required for append-only shadow candidates.",
    )
    parser.add_argument(
        "--append-edge-percentile-min",
        type=float,
        default=0.90,
        help="Minimum abs-edge percentile required for append-only shadow candidates.",
    )
    parser.add_argument(
        "--append-max-extra-plays",
        type=int,
        default=3,
        help="Maximum number of append-only shadow plays added beyond the edge base board.",
    )
    return parser.parse_args()


def american_profit_per_unit(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def recommendation_rank(label: str) -> int:
    order = {"elite": 0, "strong": 1, "consider": 2, "pass": 3}
    return order.get(str(label), 3)


def minimum_recommendation_rank(label: str) -> int:
    return {"elite": 0, "strong": 1, "consider": 2, "pass": 3}[label]


def _numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _stable_seed_from_row(row: pd.Series, base_seed: int) -> int:
    key = "|".join(
        [
            str(base_seed),
            str(row.get("player", "")),
            str(row.get("target", "")),
            str(row.get("market_date", "")),
            str(row.get("market_event_id", "")),
            str(row.get("direction", "")),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32 - 1)


def _build_game_key(df: pd.DataFrame) -> pd.Series:
    if "market_event_id" in df.columns:
        event_key = df["market_event_id"].astype(str).str.strip()
    else:
        event_key = pd.Series("", index=df.index, dtype=str)

    home = df.get("market_home_team", pd.Series("", index=df.index)).astype(str).str.strip()
    away = df.get("market_away_team", pd.Series("", index=df.index)).astype(str).str.strip()
    teams_sorted = np.where(home <= away, home + "@" + away, away + "@" + home)
    teams_sorted = pd.Series(teams_sorted, index=df.index, dtype=str)
    market_date = df.get("market_date", pd.Series("", index=df.index)).astype(str).str.slice(0, 10)
    player = df.get("player", pd.Series("", index=df.index)).astype(str).str.strip()
    target = df.get("target", pd.Series("", index=df.index)).astype(str).str.strip()
    teams_missing = home.eq("") & away.eq("")
    fallback_team_key = market_date + "|" + teams_sorted
    fallback_player_key = market_date + "|" + player + "|" + target
    fallback_key = pd.Series(
        np.where(teams_missing, fallback_player_key, fallback_team_key),
        index=df.index,
        dtype=str,
    )
    return np.where(event_key.ne("") & event_key.ne("nan"), event_key, fallback_key)


def _normalize_script_cluster(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    if text in {"", "nan", "none", "null", "unknown", "script=unknown", "uninferred", "script=uninferred"}:
        return ""
    return text


def _resolve_target_caps(
    ranked: pd.DataFrame,
    max_plays_per_target: int,
    max_target_plays: dict[str, int] | None,
) -> dict[str, int]:
    caps = {k: int(v) for k, v in (max_target_plays or {}).items()}
    if not caps and max_plays_per_target > 0:
        caps = {target: int(max_plays_per_target) for target in ranked.get("target", pd.Series(dtype=str)).astype(str).unique()}
    return caps


def _zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    std = float(numeric.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    mean = float(numeric.mean())
    return (numeric - mean) / std


def _append_rows_with_caps(
    ranked: pd.DataFrame,
    selected_rows: list[dict],
    seen_indices: set,
    player_counts: dict[str, int],
    target_counts: dict[str, int],
    game_counts: dict[str, int],
    script_cluster_counts: dict[str, int],
    caps: dict[str, int],
    max_plays_per_player: int,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
    max_total_plays: int,
    max_new_rows: int | None = None,
) -> int:
    if ranked.empty:
        return 0
    if max_new_rows is not None and int(max_new_rows) <= 0:
        return 0

    added = 0
    for row_index, row in ranked.iterrows():
        if max_total_plays > 0 and len(selected_rows) >= int(max_total_plays):
            break
        if max_new_rows is not None and added >= int(max_new_rows):
            break
        if row_index in seen_indices:
            continue

        player = str(row.get("player", ""))
        target = str(row.get("target", ""))
        game_key = str(row.get("game_key", ""))
        script_cluster = _normalize_script_cluster(row.get("script_cluster_id", ""))

        if max_plays_per_player > 0 and player_counts.get(player, 0) >= int(max_plays_per_player):
            continue
        target_cap = int(caps.get(target, 0))
        if target_cap > 0 and target_counts.get(target, 0) >= target_cap:
            continue
        if max_plays_per_game > 0 and game_counts.get(game_key, 0) >= int(max_plays_per_game):
            continue
        if (
            max_plays_per_script_cluster > 0
            and script_cluster
            and script_cluster_counts.get(script_cluster, 0) >= int(max_plays_per_script_cluster)
        ):
            continue

        selected_rows.append(row.to_dict())
        seen_indices.add(row_index)
        player_counts[player] = player_counts.get(player, 0) + 1
        target_counts[target] = target_counts.get(target, 0) + 1
        game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if script_cluster:
            script_cluster_counts[script_cluster] = script_cluster_counts.get(script_cluster, 0) + 1
        added += 1
    return added


def _selection_counters_from_rows(selected_rows: list[dict]) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    player_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    script_cluster_counts: dict[str, int] = {}
    for row in selected_rows:
        player = str(row.get("player", ""))
        target = str(row.get("target", ""))
        game_key = str(row.get("game_key", ""))
        script_cluster = _normalize_script_cluster(row.get("script_cluster_id", ""))
        player_counts[player] = player_counts.get(player, 0) + 1
        target_counts[target] = target_counts.get(target, 0) + 1
        game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if script_cluster:
            script_cluster_counts[script_cluster] = script_cluster_counts.get(script_cluster, 0) + 1
    return player_counts, target_counts, game_counts, script_cluster_counts


def _select_edge_append_shadow_board(
    candidates: pd.DataFrame,
    max_plays_per_player: int,
    max_plays_per_target: int,
    max_total_plays: int,
    max_target_plays: dict[str, int] | None,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
    append_agreement_min: int,
    append_edge_percentile_min: float,
    append_max_extra_plays: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    base_size = int(max_total_plays) if max_total_plays > 0 else int(len(candidates))
    if base_size <= 0:
        return candidates.iloc[0:0].copy()

    working = candidates.copy()
    working["_source_index"] = working.index

    # 1) Base board is pure edge, fully capped.
    edge_ranked = working.sort_values(["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).copy()
    base_board = _apply_portfolio_caps(
        edge_ranked,
        max_plays_per_player=max_plays_per_player,
        max_plays_per_target=max_plays_per_target,
        max_total_plays=base_size,
        max_target_plays=max_target_plays,
        max_plays_per_game=max_plays_per_game,
        max_plays_per_script_cluster=max_plays_per_script_cluster,
    )
    if base_board.empty:
        return base_board

    base_indices = set(pd.to_numeric(base_board["_source_index"], errors="coerce").dropna().astype(int).tolist())
    base_board = base_board.copy()
    base_board["append_shadow_added"] = False

    extra_cap = max(0, int(append_max_extra_plays))
    if extra_cap <= 0:
        base_board["append_anchor_member"] = 1
        base_board["append_agreement_count"] = np.nan
        base_board["append_edge_percentile"] = np.nan
        base_board["append_sources"] = ""
        return base_board

    # 2) Build strict append-only candidate gates.
    overfetch = int(np.clip(3 * base_size, 1, len(working)))
    idx_e = set(working.sort_values(["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).head(overfetch).index.tolist())
    idx_t = set(
        working.sort_values(["thompson_ev", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False, False])
        .head(overfetch)
        .index.tolist()
    )
    idx_v = set(working.sort_values(["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False]).head(overfetch).index.tolist())

    agreement = working.index.to_series().map(lambda idx: int(idx in idx_e) + int(idx in idx_t) + int(idx in idx_v)).astype(int)
    edge_pct = pd.to_numeric(working["abs_edge"], errors="coerce").fillna(0.0).rank(method="average", pct=True)

    working["append_agreement_count"] = agreement
    working["append_edge_percentile"] = edge_pct
    working["append_sources"] = working.apply(
        lambda row: ",".join(
            part
            for part, enabled in (
                ("E", bool(row.name in idx_e)),
                ("T", bool(row.name in idx_t)),
                ("V", bool(row.name in idx_v)),
            )
            if enabled
        ),
        axis=1,
    )
    quality_mask = (
        (_numeric_series(working, "market_books", 0.0) >= 4.0)
        & (_numeric_series(working, "history_rows", 0.0) >= 35.0)
        & (_numeric_series(working, "final_confidence", 0.0) >= 0.03)
    )
    append_mask = (
        (~working.index.isin(base_indices))
        & (working["append_agreement_count"] >= int(max(1, append_agreement_min)))
        & (working["append_edge_percentile"] >= float(np.clip(append_edge_percentile_min, 0.0, 1.0)))
        & quality_mask
    )
    append_ranked = working.loc[append_mask].sort_values(
        ["append_agreement_count", "append_edge_percentile", "edge", "expected_win_rate", "final_confidence"],
        ascending=[False, False, False, False, False],
    ).copy()

    selected_rows = [row.to_dict() for _, row in base_board.iterrows()]
    seen_indices = set(base_indices)
    player_counts, target_counts, game_counts, script_cluster_counts = _selection_counters_from_rows(selected_rows)

    caps = _resolve_target_caps(working, max_plays_per_target=max_plays_per_target, max_target_plays=max_target_plays)
    # Append mode is intentionally additive; widen target caps by extra_cap.
    widened_caps = {target: (int(cap) + extra_cap if int(cap) > 0 else 0) for target, cap in caps.items()}

    _append_rows_with_caps(
        append_ranked,
        selected_rows,
        seen_indices,
        player_counts,
        target_counts,
        game_counts,
        script_cluster_counts,
        widened_caps,
        max_plays_per_player=max_plays_per_player,
        max_plays_per_game=max_plays_per_game,
        max_plays_per_script_cluster=max_plays_per_script_cluster,
        max_total_plays=base_size + extra_cap,
        max_new_rows=extra_cap,
    )

    out = pd.DataFrame.from_records(selected_rows) if selected_rows else working.iloc[0:0].copy()
    if out.empty:
        return out
    out["_source_index"] = pd.to_numeric(out.get("_source_index"), errors="coerce").fillna(-1).astype(int)
    out["append_shadow_added"] = ~out["_source_index"].isin(base_indices)
    out["append_anchor_member"] = (~out["append_shadow_added"]).astype(int)
    out.loc[~out["append_shadow_added"], "append_sources"] = ""
    out.loc[~out["append_shadow_added"], "append_agreement_count"] = np.nan
    out.loc[~out["append_shadow_added"], "append_edge_percentile"] = np.nan
    return out


def _select_set_theory_board(
    candidates: pd.DataFrame,
    max_plays_per_player: int,
    max_plays_per_target: int,
    max_total_plays: int,
    max_target_plays: dict[str, int] | None,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    board_size = int(max_total_plays) if max_total_plays > 0 else int(len(candidates))
    if board_size <= 0:
        return candidates.iloc[0:0].copy()

    # Edge is the anchor universe. Thompson/EV are used as confirmation overlays.
    overfetch = int(np.clip(3 * board_size, 1, len(candidates)))
    edge_ranked = candidates.sort_values(["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).copy()
    thompson_ranked = candidates.sort_values(
        ["thompson_ev", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False, False]
    ).copy()
    ev_ranked = candidates.sort_values(["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False]).copy()

    edge_idx = set(edge_ranked.head(overfetch).index.tolist())
    thompson_idx = set(thompson_ranked.head(overfetch).index.tolist())
    ev_idx = set(ev_ranked.head(overfetch).index.tolist())

    scored = candidates.loc[candidates.index.isin(edge_idx)].copy()
    if scored.empty:
        return scored
    scored["in_edge_set"] = True
    scored["in_thompson_set"] = scored.index.isin(thompson_idx)
    scored["in_ev_set"] = scored.index.isin(ev_idx)
    scored["agreement_count"] = 1 + scored["in_thompson_set"].astype(int) + scored["in_ev_set"].astype(int)
    scored["set_sources"] = scored.apply(
        lambda row: ",".join(
            part
            for part, enabled in (
                ("E", bool(row.get("in_edge_set"))),
                ("T", bool(row.get("in_thompson_set"))),
                ("V", bool(row.get("in_ev_set"))),
            )
            if enabled
        ),
        axis=1,
    )

    scored["set_group"] = "anchor_fallback"
    scored.loc[scored["in_thompson_set"] | scored["in_ev_set"], "set_group"] = "strong_expansion"
    scored.loc[scored["in_thompson_set"] & scored["in_ev_set"], "set_group"] = "core"
    scored["set_strength"] = scored["set_group"].map({"core": 3, "strong_expansion": 2, "anchor_fallback": 1}).fillna(0).astype(int)

    scored["z_edge"] = _zscore_series(scored["edge"])
    scored["z_expected_win_rate"] = _zscore_series(scored["expected_win_rate"])
    scored["z_ev_adjusted"] = _zscore_series(scored["ev_adjusted"])
    scored["consensus_score"] = (
        0.45 * scored["z_edge"]
        + 0.20 * scored["z_expected_win_rate"]
        + 0.20 * scored["z_ev_adjusted"]
        + 0.15 * (scored["agreement_count"] - 1.0)
    )

    sort_consensus = ["consensus_score", "agreement_count", "edge", "expected_win_rate", "ev_adjusted", "abs_edge", "final_confidence"]
    core_ranked = scored.loc[scored["set_group"].eq("core")].sort_values(sort_consensus, ascending=[False] * len(sort_consensus)).copy()
    strong_ranked = scored.loc[scored["set_group"].eq("strong_expansion")].sort_values(sort_consensus, ascending=[False] * len(sort_consensus)).copy()
    fallback_ranked = scored.loc[scored["set_group"].eq("anchor_fallback")].sort_values(
        ["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]
    ).copy()

    selected_rows: list[dict] = []
    seen_indices: set = set()
    player_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    script_cluster_counts: dict[str, int] = {}
    caps = _resolve_target_caps(scored, max_plays_per_target=max_plays_per_target, max_target_plays=max_target_plays)

    _append_rows_with_caps(
        core_ranked,
        selected_rows,
        seen_indices,
        player_counts,
        target_counts,
        game_counts,
        script_cluster_counts,
        caps,
        max_plays_per_player=max_plays_per_player,
        max_plays_per_game=max_plays_per_game,
        max_plays_per_script_cluster=max_plays_per_script_cluster,
        max_total_plays=board_size,
    )
    _append_rows_with_caps(
        strong_ranked,
        selected_rows,
        seen_indices,
        player_counts,
        target_counts,
        game_counts,
        script_cluster_counts,
        caps,
        max_plays_per_player=max_plays_per_player,
        max_plays_per_game=max_plays_per_game,
        max_plays_per_script_cluster=max_plays_per_script_cluster,
        max_total_plays=board_size,
    )

    if len(selected_rows) < board_size:
        _append_rows_with_caps(
            fallback_ranked,
            selected_rows,
            seen_indices,
            player_counts,
            target_counts,
            game_counts,
            script_cluster_counts,
            caps,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
            max_total_plays=board_size,
            max_new_rows=board_size - len(selected_rows),
        )

    if not selected_rows:
        return scored.iloc[0:0].copy()
    return pd.DataFrame.from_records(selected_rows)


def _apply_portfolio_caps(
    ranked: pd.DataFrame,
    max_plays_per_player: int,
    max_plays_per_target: int,
    max_total_plays: int,
    max_target_plays: dict[str, int] | None,
    max_plays_per_game: int,
    max_plays_per_script_cluster: int,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    selected_rows: list[dict] = []
    player_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    script_cluster_counts: dict[str, int] = {}

    caps = _resolve_target_caps(ranked, max_plays_per_target=max_plays_per_target, max_target_plays=max_target_plays)

    for _, row in ranked.iterrows():
        if max_total_plays > 0 and len(selected_rows) >= int(max_total_plays):
            break

        player = str(row.get("player", ""))
        target = str(row.get("target", ""))
        game_key = str(row.get("game_key", ""))
        script_cluster = _normalize_script_cluster(row.get("script_cluster_id", ""))

        if max_plays_per_player > 0 and player_counts.get(player, 0) >= int(max_plays_per_player):
            continue
        target_cap = int(caps.get(target, 0))
        if target_cap > 0 and target_counts.get(target, 0) >= target_cap:
            continue
        if max_plays_per_game > 0 and game_counts.get(game_key, 0) >= int(max_plays_per_game):
            continue
        if (
            max_plays_per_script_cluster > 0
            and script_cluster
            and script_cluster_counts.get(script_cluster, 0) >= int(max_plays_per_script_cluster)
        ):
            continue

        selected_rows.append(row.to_dict())
        player_counts[player] = player_counts.get(player, 0) + 1
        target_counts[target] = target_counts.get(target, 0) + 1
        game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if script_cluster:
            script_cluster_counts[script_cluster] = script_cluster_counts.get(script_cluster, 0) + 1

    if not selected_rows:
        return ranked.iloc[0:0].copy()
    return pd.DataFrame.from_records(selected_rows)


def compute_final_board(
    plays: pd.DataFrame,
    american_odds: int = -110,
    min_ev: float = 0.0,
    min_final_confidence: float = 0.02,
    min_recommendation: str = "consider",
    selection_mode: str = "thompson_ev",
    ranking_mode: str = "ev_adjusted",
    max_plays_per_player: int = 1,
    max_plays_per_target: int = 8,
    max_total_plays: int = 20,
    max_target_plays: dict[str, int] | None = None,
    max_plays_per_game: int = 2,
    max_plays_per_script_cluster: int = 2,
    non_pts_min_gap_percentile: float = 0.90,
    edge_adjust_k: float = 0.30,
    thompson_temperature: float = 1.0,
    thompson_seed: int = 17,
    min_bet_win_rate: float = 0.57,
    medium_bet_win_rate: float = 0.60,
    full_bet_win_rate: float = 0.65,
    medium_tier_percentile: float = 0.80,
    strong_tier_percentile: float = 0.90,
    elite_tier_percentile: float = 0.95,
    small_bet_fraction: float = 0.005,
    medium_bet_fraction: float = 0.010,
    full_bet_fraction: float = 0.015,
    max_bet_fraction: float = 0.02,
    max_total_bet_fraction: float = 0.05,
    belief_uncertainty_lower: float = BELIEF_UNCERTAINTY_LOWER,
    belief_uncertainty_upper: float = BELIEF_UNCERTAINTY_UPPER,
    append_agreement_min: int = 3,
    append_edge_percentile_min: float = 0.90,
    append_max_extra_plays: int = 3,
) -> pd.DataFrame:
    out = plays.copy()
    if out.empty:
        return out

    payout = american_profit_per_unit(american_odds)
    out["expected_win_rate"] = pd.to_numeric(out["expected_win_rate"], errors="coerce").fillna(0.5)
    out["expected_push_rate"] = _numeric_series(out, "expected_push_rate", 0.0).clip(lower=0.0, upper=1.0)
    if "expected_loss_rate" in out.columns:
        out["expected_loss_rate"] = _numeric_series(out, "expected_loss_rate", 0.0)
    else:
        out["expected_loss_rate"] = np.clip(1.0 - out["expected_win_rate"] - out["expected_push_rate"], 0.0, 1.0)
    out["gap_percentile"] = pd.to_numeric(out["gap_percentile"], errors="coerce").fillna(0.0)
    out["belief_uncertainty"] = _numeric_series(out, "belief_uncertainty", 1.0)
    normalized_belief = normalize_belief_uncertainty(
        out["belief_uncertainty"],
        default=1.0,
        lower=float(belief_uncertainty_lower),
        upper=float(belief_uncertainty_upper),
    )
    belief_conf = belief_confidence_factor(
        out["belief_uncertainty"],
        default=1.0,
        lower=float(belief_uncertainty_lower),
        upper=float(belief_uncertainty_upper),
    )
    if "belief_uncertainty_normalized" in out.columns:
        out["belief_uncertainty_normalized"] = pd.to_numeric(out["belief_uncertainty_normalized"], errors="coerce").fillna(normalized_belief)
    else:
        out["belief_uncertainty_normalized"] = normalized_belief
    if "belief_confidence_factor" in out.columns:
        out["belief_confidence_factor"] = pd.to_numeric(out["belief_confidence_factor"], errors="coerce").fillna(belief_conf)
    else:
        out["belief_confidence_factor"] = belief_conf
    out["feasibility"] = _numeric_series(out, "feasibility", 0.0)
    out["abs_edge"] = _numeric_series(out, "abs_edge", 0.0)
    out["edge"] = _numeric_series(out, "edge", 0.0)
    out["posterior_alpha"] = _numeric_series(out, "posterior_alpha", 1.0)
    out["posterior_beta"] = _numeric_series(out, "posterior_beta", 1.0)
    out["posterior_variance"] = _numeric_series(out, "posterior_variance", 0.25)

    out["game_key"] = _build_game_key(out)
    out["market_prior_win_rate"] = 0.5
    base_confidence = out["gap_percentile"] * out["belief_confidence_factor"] * np.clip(out["feasibility"], 0.0, None)
    confidence_blend = np.clip(pd.to_numeric(base_confidence, errors="coerce").fillna(0.0), 0.0, 1.0)
    uncertainty_penalty = np.clip(np.sqrt(np.clip(out["posterior_variance"], 0.0, 1.0)) * 0.6, 0.0, 0.45)
    out["calibration_blend_weight"] = np.clip(confidence_blend * (1.0 - uncertainty_penalty), 0.10, 0.95)
    out["calibrated_win_rate"] = (
        out["calibration_blend_weight"] * out["expected_win_rate"]
        + (1.0 - out["calibration_blend_weight"]) * out["market_prior_win_rate"]
    )
    out["expected_win_rate"] = out["calibrated_win_rate"].clip(lower=0.0, upper=1.0 - out["expected_push_rate"])
    out["expected_loss_rate"] = np.clip(1.0 - out["expected_win_rate"] - out["expected_push_rate"], 0.0, 1.0)

    out["ev"] = out["expected_win_rate"] * payout - out["expected_loss_rate"]
    out["final_confidence"] = base_confidence
    out["recommendation_rank"] = out["recommendation"].map(recommendation_rank)
    edge_baseline = out.groupby("target")["abs_edge"].transform(lambda s: s.median() if len(s) else 1.0).replace(0.0, 1.0)
    out["edge_scale"] = (out["abs_edge"] / edge_baseline).clip(lower=0.50, upper=2.50)
    out["ev_adjusted"] = out["ev"] * (1.0 + float(edge_adjust_k) * (out["edge_scale"] - 1.0))
    out["ranking_mode"] = str(selection_mode or ranking_mode)

    temp = max(float(thompson_temperature), 1e-6)
    out["thompson_alpha"] = out["posterior_alpha"] / temp
    out["thompson_beta"] = out["posterior_beta"] / temp

    thompson_conditional: list[float] = []
    for _, sample_row in out.iterrows():
        seed = _stable_seed_from_row(sample_row, int(thompson_seed))
        rng = np.random.default_rng(seed)
        alpha = max(0.1, float(sample_row["thompson_alpha"]))
        beta = max(0.1, float(sample_row["thompson_beta"]))
        thompson_conditional.append(float(rng.beta(alpha, beta)))
    out["thompson_conditional_win_rate"] = thompson_conditional
    resolved_share = np.clip(1.0 - out["expected_push_rate"], 0.0, 1.0)
    out["thompson_win_rate"] = np.clip(out["thompson_conditional_win_rate"] * resolved_share, 0.0, resolved_share)
    out["thompson_loss_rate"] = np.clip(resolved_share - out["thompson_win_rate"], 0.0, 1.0)
    out["thompson_ev"] = out["thompson_win_rate"] * payout - out["thompson_loss_rate"]

    if "conditional_eligible_for_board" in out.columns:
        eligible_mask = pd.to_numeric(out["conditional_eligible_for_board"], errors="coerce").fillna(0).astype(bool)
        out = out.loc[eligible_mask].copy()
        if out.empty:
            return out

    out = out.loc[out["recommendation_rank"] <= minimum_recommendation_rank(min_recommendation)].copy()
    out = out.loc[out["ev"] >= float(min_ev)].copy()
    out = out.loc[out["final_confidence"] >= float(min_final_confidence)].copy()
    out = out.loc[(out["target"] == "PTS") | (out["gap_percentile"] >= float(non_pts_min_gap_percentile))].copy()
    if out.empty:
        return out

    effective_mode = str(selection_mode or ranking_mode)
    rank_columns = ["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    caps_already_applied = False
    if effective_mode == "xgb_ltr" and "xgb_ltr_score" in out.columns:
        out["xgb_ltr_score"] = pd.to_numeric(out["xgb_ltr_score"], errors="coerce").fillna(-1.0)
        rank_columns = ["xgb_ltr_score", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    elif effective_mode == "robust_reranker" and "robust_reranker_prob" in out.columns:
        out["robust_reranker_prob"] = pd.to_numeric(out["robust_reranker_prob"], errors="coerce").fillna(-1.0)
        out["robust_reranker_blend_raw"] = _numeric_series(out, "robust_reranker_blend_raw", -1.0)
        rank_columns = ["robust_reranker_prob", "robust_reranker_blend_raw", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    elif effective_mode == "edge":
        rank_columns = ["edge", "abs_edge", "expected_win_rate", "final_confidence"]
    elif effective_mode == "thompson_ev":
        rank_columns = ["thompson_ev", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    elif effective_mode == "set_theory":
        out = _select_set_theory_board(
            out,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_target=max_plays_per_target,
            max_total_plays=max_total_plays,
            max_target_plays=max_target_plays,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
        )
        caps_already_applied = True
        rank_columns = ["set_strength", "consensus_score", "agreement_count", "expected_win_rate", "ev_adjusted", "abs_edge"]
    elif effective_mode == "edge_append_shadow":
        out = _select_edge_append_shadow_board(
            out,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_target=max_plays_per_target,
            max_total_plays=max_total_plays,
            max_target_plays=max_target_plays,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
            append_agreement_min=append_agreement_min,
            append_edge_percentile_min=append_edge_percentile_min,
            append_max_extra_plays=append_max_extra_plays,
        )
        caps_already_applied = True
        rank_columns = ["append_anchor_member", "edge", "abs_edge", "expected_win_rate", "final_confidence"]

    if not caps_already_applied:
        out = out.sort_values(rank_columns, ascending=[False] * len(rank_columns)).copy()
        out = _apply_portfolio_caps(
            out,
            max_plays_per_player=max_plays_per_player,
            max_plays_per_target=max_plays_per_target,
            max_total_plays=max_total_plays,
            max_target_plays=max_target_plays,
            max_plays_per_game=max_plays_per_game,
            max_plays_per_script_cluster=max_plays_per_script_cluster,
        )
    else:
        if out.empty:
            return out
        missing_rank_cols = [column for column in rank_columns if column not in out.columns]
        if missing_rank_cols:
            return out.iloc[0:0].copy()
        out = out.sort_values(rank_columns, ascending=[False] * len(rank_columns)).copy()
    if out.empty:
        return out
    out = apply_tiered_bet_sizing(
        out,
        expected_win_rate_col="expected_win_rate",
        gap_percentile_col="gap_percentile",
        min_bet_win_rate=min_bet_win_rate,
        medium_bet_win_rate=medium_bet_win_rate,
        full_bet_win_rate=full_bet_win_rate,
        medium_tier_percentile=medium_tier_percentile,
        strong_tier_percentile=strong_tier_percentile,
        elite_tier_percentile=elite_tier_percentile,
        small_bet_fraction=small_bet_fraction,
        medium_bet_fraction=medium_bet_fraction,
        full_bet_fraction=full_bet_fraction,
        max_bet_fraction=max_bet_fraction,
        max_total_bet_fraction=max_total_bet_fraction,
    )
    out["expected_profit_fraction"] = pd.to_numeric(out["bet_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(out["ev"], errors="coerce").fillna(0.0)
    active_mask = pd.to_numeric(out["bet_fraction"], errors="coerce").fillna(0.0) > 0.0
    if active_mask.any():
        out = out.loc[active_mask].copy()
    else:
        # Fallback: keep a small fractional allocation on the best-ranked plays
        # so the board remains actionable when strict tier gates reject all rows.
        fallback_fraction = float(np.clip(small_bet_fraction, 0.0, max_bet_fraction))
        if fallback_fraction <= 0.0:
            return out.iloc[0:0].copy()
        out = out.head(max_total_plays if max_total_plays > 0 else len(out)).copy()
        out["allocation_tier"] = "fallback_small"
        out["allocation_action"] = "fallback_small"
        out["bet_fraction"] = fallback_fraction
        if max_total_bet_fraction > 0 and float(out["bet_fraction"].sum()) > float(max_total_bet_fraction):
            scale = float(max_total_bet_fraction) / float(out["bet_fraction"].sum())
            out["bet_fraction"] = out["bet_fraction"] * scale
        out["expected_profit_fraction"] = pd.to_numeric(out["bet_fraction"], errors="coerce").fillna(0.0) * pd.to_numeric(out["ev"], errors="coerce").fillna(0.0)
    out = out.sort_values(rank_columns, ascending=[False] * len(rank_columns)).copy()
    out["selected_rank"] = np.arange(1, len(out) + 1)
    if "_source_index" in out.columns:
        out = out.drop(columns=["_source_index"])
    out = out.drop(columns=["recommendation_rank"])
    out = out.reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Selector CSV not found: {csv_path}")

    plays = pd.read_csv(csv_path)
    final_board = compute_final_board(
        plays,
        american_odds=args.american_odds,
        min_ev=args.min_ev,
        min_final_confidence=args.min_final_confidence,
        min_recommendation=args.min_recommendation,
        selection_mode=args.selection_mode,
        ranking_mode=args.selection_mode,
        max_plays_per_player=args.max_plays_per_player,
        max_plays_per_target=args.max_plays_per_target,
        max_total_plays=args.max_total_plays,
        max_target_plays={"PTS": args.max_pts_plays, "TRB": args.max_trb_plays, "AST": args.max_ast_plays},
        max_plays_per_game=args.max_plays_per_game,
        max_plays_per_script_cluster=args.max_plays_per_script_cluster,
        non_pts_min_gap_percentile=args.non_pts_min_gap_percentile,
        edge_adjust_k=args.edge_adjust_k,
        thompson_temperature=args.thompson_temperature,
        thompson_seed=args.thompson_seed,
        min_bet_win_rate=args.min_bet_win_rate,
        medium_bet_win_rate=args.medium_bet_win_rate,
        full_bet_win_rate=args.full_bet_win_rate,
        medium_tier_percentile=args.medium_tier_percentile,
        strong_tier_percentile=args.strong_tier_percentile,
        elite_tier_percentile=args.elite_tier_percentile,
        small_bet_fraction=args.small_bet_fraction,
        medium_bet_fraction=args.medium_bet_fraction,
        full_bet_fraction=args.full_bet_fraction,
        max_bet_fraction=args.max_bet_fraction,
        max_total_bet_fraction=args.max_total_bet_fraction,
        belief_uncertainty_lower=args.belief_uncertainty_lower,
        belief_uncertainty_upper=args.belief_uncertainty_upper,
        append_agreement_min=args.append_agreement_min,
        append_edge_percentile_min=args.append_edge_percentile_min,
        append_max_extra_plays=args.append_max_extra_plays,
    )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    final_board.to_csv(args.csv_out, index=False)

    payload = {
        "source_csv": str(csv_path),
        "rows_in": int(len(plays)),
        "rows_out": int(len(final_board)),
        "american_odds": int(args.american_odds),
        "min_ev": float(args.min_ev),
        "min_final_confidence": float(args.min_final_confidence),
        "min_recommendation": args.min_recommendation,
        "selection_mode": args.selection_mode,
        "max_plays_per_player": int(args.max_plays_per_player),
        "max_plays_per_target": int(args.max_plays_per_target),
        "max_plays_per_game": int(args.max_plays_per_game),
        "max_plays_per_script_cluster": int(args.max_plays_per_script_cluster),
        "max_pts_plays": int(args.max_pts_plays),
        "max_trb_plays": int(args.max_trb_plays),
        "max_ast_plays": int(args.max_ast_plays),
        "max_total_plays": int(args.max_total_plays),
        "non_pts_min_gap_percentile": float(args.non_pts_min_gap_percentile),
        "edge_adjust_k": float(args.edge_adjust_k),
        "thompson_temperature": float(args.thompson_temperature),
        "thompson_seed": int(args.thompson_seed),
        "min_bet_win_rate": float(args.min_bet_win_rate),
        "medium_bet_win_rate": float(args.medium_bet_win_rate),
        "full_bet_win_rate": float(args.full_bet_win_rate),
        "medium_tier_percentile": float(args.medium_tier_percentile),
        "strong_tier_percentile": float(args.strong_tier_percentile),
        "elite_tier_percentile": float(args.elite_tier_percentile),
        "small_bet_fraction": float(args.small_bet_fraction),
        "medium_bet_fraction": float(args.medium_bet_fraction),
        "full_bet_fraction": float(args.full_bet_fraction),
        "max_bet_fraction": float(args.max_bet_fraction),
        "max_total_bet_fraction": float(args.max_total_bet_fraction),
        "belief_uncertainty_lower": float(args.belief_uncertainty_lower),
        "belief_uncertainty_upper": float(args.belief_uncertainty_upper),
        "append_agreement_min": int(args.append_agreement_min),
        "append_edge_percentile_min": float(args.append_edge_percentile_min),
        "append_max_extra_plays": int(args.append_max_extra_plays),
        "top_plays": final_board.head(20).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("FINAL MARKET PLAY BOARD")
    print("=" * 90)
    print(f"Input rows:   {len(plays)}")
    print(f"Output rows:  {len(final_board)}")
    print(f"CSV:          {args.csv_out}")
    print(f"JSON:         {args.json_out}")
    if not final_board.empty:
        show_cols = [
            "player",
            "target",
            "direction",
            "prediction",
            "market_line",
            "abs_edge",
            "expected_win_rate",
            "expected_push_rate",
            "ev",
            "ev_adjusted",
            "thompson_ev",
            "final_confidence",
            "selected_rank",
            "allocation_tier",
            "allocation_action",
            "bet_fraction",
            "recommendation",
        ]
        print("\nTop final plays:")
        print(final_board[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
