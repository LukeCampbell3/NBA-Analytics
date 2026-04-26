#!/usr/bin/env python3
"""
Export the latest daily market prediction run to a static web JSON payload.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parent.parent
SPORT_ROOT = PLAYER_PREDICTOR_ROOT.parents[1]
WORKSPACE_ROOT = SPORT_ROOT.parents[1]
DAILY_RUNS_ROOT = PLAYER_PREDICTOR_ROOT / "model" / "analysis" / "daily_runs"
DEFAULT_WEB_JSON = SPORT_ROOT / "web" / "data" / "daily_predictions.json"
DEFAULT_DIST_JSON = WORKSPACE_ROOT / "dist" / "nba" / "data" / "daily_predictions.json"
DEFAULT_CARDS_JSON = SPORT_ROOT / "web" / "data" / "cards.json"

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sports.parlay_analysis import annotate_parlay_board, evaluate_historical_parlays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily market predictions to static web JSON.")
    parser.add_argument("--manifest", type=Path, default=None, help="Explicit daily pipeline manifest JSON.")
    parser.add_argument("--daily-runs-root", type=Path, default=DAILY_RUNS_ROOT, help="Root directory containing daily run folders.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_WEB_JSON, help="Static web JSON output path (web/data).")
    parser.add_argument("--out-dist", type=Path, default=DEFAULT_DIST_JSON, help="Static dist JSON output path (dist/data).")
    parser.add_argument(
        "--cards-json",
        type=Path,
        default=DEFAULT_CARDS_JSON,
        help="Optional cards.json used to enrich output with player ids/headshot URLs.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class PlayerIdentityLookup:
    name_to_id: dict[str, int]
    abbr_to_id: dict[str, int]


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator is None:
        return None
    try:
        den = float(denominator)
        if den == 0:
            return None
        return float(numerator) / den
    except Exception:
        return None


def summarize_accuracy_bucket(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "signal_count": 0,
            "settled_count": 0,
            "graded_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "push_count": 0,
            "unresolved_count": 0,
            "win_rate": None,
            "loss_rate": None,
            "push_rate": None,
            "roi_per_graded_play": None,
            "unit_profit": 0.0,
            "mean_abs_error": None,
            "rmse": None,
            "mean_abs_edge": None,
        }

    signal_count = int(len(rows))
    settled_mask = rows["outcome"].isin(["win", "loss", "push"])
    graded_mask = rows["outcome"].isin(["win", "loss"])
    settled = rows.loc[settled_mask]
    graded = rows.loc[graded_mask]

    win_count = int((rows["outcome"] == "win").sum())
    loss_count = int((rows["outcome"] == "loss").sum())
    push_count = int((rows["outcome"] == "push").sum())
    unresolved_count = int((rows["outcome"] == "unresolved").sum())
    settled_count = int(len(settled))
    graded_count = int(len(graded))

    unit_profit = (win_count * (100.0 / 110.0)) - loss_count

    abs_error = pd.to_numeric(settled.get("abs_error"), errors="coerce")
    sq_error = pd.to_numeric(settled.get("sq_error"), errors="coerce")
    abs_edge = pd.to_numeric(rows.get("abs_edge"), errors="coerce")

    mean_abs_error = safe_float(abs_error.mean())
    rmse = safe_float(math.sqrt(sq_error.mean())) if sq_error.notna().any() else None
    mean_abs_edge = safe_float(abs_edge.mean())

    return {
        "signal_count": signal_count,
        "settled_count": settled_count,
        "graded_count": graded_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "push_count": push_count,
        "unresolved_count": unresolved_count,
        "win_rate": safe_float(safe_div(win_count, graded_count)),
        "loss_rate": safe_float(safe_div(loss_count, graded_count)),
        "push_rate": safe_float(safe_div(push_count, settled_count)),
        "roi_per_graded_play": safe_float(safe_div(unit_profit, graded_count)),
        "unit_profit": safe_float(unit_profit),
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "mean_abs_edge": mean_abs_edge,
    }


def build_accuracy_metrics(history_csv_path: Path | None) -> dict:
    if history_csv_path is None:
        return {"available": False, "reason": "history csv path not provided"}
    if not history_csv_path.exists():
        return {"available": False, "reason": f"history csv not found: {history_csv_path}"}

    try:
        history = pd.read_csv(history_csv_path)
    except Exception as exc:
        return {"available": False, "reason": f"failed reading history csv: {exc}"}

    if history.empty:
        return {"available": False, "reason": "history csv is empty"}

    row_parts: list[pd.DataFrame] = []
    for target in ("PTS", "TRB", "AST"):
        pred_col = f"pred_{target}"
        market_col = f"market_{target}"
        actual_col = f"actual_{target}"
        required_cols = {"market_date", pred_col, market_col, actual_col}
        if not required_cols.issubset(set(history.columns)):
            continue

        part = history.loc[:, [pred_col, market_col, actual_col, "market_date"]].copy()
        part["prediction"] = pd.to_numeric(part[pred_col], errors="coerce")
        part["market_line"] = pd.to_numeric(part[market_col], errors="coerce")
        part["actual"] = pd.to_numeric(part[actual_col], errors="coerce")
        part["market_date"] = pd.to_datetime(part["market_date"], errors="coerce").dt.strftime("%Y-%m-%d")

        part = part.loc[part["prediction"].notna() & part["market_line"].notna()].copy()
        if part.empty:
            continue

        part = part.loc[part["prediction"] != part["market_line"]].copy()
        if part.empty:
            continue

        part["target"] = target
        part["direction"] = part["prediction"].gt(part["market_line"]).map({True: "OVER", False: "UNDER"})
        part["outcome"] = "unresolved"
        actual_known = part["actual"].notna()
        push_mask = actual_known & part["actual"].eq(part["market_line"])
        over_win_mask = actual_known & part["direction"].eq("OVER") & part["actual"].gt(part["market_line"])
        under_win_mask = actual_known & part["direction"].eq("UNDER") & part["actual"].lt(part["market_line"])
        win_mask = over_win_mask | under_win_mask
        loss_mask = actual_known & ~push_mask & ~win_mask
        part.loc[push_mask, "outcome"] = "push"
        part.loc[win_mask, "outcome"] = "win"
        part.loc[loss_mask, "outcome"] = "loss"
        part["abs_error"] = (part["prediction"] - part["actual"]).abs()
        part["sq_error"] = (part["prediction"] - part["actual"]).pow(2)
        part["abs_edge"] = (part["prediction"] - part["market_line"]).abs()
        row_parts.append(part[["market_date", "target", "direction", "outcome", "abs_error", "sq_error", "abs_edge"]])

    if not row_parts:
        return {"available": False, "reason": "history csv did not contain usable target columns"}

    rows = pd.concat(row_parts, ignore_index=True)
    if rows.empty:
        return {"available": False, "reason": "history csv produced no usable resolved rows"}

    market_dates = pd.to_datetime(rows["market_date"], errors="coerce")
    as_of_market_date = None
    if market_dates.notna().any():
        as_of_market_date = market_dates.max().strftime("%Y-%m-%d")

    by_target = {
        target: summarize_accuracy_bucket(rows.loc[rows["target"] == target].copy())
        for target in ("PTS", "TRB", "AST")
        if not rows.loc[rows["target"] == target].empty
    }
    by_direction = {
        direction: summarize_accuracy_bucket(rows.loc[rows["direction"] == direction].copy())
        for direction in ("OVER", "UNDER")
        if not rows.loc[rows["direction"] == direction].empty
    }

    return {
        "available": True,
        "source_history_csv": str(history_csv_path),
        "as_of_market_date": as_of_market_date,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall": summarize_accuracy_bucket(rows),
        "by_target": by_target,
        "by_direction": by_direction,
    }


def heuristic_nba_leg_probability(target: str, prediction: float, market_line: float) -> float:
    gap = abs(float(prediction) - float(market_line))
    scale_map = {"PTS": 3.4, "TRB": 2.2, "AST": 2.0}
    scale = float(scale_map.get(str(target).upper(), 2.8))
    return max(0.5, min(0.86, 0.5 + (0.23 * math.tanh(gap / max(scale, 1e-6)))))


def build_parlay_validation(history_csv_path: Path | None) -> dict:
    if history_csv_path is None:
        return {"available": False, "reason": "history csv path not provided"}
    if not history_csv_path.exists():
        return {"available": False, "reason": f"history csv not found: {history_csv_path}"}

    try:
        history = pd.read_csv(history_csv_path)
    except Exception as exc:
        return {"available": False, "reason": f"failed reading history csv: {exc}"}

    if history.empty:
        return {"available": False, "reason": "history csv is empty"}

    row_parts: list[pd.DataFrame] = []
    for target in ("PTS", "TRB", "AST"):
        pred_col = f"pred_{target}"
        market_col = f"market_{target}"
        actual_col = f"actual_{target}"
        required_cols = {"market_date", pred_col, market_col, actual_col}
        if not required_cols.issubset(set(history.columns)):
            continue

        part = history.loc[:, [pred_col, market_col, actual_col, "market_date"]].copy()
        for optional_col in ("Player", "player", "Team", "Opponent", "Game_ID"):
            if optional_col in history.columns:
                part[optional_col] = history[optional_col]
        part["prediction"] = pd.to_numeric(part[pred_col], errors="coerce")
        part["market_line"] = pd.to_numeric(part[market_col], errors="coerce")
        part["actual"] = pd.to_numeric(part[actual_col], errors="coerce")
        part = part.loc[part["prediction"].notna() & part["market_line"].notna() & part["actual"].notna()].copy()
        if part.empty:
            continue

        part = part.loc[part["prediction"] != part["market_line"]].copy()
        if part.empty:
            continue

        part["market_date"] = pd.to_datetime(part["market_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        part["target"] = target
        part["direction"] = part["prediction"].gt(part["market_line"]).map({True: "OVER", False: "UNDER"})
        part["estimated_win_rate"] = [
            heuristic_nba_leg_probability(target, pred, line)
            for pred, line in zip(part["prediction"], part["market_line"])
        ]
        part["player"] = (
            part.get("Player", pd.Series("", index=part.index))
            .fillna(part.get("player", pd.Series("", index=part.index)))
            .astype(str)
            .str.strip()
        )
        part["player_display_name"] = part["player"]
        part["team"] = part.get("Team", pd.Series("", index=part.index)).fillna("").astype(str).str.strip()
        part["opponent"] = part.get("Opponent", pd.Series("", index=part.index)).fillna("").astype(str).str.strip()
        part["game_id"] = part.get("Game_ID", pd.Series("", index=part.index)).fillna("").astype(str).str.strip()
        part["result"] = "loss"
        part.loc[part["actual"] == part["market_line"], "result"] = "push"
        over_win = part["direction"].eq("OVER") & part["actual"].gt(part["market_line"])
        under_win = part["direction"].eq("UNDER") & part["actual"].lt(part["market_line"])
        part.loc[over_win | under_win, "result"] = "win"
        row_parts.append(
            part[
                [
                    "market_date",
                    "player",
                    "player_display_name",
                    "team",
                    "opponent",
                    "game_id",
                    "target",
                    "direction",
                    "estimated_win_rate",
                    "result",
                ]
            ]
        )

    if not row_parts:
        return {"available": False, "reason": "history csv did not contain usable target columns for parlay validation"}

    rows = pd.concat(row_parts, ignore_index=True)
    if rows.empty:
        return {"available": False, "reason": "history csv produced no usable pair-validation rows"}

    summary = evaluate_historical_parlays(
        rows,
        sport="nba",
        date_col="market_date",
        probability_col="estimated_win_rate",
        result_col="result",
        max_pairs_per_day=1,
    )
    summary["source_history_csv"] = str(history_csv_path)
    summary["history_row_count"] = int(len(rows))
    return summary


def find_latest_manifest(root: Path) -> Path:
    manifests = sorted(root.glob("**/daily_market_pipeline_manifest_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError(f"No daily pipeline manifest found under {root}")
    return manifests[0]


def resolve_artifact_path(raw_path: str | None, manifest_dir: Path) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    local_fallback = manifest_dir / candidate.name
    if local_fallback.exists():
        return local_fallback
    return candidate


def normalize_player_key(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace(" ", "_")
    for old, new in [(".", ""), ("'", ""), ("`", ""), (",", ""), ("/", "-"), ("\\", "-"), (":", "")]:
        text = text.replace(old, new)
    text = "_".join(part for part in text.split("_") if part)
    folded = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return ascii_text.lower()


def abbreviate_player_key(value: str) -> str:
    normalized = normalize_player_key(value)
    if not normalized:
        return ""
    parts = [part for part in normalized.split("_") if part]
    if len(parts) < 2:
        return normalized
    return f"{parts[0][0]}_{'_'.join(parts[1:])}"


def display_name_from_csv_path(csv_path: str | None) -> str:
    raw = str(csv_path or "").strip()
    if not raw:
        return ""
    folder_name = Path(raw).parent.name
    if not folder_name:
        return ""
    pretty = re.sub(r"\s+", " ", folder_name.replace("_", " ").strip())
    lowered = pretty.lower()
    if lowered in {"data proc", "data-proc"}:
        return ""
    return pretty


def build_player_identity_lookup(cards_json_path: Path | None) -> PlayerIdentityLookup:
    if cards_json_path is None or not cards_json_path.exists():
        return PlayerIdentityLookup(name_to_id={}, abbr_to_id={})

    try:
        cards_payload = json.loads(cards_json_path.read_text(encoding="utf-8"))
    except Exception:
        return PlayerIdentityLookup(name_to_id={}, abbr_to_id={})

    if not isinstance(cards_payload, list):
        return PlayerIdentityLookup(name_to_id={}, abbr_to_id={})

    name_to_id: dict[str, int] = {}
    abbr_to_ids: dict[str, set[int]] = defaultdict(set)
    for item in cards_payload:
        if not isinstance(item, dict):
            continue
        player = item.get("player", {})
        if not isinstance(player, dict):
            continue
        name = str(player.get("name", "")).strip()
        raw_id = player.get("id")
        try:
            player_id = int(raw_id)
        except Exception:
            continue
        normalized = normalize_player_key(name)
        if normalized and normalized not in name_to_id:
            name_to_id[normalized] = player_id
        abbr = abbreviate_player_key(name)
        if abbr:
            abbr_to_ids[abbr].add(player_id)

    abbr_to_id = {key: next(iter(ids)) for key, ids in abbr_to_ids.items() if len(ids) == 1}
    return PlayerIdentityLookup(name_to_id=name_to_id, abbr_to_id=abbr_to_id)


def build_player_headshot_url(player_id: int | None) -> str | None:
    if player_id is None:
        return None
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"


def resolve_player_identity(row: pd.Series, lookup: PlayerIdentityLookup) -> tuple[str, int | None, str | None]:
    player_code = str(row.get("player", "")).strip()
    display_name = display_name_from_csv_path(row.get("csv"))
    if not display_name:
        market_player_raw = str(row.get("market_player_raw", "")).replace(".", " ").strip()
        display_name = re.sub(r"\s+", " ", market_player_raw) if market_player_raw else ""
    if not display_name:
        display_name = re.sub(r"\s+", " ", player_code.replace("_", " ")).strip()

    candidates = [
        display_name,
        player_code,
        str(row.get("market_player_raw", "")).strip(),
        str(row.get("player", "")).strip().replace("_", " "),
    ]

    player_id: int | None = None
    for candidate in candidates:
        normalized = normalize_player_key(candidate)
        if normalized and normalized in lookup.name_to_id:
            player_id = lookup.name_to_id[normalized]
            break
    if player_id is None:
        for candidate in candidates:
            abbr = abbreviate_player_key(candidate)
            if abbr and abbr in lookup.abbr_to_id:
                player_id = lookup.abbr_to_id[abbr]
                break

    return display_name, player_id, build_player_headshot_url(player_id)


def safe_float(value) -> float | None:
    try:
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def build_summary(plays: pd.DataFrame) -> dict:
    if plays.empty:
        return {
            "play_count": 0,
            "avg_expected_win_rate": None,
            "avg_ev": None,
            "avg_edge": None,
            "by_target": {},
            "by_recommendation": {},
        }

    return {
        "play_count": int(len(plays)),
        "avg_expected_win_rate": safe_float(pd.to_numeric(plays.get("expected_win_rate"), errors="coerce").mean()),
        "avg_ev": safe_float(pd.to_numeric(plays.get("ev"), errors="coerce").mean()),
        "avg_edge": safe_float(pd.to_numeric(plays.get("abs_edge"), errors="coerce").mean()),
        "total_bet_fraction": safe_float(pd.to_numeric(plays.get("bet_fraction"), errors="coerce").sum()),
        "avg_bet_fraction": safe_float(pd.to_numeric(plays.get("bet_fraction"), errors="coerce").mean()),
        "expected_profit_fraction": safe_float(pd.to_numeric(plays.get("expected_profit_fraction"), errors="coerce").sum()),
        "by_target": plays.get("target", pd.Series(dtype=str)).value_counts().to_dict(),
        "by_recommendation": plays.get("recommendation", pd.Series(dtype=str)).value_counts().to_dict(),
        "by_allocation_tier": plays.get("allocation_tier", pd.Series(dtype=str)).value_counts().to_dict(),
        "by_allocation_action": plays.get("allocation_action", pd.Series(dtype=str)).value_counts().to_dict(),
    }


def normalize_play_rows(plays: pd.DataFrame, identity_lookup: PlayerIdentityLookup) -> list[dict]:
    rows: list[dict] = []
    ordered = plays.copy().reset_index(drop=True)
    ordered["rank"] = ordered.index + 1
    for _, row in ordered.iterrows():
        player_display_name, player_id, player_headshot_url = resolve_player_identity(row, identity_lookup)
        rows.append(
            {
                "rank": int(row["rank"]),
                "player": str(row.get("player", "")),
                "player_display_name": str(player_display_name or row.get("player", "")),
                "player_id": int(player_id) if player_id is not None else None,
                "player_headshot_url": player_headshot_url,
                "target": str(row.get("target", "")),
                "direction": str(row.get("direction", "")),
                "team": str(row.get("team", "")),
                "opponent": str(row.get("opponent", "")),
                "market_date": str(row.get("market_date", "")) if pd.notna(row.get("market_date")) else None,
                "market_home_team": str(row.get("market_home_team", "")),
                "market_away_team": str(row.get("market_away_team", "")),
                "last_history_date": str(row.get("last_history_date", "")) if pd.notna(row.get("last_history_date")) else None,
                "prediction": safe_float(row.get("prediction")),
                "market_line": safe_float(row.get("market_line")),
                "edge": safe_float(row.get("edge")),
                "abs_edge": safe_float(row.get("abs_edge")),
                "expected_win_rate": safe_float(row.get("expected_win_rate")),
                "expected_push_rate": safe_float(row.get("expected_push_rate")),
                "expected_loss_rate": safe_float(row.get("expected_loss_rate")),
                "raw_expected_win_rate": safe_float(row.get("raw_expected_win_rate")),
                "ev": safe_float(row.get("ev")),
                "thompson_ev": safe_float(row.get("thompson_ev")),
                "final_confidence": safe_float(row.get("final_confidence")),
                "gap_percentile": safe_float(row.get("gap_percentile")),
                "allocation_tier": str(row.get("allocation_tier", "")),
                "allocation_action": str(row.get("allocation_action", "")),
                "bet_fraction": safe_float(row.get("bet_fraction")),
                "expected_profit_fraction": safe_float(row.get("expected_profit_fraction")),
                "selected_rank": int(row.get("selected_rank", 0)) if pd.notna(row.get("selected_rank")) else None,
                "game_key": str(row.get("game_key", "")),
                "script_cluster_id": str(row.get("script_cluster_id", "")),
                "recommendation": str(row.get("recommendation", "")),
                "decision_tier": str(row.get("decision_tier", "")),
                "weak_bucket": str(row.get("weak_bucket", "")),
                "conditional_promoted": bool(row.get("conditional_promoted")) if pd.notna(row.get("conditional_promoted")) else False,
                "conditional_audit_summary": str(row.get("conditional_audit_summary", "")),
                "promotion_reason_codes": str(row.get("promotion_reason_codes", "")),
                "p_base": safe_float(row.get("p_base")),
                "p_final": safe_float(row.get("p_final")),
                "recoverability_score": safe_float(row.get("recoverability_score")),
                "contradiction_score": safe_float(row.get("contradiction_score")),
                "conditional_support": safe_float(row.get("conditional_support")),
                "history_rows": int(row.get("history_rows", 0)) if pd.notna(row.get("history_rows")) else None,
                "market_books": safe_float(row.get("market_books")),
                "uncertainty_sigma": safe_float(row.get("uncertainty_sigma")),
                "spike_probability": safe_float(row.get("spike_probability")),
            }
        )
    return rows


def load_shadow_runs(manifest: dict, manifest_path: Path, identity_lookup: PlayerIdentityLookup) -> list[dict]:
    out: list[dict] = []
    for item in manifest.get("shadow_runs", []):
        final_csv = resolve_artifact_path(item.get("final_csv"), manifest_path.parent)
        final_json = resolve_artifact_path(item.get("final_json"), manifest_path.parent)
        plays = pd.read_csv(final_csv) if final_csv and final_csv.exists() else pd.DataFrame()
        final_payload = json.loads(final_json.read_text(encoding="utf-8")) if final_json and final_json.exists() else {}
        out.append(
            {
                "policy_profile": item.get("policy_profile"),
                "final_csv": str(final_csv) if final_csv else None,
                "summary": build_summary(plays),
                "policy": final_payload.get("policy", {}),
                "top_plays": normalize_play_rows(plays.head(10), identity_lookup),
            }
        )
    return out


def build_selector_pool_fallback(plays: pd.DataFrame, *, limit: int = 12) -> pd.DataFrame:
    if plays.empty:
        return plays.copy()

    ordered = plays.copy()
    for column in ("expected_win_rate", "ev", "final_confidence", "abs_edge"):
        ordered[column] = pd.to_numeric(ordered.get(column), errors="coerce")
    ordered = ordered.sort_values(
        ["expected_win_rate", "ev", "final_confidence", "abs_edge"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    positive_ev = ordered.loc[pd.to_numeric(ordered.get("ev"), errors="coerce").fillna(-1.0) >= 0].copy()
    negative_ev = ordered.loc[~ordered.index.isin(positive_ev.index)].copy()
    if not positive_ev.empty:
        ordered = pd.concat([positive_ev, negative_ev], ignore_index=True)

    selected_rows: list[pd.Series] = []
    seen_players: set[str] = set()
    per_game_counts: dict[str, int] = {}
    for _, row in ordered.iterrows():
        player = str(row.get("player", "")).strip().lower()
        game_key = str(row.get("game_key", "") or row.get("market_event_id", "")).strip().lower()
        if player and player in seen_players:
            continue
        if game_key and per_game_counts.get(game_key, 0) >= 2:
            continue
        selected_rows.append(row)
        if player:
            seen_players.add(player)
        if game_key:
            per_game_counts[game_key] = per_game_counts.get(game_key, 0) + 1
        if len(selected_rows) >= int(limit):
            break

    if not selected_rows:
        return ordered.head(int(limit)).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def resolve_published_board(manifest: dict, manifest_path: Path) -> tuple[pd.DataFrame, dict, str]:
    final_csv = resolve_artifact_path(manifest.get("final_csv"), manifest_path.parent)
    final_json = resolve_artifact_path(manifest.get("final_json"), manifest_path.parent)
    plays = pd.read_csv(final_csv) if final_csv and final_csv.exists() else pd.DataFrame()
    final_payload = json.loads(final_json.read_text(encoding="utf-8")) if final_json and final_json.exists() else {}
    if not plays.empty:
        return plays, final_payload, "primary_final_board"

    selector_csv = resolve_artifact_path(manifest.get("selector_csv"), manifest_path.parent)
    selector_plays = pd.read_csv(selector_csv) if selector_csv and selector_csv.exists() else pd.DataFrame()
    if not selector_plays.empty:
        fallback_plays = build_selector_pool_fallback(selector_plays)
        final_payload.setdefault("policy_profile", manifest.get("policy_profile"))
        final_payload.setdefault("market_snapshot", manifest.get("current_market_snapshot"))
        return fallback_plays, final_payload, "primary_selector_pool_fallback"

    for item in manifest.get("shadow_runs", []):
        shadow_csv = resolve_artifact_path(item.get("final_csv"), manifest_path.parent)
        shadow_json = resolve_artifact_path(item.get("final_json"), manifest_path.parent)
        shadow_plays = pd.read_csv(shadow_csv) if shadow_csv and shadow_csv.exists() else pd.DataFrame()
        if shadow_plays.empty:
            shadow_selector_csv = resolve_artifact_path(item.get("selector_csv"), manifest_path.parent)
            shadow_selector = pd.read_csv(shadow_selector_csv) if shadow_selector_csv and shadow_selector_csv.exists() else pd.DataFrame()
            if shadow_selector.empty:
                continue
            shadow_payload = json.loads(shadow_json.read_text(encoding="utf-8")) if shadow_json and shadow_json.exists() else {}
            shadow_payload.setdefault("policy_profile", item.get("policy_profile"))
            return build_selector_pool_fallback(shadow_selector), shadow_payload, "shadow_selector_pool_fallback"
        shadow_payload = json.loads(shadow_json.read_text(encoding="utf-8")) if shadow_json and shadow_json.exists() else {}
        shadow_payload.setdefault("policy_profile", item.get("policy_profile"))
        return shadow_plays, shadow_payload, "shadow_final_board_fallback"

    return plays, final_payload, "primary_final_board_empty"


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve() if args.manifest else find_latest_manifest(args.daily_runs_root.resolve())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_lookup = build_player_identity_lookup(args.cards_json.resolve())

    history_csv = resolve_artifact_path(manifest.get("history_csv"), manifest_path.parent)
    plays, final_payload, published_board_source = resolve_published_board(manifest, manifest_path)

    plays_json = normalize_play_rows(plays, identity_lookup)
    parlay_payload = annotate_parlay_board(
        plays_json,
        sport="nba",
        probability_field="expected_win_rate",
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "run_date": manifest.get("run_date"),
        "season": manifest.get("season"),
        "through_date": manifest.get("through_date"),
        "published_board_source": published_board_source,
        "current_market_rows": manifest.get("current_market_rows"),
        "current_market_snapshot_meta": manifest.get("current_market_snapshot_meta", {}),
        "used_latest_manifest": manifest.get("used_latest_manifest"),
        "shadow_policy_profiles": manifest.get("shadow_policy_profiles", []),
        "market_snapshot": final_payload.get("market_snapshot") or manifest.get("current_market_snapshot"),
        "model_run_id": final_payload.get("run_id"),
        "policy_profile": final_payload.get("policy_profile"),
        "policy": final_payload.get("policy", {}),
        "input_validation": final_payload.get("input_validation", {}),
        "summary": build_summary(plays),
        "accuracy_metrics": build_accuracy_metrics(history_csv),
        "parlay_summary": parlay_payload["summary"],
        "parlay_pairs": parlay_payload["pairs"],
        "parlay_validation": build_parlay_validation(history_csv),
        "plays": parlay_payload["plays"],
        "shadow_runs": load_shadow_runs(manifest, manifest_path, identity_lookup),
    }
    payload["summary"]["parlay_tagged_plays"] = int(payload["parlay_summary"].get("tagged_play_count", 0))
    payload["summary"]["parlay_pairs"] = int(payload["parlay_summary"].get("selected_pair_count", 0))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Also copy to dist/data for the predictor page
    args.out_dist.parent.mkdir(parents=True, exist_ok=True)
    args.out_dist.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("DAILY PREDICTIONS WEB EXPORT COMPLETE")
    print("=" * 90)
    print(f"Manifest: {manifest_path}")
    print(f"Rows:     {len(plays)}")
    print(f"Output:   {args.out_json}")
    print(f"Dist:     {args.out_dist}")


if __name__ == "__main__":
    main()
