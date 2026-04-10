from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import write_csv, write_json


HITTER_TARGETS = ["H", "HR", "RBI"]
PITCHER_TARGETS = ["K", "ER", "ERA"]


@dataclass
class PoolScoreConfig:
    pool_csv: Path
    raw_dir: Path
    scored_csv_out: Path | None = None
    summary_json_out: Path | None = None


def _normalize_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_name(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return ""
    out = text
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        if np.isnan(out):
            return None
        return out
    except Exception:
        return None


def _actual_long_from_raw(raw_dir: Path) -> pd.DataFrame:
    raw_dir = raw_dir.resolve()
    hitters_path = raw_dir / "hitter_game_logs.csv"
    pitchers_path = raw_dir / "pitcher_game_logs.csv"
    parts: list[pd.DataFrame] = []

    if hitters_path.exists():
        hitters = pd.read_csv(hitters_path)
        if not hitters.empty:
            hitters["Date"] = pd.to_datetime(hitters.get("Date"), errors="coerce")
            hitters = hitters.loc[hitters["Date"].notna()].copy()
            hitters["Date"] = hitters["Date"].dt.strftime("%Y-%m-%d")
            hitters["Game_ID"] = hitters.get("Game_ID", "").astype(str)
            hitters["Player_ID_norm"] = hitters.get("Player_ID", "").map(_normalize_id)
            hitters["Player_norm"] = hitters.get("Player", "").map(_normalize_name)
            for target in HITTER_TARGETS:
                if target not in hitters.columns:
                    continue
                part = hitters[["Date", "Game_ID", "Player_ID_norm", "Player_norm", target]].copy()
                part = part.rename(columns={target: "Actual"})
                part["Player_Type"] = "hitter"
                part["Target"] = target
                parts.append(part)

    if pitchers_path.exists():
        pitchers = pd.read_csv(pitchers_path)
        if not pitchers.empty:
            pitchers["Date"] = pd.to_datetime(pitchers.get("Date"), errors="coerce")
            pitchers = pitchers.loc[pitchers["Date"].notna()].copy()
            pitchers["Date"] = pitchers["Date"].dt.strftime("%Y-%m-%d")
            pitchers["Game_ID"] = pitchers.get("Game_ID", "").astype(str)
            pitchers["Player_ID_norm"] = pitchers.get("Player_ID", "").map(_normalize_id)
            pitchers["Player_norm"] = pitchers.get("Player", "").map(_normalize_name)
            for target in PITCHER_TARGETS:
                if target not in pitchers.columns:
                    continue
                part = pitchers[["Date", "Game_ID", "Player_ID_norm", "Player_norm", target]].copy()
                part = part.rename(columns={target: "Actual"})
                part["Player_Type"] = "pitcher"
                part["Target"] = target
                parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["Date", "Game_ID", "Player_ID_norm", "Player_norm", "Actual", "Player_Type", "Target"])

    actual = pd.concat(parts, ignore_index=True)
    actual["Actual"] = pd.to_numeric(actual["Actual"], errors="coerce")
    actual = actual.drop_duplicates(
        subset=["Date", "Game_ID", "Player_Type", "Target", "Player_ID_norm", "Player_norm"],
        keep="last",
    ).reset_index(drop=True)
    return actual


def _summarize_bucket(rows: pd.DataFrame) -> dict:
    total = int(len(rows))
    resolved = rows.loc[rows["Actual"].notna()].copy()
    resolved_count = int(len(resolved))

    mae = _safe_float((resolved["Prediction"] - resolved["Actual"]).abs().mean())
    rmse = _safe_float(np.sqrt(((resolved["Prediction"] - resolved["Actual"]) ** 2).mean())) if resolved_count > 0 else None

    market_scored = resolved.loc[
        resolved["Market_Line"].notna() & resolved["Direction"].isin(["OVER", "UNDER"])
    ].copy()
    win_count = int((market_scored["Outcome"] == "win").sum())
    loss_count = int((market_scored["Outcome"] == "loss").sum())
    push_count = int((market_scored["Outcome"] == "push").sum())
    graded_count = int(win_count + loss_count)
    settled_count = int(win_count + loss_count + push_count)
    unit_profit = (win_count * (100.0 / 110.0)) - loss_count

    return {
        "rows": total,
        "resolved_rows": resolved_count,
        "mae": mae,
        "rmse": rmse,
        "market_rows": int(len(market_scored)),
        "win_count": win_count,
        "loss_count": loss_count,
        "push_count": push_count,
        "graded_count": graded_count,
        "settled_count": settled_count,
        "win_rate": _safe_float(win_count / graded_count) if graded_count > 0 else None,
        "roi_per_graded_play": _safe_float(unit_profit / graded_count) if graded_count > 0 else None,
        "unit_profit": _safe_float(unit_profit),
    }


def score_prediction_pool(config: PoolScoreConfig) -> dict:
    pool_csv = config.pool_csv.resolve()
    raw_dir = config.raw_dir.resolve()
    if not pool_csv.exists():
        raise FileNotFoundError(f"Prediction pool CSV not found: {pool_csv}")

    pool = pd.read_csv(pool_csv)
    if pool.empty:
        raise RuntimeError(f"Prediction pool CSV is empty: {pool_csv}")

    required = ["Game_Date", "Game_ID", "Player", "Player_Type", "Target", "Prediction"]
    missing = [col for col in required if col not in pool.columns]
    if missing:
        raise RuntimeError(f"Prediction pool is missing required columns: {missing}")

    pool["Game_Date"] = pd.to_datetime(pool["Game_Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pool["Game_ID"] = pool["Game_ID"].astype(str)
    pool["Player_ID_norm"] = pool.get("Player_ID", "").map(_normalize_id)
    pool["Player_norm"] = pool["Player"].map(_normalize_name)
    pool["Player_Type"] = pool["Player_Type"].astype(str).str.lower()
    pool["Target"] = pool["Target"].astype(str).str.upper()
    pool["Prediction"] = pd.to_numeric(pool["Prediction"], errors="coerce")
    pool["Market_Line"] = pd.to_numeric(pool.get("Market_Line", np.nan), errors="coerce")
    pool = pool.loc[pool["Prediction"].notna()].copy()
    if pool.empty:
        raise RuntimeError(f"No usable prediction rows in pool after numeric cleanup: {pool_csv}")

    actual = _actual_long_from_raw(raw_dir)
    if actual.empty:
        raise RuntimeError(
            f"No raw actual logs found under {raw_dir}. "
            "Expected hitter_game_logs.csv and/or pitcher_game_logs.csv."
        )

    by_id = pool.merge(
        actual[["Date", "Game_ID", "Player_Type", "Target", "Player_ID_norm", "Actual"]],
        left_on=["Game_Date", "Game_ID", "Player_Type", "Target", "Player_ID_norm"],
        right_on=["Date", "Game_ID", "Player_Type", "Target", "Player_ID_norm"],
        how="left",
    )
    by_id = by_id.drop(columns=["Date"])

    unresolved_mask = by_id["Actual"].isna()
    if unresolved_mask.any():
        unresolved = by_id.loc[unresolved_mask].drop(columns=["Actual"]).copy()
        by_name = unresolved.merge(
            actual[["Date", "Game_ID", "Player_Type", "Target", "Player_norm", "Actual"]],
            left_on=["Game_Date", "Game_ID", "Player_Type", "Target", "Player_norm"],
            right_on=["Date", "Game_ID", "Player_Type", "Target", "Player_norm"],
            how="left",
        ).drop(columns=["Date"])
        by_id.loc[unresolved_mask, "Actual"] = by_name["Actual"].to_numpy()

    scored = by_id.copy()
    scored["Actual"] = pd.to_numeric(scored["Actual"], errors="coerce")
    scored["Abs_Error"] = (scored["Prediction"] - scored["Actual"]).abs()
    scored["Sq_Error"] = (scored["Prediction"] - scored["Actual"]) ** 2

    scored["Direction"] = np.where(
        scored["Market_Line"].notna() & (scored["Prediction"] > scored["Market_Line"]),
        "OVER",
        np.where(
            scored["Market_Line"].notna() & (scored["Prediction"] < scored["Market_Line"]),
            "UNDER",
            "NONE",
        ),
    )
    scored["Outcome"] = "unresolved"
    actual_known = scored["Actual"].notna()
    has_line = scored["Market_Line"].notna()
    push_mask = actual_known & has_line & scored["Actual"].eq(scored["Market_Line"])
    over_win = actual_known & has_line & scored["Direction"].eq("OVER") & scored["Actual"].gt(scored["Market_Line"])
    under_win = actual_known & has_line & scored["Direction"].eq("UNDER") & scored["Actual"].lt(scored["Market_Line"])
    win_mask = over_win | under_win
    loss_mask = actual_known & has_line & scored["Direction"].isin(["OVER", "UNDER"]) & ~push_mask & ~win_mask
    scored.loc[push_mask, "Outcome"] = "push"
    scored.loc[win_mask, "Outcome"] = "win"
    scored.loc[loss_mask, "Outcome"] = "loss"
    scored.loc[actual_known & ~has_line, "Outcome"] = "actual_only"

    scored = scored.sort_values(["Game_Date", "Game_ID", "Player_Type", "Player", "Target"]).reset_index(drop=True)

    default_scored_csv = pool_csv.with_name(f"{pool_csv.stem}_scored.csv")
    default_summary_json = pool_csv.with_name(f"{pool_csv.stem}_scored_summary.json")
    scored_csv_out = config.scored_csv_out.resolve() if config.scored_csv_out is not None else default_scored_csv
    summary_json_out = config.summary_json_out.resolve() if config.summary_json_out is not None else default_summary_json

    write_csv(scored, scored_csv_out)

    summary = {
        "pool_csv": str(pool_csv),
        "raw_dir": str(raw_dir),
        "scored_csv": str(scored_csv_out),
        "rows": int(len(scored)),
        "resolved_rows": int(scored["Actual"].notna().sum()),
        "by_role": {
            role: _summarize_bucket(scored.loc[scored["Player_Type"] == role].copy())
            for role in sorted(scored["Player_Type"].dropna().unique().tolist())
        },
        "by_target": {
            target: _summarize_bucket(scored.loc[scored["Target"] == target].copy())
            for target in sorted(scored["Target"].dropna().unique().tolist())
        },
        "overall": _summarize_bucket(scored),
        "as_of_game_date_min": str(scored["Game_Date"].min()) if not scored.empty else None,
        "as_of_game_date_max": str(scored["Game_Date"].max()) if not scored.empty else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(summary_json_out, summary)
    return summary


def score_all_unscored_pools(
    *,
    daily_runs_root: Path,
    raw_dir: Path,
    as_of_date: str,
    overwrite: bool = False,
) -> dict:
    root = daily_runs_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Daily runs root does not exist: {root}")
    as_of = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(as_of):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    as_of = as_of.normalize()

    scanned = 0
    scored_count = 0
    skipped_existing = 0
    skipped_future = 0
    failures: list[dict] = []
    outputs: list[dict] = []

    for pool_csv in sorted(root.glob("**/daily_prediction_pool_*.csv")):
        if pool_csv.stem.endswith("_scored"):
            continue
        scanned += 1
        scored_summary_path = pool_csv.with_name(f"{pool_csv.stem}_scored_summary.json")
        if scored_summary_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        try:
            quick = pd.read_csv(pool_csv, usecols=["Game_Date"])
            max_game_date = pd.to_datetime(quick["Game_Date"], errors="coerce").max()
        except Exception:
            max_game_date = pd.NaT

        if pd.notna(max_game_date) and pd.Timestamp(max_game_date).normalize() > as_of:
            skipped_future += 1
            continue

        try:
            summary = score_prediction_pool(
                PoolScoreConfig(
                    pool_csv=pool_csv,
                    raw_dir=raw_dir,
                    scored_csv_out=pool_csv.with_name(f"{pool_csv.stem}_scored.csv"),
                    summary_json_out=scored_summary_path,
                )
            )
            scored_count += 1
            outputs.append(
                {
                    "pool_csv": str(pool_csv),
                    "summary_json": str(scored_summary_path),
                    "resolved_rows": int(summary.get("resolved_rows", 0)),
                    "rows": int(summary.get("rows", 0)),
                }
            )
        except Exception as exc:
            failures.append({"pool_csv": str(pool_csv), "error": str(exc)})

    return {
        "daily_runs_root": str(root),
        "raw_dir": str(raw_dir.resolve()),
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "scanned": scanned,
        "scored": scored_count,
        "skipped_existing": skipped_existing,
        "skipped_future": skipped_future,
        "failures": failures,
        "outputs": outputs,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
