#!/usr/bin/env python3
"""
Evaluate daily market boards over a date range using realized outcomes from
player processed CSVs. Intended for forward paper-trade monitoring.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "model" / "analysis" / "daily_runs"
DEFAULT_MODES = ["aggressive_shadow", "production_calibrated", "production_high_precision"]
DEFAULT_PAYOUT = 100.0 / 110.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forward paper-trade boards over a date range.")
    parser.add_argument("--date-from", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--date-to", type=str, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="Root directory for daily run outputs.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        help=(
            "Policy modes to evaluate. "
            "Use aggressive_shadow for primary board and policy profile names for shadow boards."
        ),
    )
    parser.add_argument(
        "--strict-aligned-only",
        action="store_true",
        help="Keep only rows where market_date == run_date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--exclude-stale-fallback",
        action="store_true",
        help="Exclude rows from runs where snapshot mode was stale_fallback.",
    )
    parser.add_argument(
        "--baseline-win-rate",
        type=float,
        default=0.62,
        help="Baseline resolved win rate used for z-score comparisons.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Rolling resolved-bet window for rolling z-score diagnostics.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of equal-width bins used for calibration curve outputs.",
    )
    parser.add_argument(
        "--logloss-eps",
        type=float,
        default=1e-6,
        help="Numerical stability epsilon for log-loss probability clipping.",
    )
    parser.add_argument("--payout", type=float, default=DEFAULT_PAYOUT, help="Per-unit profit for a winning bet.")
    parser.add_argument(
        "--kelly-start-bankroll",
        type=float,
        default=10000.0,
        help="Starting bankroll for Kelly simulation.",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier (0..1).",
    )
    parser.add_argument(
        "--kelly-max-fraction",
        type=float,
        default=0.02,
        help="Maximum bankroll fraction risked on any single simulated bet.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help=(
            "Output file prefix path (without extension). "
            "Defaults to <run-root>/forward_eval_<YYYYMMDD>_<YYYYMMDD>."
        ),
    )
    return parser.parse_args()


def board_path_for_mode(run_root: Path, run_stamp: str, mode: str) -> Path:
    if mode == "aggressive_shadow":
        return run_root / run_stamp / f"final_market_plays_{run_stamp}.csv"
    return run_root / run_stamp / "shadow" / mode / f"final_market_plays_{run_stamp}_{mode}.csv"


def manifest_path(run_root: Path, run_stamp: str) -> Path:
    return run_root / run_stamp / f"daily_market_pipeline_manifest_{run_stamp}.json"


def classify_result(direction: str, actual: float, market_line: float) -> str:
    if pd.isna(actual) or pd.isna(market_line):
        return "missing"
    side = str(direction).upper()
    if side == "OVER":
        if actual > market_line:
            return "win"
        if actual < market_line:
            return "loss"
        return "push"
    if side == "UNDER":
        if actual < market_line:
            return "win"
        if actual > market_line:
            return "loss"
        return "push"
    return "push"


def wilson_ci(wins: int, resolved_binary: int, z: float = 1.96) -> tuple[float, float]:
    if resolved_binary <= 0:
        return np.nan, np.nan
    phat = wins / resolved_binary
    denom = 1.0 + (z * z) / resolved_binary
    center = (phat + (z * z) / (2.0 * resolved_binary)) / denom
    margin = (
        z
        * sqrt(
            (phat * (1.0 - phat) / resolved_binary)
            + ((z * z) / (4.0 * resolved_binary * resolved_binary))
        )
        / denom
    )
    return center - margin, center + margin


def load_manifest_snapshot_mode(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload.get("current_market_snapshot_meta", {}).get("mode")


def infer_binary_win_probability(frame: pd.DataFrame, eps: float) -> pd.Series:
    p_win = pd.to_numeric(frame.get("expected_win_rate"), errors="coerce")
    p_push = pd.to_numeric(frame.get("expected_push_rate"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    p_loss = pd.to_numeric(frame.get("expected_loss_rate"), errors="coerce")
    fallback_loss = (1.0 - p_win - p_push).clip(lower=0.0, upper=1.0)
    p_loss = p_loss.where(p_loss.notna(), fallback_loss)

    denom = (p_win + p_loss).where((p_win + p_loss) > float(eps), np.nan)
    p_binary = (p_win / denom).clip(lower=float(eps), upper=1.0 - float(eps))
    return p_binary


def summarize_slice(
    frame: pd.DataFrame,
    slice_name: str,
    baseline_win_rate: float,
    logloss_eps: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for mode, group in frame.groupby("mode"):
        valid = group.loc[group["result"].isin(["win", "loss", "push"])].copy()
        counts = valid["result"].value_counts().to_dict()
        wins = int(counts.get("win", 0))
        losses = int(counts.get("loss", 0))
        pushes = int(counts.get("push", 0))
        resolved = int(len(valid))
        resolved_binary = wins + losses
        hit_rate = (wins / resolved) if resolved > 0 else np.nan
        resolved_hit_rate = (wins / resolved_binary) if resolved_binary > 0 else np.nan
        ci_low, ci_high = wilson_ci(wins, resolved_binary)
        p0 = float(np.clip(baseline_win_rate, 1e-6, 1.0 - 1e-6))
        if resolved_binary > 0:
            z_vs_62 = (resolved_hit_rate - p0) / sqrt(p0 * (1.0 - p0) / resolved_binary)
        else:
            z_vs_62 = np.nan

        expected_ev = pd.to_numeric(valid.get("expected_ev_unit"), errors="coerce").dropna()
        realized_ev = pd.to_numeric(valid.get("realized_ev_unit"), errors="coerce").dropna()
        expected_sum = float(expected_ev.sum()) if len(expected_ev) else np.nan
        realized_sum = float(realized_ev.sum()) if len(realized_ev) else np.nan
        if np.isfinite(expected_sum) and abs(expected_sum) > 1e-9:
            realized_vs_expected = realized_sum / expected_sum
        else:
            realized_vs_expected = np.nan

        binary = valid.loc[valid["result"].isin(["win", "loss"])].copy()
        binary["y_win"] = (binary["result"] == "win").astype(float)
        binary["p_win_binary"] = infer_binary_win_probability(binary, eps=logloss_eps)
        prob_valid = binary.loc[binary["p_win_binary"].notna()].copy()
        if prob_valid.empty:
            brier = np.nan
            log_loss = np.nan
            prob_rows = 0
        else:
            y = prob_valid["y_win"].to_numpy(dtype=float)
            p = np.clip(prob_valid["p_win_binary"].to_numpy(dtype=float), logloss_eps, 1.0 - logloss_eps)
            brier = float(np.mean((p - y) ** 2))
            log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            prob_rows = int(len(prob_valid))

        rows.append(
            {
                "slice": slice_name,
                "mode": mode,
                "rows": int(len(group)),
                "resolved": resolved,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": hit_rate,
                "resolved_hit_rate": resolved_hit_rate,
                "wilson95_low": ci_low,
                "wilson95_high": ci_high,
                "z_vs_0p62": z_vs_62,
                "mean_expected_ev_unit": float(expected_ev.mean()) if len(expected_ev) else np.nan,
                "mean_realized_ev_unit": float(realized_ev.mean()) if len(realized_ev) else np.nan,
                "sum_expected_ev_unit": expected_sum,
                "sum_realized_ev_unit": realized_sum,
                "realized_vs_expected_ratio": realized_vs_expected,
                "prob_eval_rows": prob_rows,
                "brier_score": brier,
                "log_loss": log_loss,
            }
        )
    return pd.DataFrame(rows)


def build_drawdown(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    valid = frame.loc[frame["result"].isin(["win", "loss", "push"])].copy()
    if valid.empty:
        return pd.DataFrame(columns=["mode", "n", "cum_profit_unit", "peak_profit_unit", "drawdown_unit", "max_drawdown_unit"])

    sort_cols = ["mode", "run_date", "selected_rank", "player", "target", "direction"]
    for column in sort_cols:
        if column not in valid.columns:
            valid[column] = np.nan
    valid = valid.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    for mode, group in valid.groupby("mode"):
        profits = pd.to_numeric(group["realized_ev_unit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        cum = np.cumsum(profits)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = float(dd.max()) if len(dd) else 0.0
        for idx, (cum_v, peak_v, dd_v) in enumerate(zip(cum, peak, dd), start=1):
            rows.append(
                {
                    "mode": mode,
                    "n": idx,
                    "cum_profit_unit": float(cum_v),
                    "peak_profit_unit": float(peak_v),
                    "drawdown_unit": float(dd_v),
                    "max_drawdown_unit": max_dd,
                }
            )
    return pd.DataFrame(rows)


def build_tier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    valid = frame.loc[frame["result"].isin(["win", "loss", "push"])].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "allocation_tier",
                "rows",
                "wins",
                "losses",
                "pushes",
                "hit_rate",
                "resolved_hit_rate",
                "mean_expected_ev_unit",
                "mean_realized_ev_unit",
                "sum_expected_ev_unit",
                "sum_realized_ev_unit",
                "realized_vs_expected_ratio",
                "sum_expected_profit_fraction",
                "sum_realized_profit_fraction",
            ]
        )

    valid["allocation_tier"] = valid.get("allocation_tier", pd.Series("unknown", index=valid.index)).fillna("unknown").astype(str)
    valid["expected_ev_unit"] = pd.to_numeric(valid["expected_ev_unit"], errors="coerce")
    valid["realized_ev_unit"] = pd.to_numeric(valid["realized_ev_unit"], errors="coerce")
    valid["bet_fraction"] = pd.to_numeric(valid.get("bet_fraction"), errors="coerce")
    valid["expected_profit_fraction"] = valid["expected_ev_unit"] * valid["bet_fraction"]
    valid["realized_profit_fraction"] = valid["realized_ev_unit"] * valid["bet_fraction"]

    for (mode, tier), group in valid.groupby(["mode", "allocation_tier"]):
        counts = group["result"].value_counts().to_dict()
        wins = int(counts.get("win", 0))
        losses = int(counts.get("loss", 0))
        pushes = int(counts.get("push", 0))
        resolved = wins + losses + pushes
        resolved_binary = wins + losses
        expected_sum = float(group["expected_ev_unit"].sum(skipna=True))
        realized_sum = float(group["realized_ev_unit"].sum(skipna=True))
        if abs(expected_sum) > 1e-9:
            ratio = realized_sum / expected_sum
        else:
            ratio = np.nan
        rows.append(
            {
                "mode": mode,
                "allocation_tier": tier,
                "rows": int(len(group)),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": (wins / resolved) if resolved > 0 else np.nan,
                "resolved_hit_rate": (wins / resolved_binary) if resolved_binary > 0 else np.nan,
                "mean_expected_ev_unit": float(group["expected_ev_unit"].mean(skipna=True)),
                "mean_realized_ev_unit": float(group["realized_ev_unit"].mean(skipna=True)),
                "sum_expected_ev_unit": expected_sum,
                "sum_realized_ev_unit": realized_sum,
                "realized_vs_expected_ratio": ratio,
                "sum_expected_profit_fraction": float(group["expected_profit_fraction"].sum(skipna=True)),
                "sum_realized_profit_fraction": float(group["realized_profit_fraction"].sum(skipna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "allocation_tier"]).reset_index(drop=True)


def build_calibration_curve(frame: pd.DataFrame, bins: int, eps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict] = []
    ece_rows: list[dict] = []
    n_bins = int(max(2, bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for mode, group in frame.groupby("mode"):
        binary = group.loc[group["result"].isin(["win", "loss"])].copy()
        if binary.empty:
            continue
        binary["y_win"] = (binary["result"] == "win").astype(float)
        binary["p_win_binary"] = infer_binary_win_probability(binary, eps=eps)
        binary = binary.loc[binary["p_win_binary"].notna()].copy()
        if binary.empty:
            continue

        probs = np.clip(binary["p_win_binary"].to_numpy(dtype=float), eps, 1.0 - eps)
        bin_ids = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
        binary["calibration_bin"] = bin_ids

        total_n = int(len(binary))
        weighted_abs_gap = 0.0
        max_abs_gap = 0.0
        for bin_idx, bin_group in binary.groupby("calibration_bin"):
            count = int(len(bin_group))
            wins = int((bin_group["y_win"] > 0.5).sum())
            losses = int(count - wins)
            mean_pred = float(bin_group["p_win_binary"].mean())
            observed = float(wins / count) if count > 0 else np.nan
            gap = float(observed - mean_pred) if count > 0 else np.nan
            abs_gap = abs(gap) if np.isfinite(gap) else np.nan
            if np.isfinite(abs_gap):
                weighted_abs_gap += count * abs_gap
                max_abs_gap = max(max_abs_gap, abs_gap)

            curve_rows.append(
                {
                    "mode": mode,
                    "bin_index": int(bin_idx),
                    "prob_low": float(edges[int(bin_idx)]),
                    "prob_high": float(edges[int(bin_idx) + 1]),
                    "count": count,
                    "wins": wins,
                    "losses": losses,
                    "mean_pred_win": mean_pred,
                    "observed_win_rate": observed,
                    "calibration_gap": gap,
                    "abs_calibration_gap": abs_gap,
                }
            )

        ece_rows.append(
            {
                "mode": mode,
                "calibration_rows": total_n,
                "ece": float(weighted_abs_gap / max(1, total_n)),
                "mce": float(max_abs_gap),
            }
        )

    curve_df = pd.DataFrame(curve_rows)
    if not curve_df.empty:
        curve_df = curve_df.sort_values(["mode", "bin_index"]).reset_index(drop=True)
    ece_df = pd.DataFrame(ece_rows)
    if not ece_df.empty:
        ece_df = ece_df.sort_values(["mode"]).reset_index(drop=True)
    return curve_df, ece_df


def build_rolling_z(frame: pd.DataFrame, baseline_win_rate: float, window: int) -> pd.DataFrame:
    valid = frame.loc[frame["result"].isin(["win", "loss"])].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "seq_index",
                "run_date",
                "market_date",
                "player",
                "target",
                "direction",
                "cum_n",
                "cum_wins",
                "cum_win_rate",
                "cum_z",
                "rolling_n",
                "rolling_wins",
                "rolling_win_rate",
                "rolling_z",
            ]
        )

    p0 = float(np.clip(baseline_win_rate, 1e-6, 1.0 - 1e-6))
    sort_cols = ["mode", "run_date", "selected_rank", "player", "target", "direction"]
    for column in sort_cols:
        if column not in valid.columns:
            valid[column] = np.nan
    valid["selected_rank"] = pd.to_numeric(valid["selected_rank"], errors="coerce")
    valid = valid.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    out_frames: list[pd.DataFrame] = []
    for mode, group in valid.groupby("mode", sort=False):
        g = group.copy().reset_index(drop=True)
        g["y_win"] = (g["result"] == "win").astype(float)
        g["cum_n"] = np.arange(1, len(g) + 1, dtype=float)
        g["cum_wins"] = g["y_win"].cumsum()
        g["cum_win_rate"] = g["cum_wins"] / g["cum_n"]
        g["cum_z"] = (g["cum_win_rate"] - p0) / np.sqrt((p0 * (1.0 - p0)) / g["cum_n"])

        rolling_n = g["y_win"].rolling(window=int(max(1, window)), min_periods=int(max(1, window))).count()
        rolling_wins = g["y_win"].rolling(window=int(max(1, window)), min_periods=int(max(1, window))).sum()
        rolling_rate = rolling_wins / rolling_n
        rolling_z = (rolling_rate - p0) / np.sqrt((p0 * (1.0 - p0)) / rolling_n)

        g["rolling_n"] = rolling_n
        g["rolling_wins"] = rolling_wins
        g["rolling_win_rate"] = rolling_rate
        g["rolling_z"] = rolling_z
        g["seq_index"] = np.arange(1, len(g) + 1, dtype=int)

        out_frames.append(
            g[
                [
                    "mode",
                    "seq_index",
                    "run_date",
                    "market_date",
                    "player",
                    "target",
                    "direction",
                    "cum_n",
                    "cum_wins",
                    "cum_win_rate",
                    "cum_z",
                    "rolling_n",
                    "rolling_wins",
                    "rolling_win_rate",
                    "rolling_z",
                ]
            ]
        )

    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def build_kelly_simulation(
    frame: pd.DataFrame,
    payout: float,
    start_bankroll: float,
    kelly_fraction: float,
    kelly_cap: float,
    eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.loc[frame["result"].isin(["win", "loss", "push"])].copy()
    if valid.empty:
        empty_trades = pd.DataFrame(
            columns=[
                "mode",
                "seq_index",
                "run_date",
                "market_date",
                "player",
                "target",
                "direction",
                "result",
                "p_win_binary",
                "full_kelly_fraction",
                "stake_fraction",
                "bankroll_before",
                "stake_amount",
                "profit_amount",
                "bankroll_after",
                "peak_bankroll",
                "drawdown_pct",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "mode",
                "start_bankroll",
                "end_bankroll",
                "profit_amount",
                "return_pct",
                "max_drawdown_pct",
                "num_bets",
                "num_staked_bets",
                "avg_stake_fraction",
            ]
        )
        return empty_trades, empty_summary

    sort_cols = ["mode", "run_date", "selected_rank", "player", "target", "direction"]
    for column in sort_cols:
        if column not in valid.columns:
            valid[column] = np.nan
    valid["selected_rank"] = pd.to_numeric(valid["selected_rank"], errors="coerce")
    valid = valid.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    frac_kelly = float(np.clip(kelly_fraction, 0.0, 1.0))
    cap = float(max(0.0, kelly_cap))
    b = float(max(eps, payout))
    start = float(max(eps, start_bankroll))

    trades: list[dict] = []
    summary: list[dict] = []

    for mode, group in valid.groupby("mode", sort=False):
        g = group.copy().reset_index(drop=True)
        g["p_win_binary"] = infer_binary_win_probability(g, eps=eps)

        bankroll = start
        peak_bankroll = start
        max_drawdown = 0.0
        staked_bets = 0
        stake_fracs: list[float] = []

        for idx, row in g.iterrows():
            p = row.get("p_win_binary")
            if pd.isna(p):
                full_kelly = 0.0
            else:
                p = float(np.clip(p, eps, 1.0 - eps))
                full_kelly = float(((b * p) - (1.0 - p)) / b)
            stake_fraction = float(np.clip(frac_kelly * max(0.0, full_kelly), 0.0, cap))
            bankroll_before = bankroll
            stake_amount = bankroll_before * stake_fraction
            result = str(row.get("result"))
            if result == "win":
                profit_amount = stake_amount * b
            elif result == "loss":
                profit_amount = -stake_amount
            else:
                profit_amount = 0.0
            bankroll_after = bankroll_before + profit_amount
            peak_bankroll = max(peak_bankroll, bankroll_after)
            drawdown_pct = (peak_bankroll - bankroll_after) / peak_bankroll if peak_bankroll > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown_pct)

            if stake_fraction > 0.0:
                staked_bets += 1
                stake_fracs.append(stake_fraction)

            trades.append(
                {
                    "mode": mode,
                    "seq_index": int(idx + 1),
                    "run_date": row.get("run_date"),
                    "market_date": row.get("market_date"),
                    "player": row.get("player"),
                    "target": row.get("target"),
                    "direction": row.get("direction"),
                    "result": result,
                    "p_win_binary": p,
                    "full_kelly_fraction": full_kelly,
                    "stake_fraction": stake_fraction,
                    "bankroll_before": float(bankroll_before),
                    "stake_amount": float(stake_amount),
                    "profit_amount": float(profit_amount),
                    "bankroll_after": float(bankroll_after),
                    "peak_bankroll": float(peak_bankroll),
                    "drawdown_pct": float(drawdown_pct),
                }
            )

            bankroll = bankroll_after

        summary.append(
            {
                "mode": mode,
                "start_bankroll": float(start),
                "end_bankroll": float(bankroll),
                "profit_amount": float(bankroll - start),
                "return_pct": float((bankroll / start) - 1.0),
                "max_drawdown_pct": float(max_drawdown),
                "num_bets": int(len(g)),
                "num_staked_bets": int(staked_bets),
                "avg_stake_fraction": float(np.mean(stake_fracs)) if stake_fracs else 0.0,
            }
        )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["mode", "seq_index"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("mode").reset_index(drop=True)
    return trades_df, summary_df


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()

    date_index = pd.date_range(args.date_from, args.date_to, freq="D")
    run_stamps = date_index.strftime("%Y%m%d").tolist()

    rows: list[dict] = []
    for mode in args.modes:
        for run_stamp in run_stamps:
            board_path = board_path_for_mode(run_root, run_stamp, mode)
            snapshot_mode = load_manifest_snapshot_mode(manifest_path(run_root, run_stamp))
            run_date_iso = pd.to_datetime(run_stamp, format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d")

            if not board_path.exists():
                rows.append(
                    {
                        "mode": mode,
                        "run_date": run_stamp,
                        "run_date_iso": run_date_iso,
                        "snapshot_mode": snapshot_mode,
                        "status": "missing_board_file",
                        "result": "missing",
                    }
                )
                continue

            board = pd.read_csv(board_path)
            if board.empty:
                rows.append(
                    {
                        "mode": mode,
                        "run_date": run_stamp,
                        "run_date_iso": run_date_iso,
                        "snapshot_mode": snapshot_mode,
                        "status": "empty_board",
                        "result": "missing",
                    }
                )
                continue

            for _, row in board.iterrows():
                csv_path = Path(str(row.get("csv", ""))) if str(row.get("csv", "")) else None
                target = str(row.get("target", "")).upper()
                market_date = str(row.get("market_date", ""))
                market_line = pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").iloc[0]
                expected_ev = pd.to_numeric(pd.Series([row.get("ev")]), errors="coerce").iloc[0]
                expected_win_rate = pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").iloc[0]
                expected_push_rate = pd.to_numeric(pd.Series([row.get("expected_push_rate")]), errors="coerce").iloc[0]
                expected_loss_rate = pd.to_numeric(pd.Series([row.get("expected_loss_rate")]), errors="coerce").iloc[0]
                bet_fraction = pd.to_numeric(pd.Series([row.get("bet_fraction")]), errors="coerce").iloc[0]

                actual = np.nan
                status = "ok"
                if (csv_path is None) or (not csv_path.exists()):
                    status = "missing_player_csv"
                else:
                    try:
                        history = pd.read_csv(csv_path, usecols=lambda col: col in {"Date", target})
                        history["Date"] = pd.to_datetime(history["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                        hit = history.loc[history["Date"] == market_date]
                        if hit.empty:
                            status = "missing_actual_row"
                        else:
                            actual = pd.to_numeric(pd.Series([hit.iloc[-1].get(target)]), errors="coerce").iloc[0]
                            if pd.isna(actual):
                                status = "actual_nan"
                    except Exception as exc:
                        status = f"read_error:{type(exc).__name__}"

                result = classify_result(row.get("direction"), actual, market_line) if status == "ok" else "missing"
                if result == "win":
                    realized_ev = float(args.payout)
                elif result == "loss":
                    realized_ev = -1.0
                elif result == "push":
                    realized_ev = 0.0
                else:
                    realized_ev = np.nan

                rows.append(
                    {
                        "mode": mode,
                        "run_date": run_stamp,
                        "run_date_iso": run_date_iso,
                        "snapshot_mode": snapshot_mode,
                        "market_date": market_date,
                        "player": row.get("player"),
                        "target": target,
                        "direction": row.get("direction"),
                        "market_line": market_line,
                        "actual": actual,
                        "allocation_tier": row.get("allocation_tier"),
                        "selected_rank": row.get("selected_rank"),
                        "status": status,
                        "result": result,
                        "expected_ev_unit": expected_ev,
                        "realized_ev_unit": realized_ev,
                        "expected_win_rate": expected_win_rate,
                        "expected_push_rate": expected_push_rate,
                        "expected_loss_rate": expected_loss_rate,
                        "bet_fraction": bet_fraction,
                    }
                )

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        raise RuntimeError("No rows were evaluated.")

    rows_df["is_aligned_same_day"] = rows_df["market_date"] == rows_df["run_date_iso"]
    rows_df["is_stale_fallback_date"] = rows_df["snapshot_mode"].astype(str).str.lower().eq("stale_fallback")

    filtered = rows_df.copy()
    if args.strict_aligned_only:
        filtered = filtered.loc[filtered["is_aligned_same_day"]].copy()
    if args.exclude_stale_fallback:
        filtered = filtered.loc[~filtered["is_stale_fallback_date"]].copy()

    if filtered.empty:
        raise RuntimeError("Filtered evaluation set is empty. Check flags and date range.")

    summary_all = summarize_slice(
        filtered,
        "filtered",
        baseline_win_rate=float(args.baseline_win_rate),
        logloss_eps=float(args.logloss_eps),
    )
    calibration_df, calibration_stats = build_calibration_curve(
        filtered,
        bins=int(args.calibration_bins),
        eps=float(args.logloss_eps),
    )
    if not calibration_stats.empty and not summary_all.empty:
        summary_all = summary_all.merge(calibration_stats, on="mode", how="left")

    per_day_rows: list[dict] = []
    for (mode, run_date), group in filtered.groupby(["mode", "run_date"]):
        counts = group["result"].value_counts().to_dict()
        valid = group.loc[group["result"].isin(["win", "loss", "push"])].copy()
        wins = int(counts.get("win", 0))
        losses = int(counts.get("loss", 0))
        pushes = int(counts.get("push", 0))
        resolved_binary = wins + losses
        resolved_hit_rate = (wins / resolved_binary) if resolved_binary > 0 else np.nan
        expected_sum = float(pd.to_numeric(valid["expected_ev_unit"], errors="coerce").sum(skipna=True))
        realized_sum = float(pd.to_numeric(valid["realized_ev_unit"], errors="coerce").sum(skipna=True))
        per_day_rows.append(
            {
                "mode": mode,
                "run_date": run_date,
                "snapshot_mode": group["snapshot_mode"].iloc[0] if len(group) else None,
                "rows": int(len(group)),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "resolved_hit_rate": resolved_hit_rate,
                "sum_expected_ev_unit": expected_sum,
                "sum_realized_ev_unit": realized_sum,
                "missing": int(counts.get("missing", 0)),
            }
        )
    per_day_df = pd.DataFrame(per_day_rows).sort_values(["mode", "run_date"]).reset_index(drop=True)
    tier_df = build_tier_summary(filtered)
    drawdown_df = build_drawdown(filtered)
    rolling_df = build_rolling_z(
        filtered,
        baseline_win_rate=float(args.baseline_win_rate),
        window=int(args.rolling_window),
    )
    kelly_trades_df, kelly_summary_df = build_kelly_simulation(
        filtered,
        payout=float(args.payout),
        start_bankroll=float(args.kelly_start_bankroll),
        kelly_fraction=float(args.kelly_fraction),
        kelly_cap=float(args.kelly_max_fraction),
        eps=float(args.logloss_eps),
    )

    if args.out_prefix is not None:
        prefix = args.out_prefix.resolve()
    else:
        prefix = run_root / f"forward_eval_{run_stamps[0]}_{run_stamps[-1]}"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows_path = Path(str(prefix) + "_rows.csv")
    summary_path = Path(str(prefix) + "_summary.csv")
    daily_path = Path(str(prefix) + "_daily.csv")
    tier_path = Path(str(prefix) + "_tiers.csv")
    drawdown_path = Path(str(prefix) + "_drawdown.csv")
    calibration_path = Path(str(prefix) + "_calibration.csv")
    rolling_path = Path(str(prefix) + "_rolling_z.csv")
    kelly_trades_path = Path(str(prefix) + "_kelly_trades.csv")
    kelly_summary_path = Path(str(prefix) + "_kelly_summary.csv")

    filtered.to_csv(rows_path, index=False)
    summary_all.to_csv(summary_path, index=False)
    per_day_df.to_csv(daily_path, index=False)
    tier_df.to_csv(tier_path, index=False)
    drawdown_df.to_csv(drawdown_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    rolling_df.to_csv(rolling_path, index=False)
    kelly_trades_df.to_csv(kelly_trades_path, index=False)
    kelly_summary_df.to_csv(kelly_summary_path, index=False)

    print("\n" + "=" * 100)
    print("FORWARD PAPER-TRADE EVALUATION")
    print("=" * 100)
    print(f"Date range:      {args.date_from} -> {args.date_to}")
    print(f"Run root:        {run_root}")
    print(f"Modes:           {', '.join(args.modes)}")
    print(f"Strict aligned:  {bool(args.strict_aligned_only)}")
    print(f"Exclude stale:   {bool(args.exclude_stale_fallback)}")
    print(f"Baseline p:      {float(args.baseline_win_rate):.4f}")
    print(f"Rolling window:  {int(args.rolling_window)}")
    print(f"Calibration bins:{int(args.calibration_bins)}")
    print(f"Kelly start:     {float(args.kelly_start_bankroll):.2f}")
    print(f"Kelly fraction:  {float(args.kelly_fraction):.4f}")
    print(f"Kelly cap:       {float(args.kelly_max_fraction):.4f}")
    print("\nSummary:")
    if summary_all.empty:
        print("(empty)")
    else:
        print(summary_all.to_string(index=False))
    print("\nOutputs:")
    print(f"Rows:      {rows_path}")
    print(f"Summary:   {summary_path}")
    print(f"Daily:     {daily_path}")
    print(f"Tiers:     {tier_path}")
    print(f"Drawdown:  {drawdown_path}")
    print(f"Calib:     {calibration_path}")
    print(f"Rolling Z: {rolling_path}")
    print(f"Kelly:     {kelly_trades_path}")
    print(f"Kelly Sum: {kelly_summary_path}")


if __name__ == "__main__":
    main()
