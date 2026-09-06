#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.advanced.schema import AdvancedCandidateContext, BatterProcessProfile, PitcherProcessProfile
from sports.mlb.advanced.sequential_pa_model import simulate_hitter_market

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_JSON = REPO_ROOT / "artifacts" / "mlb_sequential_pa_historical_validation.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "mlb_sequential_pa_historical_validation.md"


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_rate(num: float, den: float, default: float) -> float:
    return float(num / den) if den > 0 else float(default)


def poisson_over_probability(mean: float, line: float) -> float:
    threshold = math.floor(float(line)) + 1
    term = math.exp(-max(0.0, float(mean)))
    cdf = term
    for k in range(1, threshold):
        term *= max(0.0, float(mean)) / k
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def prior_batter_profile(history: pd.DataFrame, *, player_id: int, player_name: str, as_of_date: str) -> BatterProcessProfile:
    recent = history.tail(30)
    pa = pd.to_numeric(history.get("PA"), errors="coerce").fillna(0.0)
    ab = pd.to_numeric(history.get("AB"), errors="coerce").fillna(0.0)
    so = pd.to_numeric(history.get("SO"), errors="coerce").fillna(0.0)
    bb = pd.to_numeric(history.get("BB"), errors="coerce").fillna(0.0) if "BB" in history else pd.Series(0.0, index=history.index)
    hr = pd.to_numeric(history.get("HR"), errors="coerce").fillna(0.0)
    hits = pd.to_numeric(history.get("H"), errors="coerce").fillna(0.0)
    tb = pd.to_numeric(history.get("TB"), errors="coerce").fillna(0.0)
    pa_sum = float(pa.sum())
    ab_sum = float(ab.sum())
    k_rate = safe_rate(float(so.sum()), pa_sum, 0.225)
    bb_rate = safe_rate(float(bb.sum()), pa_sum, 0.085)
    hr_rate = safe_rate(float(hr.sum()), pa_sum, 0.030)

    non_hr_hits = max(0.0, float(hits.sum() - hr.sum()))
    fieldable_ab = max(1.0, ab_sum - float(so.sum()) - float(hr.sum()))
    contact_hit_rate = max(0.12, min(0.62, non_hr_hits / fieldable_ab))
    slg = safe_rate(float(tb.sum()), ab_sum, 0.420)
    xwoba = finite(pd.to_numeric(recent.get("xwOBA"), errors="coerce").dropna().mean() if "xwOBA" in recent else None)
    hard_hit = finite(pd.to_numeric(recent.get("HardHit%"), errors="coerce").dropna().mean() if "HardHit%" in recent else None)
    barrel = finite(pd.to_numeric(recent.get("Barrel%"), errors="coerce").dropna().mean() if "Barrel%" in recent else None)
    # Processed historical files do not preserve pitch-level expected hit type
    # probabilities for every date. The validation proxy therefore uses only
    # information available before the game and labels this limitation.
    return BatterProcessProfile(
        player_id=player_id,
        player_name=player_name,
        as_of_date=as_of_date,
        sample_pa=int(pa_sum),
        sample_bbe=int(fieldable_ab),
        k_rate=max(0.02, min(0.60, k_rate)),
        bb_rate=max(0.01, min(0.30, bb_rate)),
        hbp_rate=0.012,
        hr_rate=max(0.002, min(0.18, hr_rate)),
        contact_rate=max(0.05, min(0.98, 1.0 - k_rate)),
        whiff_rate=max(0.02, min(0.60, k_rate * 0.78)),
        woba=finite(pd.to_numeric(recent.get("wOBA"), errors="coerce").dropna().mean() if "wOBA" in recent else None),
        xwoba=xwoba,
        xba=contact_hit_rate,
        xslg=max(0.20, min(1.10, slg)),
        avg_ev=None,
        hard_hit_rate=(hard_hit / 100.0 if hard_hit is not None and hard_hit > 1 else hard_hit),
        barrel_rate=(barrel / 100.0 if barrel is not None and barrel > 1 else barrel),
        support=max(0.0, min(1.0, len(history) / 30.0)),
    )


def prior_pitcher_proxy(row: pd.Series, *, as_of_date: str) -> PitcherProcessProfile:
    k9 = finite(row.get("Opp_Pitcher_K9_3"), 8.2) or 8.2
    era = finite(row.get("Opp_Pitcher_ERA_3"), 4.1) or 4.1
    # Convert K/9 to an approximate K% using 38 batters faced per nine as a
    # neutral exposure denominator. This is a historical validation proxy only;
    # live production uses pitch-level Statcast process data.
    k_rate = max(0.05, min(0.45, k9 / 38.0))
    vulnerability_h = finite(row.get("Pitcher_Profile_H_Vulnerability"), 0.0) or 0.0
    vulnerability_tb = finite(row.get("Pitcher_Profile_TB_Vulnerability"), 0.0) or 0.0
    xba_allowed = max(0.20, min(0.46, 0.320 + 0.035 * vulnerability_h))
    xslg_allowed = max(0.32, min(0.80, 0.510 + 0.075 * vulnerability_tb))
    return PitcherProcessProfile(
        player_id=int(finite(row.get("Opp_Starter_ID"), 0) or 0),
        player_name=str(row.get("Opp_Starter_Player") or row.get("Opposing_Pitcher") or "historical starter"),
        as_of_date=as_of_date,
        sample_pa=180,
        sample_bbe=110,
        k_rate=k_rate,
        bb_rate=0.085,
        hbp_rate=0.012,
        hr_rate=0.030,
        k_minus_bb_rate=k_rate - 0.085,
        whiff_rate=max(0.08, min(0.40, k_rate * 0.95)),
        xwoba_allowed=max(0.24, min(0.42, 0.315 + 0.025 * vulnerability_tb)),
        xba_allowed=xba_allowed,
        xslg_allowed=xslg_allowed,
        era=era,
        xfip=None,
        siera=None,
        projected_ip=5.4,
        support=0.60,
    )


def legacy_projection_probability(row: pd.Series, target: str, line: float) -> tuple[float, float]:
    market = finite(row.get(f"Market_{target}"), line) or line
    gap = finite(row.get(f"{target}_market_gap"))
    rolling = finite(row.get(f"{target}_rolling_avg"))
    if gap is not None:
        prediction = max(0.0, market + gap)
    elif rolling is not None:
        prediction = max(0.0, rolling)
    else:
        prediction = max(0.0, market)
    return prediction, poisson_over_probability(prediction, line)


def brier(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([(float(row[key]) - float(row["outcome"])) ** 2 for row in rows]))


def logloss(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    values = []
    for row in rows:
        p = max(1e-6, min(1.0 - 1e-6, float(row[key])))
        y = float(row["outcome"])
        values.append(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))
    return float(np.mean(values))


def ece(rows: list[dict[str, Any]], key: str, bins: int = 10) -> float | None:
    if not rows:
        return None
    total = len(rows)
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        bucket = [row for row in rows if lower <= float(row[key]) < upper or (upper >= 1.0 and float(row[key]) == 1.0)]
        if not bucket:
            continue
        mean_p = float(np.mean([float(row[key]) for row in bucket]))
        mean_y = float(np.mean([float(row["outcome"]) for row in bucket]))
        error += len(bucket) / total * abs(mean_p - mean_y)
    return float(error)


def calibration_bins(rows: list[dict[str, Any]], key: str, bins: int = 10) -> list[dict[str, Any]]:
    out = []
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        bucket = [row for row in rows if lower <= float(row[key]) < upper or (upper >= 1.0 and float(row[key]) == 1.0)]
        if bucket:
            out.append({
                "lower": float(lower),
                "upper": float(upper),
                "n": len(bucket),
                "mean_probability": float(np.mean([float(row[key]) for row in bucket])),
                "hit_rate": float(np.mean([float(row["outcome"]) for row in bucket])),
            })
    return out


def collect_rows(data_root: Path, *, season: int, max_rows: int, trials: int, min_history: int) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    files = sorted(data_root.glob(f"*/{season}_processed_processed.csv"))
    # Deterministic interleaving across players prevents one long-tenured hitter
    # from dominating a bounded validation run.
    for file_index, path in enumerate(files):
        if len(evaluated) >= max_rows:
            break
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or str(frame.iloc[0].get("Player_Type") or "").lower() != "hitter":
            continue
        frame["_date"] = pd.to_datetime(frame.get("Date"), errors="coerce")
        frame = frame.loc[frame["_date"].notna()].sort_values(["_date", "Game_Index" if "Game_Index" in frame.columns else "_date"]).reset_index(drop=True)
        if len(frame) <= min_history:
            continue
        player_name = str(frame.iloc[0].get("Player") or path.parent.name).replace("_", " ")
        player_id = int(finite(frame.iloc[-1].get("Player_MLBAM_ID"), file_index + 1) or file_index + 1)
        # Evaluate recent rows but preserve a strict prior-history cut.
        candidate_indices = list(range(min_history, len(frame)))
        if len(candidate_indices) > 6:
            candidate_indices = candidate_indices[-6:]
        for idx in candidate_indices:
            if len(evaluated) >= max_rows:
                break
            row = frame.iloc[idx]
            history = frame.iloc[:idx]
            as_of_date = row["_date"].date().isoformat()
            batter = prior_batter_profile(history, player_id=player_id, player_name=player_name, as_of_date=as_of_date)
            pitcher = prior_pitcher_proxy(row, as_of_date=as_of_date)
            batting_order = int(finite(row.get("Batting_Order"), 6) or 6)
            context = AdvancedCandidateContext(
                game_id=str(row.get("Game_ID") or f"hist-{file_index}-{idx}"),
                run_date=as_of_date,
                batter=batter,
                pitcher=pitcher,
                direct_matchup=None,
                batting_order=batting_order,
                is_home=str(row.get("Is_Home") or "0").strip() in {"1", "true", "True"},
                team_expected_runs=None,
                park_factor=float(finite(row.get("Park_Factor"), 1.0) or 1.0),
                defense_residual=0.0,
                defense_status="HISTORICAL_AVERAGE_CONTEXT_ONLY",
                data_freshness_status="FRESH",
                missing_components=("HISTORICAL_PITCH_LEVEL_XFIP_SIERA_NOT_PRESERVED",),
            )
            for target, line, actual in (("H", 0.5, finite(row.get("H"))), ("TB", 1.5, finite(row.get("TB")))):
                if actual is None:
                    continue
                legacy_prediction, legacy_p = legacy_projection_probability(row, target, line)
                result = simulate_hitter_market(context, target=target, market_line=line, trials=trials)
                outcome = 1 if actual > line else 0
                expected_count = result.expected_hits if target == "H" else result.expected_tb
                evaluated.append({
                    "date": as_of_date,
                    "player": player_name,
                    "target": target,
                    "line": line,
                    "actual": actual,
                    "outcome": outcome,
                    "legacy_prediction": legacy_prediction,
                    "legacy_probability": legacy_p,
                    "sequential_probability": result.raw_structural_probability,
                    "sequential_usable_probability": result.usable_probability,
                    "sequential_expected_count": expected_count,
                    "p_h_0": result.p_h_0,
                    "history_rows": len(history),
                })
                if len(evaluated) >= max_rows:
                    break
    return evaluated


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target in ("H", "TB"):
        subset = [row for row in rows if row["target"] == target]
        output[target] = {
            "rows": len(subset),
            "observed_hit_rate": float(np.mean([row["outcome"] for row in subset])) if subset else None,
            "legacy": {
                "brier": brier(subset, "legacy_probability"),
                "log_loss": logloss(subset, "legacy_probability"),
                "ece": ece(subset, "legacy_probability"),
                "mae_count": float(np.mean([abs(row["legacy_prediction"] - row["actual"]) for row in subset])) if subset else None,
                "calibration_bins": calibration_bins(subset, "legacy_probability"),
            },
            "sequential_raw": {
                "brier": brier(subset, "sequential_probability"),
                "log_loss": logloss(subset, "sequential_probability"),
                "ece": ece(subset, "sequential_probability"),
                "mae_count": float(np.mean([abs(row["sequential_expected_count"] - row["actual"]) for row in subset])) if subset else None,
                "calibration_bins": calibration_bins(subset, "sequential_probability"),
            },
            "sequential_usable": {
                "brier": brier(subset, "sequential_usable_probability"),
                "log_loss": logloss(subset, "sequential_usable_probability"),
                "ece": ece(subset, "sequential_usable_probability"),
                "calibration_bins": calibration_bins(subset, "sequential_usable_probability"),
            },
        }
    h_rows = [row for row in rows if row["target"] == "H"]
    output["zero_hit"] = {
        "rows": len(h_rows),
        "predicted_zero_hit_rate": float(np.mean([row["p_h_0"] for row in h_rows])) if h_rows else None,
        "observed_zero_hit_rate": float(np.mean([1 - row["outcome"] for row in h_rows])) if h_rows else None,
    }
    return output


def markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Sequential PA historical validation",
        "",
        f"Evidence class: `{payload['evidence_class']}`",
        "",
        "This is a strict rolling-origin predictive diagnostic using only each player's rows before the evaluated game. It does **not** claim full live-model certification because historical pitch-level xFIP/SIERA/OAA snapshots were not preserved for every replay date.",
        "",
        "| Target | Rows | Legacy Brier | Seq raw Brier | Seq usable Brier | Legacy logloss | Seq raw logloss | Legacy MAE | Seq MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ("H", "TB"):
        item = summary[target]
        def f(value):
            return "n/a" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {target} | {item['rows']} | {f(item['legacy']['brier'])} | {f(item['sequential_raw']['brier'])} | "
            f"{f(item['sequential_usable']['brier'])} | {f(item['legacy']['log_loss'])} | {f(item['sequential_raw']['log_loss'])} | "
            f"{f(item['legacy']['mae_count'])} | {f(item['sequential_raw']['mae_count'])} |"
        )
    zero = summary["zero_hit"]
    lines.extend([
        "",
        "## Zero-hit calibration",
        "",
        f"Predicted zero-hit rate: `{zero['predicted_zero_hit_rate']}`; observed zero-hit rate: `{zero['observed_zero_hit_rate']}` across `{zero['rows']}` H observations.",
        "",
        "## Economic evidence",
        "",
        "No ROI claim is made unless an exact preserved decision-time price and timestamp are present. This processed-history diagnostic does not fabricate historical sportsbook prices.",
        "",
        "## Limitations",
        "",
        "- Historical processed rows preserve useful rolling hitter/process and opponent-starter fields but not the complete live Statcast/FanGraphs/OAA state now used by production.",
        "- The pitcher portion of this replay is therefore a leakage-safe process proxy, explicitly not a reconstruction of missing historical xFIP/SIERA snapshots.",
        "- Results are predictive diagnostics only and cannot promote the new model out of negative-authority/shadow status.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--max-rows", type=int, default=240)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--min-history", type=int, default=20)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.data_root, season=args.season, max_rows=args.max_rows, trials=args.trials, min_history=args.min_history)
    payload = {
        "schema_version": "mlb_sequential_pa_historical_validation_v1",
        "evidence_class": "ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION",
        "season": args.season,
        "rows": len(rows),
        "summary": summarize(rows),
        "economic_evidence": {"roi_claim": False, "reason": "exact decision-time prices are not fabricated for processed-history replay"},
        "observations": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
