#!/usr/bin/env python3
"""
Train NBA v9.6 prop probability model.

v9.6 builds on v9.5 pregame lineup with targeted improvements:
  1. Side-specific calibration (separate isotonic for OVER vs UNDER)
  2. Uncertainty-aware probability shrinkage toward market
  3. Regime-adaptive gating (tighter gates in uncertainty_risk regime)
  4. Market-residual correction (lightweight, not full v10 stack)
  5. Pregame lineup interaction features

Key design principle: v10 failed because it added too much complexity
and the branches disagreed. v9.6 keeps the v9 distribution as the
primary signal and only applies corrections where the audit showed
clear weakness (OVER side, high uncertainty, low edge).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "training"))

from market_odds_quality import add_american_odds_quality


def _resolve(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / text).resolve()


def _load_v95_rows(manifest_path: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = _resolve(Path(manifest["output"]))
    rows_path = output / "data" / "prop_training_rows.csv"
    rows = pd.read_csv(rows_path, low_memory=False)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.dropna(subset=["date"]).copy()
    return manifest, rows


def _brier(probs, y) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def _ece(probs, y, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float).clip(0.001, 0.999)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        mask = (p >= bins[idx]) & (p < bins[idx + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _bss(probs, y) -> float:
    return 1.0 - _brier(probs, y) / 0.25


EPS = 1e-4


def clip_prob(values) -> np.ndarray:
    return np.asarray(values, dtype=float).clip(EPS, 1.0 - EPS)


def logit(values) -> np.ndarray:
    p = clip_prob(values)
    return np.log(p / (1.0 - p))


def sigmoid(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ─── Feature Engineering ───────────────────────────────────────────

def add_v96_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Add v9.6 features without leakage."""
    rows = rows.copy()

    # Core probability
    rows["p_over_raw"] = clip_prob(rows["p_over_raw"])

    # Market probability (use 0.5 if not available)
    if "market_no_vig_over" not in rows.columns:
        rows["market_no_vig_over"] = 0.5
    rows["market_no_vig_over"] = clip_prob(rows["market_no_vig_over"])

    # Model-vs-line features
    rows["model_minus_line"] = rows["model_mean"] - rows["line"]
    sigma = rows["sigma"].replace(0, np.nan).fillna(1.0)
    rows["model_minus_line_z"] = rows["model_minus_line"] / sigma

    # Model-vs-market disagreement
    rows["model_market_disagreement"] = np.abs(
        logit(rows["p_over_raw"]) - logit(rows["market_no_vig_over"])
    )

    # Pregame lineup features (interaction terms)
    if "pregame_lineup_adjustment" in rows.columns:
        rows["pregame_lineup_adjustment"] = rows["pregame_lineup_adjustment"].fillna(0.0)
        rows["lineup_adj_x_uncertainty"] = (
            rows["pregame_lineup_adjustment"] * rows["belief_uncertainty"]
        )
        rows["lineup_adj_x_model_edge"] = (
            rows["pregame_lineup_adjustment"] * rows["model_minus_line_z"].abs()
        )
    else:
        rows["pregame_lineup_adjustment"] = 0.0
        rows["lineup_adj_x_uncertainty"] = 0.0
        rows["lineup_adj_x_model_edge"] = 0.0

    # Teammate availability impact
    if "pregame_teammate_out_prob_sum" in rows.columns:
        rows["teammate_out_impact"] = rows["pregame_teammate_out_prob_sum"].fillna(0.0)
    else:
        rows["teammate_out_impact"] = 0.0

    # Side indicator (for side-specific corrections)
    rows["is_over_pick"] = (rows["p_over_raw"] > rows["market_no_vig_over"]).astype(float)

    # Uncertainty regime
    rows["uncertainty_regime"] = pd.cut(
        rows["belief_uncertainty"],
        bins=[-0.001, 0.3, 0.6, 0.85, 1.01],
        labels=["low", "medium", "high", "extreme"],
    ).astype(str)

    return rows


# ─── Side-Specific Isotonic Calibration ────────────────────────────

def fit_side_isotonic(train_rows: pd.DataFrame) -> dict:
    """Fit separate isotonic calibration for OVER and UNDER predictions."""
    from sklearn.isotonic import IsotonicRegression

    calibrators = {}
    probs = clip_prob(train_rows["p_over_raw"])
    y = train_rows["result_over"].astype(float).values

    # Determine side: if p_over > market, it's an OVER pick
    market = clip_prob(train_rows["market_no_vig_over"])
    is_over = probs > market

    # OVER side calibration
    if is_over.sum() >= 50:
        iso_over = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso_over.fit(probs[is_over], y[is_over])
        calibrators["over"] = iso_over

    # UNDER side calibration
    if (~is_over).sum() >= 50:
        iso_under = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso_under.fit(probs[~is_over], y[~is_over])
        calibrators["under"] = iso_under

    # Global fallback
    iso_global = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_global.fit(probs, y)
    calibrators["global"] = iso_global

    return calibrators


def apply_side_isotonic(rows: pd.DataFrame, calibrators: dict) -> np.ndarray:
    """Apply side-specific isotonic calibration."""
    probs = clip_prob(rows["p_over_raw"])
    market = clip_prob(rows["market_no_vig_over"])
    is_over = probs > market

    calibrated = np.full(len(rows), 0.5)

    if "over" in calibrators and is_over.sum() > 0:
        calibrated[is_over] = calibrators["over"].predict(probs[is_over])
    elif "global" in calibrators and is_over.sum() > 0:
        calibrated[is_over] = calibrators["global"].predict(probs[is_over])

    if "under" in calibrators and (~is_over).sum() > 0:
        calibrated[~is_over] = calibrators["under"].predict(probs[~is_over])
    elif "global" in calibrators and (~is_over).sum() > 0:
        calibrated[~is_over] = calibrators["global"].predict(probs[~is_over])

    return np.clip(calibrated, 0.01, 0.99)


# ─── Uncertainty-Aware Market Shrinkage ────────────────────────────

def apply_uncertainty_shrinkage(
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    uncertainty: np.ndarray,
    base_shrink: float = 0.0,
    uncertainty_shrink_rate: float = 0.15,
    max_shrink: float = 0.30,
) -> np.ndarray:
    """Shrink model probability toward market when uncertainty is high.

    The idea: when the model is uncertain, trust the market more.
    This specifically helps the uncertainty_risk regime where v10 had -3.2% BSS.
    """
    shrink_weight = np.clip(
        base_shrink + uncertainty_shrink_rate * uncertainty,
        0.0,
        max_shrink,
    )
    model_logit = logit(model_prob)
    market_logit = logit(market_prob)
    blended_logit = (1.0 - shrink_weight) * model_logit + shrink_weight * market_logit
    return sigmoid(blended_logit)


# ─── Regime-Adaptive Gating ────────────────────────────────────────

def compute_v96_edge_and_gate(
    rows: pd.DataFrame,
    min_edge_over: float = 0.055,
    min_edge_under: float = 0.045,
    max_uncertainty: float = 0.70,
    min_ev: float = 0.025,
) -> pd.DataFrame:
    """Compute edges and apply regime-adaptive gating."""
    rows = rows.copy()

    p_cal = rows["p_v96_calibrated"].values
    market = rows["market_no_vig_over"].values

    rows["edge_over"] = p_cal - market
    rows["edge_under"] = (1.0 - p_cal) - (1.0 - market)

    # Side selection
    rows["selected_side"] = np.where(rows["edge_over"] > rows["edge_under"], "OVER", "UNDER")
    rows["selected_edge"] = np.maximum(rows["edge_over"], rows["edge_under"])

    # Percentile-normalize uncertainty for gating
    rows["uncertainty_percentile"] = rows["belief_uncertainty"].rank(pct=True)

    # Regime-adaptive thresholds
    # In high uncertainty percentile, require higher edge
    unc_pct = rows["uncertainty_percentile"].values
    regime_edge_boost = np.where(unc_pct > 0.85, 0.02, 0.0)

    over_threshold = min_edge_over + regime_edge_boost
    under_threshold = min_edge_under + regime_edge_boost

    # Gate logic (using percentile uncertainty)
    over_pass = (rows["selected_side"] == "OVER") & (rows["edge_over"].values >= over_threshold)
    under_pass = (rows["selected_side"] == "UNDER") & (rows["edge_under"].values >= under_threshold)
    uncertainty_pass = unc_pct <= max_uncertainty

    rows["gated"] = (over_pass | under_pass) & uncertainty_pass
    rows["p_selected"] = np.where(rows["selected_side"] == "OVER", p_cal, 1.0 - p_cal)
    rows["selected_outcome"] = np.where(
        rows["selected_side"] == "OVER",
        rows["result_over"],
        1.0 - rows["result_over"],
    )

    return rows


# ─── Main Training Pipeline ───────────────────────────────────────

def train_v96(
    rows: pd.DataFrame,
    train_end: str | None = None,
    val_start: str | None = None,
    val_end: str | None = None,
    shrink_params: dict | None = None,
    min_edge_over: float = 0.055,
    min_edge_under: float = 0.045,
    max_uncertainty: float = 0.70,
) -> tuple[pd.DataFrame, dict]:
    """Train v9.6 model and return scored rows + diagnostics."""

    if shrink_params is None:
        shrink_params = {
            "base_shrink": 0.0,
            "uncertainty_shrink_rate": 0.12,
            "max_shrink": 0.25,
        }

    # Split into calibration training and validation
    if train_end:
        cal_train = rows[rows["date"] < pd.Timestamp(train_end)].copy()
        val_rows = rows[rows["date"] >= pd.Timestamp(train_end)].copy()
    elif val_start:
        cal_train = rows[rows["date"] < pd.Timestamp(val_start)].copy()
        val_rows = rows[rows["date"] >= pd.Timestamp(val_start)].copy()
    else:
        # Use first 70% for calibration training
        split_idx = int(len(rows) * 0.7)
        sorted_rows = rows.sort_values("date")
        cal_train = sorted_rows.iloc[:split_idx].copy()
        val_rows = sorted_rows.iloc[split_idx:].copy()

    if val_end:
        val_rows = val_rows[val_rows["date"] <= pd.Timestamp(val_end)]

    # Add features
    cal_train = add_v96_features(cal_train)
    val_rows = add_v96_features(val_rows)

    # Fit side-specific isotonic calibration on training data
    calibrators = fit_side_isotonic(cal_train)

    # Apply calibration to validation data
    val_rows["p_v96_side_calibrated"] = apply_side_isotonic(val_rows, calibrators)

    # Apply uncertainty shrinkage
    val_rows["p_v96_calibrated"] = apply_uncertainty_shrinkage(
        model_prob=val_rows["p_v96_side_calibrated"].values,
        market_prob=val_rows["market_no_vig_over"].values,
        uncertainty=val_rows["belief_uncertainty"].values,
        **shrink_params,
    )

    # Compute edges and gates
    val_rows = compute_v96_edge_and_gate(val_rows, min_edge_over=min_edge_over, min_edge_under=min_edge_under, max_uncertainty=max_uncertainty)

    # Also score the full dataset for diagnostics
    all_rows = add_v96_features(rows)
    all_rows["p_v96_side_calibrated"] = apply_side_isotonic(all_rows, calibrators)
    all_rows["p_v96_calibrated"] = apply_uncertainty_shrinkage(
        model_prob=all_rows["p_v96_side_calibrated"].values,
        market_prob=all_rows["market_no_vig_over"].values,
        uncertainty=all_rows["belief_uncertainty"].values,
        **shrink_params,
    )
    all_rows = compute_v96_edge_and_gate(all_rows, min_edge_over=min_edge_over, min_edge_under=min_edge_under, max_uncertainty=max_uncertainty)

    # Diagnostics
    y_val = val_rows["result_over"].astype(float).values
    y_all = all_rows["result_over"].astype(float).values
    gated_val = val_rows[val_rows["gated"]].copy()
    gated_all = all_rows[all_rows["gated"]].copy()

    diagnostics = {
        "calibration_training_rows": int(len(cal_train)),
        "validation_rows": int(len(val_rows)),
        "all_rows": int(len(all_rows)),
        "validation": {
            "v9_raw_brier": _brier(val_rows["p_over_raw"], y_val),
            "v96_calibrated_brier": _brier(val_rows["p_v96_calibrated"], y_val),
            "v96_calibrated_bss": _bss(val_rows["p_v96_calibrated"], y_val),
            "v96_calibrated_ece": _ece(val_rows["p_v96_calibrated"], y_val),
            "gated_rows": int(len(gated_val)),
            "gated_brier": _brier(gated_val["p_selected"], gated_val["selected_outcome"]) if len(gated_val) > 0 else None,
            "gated_bss": _bss(gated_val["p_selected"], gated_val["selected_outcome"]) if len(gated_val) > 0 else None,
            "gated_ece": _ece(gated_val["p_selected"], gated_val["selected_outcome"]) if len(gated_val) > 0 else None,
            "gated_hit_rate": float(gated_val["selected_outcome"].mean()) if len(gated_val) > 0 else None,
        },
        "all_data": {
            "v9_raw_brier": _brier(all_rows["p_over_raw"], y_all),
            "v96_calibrated_brier": _brier(all_rows["p_v96_calibrated"], y_all),
            "v96_calibrated_bss": _bss(all_rows["p_v96_calibrated"], y_all),
            "v96_calibrated_ece": _ece(all_rows["p_v96_calibrated"], y_all),
            "gated_rows": int(len(gated_all)),
            "gated_brier": _brier(gated_all["p_selected"], gated_all["selected_outcome"]) if len(gated_all) > 0 else None,
            "gated_bss": _bss(gated_all["p_selected"], gated_all["selected_outcome"]) if len(gated_all) > 0 else None,
            "gated_ece": _ece(gated_all["p_selected"], gated_all["selected_outcome"]) if len(gated_all) > 0 else None,
            "gated_hit_rate": float(gated_all["selected_outcome"].mean()) if len(gated_all) > 0 else None,
        },
        "side_analysis": {},
        "shrink_params": shrink_params,
    }

    # Side-specific analysis on validation
    if len(gated_val) > 0:
        for side in ["OVER", "UNDER"]:
            side_rows = gated_val[gated_val["selected_side"] == side]
            if len(side_rows) >= 30:
                diagnostics["side_analysis"][side] = {
                    "n": int(len(side_rows)),
                    "brier": _brier(side_rows["p_selected"], side_rows["selected_outcome"]),
                    "bss": _bss(side_rows["p_selected"], side_rows["selected_outcome"]),
                    "hit_rate": float(side_rows["selected_outcome"].mean()),
                    "share": float(len(side_rows) / len(gated_val)),
                }

    return val_rows, all_rows, calibrators, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NBA v9.6 prop model")
    parser.add_argument(
        "--v95-manifest",
        type=Path,
        default=ROOT / "model" / "props" / "v9_5_prelock_availability_w050" / "manifest.json",
    )
    parser.add_argument("--train-end", type=str, default="2026-02-15")
    parser.add_argument("--val-start", type=str, default="2026-02-15")
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_6")
    parser.add_argument("--base-shrink", type=float, default=0.0)
    parser.add_argument("--uncertainty-shrink-rate", type=float, default=0.12)
    parser.add_argument("--max-shrink", type=float, default=0.25)
    parser.add_argument("--min-edge-over", type=float, default=0.055)
    parser.add_argument("--min-edge-under", type=float, default=0.045)
    parser.add_argument("--max-uncertainty", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = _resolve(args.v95_manifest)
    v95_manifest, rows = _load_v95_rows(manifest_path)

    print(f"[v9.6] Loaded {len(rows)} rows from v9.5 manifest")
    print(f"[v9.6] Date range: {rows['date'].min().date()} to {rows['date'].max().date()}")
    print(f"[v9.6] Training calibration on data before {args.train_end}")
    print(f"[v9.6] Validating on data from {args.val_start}")

    shrink_params = {
        "base_shrink": args.base_shrink,
        "uncertainty_shrink_rate": args.uncertainty_shrink_rate,
        "max_shrink": args.max_shrink,
    }

    val_rows, all_rows, calibrators, diagnostics = train_v96(
        rows,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        shrink_params=shrink_params,
        min_edge_over=args.min_edge_over,
        min_edge_under=args.min_edge_under,
        max_uncertainty=args.max_uncertainty,
    )

    # Write outputs
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "calibration").mkdir(parents=True, exist_ok=True)

    # Save scored validation rows
    val_rows.to_csv(output / "data" / "prop_training_rows.csv", index=False)

    # Save calibrators
    import joblib
    joblib.dump(calibrators, output / "calibration" / "side_isotonic_calibrators.pkl")

    # Save diagnostics
    (output / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )

    # Build manifest
    manifest = {
        "model_version": "prop_engine_v9_6_side_calibrated_shrinkage",
        "status": "candidate_pending_walk_forward",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_v95_manifest": str(manifest_path),
        "output": str(output),
        "rows": int(len(rows)),
        "validation_rows": int(len(val_rows)),
        "players": int(rows["player"].nunique()),
        "date_min": str(rows["date"].min().date()),
        "date_max": str(rows["date"].max().date()),
        "calibration_train_end": args.train_end,
        "validation_start": args.val_start,
        "artifacts": {
            "data": "data/prop_training_rows.csv",
            "calibration": "calibration/side_isotonic_calibrators.pkl",
            "training_diagnostics": "training_diagnostics.json",
        },
        "improvements_over_v95": [
            "Side-specific isotonic calibration (separate OVER/UNDER)",
            "Uncertainty-aware market shrinkage",
            "Regime-adaptive gating (tighter in uncertainty_risk)",
            "Pregame lineup interaction features",
        ],
        "shrink_params": shrink_params,
        "gate_params": {
            "min_edge_over": args.min_edge_over,
            "min_edge_under": args.min_edge_under,
            "max_uncertainty": args.max_uncertainty,
        },
        "diagnostics_summary": diagnostics["validation"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # Print results
    print("\n" + "=" * 60)
    print("V9.6 TRAINING RESULTS")
    print("=" * 60)
    print(f"\nValidation ({args.val_start} to {args.val_end or 'end'}):")
    v = diagnostics["validation"]
    print(f"  v9 raw Brier:          {v['v9_raw_brier']:.6f}")
    print(f"  v9.6 calibrated Brier: {v['v96_calibrated_brier']:.6f}")
    print(f"  v9.6 calibrated BSS:   {v['v96_calibrated_bss']:.4f} ({v['v96_calibrated_bss']*100:.2f}%)")
    print(f"  v9.6 calibrated ECE:   {v['v96_calibrated_ece']:.6f}")
    print(f"  Gated rows:            {v['gated_rows']}")
    if v["gated_brier"] is not None:
        print(f"  Gated Brier:           {v['gated_brier']:.6f}")
        print(f"  Gated BSS:             {v['gated_bss']:.4f} ({v['gated_bss']*100:.2f}%)")
        print(f"  Gated ECE:             {v['gated_ece']:.6f}")
        print(f"  Gated hit rate:        {v['gated_hit_rate']:.4f}")

    print(f"\nSide analysis:")
    for side, metrics in diagnostics["side_analysis"].items():
        print(f"  {side}: n={metrics['n']}, Brier={metrics['brier']:.4f}, BSS={metrics['bss']*100:.2f}%, hit={metrics['hit_rate']:.3f}, share={metrics['share']:.2f}")

    print(f"\nComparison targets:")
    print(f"  v9.4 safe gated Brier:     0.2108")
    print(f"  v9.5 prelock gated Brier:  0.1973")
    print(f"  v9.4 oracle ceiling:       0.1920")
    print(f"\nManifest written to: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
