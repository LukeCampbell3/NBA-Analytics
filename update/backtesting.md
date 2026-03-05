# Backtesting & Validation Plan — Breakout Detector (Robust, Calibrated, Self-Auditing)

This document is a build-spec for turning your breakout detector from a **heuristic gate** into a **calibrated, decision-grade** module.

---

## 0) Goal (what “validated” means)

You want:

- **Operational labels** for breakouts that can be measured from available data.
- A **repeatable backtest pipeline** that:
  1) generates predictions from your tools,
  2) builds ground-truth labels from later-season outcomes / priors,
  3) evaluates accuracy,
  4) **calibrates confidence** so predicted confidence matches empirical hit rates,
  5) detects systematic drift by archetype/team/role.

The key outcome is not only “hit rate,” but **truthful confidence**:
> If the system outputs 0.70 confidence, it should be ~70% correct under the defined label.

---

## 1) What you need (data inputs)

### 1.1 Minimum required (works with limited data)
For each player-season `t` (e.g., 2023 season) you need:
- Your player card fields at season `t` (your existing JSON cards).
- Next-season outcomes for the same player at season `t+1`:
  - box/role: MPG, USG, PTS, AST, TS% (or approximations).
  - or “impact prior” value at `t` and `t+1` if available.

**If you do not have a true impact prior dataset yet:** you can still backtest *role/box breakouts* and later add impact-based labels.

### 1.2 Strongly recommended (turns this into “adult supervision”)
Add at least one **impact prior family** series:
- EPM-like, LEBRON-like, DARKO-like, RAPM-like.
Even one is enough to start; multiple is better for consensus.

### 1.3 Optional but high leverage
- Age (or DOB) so you can backtest aging curve behavior.
- Contract info (AAV, years remaining) if you want valuation backtests too.
- Team context / roster proxies (even simple) to interpret false positives/negatives.

---

## 2) Define breakout labels (ground truth)

You need labels you can compute from available data. Use **three breakout types**:

### 2.1 Role Breakout (deployment shift)
A player is a **role breakout** if:
- `MPG_{t+1} - MPG_t >= mpg_delta_threshold`
AND at least one:
- `USG_{t+1} - USG_t >= usg_delta_threshold`
- role band changes (bench → starter; secondary → primary; etc. if you track it)

Suggested defaults (tune later):
- `mpg_delta_threshold = +3.0`
- `usg_delta_threshold = +0.03`

### 2.2 Efficiency Breakout (production quality)
A player is an **efficiency breakout** if:
- `TS%_{t+1} - TS%_t >= ts_delta_threshold`  
OR if TS% not available, use a proxy:
- `eFG%` or `3P%` uplift with volume stability
- turnover rate improvement with stable usage

Suggested default:
- `ts_delta_threshold = +0.015` (1.5 TS points)

### 2.3 Impact Breakout (reality anchor)
A player is an **impact breakout** if:
- `ImpactPrior_{t+1} - ImpactPrior_t >= impact_delta_threshold`

Suggested default:
- `impact_delta_threshold = +0.75 points/100` (tune based on distribution)

### 2.4 Ecosystem Breakout (your unique niche)
This is the most important label for your “hidden value” concept:

A player is an **ecosystem breakout** if:
- `ImpactPrior_{t+1} - ImpactPrior_t >= eco_impact_delta_threshold`
AND
- box/role lift is small:
  - `|PTS_{t+1} - PTS_t| < pts_small_delta`
  - `|USG_{t+1} - USG_t| < usg_small_delta`
  - `|MPG_{t+1} - MPG_t| < mpg_small_delta`

Suggested defaults:
- `eco_impact_delta_threshold = +0.75 points/100`
- `pts_small_delta = 1.5`
- `usg_small_delta = 0.02`
- `mpg_small_delta = 2.0`

This label explicitly tests: “impact rises without box inflation.”

---

## 3) Prediction targets (what the model must output)

For each player-season `t`, your breakout detector should output:

### 3.1 Continuous scores (not just booleans)
- `p_visible_breakout` in [0,1]
- `p_ecosystem_breakout` in [0,1]
- `p_any_breakout` = max(p_visible, p_ecosystem) or calibrated fusion
- `confidence` in [0,1] (or same as probability post-calibration)

### 3.2 Evidence + trace (required)
- top candidate team environment
- projected deltas (MPG/USG/PTS/AST + hidden delta)
- clamp list + clamp severity
- isolation penalties
- archetype priors used
- impact prior disagreement (when priors are available)

If you only output booleans, calibration becomes fragile.

---

## 4) Dataset splits (avoid leakage)

Use **time-based splits**.

Example:
- Train/calibrate: 2019–2022 seasons (predict t→t+1)
- Validate: 2023–2024 seasons

If you only have a short window:
- Use rolling evaluation:
  - train/calibrate on years <= Y
  - test on year Y+1
  - move forward

Never shuffle across years; that leaks future regime information.

---

## 5) Scoring metrics (what you evaluate)

You need both classification performance and calibration.

### 5.1 Classification
For each label type (role/efficiency/impact/ecosystem):
- Precision, Recall, F1
- ROC-AUC (optional if probabilities are good)
- PR-AUC (usually more informative for rare breakouts)

### 5.2 Calibration (mandatory)
- Reliability diagram: predicted probability bins vs empirical success rate
- ECE (Expected Calibration Error)
- Brier score

Target: confidence bins should be aligned with hit rates.

### 5.3 Drift & bias diagnostics (mandatory)
Compute error metrics by:
- archetype
- usage band
- team (current team and predicted best team)
- comp density bands (low/med/high)
- data quality tiers (trust score bins)
- age bands

This reveals systematic failure modes.

---

## 6) Calibration loop (robust validation)

### 6.1 Core calibration approach
Start with **post-hoc calibration** on the raw probability outputs:
- Isotonic regression (good for flexible calibration)
- Platt scaling (logistic) as simpler baseline

If you don’t have enough data for isotonic, use Platt.

### 6.2 Confidence governance rules
After calibration, apply governance:
- If impact prior disagreement is high → reduce probability / require review flag
- If comp density is low → widen uncertainty / reduce confidence
- If clamp severity is high → cap breakout probability

These are deterministic, traceable controls.

---

## 7) Robust validation loop (step-by-step)

### Step A — Build dataset table
Construct a row per player-season `t` with:
- features from card_t (your tool outputs)
- predictions from breakout detector at time t
- next-year outcomes from t+1
- label(s) computed from outcomes

Store as `backtests/breakout_dataset.parquet` (or CSV).

### Step B — Train / fit calibrator
Using the calibration split:
- Fit calibrator mapping from raw `p_any_breakout` → calibrated `p_cal`
- Fit separate calibrators per label type if needed (visible vs ecosystem)

### Step C — Evaluate on held-out year(s)
Compute classification + calibration metrics.

### Step D — Auto-adjust thresholds (optional)
If you still want boolean “breakout” tags:
- choose a threshold `τ` that achieves your desired precision level
  - e.g., τ chosen so precision >= 0.70 on validation

### Step E — Drift report
Generate a drift report:
- biggest archetype failure modes
- teams where team environment proxy is unstable
- high-confidence false positives (most important review targets)

### Step F — Update priors/weights conservatively
Only adjust one of:
- archetype prior table
- confidence component weights
- threshold constants
- clamp severity mapping

Then rerun backtest, compare deltas. Keep a changelog.

---

## 8) Directory layout (suggested)

```
core/
  backtests/
    data/
      outcomes/              # t+1 outcome tables by season
      impact_priors/         # epm/lebron/darko proxies
    runs/
      2026-03-04_run001/
        predictions.parquet
        labels.parquet
        metrics.json
        calibration.json
        drift_report.md
        reliability.png
    breakout_backtest.py
    calibration.py
    metrics.py
    drift.py
    config.yaml
```

---

## 9) Implementation skeleton (validation runner behavior)

### 9.1 breakout_backtest.py responsibilities
1) Discover cards by season `t`
2) Run breakout detector → predictions
3) Join `t+1` outcomes → labels
4) Split by year
5) Fit calibrator(s)
6) Score metrics
7) Emit artifacts

### 9.2 Artifacts to save every run
- `predictions.parquet`: player-season with raw probabilities + traces
- `labels.parquet`: label columns
- `metrics.json`: precision/recall/F1 + calibration
- `reliability.png`: reliability diagram
- `drift_report.md`: archetype/team failure analysis
- `calibration.json`: calibrator params
- `run_manifest.json`: config + git hash + tool versions

---

## 10) Label computation details (recommended rules)

### 10.1 Guardrails to prevent label noise
Only label a player-season if:
- player has sufficient games played in both seasons (e.g., GP >= 30)
- or has a “data quality tier” high enough

If missing, label as `unknown` and exclude from scoring.

### 10.2 Breakout rarity
Breakouts are rare; avoid thresholds that tag half the league.
Use distributional cutoffs:
- define impact breakout as top X% delta (e.g., 12–18%)
- then compare to fixed-point threshold

This reduces sensitivity to era changes.

---

## 11) How to incorporate archetype breakout% per archetype (correctly)

Instead of hardcoding:
- compute empirical breakout rate by archetype on train window:
  - `P(breakout | archetype)`
- compute repeat-breakout rate:
  - `P(breakout | archetype, already_established)`

Then use shrinkage:
- if archetype has low sample, shrink to league mean.

This turns your archetype priors into learned priors.

---

## 12) “Limited data” upgrades that still work

Even without tracking/lineups, you can improve robustness by:
- Calibrating confidence
- Adding impact prior disagreement flags
- Using shrinkage for team profiles + archetype priors
- Adding comp density penalties
- Tracking clamp severity and penalizing over-constrained counterfactuals

These are methodology wins, not data wins.

---

## 13) Acceptance criteria (when the system is “good enough”)

Pick clear criteria:
- Calibration: ECE <= 0.05 for p_any_breakout (or comparable)
- Precision at chosen threshold τ >= 0.70 for “breakout” tag
- Ecosystem breakout precision >= 0.60 (harder problem)
- Drift: no archetype with precision < 0.40 if confidence >= 0.75 (bias red flag)

---

## 14) Next step checklist (actionable)

1) Decide label thresholds for your first backtest window.
2) Create an outcomes table for `t+1` (box + minutes + usage; impact if possible).
3) Build `breakout_backtest.py` that outputs dataset rows and metrics.
4) Fit a calibrator and produce reliability plots.
5) Iterate: adjust priors/weights/thresholds *one change at a time*.

---

## Appendix A — Minimal dataset columns (recommended)

### From card_t / tools (features)
- player_id, player_name, team, season
- archetype, archetype_confidence
- comp_density
- trust_score, uncertainty
- current metrics: mpg, usg, ppg, apg, etc.
- projected deltas to best team: mpg/usg/ppg/apg deltas
- hidden value: expected_non_box_team_impact_delta, suppression_relief
- clamp severity, violations count
- impact prior at t (if available)
- impact disagreement (if available)

### Outcomes at t+1 (for labels)
- mpg_{t+1}, usg_{t+1}, ppg_{t+1}, apg_{t+1}, TS%_{t+1}
- impact_{t+1} (if available)

---

## Appendix B — Example label columns

- label_role_breakout (0/1/NA)
- label_efficiency_breakout (0/1/NA)
- label_impact_breakout (0/1/NA)
- label_ecosystem_breakout (0/1/NA)
- label_any_breakout = OR of the above (or separate reporting)

---

If you want, I can turn this spec into an actual runnable `breakout_backtest.py` + `calibration.py` + `metrics.py` scaffold that:
- reads your card directory by season,
- joins an outcomes CSV,
- runs breakout detector,
- outputs reliability plots and drift reports.
