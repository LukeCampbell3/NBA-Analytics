# Repository Inventory — Universal Multi-Sport Model

Generated as step 1-3 of the DRM-guided universal multi-sport build order.
This is a factual survey of what actually exists in this repository before
any universal-model code was written. All counts below were measured
directly (`wc -l`, `pandas` header/row inspection, `find`/`du`), not
estimated.

## 0. Repositories inspected

- **`LukeCampbell3/NBA-Analytics`** (this repo) — fully inspected: `sports/`,
  `sports/shared/`, `sports/validation/`, `sports/mlb/`, `sports/nba/`,
  `sports/nfl/`, `sports/f1/`, `sports/golf/`, existing prediction datasets,
  feature generation code, odds/market ingestion, validation protocols,
  historical partitions, existing model code, settlement/betting outputs
  (`parlay_certification_v2`).
- **`LukeCampbell3/drm-linux-kernel-coder`** — **NOT inspected.** A prior
  request in this session to attach this repository (read-only) for
  inspection was explicitly denied. The DRM structural controller
  (`sports/universal_model/drm_controller/`) is therefore implemented from
  the architectural principles as described in the mission specification
  text itself (OBSERVE/DERIVE/COMMIT, provisional vs. permanent structure,
  bounded mutation vocabulary, function-preserving expert birth, deferred
  consolidation, rollback-on-rejection) rather than from the DRM repository's
  actual source. This is a real limitation, not a stylistic choice, and is
  carried into `FINAL_REPORT.md` rather than hidden.

## 1. Per-sport real historical data (measured)

| Sport | Raw archive | Compiled per-observation history w/ real outcomes | Date span | Verdict |
|---|---|---|---|---|
| MLB | `sports/mlb/data/raw/` — 2,221 files, 150 MB (ESPN team games, StatsAPI pitcher boxscores, market odds) | `sports/mlb/data/predictions/calibration/historical_pool_universe_2026.csv` — **242,425 rows**, real `Result`/`Actual` columns (170,593 win / 71,678 loss / 154 push), 6 target families (H, TB, R, HR, RBI, K, ER), real market odds (line, price, book, price timestamp). Also `mlb_walk_forward_candidate_ledger.csv` (72,854 rows) and `mlb_walk_forward_backtest_rows.csv` (7,078 rows). | 2026-03-01 → 2026-08-05 (one real season in progress, ~5 months) | **INCLUDE — primary sport.** By far the largest, most complete, outcome-labeled, market-priced dataset in the repo. |
| NFL | `sports/nfl/data/raw/` — only 4 files, 152 KB | `sports/nfl/data/evaluation/backtest_rows.csv` — **5,328 rows**, real `season`/`week`/`actual`/`prediction` columns | season 2025, weeks 1–18 (one season) | **INCLUDE — small, real.** Not enough to stand alone as a full training sport, but genuinely real and dated; usable for leave-one-sport-out transfer and small-data-regime tests. |
| NBA | `sports/nba/data/raw/` — 81 files, 2.9 MB; large ancillary trees under `sports/nba/predictions/Player-Predictor/` (parquet game logs, prop odds history) | No compiled per-observation, outcome-labeled backtest ledger was found. `sports/nba/validation/production_shadow/player_simulation/backtests/2025_preseason/simulation_backtest_2025_preseason_rows.csv` exists but contains **0 data rows** (header only). `player_simulation_summary.csv` (6,985 rows) is a per-player aggregate *simulation* summary, not dated per-event observations with settled outcomes. | n/a | **EXCLUDE from training data.** Adapter is still implemented against the raw/prop sources for architecture-compatibility and the new-sport-onboarding acceptance test, but is not backed by enough settled, dated observations to contribute real training signal. Reported honestly rather than forced in. |
| Golf | `sports/golf/data/raw/` — 17 files, 1.3 MB, all current-tournament ESPN leaderboard snapshots | None found. `sports/golf/predictions/score_model.py` is a live scoring-projection model with no historical settled-outcome ledger. | n/a (current tournament only, no history) | **EXCLUDE from training data.** |
| F1 | No `data/raw/` directory at all; `sports/f1/predictions/data_source.py` fetches live | None found. | n/a | **EXCLUDE from training data.** |

**Honest conclusion:** this repository currently supports genuine multi-sport
*architecture* (a shared schema, adapters, and a shared checkpoint can
legitimately span 5 sports), but genuine multi-sport *training data* is
effectively MLB (large) + NFL (small) only. NBA/golf/F1 adapters are real and
satisfy the `SportAdapter` contract, but are excluded from DERIVE/SELECT/TEST
training because there is no settled, dated, per-observation outcome ledger
to leak-check and split chronologically. This directly shapes which of the
mission's pre-authorized honest outcome labels are reachable — see
`FINAL_REPORT.md` for the actual final call.

## 2. Existing validated feature generation (do not re-derive)

- MLB: `sports/mlb/predictions/inference/`, `sports/mlb/predictions/odds/`,
  `sports/mlb/research/{h_over_ranker,joint_position_builder_v2,parlay_certification_v2}/`
  — already-validated pregame feature construction, leakage-safe cumulative
  starter/bullpen ERA modeling (per project history), joint Monte Carlo game
  simulation, walk-forward backtesting, calibration ledgers.
- NFL: `sports/nfl/parlay_v2/calibration/{schema.py,pair_schema.py}`,
  `sports/nfl/research/parlay_certification_v2/`.
- NBA: `sports/nba/analytics/advantage_routing/`, `sports/nba/analytics/features/`,
  `sports/nba/analytics/schema/vector_schema.py`.
- Golf/F1: `sports/golf/parlay_v2/calibration/schema.py`,
  `sports/f1/predictions/model.py` (self-contained per-sport models, no
  cross-sport feature registry).

The universal `feature_registry.json` (step 4) reuses these existing pregame
feature definitions and their existing leakage discipline rather than
re-deriving feature semantics from raw sources.

## 3. Outer betting-certification system (must remain untouched by the model)

`sports/mlb/research/parlay_certification_v2/` implements `PolicyStatus`,
`state_machine.py`, `theory.py`, `world_certificate.py`,
`prospective_boundary.py`, `settle_evidence.py` — the real
OBSERVE→prospective-evidence→certify pipeline referred to in the mission
spec as the system the universal model and DRM controller must never
optimize against or backpropagate into. This survey confirms it exists and
is MLB-specific today; the universal model integrates as a **shadow
predictor** alongside it (never replacing it, never feeding it a gradient
signal), per mission section 31/52/54.

## 4. Environment

- Python 3.11.15, pandas 3.0.5, numpy 2.4.6, pyarrow 25.0.1, scikit-learn 1.9.0.
- `torch` was not preinstalled; installed CPU-only build (`torch==2.13.0+cpu`)
  for this work. **No GPU is available** (`torch.cuda.is_available() == False`,
  4 CPU cores, 15 GB RAM). All training in this build is necessarily
  reduced-scale; see `FINAL_REPORT.md` for what was and was not executed at
  full scale, per mission section 59 ("do not fabricate training
  completion").
