# NBA Conditional Chain V1.1

This package implements the report's NBA research architecture without changing the existing predictor or publishing policy.

## Frozen contracts

- Selector: `ROBUST_STATE_INTERSECTION_Q25_V1`
- Allocation representation: `NBA_ALLOCATION_PATH_V1_1_FROZEN`
- Chain policy: `NBA_CONDITIONAL_CHAIN_V0_SHADOW`
- Survival policy: `NBA_RECENT_REGIME_SURVIVAL_V1_SHADOW`
- Representation unit: one event/team/market
- Statistical unit: one game event for path confirmation and one final slate decision for parlay evaluation

The selector corrects UNDER probability semantics, uses the prior 20 date-safe outcomes against the current exact line, applies a Jeffreys 10% lower credible bound, and keeps the exact four-leg publication floor from the project report.

The path builder uses player-points allocation at `T-240`, `T-120`, `T-60`, `T-30`, and `T-5` minutes. A quote must be observed no later than its checkpoint and be at most 20 minutes old. Every player coordinate requires at least two independent pricing engines. Pregame team identity is mandatory and cannot be repaired from outcomes.

## Confirmation rule

T3 compares identical `StandardScaler + Ridge(alpha=1)` models. The endpoint model receives closing state only; the path model receives closing state plus trajectory features. Both are evaluated with chronological expanding windows.

Path information passes only at the frozen 20/30/50-event checkpoints when:

```text
one-sided paired-bootstrap LCB(mean endpoint MAE - path MAE) > 0.005
and
one-sided sign-flip p < 0.0167
```

The per-checkpoint alpha is `0.05 / 3` so the frozen 20/30/50-event looks retain a familywise 5% error budget.

Passing T3 does not authorize a parlay. It only opens chain-policy development. The chain resolver remains shadow-only until a separately frozen policy passes prospective validation.

The downstream extension model is trained only on earlier final decisions whose preceding prefix survived. It estimates each extension conditionally, but its evidence remains clustered and reported at one final slate decision, not at the generated-combination level.

## Parlay hit-rate finding

The recovered 53,957-row, 130-date ledger reproduces the report's final research block exactly:

- frozen four-leg control: 13/27 (48.15%);
- frozen control legs: 89/108 (82.41%);
- independence benchmark: `0.8241 ** 4 = 46.13%`;
- every one of the 27 slates contains at least four winners in the frozen top-10 reservoir.

The four-leg result is therefore close to what its marginal leg rate implies. The candidate reservoir contains enough winners, but the static features tested so far do not identify the compatible chain reliably. A date-safe role/allocation reranker improved its calibration block and then regressed to 10/27 on the reported block, so it is rejected.

A same-version rank-reliability policy, frozen only on the 34 earlier published slates, selected ranks 1 and 4 as a two-leg research core. It hit 21/27 (77.78%) on the later block. It remains shadow-only: applying those ranks to the different April predictor produced only 6/17, proving that rank reliability cannot be transferred across model or selector versions. Each exact version must earn its own core policy and prospective certificate.

The replacement survival policy does not use rank or model-version identity as a feature. It adjusts the frozen robust score with a date-safe 30-day market/side posterior, shrunk toward the recent overall hit rate with prior strength 20. It then evaluates the full combination set using an independence reference and a Frechet lower reference. Two legs are the primary board, three legs are a separately measured extension, and four legs are rejected.

Across the 41-slate expanding replay plus the 17-slate predictor-transfer block:

- survival pair: 40/58 (68.97%) versus 36/58 (62.07%) for the ordinary top-two control;
- April transfer pair: 12/17 (70.59%) versus 9/17 for top two and 6/17 for the rejected fixed-rank core;
- three-leg extension: 33/58 (56.90%) versus 29/58 (50.00%), but no transfer-block improvement;
- selective pair at the frozen 0.42 lower-reference floor: 23/31 (74.19%) at 53.45% slate coverage, versus 19/31 (61.29%) for the control on the same action slates.

The selective paired result has `p=0.0625`, so it is promising but not confirmation-grade. These windows and synthetic thresholds have been repeatedly inspected. The policy remains shadow-only and must reproduce prospectively on executable prices before it can replace publication logic.

A hierarchical pair-lift tree, forced game diversification, and a generic residual Gaussian copula were also tested and rejected. The pair tree regressed the earlier holdout, diversification did not transfer, and the strongest pooled relation correlation across 802 prior games was only 0.051. Their results are retained in `artifacts/survival_replay/rejected_joint_models.json`.

The historical Q25 ledger is not executable market evidence. Only 13.0% of the reported control lines are integers or half-points, every row uses an assumed `-110`, and the rows have no book, quote timestamp, or raw-source hash. It is labeled `SYNTHETIC_THRESHOLD_HISTORY` and cannot authorize publication.

Publication now requires all of the following:

- the exact candidate policy version matches an active prospective certificate;
- the path representation has independently passed its incremental-value test;
- every candidate carries a fresh book quote, executable decimal price, timestamp, parser version, and raw-source hash;
- candidate lineup, player, identity, support, feature-cutoff, model-version, and exposure gates pass;
- minimum prospective action-slate, selection, and coverage evidence is satisfied;
- staking remains disabled independently of candidate publication.

This prevents a high synthetic hit rate or a high model score from being presented as a validated betting parlay.

## Commands

```powershell
python -m sports.nba.conditional_chain.cli repository-audit `
  --output sports/nba/conditional_chain/artifacts/repository_path_audit.json

python -m sports.nba.conditional_chain.cli build-dataset `
  --quotes path/to/repeated_quotes.csv `
  --outcomes path/to/independent_outcomes.csv `
  --output-dir out/nba_allocation_path_v1_1

python -m sports.nba.conditional_chain.cli confirm `
  --settled-features out/nba_allocation_path_v1_1/allocation_path_settled_player_features.csv `
  --output-dir out/nba_allocation_path_v1_1/confirmation

python -m sports.nba.conditional_chain.cli synthetic-audit `
  --output sports/nba/conditional_chain/artifacts/synthetic_null_power_audit.json

python -m sports.nba.conditional_chain.cli backtest-selector `
  --candidate-pool sports/validation/validation_recent_pool_selector_20260406_20260430_rows.csv `
  --data-proc-dir sports/nba/predictions/Player-Predictor/Data-Proc `
  --output-dir sports/nba/conditional_chain/artifacts/selector_replay

python -m sports.nba.conditional_chain.cli backtest-master `
  --master-ledger C:/Users/jcthi/Code/Predictor/backtest/data/backtest_master_dataset.csv `
  --holdout-start 2026-02-11 `
  --output-dir sports/nba/conditional_chain/artifacts/full_replay

python -m sports.nba.conditional_chain.cli backtest-survival `
  --research-reservoir sports/nba/conditional_chain/artifacts/full_replay/frozen_reservoir_replay.csv `
  --transfer-candidate-pool sports/validation/validation_recent_pool_selector_20260406_20260430_rows.csv `
  --data-proc-dir sports/nba/predictions/Player-Predictor/Data-Proc `
  --warmup-slates 20 `
  --output-dir sports/nba/conditional_chain/artifacts/survival_replay
```

The documented `gs://nba-scraped-data/odds-api/player-props-history` archive is a supported upstream shape once exported to CSV or Parquet, but it returns HTTP 403 without project credentials. The committed repository sequence is audited honestly and is not promoted to confirmation evidence when fixed-checkpoint, identity, or quote-provenance gates fail.
