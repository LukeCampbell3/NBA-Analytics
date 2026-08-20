# NBA Conditional Chain V1.1

This package implements the report's NBA research architecture without changing the existing predictor or publishing policy.

## Frozen contracts

- Selector: `ROBUST_STATE_INTERSECTION_Q25_V1`
- Allocation representation: `NBA_ALLOCATION_PATH_V1_1_FROZEN`
- Chain policy: `NBA_CONDITIONAL_CHAIN_V0_SHADOW`
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
```

The documented `gs://nba-scraped-data/odds-api/player-props-history` archive is a supported upstream shape once exported to CSV or Parquet, but it is not anonymously readable. The committed repository sequence is audited honestly and is not promoted to confirmation evidence when fixed-checkpoint or identity gates fail.
