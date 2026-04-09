# NFL Replication Plan

## Objective
Replicate the NBA player-prop modeling workflow for NFL while keeping the same conceptual system:
- enrichment data collection
- market line ingestion + normalization
- processed per-player feature files
- model training/inference on top of processed data

The feature space is NFL-specific, so model artifacts and training configs must be rebuilt for NFL.

## What Is Implemented In This Pass
1. `scripts/fetch_nfl_enrichment.py`
- Pulls season-scoped NFL weekly player stats, schedules, next-gen splits, and optional roster/depth tables.
- Writes `data copy/raw/nfl_enrichment/season=<YEAR>/*.parquet` with manifest tracking.

2. `scripts/fetch_nfl_market_props.py`
- Pulls Odds API NFL props or ingests snapshot files.
- Writes normalized long/wide snapshots and append-only history under `data copy/raw/market_odds/nfl`.
- Uses a stable NFL market contract centered on:
  - `PASS_YDS`
  - `RUSH_YDS`
  - `REC_YDS`

3. `scripts/update_nfl_processed_data.py`
- Builds `Data-Proc-NFL/<Player>/<season>_processed_processed.csv`.
- Merges weekly stats + schedule context + next-gen metrics.
- Computes rolling/lag/context features and market-gap fields.
- Applies synthetic baseline fallback for missing market lines with explicit `Market_Source_*`.

## Current NFL Data Contract (Processed Files)
Primary targets:
- `PASS_YDS`, `RUSH_YDS`, `REC_YDS`

Key context:
- player/team/opponent ids and game context
- rest features, home/away, calendar encodings
- opponent rolling allowed-yard features
- optional next-gen metrics

Market contract:
- real lines, synthetic fallback lines, source tags
- books/prices/std metadata
- market gap columns for each target

## Next Build Steps
1. NFL trainer + inference scaffolding
- clone `training`/`inference` structure to NFL-specific modules
- wire to `Data-Proc-NFL` and NFL targets

2. Feature schema formalization
- freeze required NFL columns and metadata schema signature
- add contract validation on train/inference boundaries

3. Historical market alignment (NFL)
- add NFL equivalent of historical market alignment/backfill flow
- ensure training windows have stable market coverage provenance

4. Evaluation + policy adaptation
- port board-selection logic to NFL stat families and cadence
- recalibrate edge thresholds for lower game volume / positional sparsity
