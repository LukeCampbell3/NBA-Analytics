# MLB Pipeline

MLB-native data collection, feature engineering, validation, and training stack.
The trainer enforces pregame-safe features only and compares ML models against rolling baselines.

## Layout

- `schema/mlb_native_player_schema_v1.json`: machine-readable MLB contract.
- `MLB_MODEL_DATA_REQUIREMENTS.md`: human-readable requirements.
- `data/raw/`: collected raw game logs.
- `data/raw/market_odds/mlb/`: raw + normalized MLB betting line snapshots.
- `data/processed/`: per-player processed files + aggregate role files.
- `scripts/collect_mlb_data.py`: collect raw logs from MLB Stats API.
- `scripts/collect_mlb_market_lines.py`: collect current MLB player props from `odds-api.io` or `SportsGameOdds`.
- `scripts/run_daily_mlb_market_lines.py`: pull daily lines, compare providers, and publish the best file as `latest_player_props_wide.csv`.
- `scripts/build_mlb_features.py`: build processed training features.
- `scripts/validate_mlb_processed_contract.py`: validate processed contract.
- `scripts/train_mlb_models.py`: train hitter and pitcher models.
- `scripts/run_mlb_pipeline.py`: orchestrate full pipeline.

## Quick Start

```bash
python sports/mlb/scripts/collect_mlb_data.py --start-date 2026-03-20 --end-date 2026-04-08
python sports/mlb/scripts/build_mlb_features.py --season 2026
python sports/mlb/scripts/validate_mlb_processed_contract.py
python sports/mlb/scripts/train_mlb_models.py --season 2026 --min-rows 200 --allow-synthetic-market-only
```

Collect market lines:

```bash
# Free Covers scrape for current MLB props
python sports/mlb/scripts/collect_mlb_market_lines.py --provider covers --event-date 2026-04-09

# odds-api.io single-provider pull
$env:ODDS_API_IO_KEY="YOUR_API_KEY"
python sports/mlb/scripts/collect_mlb_market_lines.py --provider odds_api_io --event-date 2026-04-09

# SportsGameOdds single-provider pull
$env:SPORTSGAMEODDS_API_KEY="YOUR_API_KEY"
python sports/mlb/scripts/collect_mlb_market_lines.py --provider sportsgameodds --event-date 2026-04-09
```

Daily current-line workflow:

```bash
# Free-first daily workflow
python sports/mlb/scripts/run_daily_mlb_market_lines.py --providers covers --event-date 2026-04-09

# Mixed provider workflow
$env:ODDS_API_IO_KEY="YOUR_API_KEY"
$env:SPORTSGAMEODDS_API_KEY="YOUR_API_KEY"
python sports/mlb/scripts/run_daily_mlb_market_lines.py --providers covers,odds_api_io,sportsgameodds --event-date 2026-04-09
```

Build features with real market lines:

```bash
python sports/mlb/scripts/build_mlb_features.py --season 2026 --market-file sports/mlb/data/raw/market_odds/mlb/latest_player_props_wide.csv
```

Train only when matched real market lines exist in the processed training rows:

```bash
python sports/mlb/scripts/train_mlb_models.py --season 2026 --min-rows 200 --min-real-market-rows 1 --min-real-market-dates 1
```

Full orchestrator:

```bash
python sports/mlb/scripts/run_mlb_pipeline.py --start-date 2026-03-20 --end-date 2026-04-08 --min-train-rows 200 --allow-synthetic-market-only
```

Short-range smoke test (small date windows):

```bash
python sports/mlb/scripts/run_mlb_pipeline.py --start-date 2026-04-01 --end-date 2026-04-03 --min-processed-rows 1 --min-train-rows 200
```

Pipeline with market file:

```bash
python sports/mlb/scripts/run_mlb_pipeline.py --start-date 2026-03-20 --end-date 2026-04-08 --market-file sports/mlb/data/raw/market_odds/mlb/latest_player_props_wide.csv --min-train-rows 200 --min-real-market-rows 1 --min-real-market-dates 1
```

Notes:

- `covers` is the default provider in this repo now, because it works without an API key and returns current MLB props from public matchup pages.
- `odds-api.io` remains available, but its free tier is much more restrictive and may require bookmaker-selection management.
- `SportsGameOdds` is also wired and useful as a comparison feed; the daily runner writes provider-level manifests plus `provider_comparison_latest.json`.
- Free-provider support here is focused on current lines. Historical prop archiving still needs you to save snapshots over time.
- Training now checks for matched real market-line overlap in the processed rows. A market file dated outside the training window will not satisfy that gate.
- `--allow-synthetic-market-only` is available for smoke tests and debugging, but it bypasses the real-line requirement on purpose.
